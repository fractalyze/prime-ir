/* Copyright 2025 The PrimeIR Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "prime_ir/Dialect/Ring/Conversions/RingToModArith/RingToModArith.h"

#include <cstdint>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/DialectConversion.h"
#include "prime_ir/Dialect/Field/IR/FieldDialect.h"
#include "prime_ir/Dialect/Field/IR/FieldOps.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithDialect.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithOps.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithTypes.h"
#include "prime_ir/Dialect/Ring/IR/RingOps.h"
#include "prime_ir/Dialect/Ring/IR/RingTypes.h"
#include "prime_ir/Utils/APIntUtils.h"
#include "prime_ir/Utils/ConversionUtils.h"

namespace mlir::prime_ir::ring {

#define GEN_PASS_DEF_RINGTOMODARITH
#include "prime_ir/Dialect/Ring/Conversions/RingToModArith/RingToModArith.h.inc"

namespace {

// The RNS ring lowers to [L, N] raw residue words (limb-major).
static RankedTensorType convertRqType(RqType type) {
  int64_t l = type.getModuli().asArrayRef().size();
  int64_t n = type.getRingDegree().getValue().getSExtValue();
  return RankedTensorType::get({l, n}, type.getStorageType());
}

// The lowered ring is a [L, N] residue tensor, so every pattern here slices row
// i out, works in mod_arith<q_i>, and writes the row back. This is that shape,
// held once instead of restated per pattern.
class LimbRows {
public:
  LimbRows(ImplicitLocOpBuilder &b, RqType ty)
      : b(b), word(ty.getStorageType()), width(ty.getStorageWidth()),
        n(ty.getRingDegree().getValue().getSExtValue()),
        rowWord(RankedTensorType::get({n}, word)),
        strides(2, b.getIndexAttr(1)),
        sizes{b.getIndexAttr(1), b.getIndexAttr(n)} {}

  RankedTensorType getRowType() const { return rowWord; }
  int64_t getDegree() const { return n; }

  // Residues are stored canonically; Montgomery form would need the transform
  // around every op, which the ring dialect leaves to its caller.
  RankedTensorType modType(int64_t q) const {
    return RankedTensorType::get(
        {n},
        mod_arith::ModArithType::get(b.getContext(), IntegerAttr::get(word, q),
                                     /*isMontgomery=*/false));
  }
  Value splat(int64_t q, uint64_t v) const {
    return mod_arith::ConstantOp::create(
        b, modType(q), DenseIntElementsAttr::get(rowWord, APInt(width, v)));
  }
  Value asMod(int64_t q, Value rawRow) const {
    return mod_arith::BitcastOp::create(b, modType(q), rawRow);
  }
  // A residue of another modulus only names a mod_arith<q> value once it is
  // below q -- the reducers state canonical operands as a precondition, and the
  // Mersenne fast path in particular subtracts q at most once. The source bound
  // is static, so the divide is emitted only where it can actually fire; a
  // Barrett constant would replace it if this ever showed up hot.
  Value asModCanonical(int64_t q, Value rawRow, uint64_t srcBound) const {
    if (srcBound > static_cast<uint64_t>(q)) {
      rawRow = arith::RemUIOp::create(b, rawRow, wordSplat(q));
    }
    return asMod(q, rawRow);
  }
  Value wordSplat(uint64_t v) const {
    return arith::ConstantOp::create(
        b, DenseIntElementsAttr::get(rowWord, APInt(width, v)));
  }
  Value asWord(Value modRow) const {
    return mod_arith::BitcastOp::create(b, rowWord, modRow);
  }
  Value emptyRows(int64_t rows) const {
    return tensor::EmptyOp::create(b, RankedTensorType::get({rows, n}, word),
                                   ValueRange{});
  }
  Value row(Value rows, int64_t i) const {
    return tensor::ExtractSliceOp::create(b, rowWord, rows, offsets(i), sizes,
                                          strides);
  }
  Value setRow(Value rows, int64_t i, Value r) const {
    return tensor::InsertSliceOp::create(b, r, rows, offsets(i), sizes,
                                         strides);
  }

private:
  SmallVector<OpFoldResult> offsets(int64_t i) const {
    return {b.getIndexAttr(i), b.getIndexAttr(0)};
  }

  ImplicitLocOpBuilder &b;
  Type word;
  unsigned width;
  int64_t n;
  RankedTensorType rowWord;
  SmallVector<OpFoldResult> strides;
  SmallVector<OpFoldResult> sizes;
};

class RingToModArithTypeConverter : public TypeConverter {
public:
  explicit RingToModArithTypeConverter(MLIRContext *ctx) {
    // The catch-all must not swallow a container of rings: a ring already
    // occupies two tensor dimensions, so tensor<k x !ring.rq> would have to
    // become a rank-(k+2) residue tensor, which no pattern here builds. Failing
    // to convert makes the pass say so; the identity rule let it through with
    // the ring type intact and the pass reporting success.
    addConversion([](Type t) -> std::optional<Type> {
      auto shaped = dyn_cast<ShapedType>(t);
      if (shaped && isa<RqType>(shaped.getElementType())) {
        return std::nullopt;
      }
      return t;
    });
    addConversion([](RqType t) -> Type { return convertRqType(t); });
  }
};

struct ConvertBaseConvert : public OpConversionPattern<BaseConvertOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BaseConvertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto inTy = cast<RqType>(op.getInput().getType());
    auto outTy = cast<RqType>(op.getOutput().getType());
    ArrayRef<int64_t> inM = inTy.getModuli().asArrayRef();
    ArrayRef<int64_t> outM = outTy.getModuli().asArrayRef();
    unsigned L = inM.size(), Lp = outM.size();

    // CRT fast-basis-extension constants (host bignum for Q = prod q_i):
    //   yHatInv_i = (Q/q_i)^{-1} mod q_i,  table_ij = (Q/q_i) mod p_j.
    unsigned qw = 64 * (L + 2);
    APInt Q(qw, 1);
    for (int64_t q : inM)
      Q *= APInt(qw, static_cast<uint64_t>(q));
    SmallVector<uint64_t> yHatInv(L);
    SmallVector<SmallVector<uint64_t>> table(L, SmallVector<uint64_t>(Lp));
    for (unsigned i = 0; i < L; ++i) {
      // The single-word divisor overloads keep this off APInt's multi-word
      // division path, which the Lp inner iterations would otherwise pay for.
      auto qi = static_cast<uint64_t>(inM[i]);
      APInt qHat = Q.udiv(qi);
      APInt qiWide(qw, qi);
      yHatInv[i] = multiplicativeInverse(APInt(qw, qHat.urem(qi)), qiWide)
                       .getZExtValue();
      for (unsigned j = 0; j < Lp; ++j)
        table[i][j] = qHat.urem(static_cast<uint64_t>(outM[j]));
    }

    LimbRows rows(b, inTy);
    Value in = adaptor.getInput(); // tensor<LxNxiW>

    // y_i = (x_i * yHatInv_i) mod q_i, held as a raw word (< q_i).
    SmallVector<Value> y(L);
    for (unsigned i = 0; i < L; ++i) {
      Value xiMa = rows.asMod(inM[i], rows.row(in, i));
      y[i] = rows.asWord(
          mod_arith::MulOp::create(b, xiMa, rows.splat(inM[i], yHatInv[i])));
    }

    // out_j = (sum_i y_i * table_ij) mod p_j, all in mod_arith<p_j>. y_i is a
    // residue mod q_i, so it needs reducing before it can stand for a value in
    // mod_arith<p_j> whenever q_i can exceed p_j.
    Value result = rows.emptyRows(Lp);
    for (unsigned j = 0; j < Lp; ++j) {
      Value acc;
      for (unsigned i = 0; i < L; ++i) {
        Value term = mod_arith::MulOp::create(
            b,
            rows.asModCanonical(outM[j], y[i], static_cast<uint64_t>(inM[i])),
            rows.splat(outM[j], table[i][j]));
        acc =
            i == 0 ? term : mod_arith::AddOp::create(b, acc, term).getResult();
      }
      result = rows.setRow(result, j, rows.asWord(acc));
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

// rescale: exact division by the trailing modulus q_last. Per output limb i,
// out_i = (x_i - x_last) * q_last^{-1} mod q_i, where x_last is the dropped
// limb's residue reduced into q_i.
struct ConvertRescale : public OpConversionPattern<RescaleOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RescaleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto inTy = cast<RqType>(op.getInput().getType());
    ArrayRef<int64_t> inM = inTy.getModuli().asArrayRef();
    unsigned L = inM.size();
    uint64_t qLast = static_cast<uint64_t>(inM[L - 1]);

    LimbRows rows(b, inTy);
    Value in = adaptor.getInput();
    Value xLast = rows.row(in, L - 1);

    Value result = rows.emptyRows(static_cast<int64_t>(L) - 1);
    for (unsigned i = 0; i + 1 < L; ++i) {
      auto qi = static_cast<uint64_t>(inM[i]);
      unsigned qw = inTy.getStorageWidth();
      uint64_t qInv =
          multiplicativeInverse(APInt(qw, qLast % qi), APInt(qw, qi))
              .getZExtValue();
      Value qInvC = rows.splat(qi, qInv);
      Value t1 =
          mod_arith::MulOp::create(b, rows.asMod(qi, rows.row(in, i)), qInvC);
      Value t2 = mod_arith::MulOp::create(
          b, rows.asModCanonical(qi, xLast, qLast), qInvC);
      result = rows.setRow(result, i,
                           rows.asWord(mod_arith::SubOp::create(b, t1, t2)));
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

// automorphism sigma_g: X->X^g. Signed static coefficient permutation per limb:
// a_k -> position (g*k mod N), sign -1 iff (g*k mod 2N) >= N. The permutation
// is a constant index table gathered in one linalg.generic, then one
// mod_arith.mul by a +/-1 sign tensor. Unrolling it into per-coefficient
// extract/insert would emit O(L*N) ops, and a CKKS-sized N = 2^15 makes that
// unusable.
struct ConvertAutomorphism : public OpConversionPattern<AutomorphismOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AutomorphismOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    MLIRContext *ctx = op.getContext();
    auto ty = cast<RqType>(op.getInput().getType());
    ArrayRef<int64_t> mod = ty.getModuli().asArrayRef();
    int64_t n = ty.getRingDegree().getValue().getSExtValue();
    unsigned L = mod.size();
    int64_t g = op.getExponent();
    int64_t psiOrder = 2 * n;

    // sigma_g depends only on g mod 2N, and reducing first keeps the products
    // below from overflowing for a large exponent.
    int64_t gr = g % psiOrder;
    bool evalBasis = ty.isEval();

    // Both bases end up as "output j reads input src[j]", which is what a
    // gather wants. Evaluation basis: slot j evaluates at psi^(2j+1), and
    // sigma_g carries that to psi^(g(2j+1)); the exponent reduces exactly mod
    // 2N, which is why no sign survives on this side. Coefficient basis: the
    // rule is stated forwards (coefficient k lands at (g*k) mod N, negated on
    // the X^N = -1 wrap), so it is inverted here -- sigma_g is a bijection, so
    // every output position is written exactly once.
    SmallVector<int64_t> src(n);
    SmallVector<bool> negAt(n, false);
    if (evalBasis) {
      for (int64_t j = 0; j < n; ++j) {
        src[j] = ((gr * (2 * j + 1)) % psiOrder - 1) / 2;
      }
    } else {
      for (int64_t k = 0; k < n; ++k) {
        int64_t dest = (gr * k) % psiOrder;
        src[dest % n] = k;
        negAt[dest % n] = dest >= n;
      }
    }

    LimbRows rows(b, ty);
    Value in = adaptor.getInput();

    // One index table and one gather for the whole [L, N] tensor: the
    // permutation is the same in every limb, and a single 2-D loop nest tiles
    // and fuses where L separate 1-D nests would not. The table is i32 because
    // the degree is a power of two far below 2^31.
    auto idxTensorTy = RankedTensorType::get({n}, b.getI32Type());
    SmallVector<int32_t> srcIdx(src.begin(), src.end());
    Value gatherIdx = arith::ConstantOp::create(
        b, DenseIntElementsAttr::get(idxTensorTy, ArrayRef<int32_t>(srcIdx)));

    AffineExpr d0, d1;
    bindDims(ctx, d0, d1);
    SmallVector<AffineMap> maps = {AffineMap::get(2, 0, {d1}, ctx),
                                   AffineMap::getMultiDimIdentityMap(2, ctx)};
    SmallVector<utils::IteratorType> iters(2, utils::IteratorType::parallel);
    Value gathered =
        linalg::GenericOp::create(
            b, TypeRange{convertRqType(ty)}, ValueRange{gatherIdx},
            ValueRange{rows.emptyRows(L)}, maps, iters,
            [&](OpBuilder &nested, Location loc, ValueRange args) {
              ImplicitLocOpBuilder lb(loc, nested);
              Value limb = linalg::IndexOp::create(lb, 0);
              Value pos =
                  arith::IndexCastOp::create(lb, lb.getIndexType(), args[0]);
              Value v =
                  tensor::ExtractOp::create(lb, in, ValueRange{limb, pos});
              linalg::YieldOp::create(lb, ValueRange{v});
            })
            .getResult(0);

    if (evalBasis) {
      rewriter.replaceOp(op, gathered);
      return success();
    }

    // The X^N = -1 wrap negates a fixed set of output positions, and that set
    // is the same in every limb -- only the encoding of -1 would differ. So the
    // mask is one shared i1 tensor and each limb selects between its gathered
    // row and the negation of it, rather than carrying a dense +/-1 constant of
    // its own.
    Value wrapMask = arith::ConstantOp::create(
        b, DenseElementsAttr::get(RankedTensorType::get({n}, b.getI1Type()),
                                  ArrayRef<bool>(negAt)));
    Value result = rows.emptyRows(L);
    for (unsigned i = 0; i < L; ++i) {
      Value rowRaw = rows.row(gathered, i);
      Value negated = rows.asWord(
          mod_arith::NegateOp::create(b, rows.asMod(mod[i], rowRaw)));
      result = rows.setRow(
          result, i, arith::SelectOp::create(b, wrapMask, negated, rowRaw));
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

// Componentwise same-basis binary op: per limb i, ModOpT on mod_arith<q_i>.
// Multiplication qualifies because the verifier admits only eval-basis
// operands, where CRT has already diagonalised the ring; reaching that basis is
// the caller's business, not this pass's.
template <typename RingOpT, typename ModOpT>
struct ConvertLimbwiseBinOp : public OpConversionPattern<RingOpT> {
  using OpConversionPattern<RingOpT>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<RingOpT>::OpAdaptor;

  LogicalResult
  matchAndRewrite(RingOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto ty = cast<RqType>(op.getLhs().getType());
    ArrayRef<int64_t> mod = ty.getModuli().asArrayRef();
    unsigned L = mod.size();

    LimbRows rows(b, ty);
    Value lhs = adaptor.getLhs(), rhs = adaptor.getRhs();
    Value result = rows.emptyRows(L);
    for (unsigned i = 0; i < L; ++i) {
      Value sm = ModOpT::create(b, rows.asMod(mod[i], rows.row(lhs, i)),
                                rows.asMod(mod[i], rows.row(rhs, i)));
      result = rows.setRow(result, i, rows.asWord(sm));
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

// Componentwise same-basis unary op: per limb i, ModOpT on mod_arith<q_i>.
// Basis-agnostic — negation commutes with the CRT map, so it needs no check.
template <typename RingOpT, typename ModOpT>
struct ConvertUnaryOp : public OpConversionPattern<RingOpT> {
  using OpConversionPattern<RingOpT>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<RingOpT>::OpAdaptor;

  LogicalResult
  matchAndRewrite(RingOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto ty = cast<RqType>(op.getInput().getType());
    ArrayRef<int64_t> mod = ty.getModuli().asArrayRef();
    unsigned L = mod.size();

    LimbRows rows(b, ty);
    Value in = adaptor.getInput();
    Value result = rows.emptyRows(L);
    for (unsigned i = 0; i < L; ++i) {
      Value rm = ModOpT::create(b, rows.asMod(mod[i], rows.row(in, i)));
      result = rows.setRow(result, i, rows.asWord(rm));
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

// from_tensor / to_tensor bridge the (converted) tensor representation and the
// ring type; after conversion both sides are the same tensor, so they fold
// away.
struct ConvertFromTensor : public OpConversionPattern<FromTensorOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(FromTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};

// The limb bridge is the one place this pass touches the field dialect: a limb
// arrives typed `!field.pf<q_i>` so its modulus is checked against the ring,
// and reaching the [L, N] residue tensor means dropping to that field's storage
// integers. field.bitcast is exactly that reinterpret, and FieldToModArith
// takes it from there.
struct ConvertFromLimbs : public OpConversionPattern<FromLimbsOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(FromLimbsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto ty = cast<RqType>(op.getOutput().getType());
    int64_t n = ty.getRingDegree().getValue().getSExtValue();
    auto L = static_cast<int64_t>(ty.getModuli().size());

    Type word = ty.getStorageType();
    auto rowWord = RankedTensorType::get({n}, word);
    auto outTy = RankedTensorType::get({L, n}, word);
    SmallVector<OpFoldResult> strides(2, b.getIndexAttr(1));
    SmallVector<OpFoldResult> sizes = {b.getIndexAttr(1), b.getIndexAttr(n)};

    Value result = tensor::EmptyOp::create(b, outTy, ValueRange{});
    for (auto [i, limb] : llvm::enumerate(adaptor.getLimbs())) {
      Value storage = field::BitcastOp::create(b, rowWord, limb);
      SmallVector<OpFoldResult> offs = {b.getIndexAttr(i), b.getIndexAttr(0)};
      result = tensor::InsertSliceOp::create(b, storage, result, offs, sizes,
                                             strides);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ConvertToLimbs : public OpConversionPattern<ToLimbsOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ToLimbsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto ty = cast<RqType>(op.getInput().getType());
    int64_t n = ty.getRingDegree().getValue().getSExtValue();

    Type word = ty.getStorageType();
    auto rowWord = RankedTensorType::get({n}, word);
    SmallVector<OpFoldResult> strides(2, b.getIndexAttr(1));
    SmallVector<OpFoldResult> sizes = {b.getIndexAttr(1), b.getIndexAttr(n)};

    SmallVector<Value> limbs;
    for (auto [i, limbType] : llvm::enumerate(op.getLimbs().getTypes())) {
      SmallVector<OpFoldResult> offs = {b.getIndexAttr(i), b.getIndexAttr(0)};
      Value row = tensor::ExtractSliceOp::create(b, rowWord, adaptor.getInput(),
                                                 offs, sizes, strides);
      limbs.push_back(field::BitcastOp::create(b, limbType, row));
    }
    rewriter.replaceOp(op, limbs);
    return success();
  }
};

struct ConvertToTensor : public OpConversionPattern<ToTensorOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ToTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};

struct RingToModArith : impl::RingToModArithBase<RingToModArith> {
  using RingToModArithBase::RingToModArithBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();
    RingToModArithTypeConverter typeConverter(context);

    ConversionTarget target(*context);
    target.addIllegalDialect<RingDialect>();
    // field is legal only as an exit: the limb bridge emits field.bitcast to
    // reach a limb's storage integers, and FieldToModArith lowers it later.
    target.addLegalDialect<mod_arith::ModArithDialect, tensor::TensorDialect,
                           field::FieldDialect>();

    RewritePatternSet patterns(context);
    patterns.add<ConvertBaseConvert, ConvertRescale, ConvertAutomorphism,
                 ConvertFromTensor, ConvertToTensor, ConvertFromLimbs,
                 ConvertToLimbs, ConvertLimbwiseBinOp<AddOp, mod_arith::AddOp>,
                 ConvertLimbwiseBinOp<SubOp, mod_arith::SubOp>,
                 ConvertLimbwiseBinOp<MulOp, mod_arith::MulOp>,
                 ConvertUnaryOp<NegateOp, mod_arith::NegateOp>>(typeConverter,
                                                                context);
    addStructuralConversionPatterns(typeConverter, patterns, target);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::prime_ir::ring
