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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/DialectConversion.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithDialect.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithOps.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithTypes.h"
#include "prime_ir/Dialect/Ring/IR/RingOps.h"
#include "prime_ir/Dialect/Ring/IR/RingTypes.h"
#include "prime_ir/Utils/ConversionUtils.h"

namespace mlir::prime_ir::ring {

#define GEN_PASS_DEF_RINGTOMODARITH
#include "prime_ir/Dialect/Ring/Conversions/RingToModArith/RingToModArith.h.inc"

namespace {

// The RNS ring lowers to [L, N] raw i64 residues (limb-major).
static RankedTensorType convertRqType(RqType type) {
  int64_t l = type.getModuli().asArrayRef().size();
  int64_t n = type.getRingDegree().getValue().getSExtValue();
  return RankedTensorType::get({l, n}, IntegerType::get(type.getContext(), 64));
}

// a^{-1} mod m via extended Euclid (m fits 64 bits; __int128 avoids overflow).
static uint64_t modInverse(uint64_t a, uint64_t m) {
  __int128 t = 0, newt = 1;
  __int128 r = (__int128)m, newr = (__int128)(a % m);
  while (newr != 0) {
    __int128 q = r / newr;
    __int128 tmp = t - q * newt;
    t = newt;
    newt = tmp;
    tmp = r - q * newr;
    r = newr;
    newr = tmp;
  }
  if (t < 0)
    t += m;
  return static_cast<uint64_t>(t);
}

class RingToModArithTypeConverter : public TypeConverter {
public:
  explicit RingToModArithTypeConverter(MLIRContext *ctx) {
    addConversion([](Type t) { return t; });
    addConversion([](RqType t) -> Type { return convertRqType(t); });
  }
};

struct ConvertBaseConvert : public OpConversionPattern<BaseConvertOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BaseConvertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    MLIRContext *ctx = getContext();
    auto inTy = cast<RqType>(op.getInput().getType());
    auto outTy = cast<RqType>(op.getOutput().getType());
    ArrayRef<int64_t> inM = inTy.getModuli().asArrayRef();
    ArrayRef<int64_t> outM = outTy.getModuli().asArrayRef();
    int64_t n = inTy.getRingDegree().getValue().getSExtValue();
    unsigned L = inM.size(), Lp = outM.size();

    // CRT fast-basis-extension constants (host bignum for Q = prod q_i):
    //   yHatInv_i = (Q/q_i)^{-1} mod q_i,  table_ij = (Q/q_i) mod p_j.
    unsigned qw = 64 * (L + 2);
    APInt Q(qw, 1);
    SmallVector<APInt> qWide;
    for (int64_t q : inM) {
      APInt qi(qw, static_cast<uint64_t>(q));
      qWide.push_back(qi);
      Q *= qi;
    }
    SmallVector<uint64_t> yHatInv(L);
    SmallVector<SmallVector<uint64_t>> table(L, SmallVector<uint64_t>(Lp));
    for (unsigned i = 0; i < L; ++i) {
      APInt qHat = Q.udiv(qWide[i]);
      yHatInv[i] = modInverse(qHat.urem(qWide[i]).getZExtValue(),
                              static_cast<uint64_t>(inM[i]));
      for (unsigned j = 0; j < Lp; ++j)
        table[i][j] =
            qHat.urem(APInt(qw, static_cast<uint64_t>(outM[j]))).getZExtValue();
    }

    Type i64 = IntegerType::get(ctx, 64);
    auto rowI64 = RankedTensorType::get({n}, i64);
    // Residues stored as `mod_arith.int<q : i64>` (i64 storage -> Barrett).
    auto maTensor = [&](int64_t q) {
      return RankedTensorType::get(
          {n}, mod_arith::ModArithType::get(ctx, IntegerAttr::get(i64, q),
                                            /*isMontgomery=*/false));
    };
    auto maConst = [&](int64_t q, uint64_t v) -> Value {
      return mod_arith::ConstantOp::create(
          b, maTensor(q), DenseIntElementsAttr::get(rowI64, APInt(64, v)));
    };
    SmallVector<OpFoldResult> strides(2, b.getIndexAttr(1));
    auto sliceOffsets = [&](int64_t row) -> SmallVector<OpFoldResult> {
      return {b.getIndexAttr(row), b.getIndexAttr(0)};
    };
    SmallVector<OpFoldResult> sizes = {b.getIndexAttr(1), b.getIndexAttr(n)};

    Value in = adaptor.getInput(); // tensor<LxNxi64>

    // y_i = (x_i * yHatInv_i) mod q_i, held as raw i64 (< q_i).
    SmallVector<Value> y(L);
    for (unsigned i = 0; i < L; ++i) {
      Value xi = tensor::ExtractSliceOp::create(b, rowI64, in, sliceOffsets(i),
                                                sizes, strides);
      Value xiMa = mod_arith::BitcastOp::create(b, maTensor(inM[i]), xi);
      Value yiMa =
          mod_arith::MulOp::create(b, xiMa, maConst(inM[i], yHatInv[i]));
      y[i] = mod_arith::BitcastOp::create(b, rowI64, yiMa);
    }

    // out_j = (sum_i y_i * table_ij) mod p_j, all in mod_arith<p_j>. Barrett
    // reduces each product, so reinterpreting a raw y_i into mod_arith<p_j>
    // (possibly non-canonical) still yields the correct residue.
    auto outTensorTy =
        RankedTensorType::get({static_cast<int64_t>(Lp), n}, i64);
    Value result = tensor::EmptyOp::create(b, outTensorTy, ValueRange{});
    for (unsigned j = 0; j < Lp; ++j) {
      Value acc;
      for (unsigned i = 0; i < L; ++i) {
        Value yiPj = mod_arith::BitcastOp::create(b, maTensor(outM[j]), y[i]);
        Value term =
            mod_arith::MulOp::create(b, yiPj, maConst(outM[j], table[i][j]));
        acc =
            i == 0 ? term : mod_arith::AddOp::create(b, acc, term).getResult();
      }
      Value outj = mod_arith::BitcastOp::create(b, rowI64, acc);
      result = tensor::InsertSliceOp::create(b, outj, result, sliceOffsets(j),
                                             sizes, strides);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

// rescale: exact division by the trailing modulus q_last. Per output limb i,
// out_i = (x_i - x_last) * q_last^{-1} mod q_i (mod_arith.mul Barrett-reduces
// the x_last*qInv product, so the non-canonical x_last reinterpret is fine).
struct ConvertRescale : public OpConversionPattern<RescaleOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RescaleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    MLIRContext *ctx = op.getContext();
    auto inTy = cast<RqType>(op.getInput().getType());
    ArrayRef<int64_t> inM = inTy.getModuli().asArrayRef();
    int64_t n = inTy.getRingDegree().getValue().getSExtValue();
    unsigned L = inM.size();
    uint64_t qLast = static_cast<uint64_t>(inM[L - 1]);

    Type i64 = IntegerType::get(ctx, 64);
    auto rowI64 = RankedTensorType::get({n}, i64);
    auto maTensor = [&](int64_t q) {
      return RankedTensorType::get(
          {n}, mod_arith::ModArithType::get(ctx, IntegerAttr::get(i64, q),
                                            /*isMontgomery=*/false));
    };
    auto maConst = [&](int64_t q, uint64_t v) -> Value {
      return mod_arith::ConstantOp::create(
          b, maTensor(q), DenseIntElementsAttr::get(rowI64, APInt(64, v)));
    };
    SmallVector<OpFoldResult> strides(2, b.getIndexAttr(1));
    SmallVector<OpFoldResult> sizes = {b.getIndexAttr(1), b.getIndexAttr(n)};
    auto offs = [&](int64_t r) -> SmallVector<OpFoldResult> {
      return {b.getIndexAttr(r), b.getIndexAttr(0)};
    };
    Value in = adaptor.getInput();
    Value xLast = tensor::ExtractSliceOp::create(b, rowI64, in, offs(L - 1),
                                                 sizes, strides);

    auto outTy = RankedTensorType::get({static_cast<int64_t>(L) - 1, n}, i64);
    Value result = tensor::EmptyOp::create(b, outTy, ValueRange{});
    for (unsigned i = 0; i + 1 < L; ++i) {
      uint64_t qi = static_cast<uint64_t>(inM[i]);
      uint64_t qInv = modInverse(qLast % qi, qi);
      Value xi = tensor::ExtractSliceOp::create(b, rowI64, in, offs(i), sizes,
                                                strides);
      Value t1 = mod_arith::MulOp::create(
          b, mod_arith::BitcastOp::create(b, maTensor(qi), xi),
          maConst(qi, qInv));
      Value t2 = mod_arith::MulOp::create(
          b, mod_arith::BitcastOp::create(b, maTensor(qi), xLast),
          maConst(qi, qInv));
      Value diff = mod_arith::SubOp::create(b, t1, t2);
      Value outi = mod_arith::BitcastOp::create(b, rowI64, diff);
      result = tensor::InsertSliceOp::create(b, outi, result, offs(i), sizes,
                                             strides);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ConvertGadgetDecompose : public OpConversionPattern<GadgetDecomposeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(GadgetDecomposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    int64_t baseBits = op.getBaseBits(), levels = op.getLevels();
    Value in = adaptor.getInput(); // tensor<LxNxi64>
    auto tType = cast<RankedTensorType>(in.getType());
    auto splat = [&](uint64_t v) -> Value {
      return arith::ConstantOp::create(
          b, DenseElementsAttr::get(tType, APInt(64, v)));
    };
    Value mask = splat((static_cast<uint64_t>(1) << baseBits) - 1);
    SmallVector<Value> digits;
    for (int64_t j = 0; j < levels; ++j) {
      Value shifted = arith::ShRUIOp::create(
          b, in, splat(static_cast<uint64_t>(j * baseBits)));
      digits.push_back(arith::AndIOp::create(b, shifted, mask));
    }
    rewriter.replaceOp(op, digits);
    return success();
  }
};

// automorphism sigma_g: X->X^g. Signed static coefficient permutation per limb:
// a_k -> position (g*k mod N), sign -1 iff (g*k mod 2N) >= N. Emitted as a
// scalar permute (extract/insert) + one mod_arith.mul by a +/-1 sign tensor.
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

    // Coefficient basis: pos[k] = destination of coefficient k, neg[k] =
    // whether the X^N = -1 wrap flips its sign. Evaluation basis: slot j
    // evaluates at psi^(2j+1), and sigma_g carries that to psi^(g(2j+1)), so
    // slot j gathers from src[j]. The exponent reduces exactly mod 2N, which is
    // why no sign survives on this side.
    SmallVector<int64_t> pos(n), src(n);
    SmallVector<bool> neg(n, false);
    if (evalBasis) {
      for (int64_t j = 0; j < n; ++j) {
        src[j] = ((gr * (2 * j + 1)) % psiOrder - 1) / 2;
      }
    } else {
      for (int64_t k = 0; k < n; ++k) {
        int64_t dest = (gr * k) % psiOrder;
        pos[k] = dest % n;
        neg[k] = dest >= n;
      }
    }

    Type i64 = IntegerType::get(ctx, 64);
    auto rowI64 = RankedTensorType::get({n}, i64);
    SmallVector<OpFoldResult> strides(2, b.getIndexAttr(1));
    SmallVector<OpFoldResult> sizes = {b.getIndexAttr(1), b.getIndexAttr(n)};
    auto offs = [&](int64_t r) -> SmallVector<OpFoldResult> {
      return {b.getIndexAttr(r), b.getIndexAttr(0)};
    };
    SmallVector<Value> idx(n);
    for (int64_t j = 0; j < n; ++j)
      idx[j] = arith::ConstantIndexOp::create(b, j);
    Value in = adaptor.getInput();

    auto outTy = RankedTensorType::get({static_cast<int64_t>(L), n}, i64);
    Value result = tensor::EmptyOp::create(b, outTy, ValueRange{});
    for (unsigned i = 0; i < L; ++i) {
      uint64_t q = static_cast<uint64_t>(mod[i]);
      Value xi = tensor::ExtractSliceOp::create(b, rowI64, in, offs(i), sizes,
                                                strides);
      if (evalBasis) {
        Value gathered = tensor::EmptyOp::create(b, rowI64, ValueRange{});
        for (int64_t j = 0; j < n; ++j) {
          Value v = tensor::ExtractOp::create(b, xi, idx[src[j]]);
          gathered = tensor::InsertOp::create(b, v, gathered, idx[j]);
        }
        result = tensor::InsertSliceOp::create(b, gathered, result, offs(i),
                                               sizes, strides);
        continue;
      }
      // permute coefficients (no sign yet)
      Value perm = tensor::EmptyOp::create(b, rowI64, ValueRange{});
      SmallVector<APInt> signs(n, APInt(64, 1));
      for (int64_t k = 0; k < n; ++k) {
        Value v = tensor::ExtractOp::create(b, xi, idx[k]);
        perm = tensor::InsertOp::create(b, v, perm, idx[pos[k]]);
        if (neg[k])
          signs[pos[k]] = APInt(64, q - 1); // -1 mod q
      }
      auto maTensor = RankedTensorType::get(
          {n}, mod_arith::ModArithType::get(ctx, IntegerAttr::get(i64, mod[i]),
                                            /*isMontgomery=*/false));
      Value signConst = mod_arith::ConstantOp::create(
          b, maTensor, DenseIntElementsAttr::get(rowI64, signs));
      Value permMa = mod_arith::BitcastOp::create(b, maTensor, perm);
      Value outMa = mod_arith::MulOp::create(b, permMa, signConst);
      Value oi = mod_arith::BitcastOp::create(b, rowI64, outMa);
      result =
          tensor::InsertSliceOp::create(b, oi, result, offs(i), sizes, strides);
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
struct ConvertBinOp : public OpConversionPattern<RingOpT> {
  using OpConversionPattern<RingOpT>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<RingOpT>::OpAdaptor;

  LogicalResult
  matchAndRewrite(RingOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    MLIRContext *ctx = op.getContext();
    auto ty = cast<RqType>(op.getLhs().getType());
    ArrayRef<int64_t> mod = ty.getModuli().asArrayRef();
    int64_t n = ty.getRingDegree().getValue().getSExtValue();
    unsigned L = mod.size();

    Type i64 = IntegerType::get(ctx, 64);
    auto rowI64 = RankedTensorType::get({n}, i64);
    auto maTensor = [&](int64_t q) {
      return RankedTensorType::get(
          {n}, mod_arith::ModArithType::get(ctx, IntegerAttr::get(i64, q),
                                            /*isMontgomery=*/false));
    };
    SmallVector<OpFoldResult> strides(2, b.getIndexAttr(1));
    SmallVector<OpFoldResult> sizes = {b.getIndexAttr(1), b.getIndexAttr(n)};
    auto offs = [&](int64_t r) -> SmallVector<OpFoldResult> {
      return {b.getIndexAttr(r), b.getIndexAttr(0)};
    };
    Value lhs = adaptor.getLhs(), rhs = adaptor.getRhs();

    auto outTy = RankedTensorType::get({static_cast<int64_t>(L), n}, i64);
    Value result = tensor::EmptyOp::create(b, outTy, ValueRange{});
    for (unsigned i = 0; i < L; ++i) {
      Value li = tensor::ExtractSliceOp::create(b, rowI64, lhs, offs(i), sizes,
                                                strides);
      Value ri = tensor::ExtractSliceOp::create(b, rowI64, rhs, offs(i), sizes,
                                                strides);
      Value lm = mod_arith::BitcastOp::create(b, maTensor(mod[i]), li);
      Value rm = mod_arith::BitcastOp::create(b, maTensor(mod[i]), ri);
      Value sm = ModOpT::create(b, lm, rm);
      Value si = mod_arith::BitcastOp::create(b, rowI64, sm);
      result =
          tensor::InsertSliceOp::create(b, si, result, offs(i), sizes, strides);
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
    MLIRContext *ctx = op.getContext();
    auto ty = cast<RqType>(op.getInput().getType());
    ArrayRef<int64_t> mod = ty.getModuli().asArrayRef();
    int64_t n = ty.getRingDegree().getValue().getSExtValue();
    unsigned L = mod.size();

    Type i64 = IntegerType::get(ctx, 64);
    auto rowI64 = RankedTensorType::get({n}, i64);
    SmallVector<OpFoldResult> strides(2, b.getIndexAttr(1));
    SmallVector<OpFoldResult> sizes = {b.getIndexAttr(1), b.getIndexAttr(n)};
    auto offs = [&](int64_t r) -> SmallVector<OpFoldResult> {
      return {b.getIndexAttr(r), b.getIndexAttr(0)};
    };
    Value in = adaptor.getInput();

    auto outTy = RankedTensorType::get({static_cast<int64_t>(L), n}, i64);
    Value result = tensor::EmptyOp::create(b, outTy, ValueRange{});
    for (unsigned i = 0; i < L; ++i) {
      auto maTensor = RankedTensorType::get(
          {n}, mod_arith::ModArithType::get(ctx, IntegerAttr::get(i64, mod[i]),
                                            /*isMontgomery=*/false));
      Value xi = tensor::ExtractSliceOp::create(b, rowI64, in, offs(i), sizes,
                                                strides);
      Value xm = mod_arith::BitcastOp::create(b, maTensor, xi);
      Value rm = ModOpT::create(b, xm);
      Value ri = mod_arith::BitcastOp::create(b, rowI64, rm);
      result =
          tensor::InsertSliceOp::create(b, ri, result, offs(i), sizes, strides);
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
    target.addLegalDialect<mod_arith::ModArithDialect, tensor::TensorDialect>();

    RewritePatternSet patterns(context);
    patterns.add<ConvertBaseConvert, ConvertRescale, ConvertAutomorphism,
                 ConvertGadgetDecompose, ConvertFromTensor, ConvertToTensor,
                 ConvertBinOp<AddOp, mod_arith::AddOp>,
                 ConvertBinOp<SubOp, mod_arith::SubOp>,
                 ConvertBinOp<MulOp, mod_arith::MulOp>,
                 ConvertUnaryOp<NegateOp, mod_arith::NegateOp>>(typeConverter,
                                                                context);
    addStructuralConversionPatterns(typeConverter, patterns, target);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::prime_ir::ring
