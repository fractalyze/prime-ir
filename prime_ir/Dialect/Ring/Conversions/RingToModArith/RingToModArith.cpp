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
#include "prime_ir/Dialect/Field/IR/FieldAttributes.h"
#include "prime_ir/Dialect/Field/IR/FieldDialect.h"
#include "prime_ir/Dialect/Field/IR/FieldOps.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithDialect.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithOps.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithTypes.h"
#include "prime_ir/Dialect/Poly/IR/PolyDialect.h"
#include "prime_ir/Dialect/Poly/IR/PolyOps.h"
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
  return (uint64_t)t;
}

// A primitive `order`-th root of unity mod q, 0 if none (order ∤ q-1). For a
// candidate c, r = c^((q-1)/order) is a root whose order divides `order`; it is
// primitive iff r^(order/2) != 1. Trying c = 2,3,... finds one in a few steps
// (one powmod each) — works for real word-size primes, unlike an O(q) scan.
static uint64_t primitiveNthRoot(int64_t order, uint64_t q) {
  auto powmod = [](unsigned __int128 base, uint64_t e, uint64_t m) -> uint64_t {
    unsigned __int128 r = 1;
    base %= m;
    while (e) {
      if (e & 1)
        r = r * base % m;
      base = base * base % m;
      e >>= 1;
    }
    return (uint64_t)r;
  };
  if (order <= 0 || (q - 1) % (uint64_t)order != 0)
    return 0;
  uint64_t exp = (q - 1) / (uint64_t)order;
  for (uint64_t c = 2; c < q; ++c) {
    uint64_t r = powmod(c, exp, q);
    if (r > 1 && powmod(r, (uint64_t)order / 2, q) != 1)
      return r;
  }
  return 0;
}

static uint64_t powmod64(unsigned __int128 base, uint64_t e, uint64_t m) {
  unsigned __int128 r = 1;
  base %= m;
  while (e) {
    if (e & 1)
      r = r * base % m;
    base = base * base % m;
    e >>= 1;
  }
  return (uint64_t)r;
}

// Negacyclic (mod X^N+1) transform machinery for one prime limb, precomputed
// once so callers can transform many operands and accumulate in the evaluation
// domain before a single inverse. psi = 2N-th root, omega = psi^2 the N-th root;
// forward() = twist by psi^k then cyclic NTT; inverse() = cyclic iNTT then untwist
// by psi^-k. `valid` is false if q has no 2N-th root of unity (2N ∤ q-1).
struct NegacyclicNTT {
  ImplicitLocOpBuilder &b;
  MLIRContext *ctx;
  int64_t n;
  uint64_t omega = 0;
  field::PrimeFieldType pfTy;
  RankedTensorType pfTensor;
  Value twistPsi, twistPsiInv;
  bool valid = false;

  NegacyclicNTT(ImplicitLocOpBuilder &b, MLIRContext *ctx, int64_t modulus,
                int64_t n)
      : b(b), ctx(ctx), n(n) {
    uint64_t q = (uint64_t)modulus;
    uint64_t psi = primitiveNthRoot(2 * n, q);
    if (psi == 0)
      return;
    omega = powmod64(psi, 2, q);
    Type i64 = IntegerType::get(ctx, 64);
    pfTy = field::PrimeFieldType::get(ctx, IntegerAttr::get(i64, modulus),
                                      /*isMontgomery=*/false);
    pfTensor = RankedTensorType::get({n}, pfTy);
    twistPsi = makeTwist(q, psi);
    twistPsiInv = makeTwist(q, modInverse(psi, q));
    valid = true;
  }

  // [psi^0, psi^1, ..., psi^{N-1}] as a field.pf coefficient tensor.
  Value makeTwist(uint64_t q, uint64_t basePow) {
    Type i64 = IntegerType::get(ctx, 64);
    auto rowI64 = RankedTensorType::get({n}, i64);
    SmallVector<APInt> vals;
    uint64_t acc = 1;
    for (int64_t k = 0; k < n; ++k) {
      vals.push_back(APInt(64, acc));
      acc = (uint64_t)((unsigned __int128)acc * basePow % q);
    }
    Value ci = arith::ConstantOp::create(b, DenseElementsAttr::get(rowI64, vals));
    return field::BitcastOp::create(b, pfTensor, ci);
  }

  Value ntt(Value src, bool inverse) {
    Type i64 = IntegerType::get(ctx, 64);
    Value dest = tensor::EmptyOp::create(b, pfTensor, ValueRange{});
    auto root = field::RootOfUnityAttr::get(ctx, pfTy,
                                            IntegerAttr::get(i64, (int64_t)omega),
                                            IntegerAttr::get(i64, n));
    return poly::NTTOp::create(b, src, dest, /*twiddles=*/Value(), root,
                               /*tileX=*/IntegerAttr(), /*gridSize=*/IntegerAttr(),
                               /*bitReverse=*/b.getBoolAttr(true),
                               /*inverse=*/b.getBoolAttr(inverse))
        .getOutput();
  }

  // Coefficient tensor -> evaluation domain (twist by psi^k, then NTT).
  Value forward(Value field) {
    return ntt(field::MulOp::create(b, field, twistPsi), /*inverse=*/false);
  }
  // Evaluation domain -> coefficient tensor (iNTT, then untwist by psi^-k).
  Value inverse(Value eval) {
    return field::MulOp::create(b, ntt(eval, /*inverse=*/true), twistPsiInv);
  }
};

// Negacyclic product of two field.pf<q> coefficient tensors (mod X^N+1).
// Returns null if q has no 2N-th root of unity.
static Value negacyclicMulLimb(ImplicitLocOpBuilder &b, MLIRContext *ctx,
                               int64_t modulus, int64_t n, Value aField,
                               Value bField) {
  NegacyclicNTT t(b, ctx, modulus, n);
  if (!t.valid)
    return Value();
  Value prod = field::MulOp::create(b, t.forward(aField), t.forward(bField));
  return t.inverse(prod);
}

class RingToModArithTypeConverter : public TypeConverter {
public:
  RingToModArithTypeConverter(MLIRContext *ctx) {
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
      APInt qi(qw, (uint64_t)q);
      qWide.push_back(qi);
      Q *= qi;
    }
    SmallVector<uint64_t> yHatInv(L);
    SmallVector<SmallVector<uint64_t>> table(L, SmallVector<uint64_t>(Lp));
    for (unsigned i = 0; i < L; ++i) {
      APInt qHat = Q.udiv(qWide[i]);
      yHatInv[i] =
          modInverse(qHat.urem(qWide[i]).getZExtValue(), (uint64_t)inM[i]);
      for (unsigned j = 0; j < Lp; ++j)
        table[i][j] = qHat.urem(APInt(qw, (uint64_t)outM[j])).getZExtValue();
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
    auto outTensorTy = RankedTensorType::get({(int64_t)Lp, n}, i64);
    Value result = tensor::EmptyOp::create(b, outTensorTy, ValueRange{});
    for (unsigned j = 0; j < Lp; ++j) {
      Value acc;
      for (unsigned i = 0; i < L; ++i) {
        Value yiPj = mod_arith::BitcastOp::create(b, maTensor(outM[j]), y[i]);
        Value term =
            mod_arith::MulOp::create(b, yiPj, maConst(outM[j], table[i][j]));
        acc = i == 0 ? term : mod_arith::AddOp::create(b, acc, term).getResult();
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
// out_i = (x_i - x_last) * q_last^{-1} mod q_i (mod_arith.mul Barrett-reduces the
// x_last*qInv product, so the non-canonical x_last reinterpret is fine).
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
    uint64_t qLast = (uint64_t)inM[L - 1];

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

    auto outTy = RankedTensorType::get({(int64_t)L - 1, n}, i64);
    Value result = tensor::EmptyOp::create(b, outTy, ValueRange{});
    for (unsigned i = 0; i + 1 < L; ++i) {
      uint64_t qi = (uint64_t)inM[i];
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

// ntt: per limb, field.bitcast i64 -> field.pf<q_i>, poly.ntt (reusing the
// shared kernel), bitcast back. Cyclic NTT with an N-th root mod q_i.
struct ConvertNTT : public OpConversionPattern<NTTOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(NTTOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    MLIRContext *ctx = op.getContext();
    auto ty = cast<RqType>(op.getInput().getType());
    ArrayRef<int64_t> mod = ty.getModuli().asArrayRef();
    int64_t n = ty.getRingDegree().getValue().getSExtValue();
    unsigned L = mod.size();
    bool inverse = op.getInverse();

    Type i64 = IntegerType::get(ctx, 64);
    auto rowI64 = RankedTensorType::get({n}, i64);
    SmallVector<OpFoldResult> strides(2, b.getIndexAttr(1));
    SmallVector<OpFoldResult> sizes = {b.getIndexAttr(1), b.getIndexAttr(n)};
    auto offs = [&](int64_t r) -> SmallVector<OpFoldResult> {
      return {b.getIndexAttr(r), b.getIndexAttr(0)};
    };
    Value in = adaptor.getInput();

    auto outTy = RankedTensorType::get({(int64_t)L, n}, i64);
    Value result = tensor::EmptyOp::create(b, outTy, ValueRange{});
    for (unsigned i = 0; i < L; ++i) {
      uint64_t q = (uint64_t)mod[i];
      uint64_t w = primitiveNthRoot(n, q);
      if (w == 0)
        return op.emitOpError("modulus ") << mod[i] << " has no N-th root of "
                                             "unity (N must divide q-1)";
      auto pfTy = field::PrimeFieldType::get(ctx, IntegerAttr::get(i64, mod[i]),
                                             /*isMontgomery=*/false);
      auto pfTensor = RankedTensorType::get({n}, pfTy);
      Value xi = tensor::ExtractSliceOp::create(b, rowI64, in, offs(i), sizes,
                                                strides);
      Value fi = field::BitcastOp::create(b, pfTensor, xi);
      Value dest = tensor::EmptyOp::create(b, pfTensor, ValueRange{});
      auto root = field::RootOfUnityAttr::get(
          ctx, pfTy, IntegerAttr::get(i64, (int64_t)w), IntegerAttr::get(i64, n));
      Value ev = poly::NTTOp::create(b, /*source=*/fi, /*dest=*/dest,
                                     /*twiddles=*/Value(), /*root=*/root,
                                     /*tileX=*/IntegerAttr(),
                                     /*gridSize=*/IntegerAttr(),
                                     /*bitReverse=*/b.getBoolAttr(true),
                                     /*inverse=*/b.getBoolAttr(inverse))
                     .getOutput();
      Value oi = field::BitcastOp::create(b, rowI64, ev);
      result = tensor::InsertSliceOp::create(b, oi, result, offs(i), sizes,
                                             strides);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

// mul: negacyclic polynomial product per limb, via the psi-twisted NTT
// (twist by psi^k, cyclic NTT with omega=psi^2, pointwise, inverse NTT, untwist
// by psi^-k). Reuses poly.ntt for the transforms; field.mul for twist/pointwise.
struct ConvertMul : public OpConversionPattern<MulOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    MLIRContext *ctx = op.getContext();
    auto ty = cast<RqType>(op.getLhs().getType());
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
    Value lhs = adaptor.getLhs(), rhs = adaptor.getRhs();

    auto outTy = RankedTensorType::get({(int64_t)L, n}, i64);
    Value result = tensor::EmptyOp::create(b, outTy, ValueRange{});
    for (unsigned i = 0; i < L; ++i) {
      auto pfTy = field::PrimeFieldType::get(ctx, IntegerAttr::get(i64, mod[i]),
                                             /*isMontgomery=*/false);
      auto pfTensor = RankedTensorType::get({n}, pfTy);
      auto limbField = [&](Value whole) -> Value {
        Value s = tensor::ExtractSliceOp::create(b, rowI64, whole, offs(i),
                                                 sizes, strides);
        return field::BitcastOp::create(b, pfTensor, s);
      };
      Value ci = negacyclicMulLimb(b, ctx, mod[i], n, limbField(lhs),
                                   limbField(rhs));
      if (!ci)
        return op.emitOpError("modulus ")
               << mod[i] << " has no 2N-th root of unity (need 2N | q-1)";
      Value oi = field::BitcastOp::create(b, rowI64, ci);
      result = tensor::InsertSliceOp::create(b, oi, result, offs(i), sizes,
                                             strides);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

// gadget_product: sum_j decompose(x)_j * keys[j]. Digit-decompose x (bit-slice),
// then per limb accumulate the negacyclic products in the NTT (evaluation) domain:
// forward-transform each digit and key once, sum the pointwise products, and do a
// SINGLE inverse per limb instead of one per term. Valid because the inverse
// transform and the psi^-k untwist are linear, so sum(inverse(P_j)) =
// inverse(sum(P_j)) — this collapses the memory-bound iNTT of the key-switch chain
// from `levels` down to 1 per limb (the fused key-switch lever).
struct ConvertGadgetProduct : public OpConversionPattern<GadgetProductOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(GadgetProductOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    MLIRContext *ctx = op.getContext();
    auto ty = cast<RqType>(op.getInput().getType());
    ArrayRef<int64_t> mod = ty.getModuli().asArrayRef();
    int64_t n = ty.getRingDegree().getValue().getSExtValue();
    unsigned L = mod.size();
    int64_t baseBits = op.getBaseBits();
    ValueRange keys = adaptor.getKeys();
    unsigned levels = keys.size();

    Type i64 = IntegerType::get(ctx, 64);
    auto rowI64 = RankedTensorType::get({n}, i64);
    Value in = adaptor.getInput();
    auto wholeTy = cast<RankedTensorType>(in.getType());
    auto splat = [&](uint64_t v) -> Value {
      return arith::ConstantOp::create(
          b, DenseElementsAttr::get(wholeTy, APInt(64, v)));
    };
    Value mask = splat((uint64_t(1) << baseBits) - 1);
    SmallVector<Value> digits;
    for (unsigned j = 0; j < levels; ++j)
      digits.push_back(arith::AndIOp::create(
          b, arith::ShRUIOp::create(b, in, splat((uint64_t)(j * baseBits))),
          mask));

    SmallVector<OpFoldResult> strides(2, b.getIndexAttr(1));
    SmallVector<OpFoldResult> sizes = {b.getIndexAttr(1), b.getIndexAttr(n)};
    auto offs = [&](int64_t r) -> SmallVector<OpFoldResult> {
      return {b.getIndexAttr(r), b.getIndexAttr(0)};
    };

    auto outTy = RankedTensorType::get({(int64_t)L, n}, i64);
    Value result = tensor::EmptyOp::create(b, outTy, ValueRange{});
    for (unsigned i = 0; i < L; ++i) {
      NegacyclicNTT t(b, ctx, mod[i], n);
      if (!t.valid)
        return op.emitOpError("modulus ")
               << mod[i] << " has no 2N-th root of unity";
      auto limbField = [&](Value whole) -> Value {
        Value s = tensor::ExtractSliceOp::create(b, rowI64, whole, offs(i),
                                                 sizes, strides);
        return field::BitcastOp::create(b, t.pfTensor, s);
      };
      // Accumulate d_j*k_j pointwise in the evaluation domain, one inverse total.
      Value accHat;
      for (unsigned j = 0; j < levels; ++j) {
        Value prod = field::MulOp::create(b, t.forward(limbField(digits[j])),
                                          t.forward(limbField(keys[j])));
        accHat =
            j == 0 ? prod : field::AddOp::create(b, accHat, prod).getResult();
      }
      Value oi = field::BitcastOp::create(b, rowI64, t.inverse(accHat));
      result = tensor::InsertSliceOp::create(b, oi, result, offs(i), sizes,
                                             strides);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

// gadget_decompose: base-2^baseBits digit split of each residue. Pure bit
// manipulation on the whole [L,N] tensor: digit_j = (x >> j*baseBits) & (B-1).
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
    Value mask = splat((uint64_t(1) << baseBits) - 1);
    SmallVector<Value> digits;
    for (int64_t j = 0; j < levels; ++j) {
      Value shifted =
          arith::ShRUIOp::create(b, in, splat((uint64_t)(j * baseBits)));
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
    int64_t twoN = 2 * n;

    // pos[k] = destination of coefficient k; neg[k] = whether it flips sign.
    SmallVector<int64_t> pos(n);
    SmallVector<bool> neg(n);
    for (int64_t k = 0; k < n; ++k) {
      int64_t dest = (g * k) % twoN;
      pos[k] = dest % n;
      neg[k] = dest >= n;
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

    auto outTy = RankedTensorType::get({(int64_t)L, n}, i64);
    Value result = tensor::EmptyOp::create(b, outTy, ValueRange{});
    for (unsigned i = 0; i < L; ++i) {
      uint64_t q = (uint64_t)mod[i];
      Value xi = tensor::ExtractSliceOp::create(b, rowI64, in, offs(i), sizes,
                                                strides);
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
      result = tensor::InsertSliceOp::create(b, oi, result, offs(i), sizes,
                                             strides);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

// Componentwise same-basis binary op: per limb i, ModOpT on mod_arith<q_i>.
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

    auto outTy = RankedTensorType::get({(int64_t)L, n}, i64);
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
      result = tensor::InsertSliceOp::create(b, si, result, offs(i), sizes,
                                             strides);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

// from_tensor / to_tensor bridge the (converted) tensor representation and the
// ring type; after conversion both sides are the same tensor, so they fold away.
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
    target.addLegalDialect<mod_arith::ModArithDialect, tensor::TensorDialect,
                           field::FieldDialect, poly::PolyDialect>();

    RewritePatternSet patterns(context);
    patterns.add<ConvertBaseConvert, ConvertRescale, ConvertNTT, ConvertMul,
                 ConvertAutomorphism, ConvertGadgetDecompose,
                 ConvertGadgetProduct, ConvertFromTensor, ConvertToTensor,
                 ConvertBinOp<AddOp, mod_arith::AddOp>,
                 ConvertBinOp<SubOp, mod_arith::SubOp>>(typeConverter, context);
    addStructuralConversionPatterns(typeConverter, patterns, target);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::prime_ir::ring
