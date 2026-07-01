/* Copyright 2026 The PrimeIR Authors.

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

#include "prime_ir/Dialect/Field/Conversions/BinaryFieldToArith/BinaryFieldToArith.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include "prime_ir/Dialect/Field/Conversions/BinaryFieldToArith/BinaryFieldCodeGen.h"
#include "prime_ir/Dialect/Field/IR/FieldDialect.h"
#include "prime_ir/Dialect/Field/IR/FieldOps.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"
#include "prime_ir/Utils/ConversionUtils.h"
#include "prime_ir/Utils/ShapedTypeConverter.h"

namespace mlir::prime_ir::field {

#define GEN_PASS_DEF_BINARYFIELDTOARITH
#include "prime_ir/Dialect/Field/Conversions/BinaryFieldToArith/BinaryFieldToArith.h.inc"

namespace {

/// Type converter for binary field to arith conversion.
/// Converts BinaryFieldType to IntegerType with appropriate bit width.
class BinaryFieldToArithTypeConverter : public ShapedTypeConverter {
public:
  explicit BinaryFieldToArithTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion(
        [](BinaryFieldType type) -> Type { return type.getStorageType(); });
    addConversion([](ShapedType type) -> Type {
      if (auto bfType = dyn_cast<BinaryFieldType>(type.getElementType())) {
        return convertShapedType(type, type.getShape(),
                                 bfType.getStorageType());
      }
      return type;
    });
  }
};

/// Check if a type contains a binary field this pass lowers: the tower `bf` or
/// the flat-basis `ghash` (both GF(2^n), both lowered here).
bool containsBinaryFieldType(Type type) {
  Type elemType = getElementTypeOrSelf(type);
  return isa<BinaryFieldType>(elemType);
}

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

// GHASH-basis multiply (`bf<7, ghash>`), defined below; used by the mul/square
// patterns to split on basis.
Value emitGhashMul(ImplicitLocOpBuilder &b, Value a, Value bv);

struct ConvertBinaryFieldConstant : public OpConversionPattern<ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // bf and ghash constants both lower to an integer arith.constant of the
    // storage type; only the multiply differs between the two bases.
    if (!containsBinaryFieldType(op.getType())) {
      return failure();
    }

    Type convertedType = typeConverter->convertType(op.getType());
    if (!convertedType) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        op, convertedType, cast<TypedAttr>(op.getValueAttr()));
    return success();
  }
};

struct ConvertBinaryFieldAdd : public OpConversionPattern<AddOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto bfType = dyn_cast<BinaryFieldType>(getElementTypeOrSelf(op.getType()));
    if (!bfType) {
      return failure();
    }

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    BinaryFieldCodeGen lhs(bfType, adaptor.getLhs(), b);
    BinaryFieldCodeGen rhs(bfType, adaptor.getRhs(), b);
    BinaryFieldCodeGen result = lhs + rhs;
    rewriter.replaceOp(op, result.getValue());
    return success();
  }
};

struct ConvertBinaryFieldSub : public OpConversionPattern<SubOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SubOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto bfType = dyn_cast<BinaryFieldType>(getElementTypeOrSelf(op.getType()));
    if (!bfType) {
      return failure();
    }

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    BinaryFieldCodeGen lhs(bfType, adaptor.getLhs(), b);
    BinaryFieldCodeGen rhs(bfType, adaptor.getRhs(), b);
    BinaryFieldCodeGen result = lhs - rhs;
    rewriter.replaceOp(op, result.getValue());
    return success();
  }
};

struct ConvertBinaryFieldNegate : public OpConversionPattern<NegateOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(NegateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto bfType = dyn_cast<BinaryFieldType>(getElementTypeOrSelf(op.getType()));
    if (!bfType) {
      return failure();
    }

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    BinaryFieldCodeGen input(bfType, adaptor.getInput(), b);
    BinaryFieldCodeGen result = -input;
    rewriter.replaceOp(op, result.getValue());
    return success();
  }
};

struct ConvertBinaryFieldDouble : public OpConversionPattern<DoubleOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DoubleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto bfType = dyn_cast<BinaryFieldType>(getElementTypeOrSelf(op.getType()));
    if (!bfType) {
      return failure();
    }

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    BinaryFieldCodeGen input(bfType, adaptor.getInput(), b);
    BinaryFieldCodeGen result = input.dbl();
    rewriter.replaceOp(op, result.getValue());
    return success();
  }
};

struct ConvertBinaryFieldMul : public OpConversionPattern<MulOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto bfType = dyn_cast<BinaryFieldType>(getElementTypeOrSelf(op.getType()));
    if (!bfType) {
      return failure();
    }

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    if (bfType.isGhash()) {
      rewriter.replaceOp(op,
                         emitGhashMul(b, adaptor.getLhs(), adaptor.getRhs()));
      return success();
    }
    BinaryFieldCodeGen lhs(bfType, adaptor.getLhs(), b);
    BinaryFieldCodeGen rhs(bfType, adaptor.getRhs(), b);
    BinaryFieldCodeGen result = lhs * rhs;
    rewriter.replaceOp(op, result.getValue());
    return success();
  }
};

struct ConvertBinaryFieldSquare : public OpConversionPattern<SquareOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SquareOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto bfType = dyn_cast<BinaryFieldType>(getElementTypeOrSelf(op.getType()));
    if (!bfType) {
      return failure();
    }

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    if (bfType.isGhash()) {
      rewriter.replaceOp(
          op, emitGhashMul(b, adaptor.getInput(), adaptor.getInput()));
      return success();
    }
    BinaryFieldCodeGen input(bfType, adaptor.getInput(), b);
    BinaryFieldCodeGen result = input.square();
    rewriter.replaceOp(op, result.getValue());
    return success();
  }
};

struct ConvertBinaryFieldInverse : public OpConversionPattern<InverseOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(InverseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto bfType = dyn_cast<BinaryFieldType>(getElementTypeOrSelf(op.getType()));
    if (!bfType) {
      return failure();
    }
    // GHASH-basis inverse is not yet lowered (the GPU prover path is
    // mul/add-heavy); fail to legalize rather than apply the tower inverse.
    if (bfType.isGhash()) {
      return failure();
    }

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    BinaryFieldCodeGen input(bfType, adaptor.getInput(), b);
    BinaryFieldCodeGen result = input.inverse();
    rewriter.replaceOp(op, result.getValue());
    return success();
  }
};

struct ConvertBinaryFieldCmp : public OpConversionPattern<CmpOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CmpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!containsBinaryFieldType(op.getLhs().getType())) {
      return failure();
    }

    // Binary field comparison is just integer comparison
    rewriter.replaceOpWithNewOp<arith::CmpIOp>(
        op, op.getPredicate(), adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

// Convert unrealized_conversion_cast ops involving binary field types.
// After type conversion, casts like bf<6> -> i64 become i64 -> i64 (no-op).
// This handles casts created by PCLMULQDQ/GFNI specialization passes.
struct ConvertBinaryFieldUnrealizedCast
    : public OpConversionPattern<UnrealizedConversionCastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only handle single-input single-output casts
    if (adaptor.getInputs().size() != 1 || op.getResults().size() != 1)
      return failure();

    Type origInputType = op.getInputs()[0].getType();
    Type origOutputType = op.getResultTypes()[0];

    // Only convert casts involving binary field types
    if (!containsBinaryFieldType(origInputType) &&
        !containsBinaryFieldType(origOutputType)) {
      return failure();
    }

    Type convertedOutputType = typeConverter->convertType(origOutputType);
    if (!convertedOutputType)
      return failure();

    Value convertedInput = adaptor.getInputs()[0];

    // If types match after conversion, this cast becomes a no-op
    if (convertedInput.getType() == convertedOutputType) {
      rewriter.replaceOp(op, convertedInput);
      return success();
    }

    // Otherwise, create a new cast with the converted types
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        op, convertedOutputType, convertedInput);
    return success();
  }
};

struct ConvertBinaryFieldBitcast : public OpConversionPattern<BitcastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BitcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type inputElemType = getElementTypeOrSelf(op.getInput().getType());
    Type outputElemType = getElementTypeOrSelf(op.getOutput().getType());

    // Only handle bitcasts involving a binary field type (tower bf or ghash)
    if (!isa<BinaryFieldType>(inputElemType) &&
        !isa<BinaryFieldType>(outputElemType)) {
      return failure();
    }

    // After type conversion, both sides are integers with same bitwidth (no-op)
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// GHASH-basis multiply (`bf<7, ghash>`)
//===----------------------------------------------------------------------===//
//
// `bf<7, ghash>` is GF(2)[x]/(x¹²⁸ + x⁷ + x² + x + 1) in the monomial basis
// (NOT the x²+x+α tower of the default `bf<7>`). Only the multiply differs;
// it is a carryless product reduced mod the GHASH polynomial. This portable
// lowering emits the carryless multiply as shift-XOR so it runs on targets
// without a CLMUL instruction (e.g. GPU); the x86/ARM specializers swap in
// PCLMULQDQ/PMULL.

Value i64Const(ImplicitLocOpBuilder &b, uint64_t v) {
  return arith::ConstantIntOp::create(b, static_cast<int64_t>(v), 64);
}

// Portable 64x64 -> (lo, hi) carryless (GF(2)[x]) product, bit-serial shift-XOR
// (mirrors flock's software clmul64). For bit i of `bv`, XOR in `a << i` (low
// limb) and `a >> (64 - i)` (high limb); i == 0 contributes nothing to `hi`.
std::pair<Value, Value> emitClmul64(ImplicitLocOpBuilder &b, Value a,
                                    Value bv) {
  Value zero = i64Const(b, 0);
  Value one = i64Const(b, 1);
  Value lo = zero;
  Value hi = zero;
  for (unsigned i = 0; i < 64; ++i) {
    Value bShifted = bv;
    Value aShl = a;
    if (i != 0) {
      Value shiftI = i64Const(b, i);
      bShifted = arith::ShRUIOp::create(b, bv, shiftI);
      aShl = arith::ShLIOp::create(b, a, shiftI);
    }
    Value bit = arith::AndIOp::create(b, bShifted, one);
    Value mask = arith::SubIOp::create(b, zero, bit); // 0 or all-ones
    lo = arith::XOrIOp::create(b, lo, arith::AndIOp::create(b, mask, aShl));
    if (i != 0) {
      Value aShr = arith::ShRUIOp::create(b, a, i64Const(b, 64 - i));
      hi = arith::XOrIOp::create(b, hi, arith::AndIOp::create(b, mask, aShr));
    }
  }
  return {lo, hi};
}

// Reduce the 256-bit carryless product (r0..r3, low-to-high i64 limbs) modulo
// x^128 + x^7 + x^2 + x + 1, returning the 128-bit result as (lo, hi). This is
// flock's ghash_reduce: fold the high half via x^128 == x^7 + x^2 + x + 1, with
// a 7-bit overflow correction for bits pushed past x^127.
std::pair<Value, Value> emitGhashReduce(ImplicitLocOpBuilder &b, Value r0,
                                        Value r1, Value r2, Value r3) {
  auto shl = [&](Value v, unsigned s) {
    return arith::ShLIOp::create(b, v, i64Const(b, s));
  };
  auto shr = [&](Value v, unsigned s) {
    return arith::ShRUIOp::create(b, v, i64Const(b, s));
  };
  auto x = [&](Value p, Value q) { return arith::XOrIOp::create(b, p, q); };
  auto orr = [&](Value p, Value q) { return arith::OrIOp::create(b, p, q); };

  Value s1Lo = shl(r2, 1), s1Hi = orr(shl(r3, 1), shr(r2, 63));
  Value s2Lo = shl(r2, 2), s2Hi = orr(shl(r3, 2), shr(r2, 62));
  Value s7Lo = shl(r2, 7), s7Hi = orr(shl(r3, 7), shr(r2, 57));
  Value foldLo = x(x(x(r2, s1Lo), s2Lo), s7Lo);
  Value foldHi = x(x(x(r3, s1Hi), s2Hi), s7Hi);
  Value ov = x(x(shr(r3, 63), shr(r3, 62)), shr(r3, 57));
  Value corr = x(x(x(ov, shl(ov, 1)), shl(ov, 2)), shl(ov, 7));
  return {x(x(r0, foldLo), corr), x(r1, foldHi)};
}

// Multiply two GHASH-basis i128 values: 128×128 carryless product reduced mod
// x¹²⁸ + x⁷ + x² + x + 1. Karatsuba — 3 clmul64 (the middle cross term
// a₀b₁ + a₁b₀ = (a₀+a₁)(b₀+b₁) + a₀b₀ + a₁b₁), which matters because clmul64 is
// fully unrolled.
Value emitGhashMul(ImplicitLocOpBuilder &b, Value a, Value bv) {
  auto i64Ty = b.getI64Type();
  auto i128Ty = b.getIntegerType(128);
  Value sh64 = arith::ConstantIntOp::create(b, 64, 128);
  Value a0 = arith::TruncIOp::create(b, i64Ty, a);
  Value a1 =
      arith::TruncIOp::create(b, i64Ty, arith::ShRUIOp::create(b, a, sh64));
  Value b0 = arith::TruncIOp::create(b, i64Ty, bv);
  Value b1 =
      arith::TruncIOp::create(b, i64Ty, arith::ShRUIOp::create(b, bv, sh64));

  Value aXor = arith::XOrIOp::create(b, a0, a1);
  Value bXor = arith::XOrIOp::create(b, b0, b1);
  auto [llLo, llHi] = emitClmul64(b, a0, b0);
  auto [hhLo, hhHi] = emitClmul64(b, a1, b1);
  auto [mLo, mHi] = emitClmul64(b, aXor, bXor);
  // cross = m + ll + hh = a₀b₁ + a₁b₀
  Value crLo =
      arith::XOrIOp::create(b, arith::XOrIOp::create(b, mLo, llLo), hhLo);
  Value crHi =
      arith::XOrIOp::create(b, arith::XOrIOp::create(b, mHi, llHi), hhHi);
  Value r0 = llLo;
  Value r1 = arith::XOrIOp::create(b, llHi, crLo);
  Value r2 = arith::XOrIOp::create(b, hhLo, crHi);
  Value r3 = hhHi;
  auto [lo, hi] = emitGhashReduce(b, r0, r1, r2, r3);

  Value loExt = arith::ExtUIOp::create(b, i128Ty, lo);
  Value hiExt = arith::ExtUIOp::create(b, i128Ty, hi);
  return arith::OrIOp::create(b, loExt, arith::ShLIOp::create(b, hiExt, sh64));
}

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct BinaryFieldToArith : impl::BinaryFieldToArithBase<BinaryFieldToArith> {
  using BinaryFieldToArithBase::BinaryFieldToArithBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();
    BinaryFieldToArithTypeConverter typeConverter(context);

    ConversionTarget target(*context);

    // Binary field operations are illegal
    target.addDynamicallyLegalOp<ConstantOp>(
        [](ConstantOp op) { return !containsBinaryFieldType(op.getType()); });
    target.addDynamicallyLegalOp<AddOp>(
        [](AddOp op) { return !containsBinaryFieldType(op.getType()); });
    target.addDynamicallyLegalOp<SubOp>(
        [](SubOp op) { return !containsBinaryFieldType(op.getType()); });
    target.addDynamicallyLegalOp<NegateOp>(
        [](NegateOp op) { return !containsBinaryFieldType(op.getType()); });
    target.addDynamicallyLegalOp<DoubleOp>(
        [](DoubleOp op) { return !containsBinaryFieldType(op.getType()); });
    target.addDynamicallyLegalOp<MulOp>(
        [](MulOp op) { return !containsBinaryFieldType(op.getType()); });
    target.addDynamicallyLegalOp<SquareOp>(
        [](SquareOp op) { return !containsBinaryFieldType(op.getType()); });
    target.addDynamicallyLegalOp<InverseOp>(
        [](InverseOp op) { return !containsBinaryFieldType(op.getType()); });
    target.addDynamicallyLegalOp<CmpOp>([](CmpOp op) {
      return !containsBinaryFieldType(op.getLhs().getType());
    });
    target.addDynamicallyLegalOp<BitcastOp>([](BitcastOp op) {
      Type inputElemType = getElementTypeOrSelf(op.getInput().getType());
      Type outputElemType = getElementTypeOrSelf(op.getOutput().getType());
      return !isa<BinaryFieldType>(inputElemType) &&
             !isa<BinaryFieldType>(outputElemType);
    });

    // Arith dialect is legal
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<scf::SCFDialect>();

    // Unrealized casts involving binary field types need to be converted.
    // Casts created by PCLMULQDQ/GFNI specialization (bf<6> -> i64) become
    // no-ops after type conversion (i64 -> i64). Other casts remain legal.
    target.addDynamicallyLegalOp<UnrealizedConversionCastOp>(
        [](UnrealizedConversionCastOp op) {
          for (Type t : op.getOperandTypes()) {
            if (containsBinaryFieldType(t))
              return false;
          }
          for (Type t : op.getResultTypes()) {
            if (containsBinaryFieldType(t))
              return false;
          }
          return true;
        });

    RewritePatternSet patterns(context);
    patterns.add<
        // clang-format off
        ConvertBinaryFieldConstant,
        ConvertBinaryFieldAdd,
        ConvertBinaryFieldSub,
        ConvertBinaryFieldNegate,
        ConvertBinaryFieldDouble,
        ConvertBinaryFieldMul,
        ConvertBinaryFieldSquare,
        ConvertBinaryFieldInverse,
        ConvertBinaryFieldCmp,
        ConvertBinaryFieldUnrealizedCast,
        ConvertBinaryFieldBitcast
        // clang-format on
        >(typeConverter, context);

    // Catch-all: converts any op whose operands/results carry binary field
    // types.
    patterns.add<ConvertAny<void>>(typeConverter, context);

    addStructuralConversionPatterns(typeConverter, patterns, target);

    target.markUnknownOpDynamicallyLegal(
        [&](Operation *op) { return typeConverter.isLegal(op); });

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::prime_ir::field
