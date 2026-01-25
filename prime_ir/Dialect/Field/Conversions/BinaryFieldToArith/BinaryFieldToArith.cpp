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

/// Check if a type contains BinaryFieldType
bool containsBinaryFieldType(Type type) {
  Type elemType = getElementTypeOrSelf(type);
  return isa<BinaryFieldType>(elemType);
}

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

struct ConvertBinaryFieldConstant : public OpConversionPattern<ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto bfType = dyn_cast<BinaryFieldType>(getElementTypeOrSelf(op.getType()));
    if (!bfType) {
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
    auto bfType =
        dyn_cast<BinaryFieldType>(getElementTypeOrSelf(op.getLhs().getType()));
    if (!bfType) {
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
        ConvertAny<tensor::ExtractOp>,
        ConvertAny<tensor::FromElementsOp>,
        ConvertAny<tensor::InsertOp>
        // clang-format on
        >(typeConverter, context);

    addStructuralConversionPatterns(typeConverter, patterns, target);

    target.addDynamicallyLegalOp<
        // clang-format off
        tensor::ExtractOp,
        tensor::FromElementsOp,
        tensor::InsertOp
        // clang-format on
        >([&](auto op) { return typeConverter.isLegal(op); });

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::prime_ir::field
