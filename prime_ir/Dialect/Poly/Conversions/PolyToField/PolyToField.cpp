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

#include "prime_ir/Dialect/Poly/Conversions/PolyToField/PolyToField.h"

#include <utility>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "prime_ir/Dialect/Field/IR/FieldAttributes.h"
#include "prime_ir/Dialect/Field/IR/FieldOperation.h"
#include "prime_ir/Dialect/Field/IR/FieldOps.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithTypes.h"
#include "prime_ir/Dialect/Poly/IR/PolyDialect.h"
#include "prime_ir/Dialect/Poly/IR/PolyOps.h"
#include "prime_ir/Dialect/Poly/IR/PolyTypes.h"
#include "prime_ir/Dialect/TensorExt/IR/TensorExtOps.h"
#include "prime_ir/Utils/ConversionUtils.h"

namespace mlir::prime_ir::poly {

#define GEN_PASS_DEF_POLYTOFIELD
#include "prime_ir/Dialect/Poly/Conversions/PolyToField/PolyToField.h.inc"

static RankedTensorType convertPolyType(PolyType type) {
  int64_t maxDegree = type.getMaxDegree().getValue().getSExtValue();
  return RankedTensorType::get({static_cast<int64_t>(maxDegree + 1)},
                               type.getBaseField());
}

class PolyToFieldTypeConverter : public TypeConverter {
public:
  explicit PolyToFieldTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion([](PolyType type) -> Type { return convertPolyType(type); });
  }
};

struct CommonConversionInfo {
  PolyType polyType;

  field::PrimeFieldType coefficientType;
  Type coefficientStorageType;

  RankedTensorType tensorType;
};

static FailureOr<CommonConversionInfo>
getCommonConversionInfo(Operation *op, const TypeConverter *typeConverter) {
  // Most ops have a single result type that is a polynomial
  PolyType polyTy = dyn_cast<PolyType>(op->getResult(0).getType());

  if (!polyTy) {
    op->emitError("Can't directly lower for a tensor of polynomials. "
                  "First run --convert-elementwise-to-affine.");
    return failure();
  }

  CommonConversionInfo info;
  info.polyType = polyTy;
  info.coefficientType = dyn_cast<field::PrimeFieldType>(polyTy.getBaseField());
  if (!info.coefficientType) {
    op->emitError("Polynomial base field must be of field type");
    return failure();
  }
  info.tensorType = cast<RankedTensorType>(typeConverter->convertType(polyTy));
  info.coefficientStorageType = info.coefficientType.getStorageType();
  return std::move(info);
}

struct ConvertToTensor : public OpConversionPattern<ToTensorOp> {
  explicit ConvertToTensor(MLIRContext *context)
      : OpConversionPattern<ToTensorOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ToTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};

struct ConvertFromTensor : public OpConversionPattern<FromTensorOp> {
  explicit ConvertFromTensor(MLIRContext *context)
      : OpConversionPattern<FromTensorOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FromTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto res = getCommonConversionInfo(op, typeConverter);
    if (failed(res))
      return failure();
    auto typeInfo = res.value();

    auto resultShape = typeInfo.tensorType.getShape()[0];
    auto inputTensorTy = op.getInput().getType();
    auto inputShape = inputTensorTy.getShape()[0];

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto coeffValue = adaptor.getInput();

    // Zero pad the tensor if the coefficients' size is less than the polynomial
    // degree.
    if (inputShape < resultShape) {
      SmallVector<OpFoldResult, 1> low, high;
      low.push_back(rewriter.getIndexAttr(0));
      high.push_back(rewriter.getIndexAttr(resultShape - inputShape));

      Value padValue = field::createFieldZero(typeInfo.coefficientType, b);
      coeffValue = tensor::PadOp::create(b, typeInfo.tensorType, coeffValue,
                                         low, high, padValue,
                                         /*nofold=*/false);
    }

    rewriter.replaceOp(op, coeffValue);
    return success();
  }
};

struct PolyToField : impl::PolyToFieldBase<PolyToField> {
  using PolyToFieldBase::PolyToFieldBase;

  void runOnOperation() override;
};

void PolyToField::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();
  PolyToFieldTypeConverter typeConverter(context);

  ConversionTarget target(*context);

  target.addIllegalDialect<PolyDialect>();
  target.addLegalDialect<field::FieldDialect>();
  RewritePatternSet patterns(context);

  patterns.add<
      // clang-format off
      ConvertFromTensor,
      ConvertToTensor,
      ConvertAny<bufferization::AllocTensorOp>
      // clang-format on
      >(typeConverter, context);

  addStructuralConversionPatterns(typeConverter, patterns, target);

  target.addDynamicallyLegalOp<bufferization::AllocTensorOp>(
      [&](auto op) { return typeConverter.isLegal(op); });

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

} // namespace mlir::prime_ir::poly
