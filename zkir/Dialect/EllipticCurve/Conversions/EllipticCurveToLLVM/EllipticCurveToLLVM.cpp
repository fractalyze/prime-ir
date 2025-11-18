/* Copyright 2025 The ZKIR Authors.

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

#include "zkir/Dialect/EllipticCurve/Conversions/EllipticCurveToLLVM/EllipticCurveToLLVM.h"

#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveDialect.h"
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveOps.h"
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveTypes.h"
#include "zkir/Dialect/Field/Conversions/ExtFieldToLLVM/ExtFieldToLLVM.h"
#include "zkir/Dialect/Field/IR/FieldTypes.h"
#include "zkir/Utils/SimpleStructBuilder.h"

namespace mlir::zkir::elliptic_curve {

#define GEN_PASS_DEF_ELLIPTICCURVETOLLVM
#include "zkir/Dialect/EllipticCurve/Conversions/EllipticCurveToLLVM/EllipticCurveToLLVM.h.inc"

using namespace mlir::LLVM;

//===----------------------------------------------------------------------===//
// Conversion patterns.
//===----------------------------------------------------------------------===//
namespace {
template <typename T>
Type convertPointType(T type, LLVMTypeConverter &typeConverter) {
  Type baseFieldType = type.getCurve().getBaseField();
  Type coordType;
  if (auto pfType = dyn_cast<field::PrimeFieldType>(baseFieldType)) {
    coordType = pfType.getStorageType();
  } else {
    coordType = typeConverter.convertType(baseFieldType);
  }
  if constexpr (std::is_same_v<T, AffineType>) {
    return LLVM::LLVMStructType::getLiteral(type.getContext(),
                                            {coordType, coordType});
  } else if constexpr (std::is_same_v<T, JacobianType>) {
    return LLVM::LLVMStructType::getLiteral(type.getContext(),
                                            {coordType, coordType, coordType});

  } else if constexpr (std::is_same_v<T, XYZZType>) {
    return LLVM::LLVMStructType::getLiteral(
        type.getContext(), {coordType, coordType, coordType, coordType});
  } else {
    return type;
  }
}

struct ConvertPoint : public ConvertOpToLLVMPattern<PointOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(PointOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto structType = typeConverter->convertType(op.getType());
    if (isa<AffineType>(op.getType())) {
      auto pointStruct = SimpleStructBuilder<2>::initialized(
          rewriter, loc, structType, adaptor.getCoords());
      rewriter.replaceOp(op, {pointStruct});
    } else if (isa<JacobianType>(op.getType())) {
      auto pointStruct = SimpleStructBuilder<3>::initialized(
          rewriter, loc, structType, adaptor.getCoords());
      rewriter.replaceOp(op, {pointStruct});
    } else if (isa<XYZZType>(op.getType())) {
      auto pointStruct = SimpleStructBuilder<4>::initialized(
          rewriter, loc, structType, adaptor.getCoords());
      rewriter.replaceOp(op, {pointStruct});
    }
    return success();
  }
};

struct ConvertExtract : public ConvertOpToLLVMPattern<ExtractOp> {
  using ConvertOpToLLVMPattern<ExtractOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isa<AffineType>(op.getInput().getType())) {
      SimpleStructBuilder<2> affineStruct(adaptor.getInput());
      SmallVector<Value> coords = affineStruct.getValues(rewriter, op.getLoc());
      rewriter.replaceOpWithMultiple(op, coords);
    } else if (isa<JacobianType>(op.getInput().getType())) {
      SimpleStructBuilder<3> jacobianStruct(adaptor.getInput());
      SmallVector<Value> coords =
          jacobianStruct.getValues(rewriter, op.getLoc());
      rewriter.replaceOpWithMultiple(op, coords);
    } else if (isa<XYZZType>(op.getInput().getType())) {
      SimpleStructBuilder<4> xyzzStruct(adaptor.getInput());
      SmallVector<Value> coords = xyzzStruct.getValues(rewriter, op.getLoc());
      rewriter.replaceOpWithMultiple(op, coords);
    }
    return success();
  }
};

#include "zkir/Dialect/EllipticCurve/Conversions/EllipticCurveToLLVM/EllipticCurveToLLVM.cpp.inc"
} // namespace

void populateEllipticCurveToLLVMTypeConversion(
    LLVMTypeConverter &typeConverter) {
  typeConverter.addConversion(
      [&](AffineType type) { return convertPointType(type, typeConverter); });
  typeConverter.addConversion(
      [&](JacobianType type) { return convertPointType(type, typeConverter); });
  typeConverter.addConversion(
      [&](XYZZType type) { return convertPointType(type, typeConverter); });
}

void populateEllipticCurveToLLVMConversionPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<
      // clang-format off
      ConvertExtract,
      ConvertPoint
      // clang-format on
      >(converter);
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
struct EllipticCurveToLLVM
    : impl::EllipticCurveToLLVMBase<EllipticCurveToLLVM> {
  using EllipticCurveToLLVMBase::EllipticCurveToLLVMBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    LLVMConversionTarget target(*context);
    LLVMTypeConverter typeConverter(context);
    RewritePatternSet patterns(context);

    field::populateExtFieldToLLVMTypeConversion(typeConverter);
    populateEllipticCurveToLLVMTypeConversion(typeConverter);
    populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
    populateEllipticCurveToLLVMConversionPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// ConvertToLLVMPatternInterface implementation
//===----------------------------------------------------------------------===//

struct EllipticCurveToLLVMDialectInterface
    : public mlir::ConvertToLLVMPatternInterface {
  using ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;

  void loadDependentDialects(mlir::MLIRContext *context) const final {
    context->loadDialect<LLVM::LLVMDialect>();
  }

  // Hook for derived dialect interface to provide conversion patterns and mark
  // dialect legal for the conversion target.
  void populateConvertToLLVMConversionPatterns(
      ConversionTarget &target, LLVMTypeConverter &typeConverter,
      RewritePatternSet &patterns) const final {
    populateEllipticCurveToLLVMTypeConversion(typeConverter);
    populateEllipticCurveToLLVMConversionPatterns(typeConverter, patterns);
  }
};

void registerConvertEllipticCurveToLLVMInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, EllipticCurveDialect *dialect) {
    dialect->addInterfaces<EllipticCurveToLLVMDialectInterface>();
  });
}
} // namespace mlir::zkir::elliptic_curve
