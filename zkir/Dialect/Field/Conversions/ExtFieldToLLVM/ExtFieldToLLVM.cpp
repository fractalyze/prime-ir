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

#include "zkir/Dialect/Field/Conversions/ExtFieldToLLVM/ExtFieldToLLVM.h"

#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "zkir/Dialect/Field/IR/FieldDialect.h"
#include "zkir/Dialect/Field/IR/FieldOps.h"
#include "zkir/Dialect/Field/IR/FieldTypes.h"

namespace mlir::zkir::field {

#define GEN_PASS_DEF_EXTFIELDTOLLVM
#include "zkir/Dialect/Field/Conversions/ExtFieldToLLVM/ExtFieldToLLVM.h.inc"

using namespace mlir::LLVM;

//===----------------------------------------------------------------------===//
// Conversion patterns.
//===----------------------------------------------------------------------===//
namespace {
template <typename T>
Type convertExtFieldType(T type) {
  if (auto extField = dyn_cast<ExtensionFieldTypeInterface>(type)) {
    Type baseFieldType = extField.getBaseFieldType();
    Type elementType;

    // Handle tower of extensions: if base is also an extension field,
    // recursively convert it to a nested struct
    if (auto baseExt = dyn_cast<ExtensionFieldTypeInterface>(baseFieldType)) {
      elementType = convertExtFieldType(baseExt);
    } else {
      // Base is a prime field, use its storage type
      auto pfType = cast<PrimeFieldType>(baseFieldType);
      elementType = pfType.getStorageType();
    }

    unsigned n = extField.getDegreeOverBase();
    SmallVector<Type> fields(n, elementType);
    return LLVM::LLVMStructType::getLiteral(type.getContext(), fields);
  }
  return type;
}

struct ConvertExtFromCoeffs : public ConvertOpToLLVMPattern<ExtFromCoeffsOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ExtFromCoeffsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto structType = typeConverter->convertType(op.getType());

    auto extField = cast<ExtensionFieldTypeInterface>(op.getType());
    SmallVector<Value> coeffs(adaptor.getInput().begin(),
                              adaptor.getInput().end());
    auto extFieldStruct = extField.buildStructFromCoeffs(b, structType, coeffs);
    rewriter.replaceOp(op, {extFieldStruct});
    return success();
  }
};

struct ConvertExtToCoeffs : public ConvertOpToLLVMPattern<ExtToCoeffsOp> {
  using ConvertOpToLLVMPattern<ExtToCoeffsOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ExtToCoeffsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    if (auto extField =
            dyn_cast<ExtensionFieldTypeInterface>(op.getInput().getType())) {
      SmallVector<Value> coeffs =
          extField.extractCoeffsFromStruct(b, adaptor.getInput());
      rewriter.replaceOpWithMultiple(op, coeffs);
      return success();
    }
    return op.emitOpError("unsupported input type");
  }
};

#include "zkir/Dialect/Field/Conversions/ExtFieldToLLVM/ExtFieldToLLVM.cpp.inc"
} // namespace

void populateExtFieldToLLVMTypeConversion(LLVMTypeConverter &typeConverter) {
  typeConverter.addConversion([](ExtensionFieldTypeInterface type) {
    return convertExtFieldType(type);
  });
}

void populateExtFieldToLLVMConversionPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<
      // clang-format off
      ConvertExtFromCoeffs,
      ConvertExtToCoeffs
      // clang-format on
      >(converter);
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
struct ExtFieldToLLVM : impl::ExtFieldToLLVMBase<ExtFieldToLLVM> {
  using ExtFieldToLLVMBase::ExtFieldToLLVMBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    LLVMConversionTarget target(*context);
    LLVMTypeConverter typeConverter(context);
    RewritePatternSet patterns(context);

    populateExtFieldToLLVMTypeConversion(typeConverter);
    populateExtFieldToLLVMConversionPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// ConvertToLLVMPatternInterface implementation
//===----------------------------------------------------------------------===//

struct ExtFieldToLLVMDialectInterface : public ConvertToLLVMPatternInterface {
  using ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;

  void loadDependentDialects(MLIRContext *context) const final {
    context->loadDialect<LLVM::LLVMDialect>();
  }

  // Hook for derived dialect interface to provide conversion patterns and mark
  // dialect legal for the conversion target.
  void populateConvertToLLVMConversionPatterns(
      ConversionTarget &target, LLVMTypeConverter &typeConverter,
      RewritePatternSet &patterns) const final {
    populateExtFieldToLLVMTypeConversion(typeConverter);
    populateExtFieldToLLVMConversionPatterns(typeConverter, patterns);
  }
};

void registerConvertExtFieldToLLVMInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, FieldDialect *dialect) {
    dialect->addInterfaces<ExtFieldToLLVMDialectInterface>();
  });
}
} // namespace mlir::zkir::field
