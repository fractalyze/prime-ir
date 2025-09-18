#include "zkir/Dialect/Field/Conversions/ExtFieldToLLVM/ExtFieldToLLVM.h"

#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "zkir/Dialect/Field/IR/FieldDialect.h"
#include "zkir/Dialect/Field/IR/FieldOps.h"
#include "zkir/Dialect/Field/IR/FieldTypes.h"
#include "zkir/Utils/SimpleStructBuilder.h"

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
  PrimeFieldType baseFieldType = type.getBaseField();
  Type integerType = baseFieldType.getStorageType();
  // TODO(batzor): Add support for more extension fields.
  if constexpr (std::is_same_v<T, QuadraticExtFieldType>) {
    return LLVM::LLVMStructType::getLiteral(type.getContext(),
                                            {integerType, integerType});
  } else {
    return type;
  }
}

struct ConvertExtFromCoeffs : public ConvertOpToLLVMPattern<ExtFromCoeffsOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ExtFromCoeffsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto structType = typeConverter->convertType(op.getType());
    if (isa<QuadraticExtFieldType>(op.getType())) {
      if (adaptor.getInput().size() != 2)
        return op.emitOpError(
            "expected two input types for quadratic extension field");
      auto extFieldStruct = SimpleStructBuilder<2>::initialized(
          rewriter, loc, structType, adaptor.getInput());
      rewriter.replaceOp(op, {extFieldStruct});
      return success();
    }
    return op.emitOpError("unsupported output type");
  }
};

struct ConvertExtToCoeffs : public ConvertOpToLLVMPattern<ExtToCoeffsOp> {
  using ConvertOpToLLVMPattern<ExtToCoeffsOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ExtToCoeffsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isa<QuadraticExtFieldType>(op.getInput().getType())) {
      SimpleStructBuilder<2> extFieldStruct(adaptor.getInput());
      SmallVector<Value> coeffs =
          extFieldStruct.getValues(rewriter, op.getLoc());
      rewriter.replaceOpWithMultiple(op, coeffs);
      return success();
    }
    return op.emitOpError("unsupported input type");
  }
};

#include "zkir/Dialect/Field/Conversions/ExtFieldToLLVM/ExtFieldToLLVM.cpp.inc"
} // namespace

void populateExtFieldToLLVMTypeConversion(LLVMTypeConverter &typeConverter) {
  typeConverter.addConversion(
      [](QuadraticExtFieldType type) { return convertExtFieldType(type); });
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

struct ExtFieldToLLVMDialectInterface
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
    populateExtFieldToLLVMTypeConversion(typeConverter);
    populateExtFieldToLLVMConversionPatterns(typeConverter, patterns);
  }
};

void registerConvertExtFieldToLLVMInterface(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, zkir::field::FieldDialect *dialect) {
        dialect->addInterfaces<ExtFieldToLLVMDialectInterface>();
      });
}
} // namespace mlir::zkir::field
