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

#include "prime_ir/Dialect/Field/Conversions/ExtFieldToLLVM/ExtFieldToLLVM.h"

#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "prime_ir/Dialect/Field/IR/FieldDialect.h"
#include "prime_ir/Dialect/Field/IR/FieldOps.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"

namespace mlir::prime_ir::field {

#define GEN_PASS_DEF_EXTFIELDTOLLVM
#include "prime_ir/Dialect/Field/Conversions/ExtFieldToLLVM/ExtFieldToLLVM.h.inc"

using namespace mlir::LLVM;

//===----------------------------------------------------------------------===//
// Conversion patterns.
//===----------------------------------------------------------------------===//
namespace {
template <typename T>
Type convertExtFieldType(T type) {
  if (auto extField = dyn_cast<ExtensionFieldType>(type)) {
    Type baseFieldType = extField.getBaseField();
    Type elementType;

    // Handle tower of extensions: if base is also an extension field,
    // recursively convert it to a nested struct
    if (auto baseExt = dyn_cast<ExtensionFieldType>(baseFieldType)) {
      elementType = convertExtFieldType(baseExt);
    } else {
      // Base is a prime field, use its storage type
      auto pfType = cast<PrimeFieldType>(baseFieldType);
      elementType = pfType.getStorageType();
    }

    unsigned n = extField.getDegree();
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

    auto extField = cast<ExtensionFieldType>(op.getType());
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

    if (auto extField = dyn_cast<ExtensionFieldType>(op.getInput().getType())) {
      SmallVector<Value> coeffs =
          extField.extractCoeffsFromStruct(b, adaptor.getInput());
      rewriter.replaceOpWithMultiple(op, coeffs);
      return success();
    }
    return op.emitOpError("unsupported input type");
  }
};

struct ConvertBitcast : public ConvertOpToLLVMPattern<BitcastOp> {
  using ConvertOpToLLVMPattern<BitcastOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(BitcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type inputType = op.getInput().getType();
    Type outputType = op.getType();

    // Handle scalar EF <-> integer bitcasts.
    if (!isa<ShapedType>(inputType) && !isa<ShapedType>(outputType)) {
      auto inputEF =
          dyn_cast<ExtensionFieldType>(getElementTypeOrSelf(inputType));
      auto outputEF =
          dyn_cast<ExtensionFieldType>(getElementTypeOrSelf(outputType));
      auto inputInt = dyn_cast<IntegerType>(inputType);
      auto outputInt = dyn_cast<IntegerType>(outputType);

      if (inputEF && outputInt)
        return convertEFToInteger(op, adaptor, rewriter, inputEF, outputInt);
      if (inputInt && outputEF)
        return convertIntegerToEF(op, adaptor, rewriter, inputInt, outputEF);
    }

    // Handle tensor reinterpret bitcasts only.
    if (!isTensorReinterpretBitcast(inputType, outputType))
      return failure();

    Type convertedOutputType = typeConverter->convertType(op.getType());
    if (!convertedOutputType) {
      return op.emitOpError("failed to convert output type");
    }

    // Memref bitcasts (produced by bufferization) require rebuilding the
    // descriptor with correct sizes/strides/offset for the output element type.
    // A plain unrealized_conversion_cast would preserve the input descriptor
    // as-is, leaving sizes in input-element units — causing memref.copy to
    // compute the wrong byte count (heap corruption).
    // NOTE: This must come before the same-type check below because memref
    // descriptors of the same rank share the same LLVM struct type regardless
    // of element type or shape — the early return would skip the rebuild.
    if (isa<MemRefType>(inputType))
      return convertMemRefBitcast(op, adaptor, rewriter, convertedOutputType);

    // If the converted types are the same, we can directly replace with the
    // input.
    if (adaptor.getInput().getType() == convertedOutputType) {
      rewriter.replaceOp(op, adaptor.getInput());
      return success();
    }
    return failure();
  }

private:
  // Returns the degree-over-prime for a field element type.
  // PrimeFieldType / ModArithType / IntegerType → 1.
  // ExtensionFieldType → total degree over its base prime field.
  static unsigned getDegreeOverPrime(Type elementType) {
    if (auto ef = dyn_cast<ExtensionFieldType>(elementType))
      return ef.getDegreeOverPrime();
    return 1;
  }

  // EF struct → iN: type-pun through stack memory (alloca + store + load).
  // LLVM's SROA eliminates the alloca at -O1+, making this a register-level
  // noop. This assumes little-endian layout (x86/ARM targets) so that struct
  // field 0 maps to the low bits of the loaded integer.
  LogicalResult convertEFToInteger(BitcastOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   ExtensionFieldType efType,
                                   IntegerType intType) const {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    auto structType = cast<LLVM::LLVMStructType>(input.getType());
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    Value one = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(1));
    Value alloca =
        rewriter.create<LLVM::AllocaOp>(loc, ptrType, structType, one);
    rewriter.create<LLVM::StoreOp>(loc, input, alloca);
    Value result = rewriter.create<LLVM::LoadOp>(loc, intType, alloca);

    rewriter.replaceOp(op, result);
    return success();
  }

  // iN → EF struct: type-pun through stack memory (alloca + store + load).
  // See convertEFToInteger for rationale.
  LogicalResult convertIntegerToEF(BitcastOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   IntegerType intType,
                                   ExtensionFieldType efType) const {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    Type convertedEFType = typeConverter->convertType(efType);
    auto structType = cast<LLVM::LLVMStructType>(convertedEFType);
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    Value one = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(1));
    Value alloca = rewriter.create<LLVM::AllocaOp>(loc, ptrType, intType, one);
    rewriter.create<LLVM::StoreOp>(loc, input, alloca);
    Value result = rewriter.create<LLVM::LoadOp>(loc, structType, alloca);

    rewriter.replaceOp(op, result);
    return success();
  }

  LogicalResult convertMemRefBitcast(BitcastOp op, OpAdaptor adaptor,
                                     ConversionPatternRewriter &rewriter,
                                     Type convertedOutputType) const {
    Location loc = op.getLoc();
    auto inputMemRef = cast<MemRefType>(op.getInput().getType());
    auto outputMemRef = cast<MemRefType>(op.getType());

    auto llvmDescTy = dyn_cast<LLVM::LLVMStructType>(convertedOutputType);
    if (!llvmDescTy)
      return failure();

    // Extract input descriptor fields.
    MemRefDescriptor inputDesc(adaptor.getInput());

    // Same-shape bitcast (e.g., memref<2x4xEF4> → memref<2x4xi128>): both
    // element types have the same byte size, so the descriptor is identical.
    if (inputMemRef.getShape() == outputMemRef.getShape()) {
      rewriter.replaceOp(op, adaptor.getInput());
      return success();
    }

    // Different-shape reinterpret (e.g., memref<2xEF4> → memref<8xPF>):
    // rebuild the descriptor with adjusted offset/sizes/strides.
    auto outputDesc = MemRefDescriptor::poison(rewriter, loc, llvmDescTy);

    // Same underlying memory — pointers are shared.
    outputDesc.setAllocatedPtr(rewriter, loc,
                               inputDesc.allocatedPtr(rewriter, loc));
    outputDesc.setAlignedPtr(rewriter, loc,
                             inputDesc.alignedPtr(rewriter, loc));

    // Adjust offset: offset_out = offset_in × degIn / degOut, where deg is
    // the degree-over-prime of the respective element type.
    unsigned degIn = getDegreeOverPrime(inputMemRef.getElementType());
    unsigned degOut = getDegreeOverPrime(outputMemRef.getElementType());

    Value offset = inputDesc.offset(rewriter, loc);
    if (degIn != degOut) {
      Type idxTy = getTypeConverter()->getIndexType();
      if (degOut > degIn) {
        // PF → EF: divide offset.
        Value ratio =
            createIndexAttrConstant(rewriter, loc, idxTy, degOut / degIn);
        offset = rewriter.create<LLVM::SDivOp>(loc, offset, ratio);
      } else {
        // EF → PF: multiply offset.
        Value ratio =
            createIndexAttrConstant(rewriter, loc, idxTy, degIn / degOut);
        offset = rewriter.create<LLVM::MulOp>(loc, offset, ratio);
      }
    }
    outputDesc.setOffset(rewriter, loc, offset);

    // Sizes: from the static output shape.
    for (int64_t i = 0, e = outputMemRef.getRank(); i < e; ++i)
      outputDesc.setConstantSize(rewriter, loc, i, outputMemRef.getDimSize(i));

    // Strides: output has identity layout (contiguous), stride = 1.
    for (int64_t i = 0, e = outputMemRef.getRank(); i < e; ++i)
      outputDesc.setConstantStride(rewriter, loc, i, 1);

    rewriter.replaceOp(op, Value(outputDesc));
    return success();
  }
};

#include "prime_ir/Dialect/Field/Conversions/ExtFieldToLLVM/ExtFieldToLLVM.cpp.inc"
} // namespace

void populateExtFieldToLLVMTypeConversion(LLVMTypeConverter &typeConverter) {
  // Scalar extension field
  typeConverter.addConversion(
      [](ExtensionFieldType type) { return convertExtFieldType(type); });

  // Tensor types with extension field element type.
  // Note: MemRef and Vector types cannot use !llvm.struct as element type, so
  // no shaped-type conversions are registered for them.  For MemRef, the
  // LLVMTypeConverter's built-in memref conversion handles memref<Nx!EF>
  // correctly: it converts the element type via the scalar conversion above
  // and builds the LLVM descriptor struct directly (bypassing MemRefType::get).
  // ConvertBitcast rebuilds the descriptor with correct sizes/strides/offset.
  typeConverter.addConversion(
      [](RankedTensorType tensorType) -> std::optional<Type> {
        auto efType = dyn_cast<ExtensionFieldType>(tensorType.getElementType());
        if (!efType)
          return std::nullopt;
        return RankedTensorType::get(tensorType.getShape(),
                                     convertExtFieldType(efType));
      });
  typeConverter.addConversion(
      [](UnrankedTensorType tensorType) -> std::optional<Type> {
        auto efType = dyn_cast<ExtensionFieldType>(tensorType.getElementType());
        if (!efType)
          return std::nullopt;
        return UnrankedTensorType::get(convertExtFieldType(efType));
      });
}

void populateExtFieldToLLVMConversionPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<
      // clang-format off
      ConvertBitcast,
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
} // namespace mlir::prime_ir::field
