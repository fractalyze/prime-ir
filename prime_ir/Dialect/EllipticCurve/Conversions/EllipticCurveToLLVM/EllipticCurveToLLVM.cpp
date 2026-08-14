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

#include "prime_ir/Dialect/EllipticCurve/Conversions/EllipticCurveToLLVM/EllipticCurveToLLVM.h"

#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveDialect.h"
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveOps.h"
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveTypes.h"
#include "prime_ir/Dialect/Field/Conversions/ExtFieldToLLVM/ExtFieldToLLVM.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"
#include "prime_ir/Utils/SimpleStructBuilder.h"

namespace mlir::prime_ir::elliptic_curve {

#define GEN_PASS_DEF_ELLIPTICCURVETOLLVM
#include "prime_ir/Dialect/EllipticCurve/Conversions/EllipticCurveToLLVM/EllipticCurveToLLVM.h.inc"

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

struct ConvertFromCoords : public ConvertOpToLLVMPattern<FromCoordsOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(FromCoordsOp op, OpAdaptor adaptor,
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

struct ConvertToCoords : public ConvertOpToLLVMPattern<ToCoordsOp> {
  using ConvertOpToLLVMPattern<ToCoordsOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ToCoordsOp op, OpAdaptor adaptor,
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

// The pattern and the legality rule have to agree on which bitcasts this pass
// owns, so they read it from the same place.
static bool isMemRefBitcast(BitcastOp op) {
  return isa<MemRefType>(op.getInput().getType()) &&
         isa<MemRefType>(op.getType());
}

struct ConvertBitcast : public ConvertOpToLLVMPattern<BitcastOp> {
  using ConvertOpToLLVMPattern<BitcastOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(BitcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Memref bitcasts (produced by bufferization) reinterpret one buffer as a
    // tensor of N points <-> N*K coordinates. The element COUNT changes, so
    // the descriptor must be rebuilt with sizes/strides/offset in the output
    // element's units. A plain unrealized_conversion_cast preserves the input
    // descriptor -- sizes still in input-element units -- and any later
    // dealloc/copy then computes the wrong byte count (heap corruption). A
    // still-tensor bitcast has no descriptor to rebuild and is left to
    // bufferization.
    if (!isMemRefBitcast(op)) {
      return failure();
    }
    auto inputMemRef = cast<MemRefType>(op.getInput().getType());
    auto outputMemRef = cast<MemRefType>(op.getType());
    Type convertedOutputType = typeConverter->convertType(outputMemRef);
    if (!convertedOutputType) {
      return op.emitOpError("failed to convert output type");
    }
    return convertMemRefBitcast(op, adaptor, rewriter, inputMemRef,
                                outputMemRef, convertedOutputType);
  }

private:
  // A point occupies K = coords * extension-degree consecutive field elements,
  // so the two memrefs never agree on extent and the input descriptor cannot be
  // forwarded: its sizes are in the wrong element unit and its offset counts
  // the wrong thing.
  LogicalResult convertMemRefBitcast(BitcastOp op, OpAdaptor adaptor,
                                     ConversionPatternRewriter &rewriter,
                                     MemRefType inputMemRef,
                                     MemRefType outputMemRef,
                                     Type convertedOutputType) const {
    Location loc = op.getLoc();
    auto llvmDescTy = dyn_cast<LLVM::LLVMStructType>(convertedOutputType);
    if (!llvmDescTy) {
      return failure();
    }

    auto pointType =
        dyn_cast<PointTypeInterface>(outputMemRef.getElementType());
    bool toPoints = pointType != nullptr;
    if (!toPoints) {
      pointType = dyn_cast<PointTypeInterface>(inputMemRef.getElementType());
    }
    if (!pointType) {
      return op.emitOpError("expected a point type on one side of the bitcast");
    }
    unsigned k = pointType.getNumCoords() *
                 field::getExtensionDegree(pointType.getBaseFieldType());

    MemRefDescriptor inputDesc(adaptor.getInput());
    auto outputDesc = MemRefDescriptor::poison(rewriter, loc, llvmDescTy);
    outputDesc.setAllocatedPtr(rewriter, loc,
                               inputDesc.allocatedPtr(rewriter, loc));
    outputDesc.setAlignedPtr(rewriter, loc,
                             inputDesc.alignedPtr(rewriter, loc));

    // The offset counts elements, so it converts with the element size: k field
    // elements per point going one way, the reverse going the other.
    Value offset = inputDesc.offset(rewriter, loc);
    if (k != 1) {
      Type idxTy = getTypeConverter()->getIndexType();
      Value ratio = createIndexAttrConstant(rewriter, loc, idxTy, k);
      offset =
          toPoints
              ? LLVM::SDivOp::create(rewriter, loc, offset, ratio).getResult()
              : LLVM::MulOp::create(rewriter, loc, offset, ratio).getResult();
    }
    outputDesc.setOffset(rewriter, loc, offset);

    for (int64_t i = 0, e = outputMemRef.getRank(); i < e; ++i) {
      outputDesc.setConstantSize(rewriter, loc, i, outputMemRef.getDimSize(i));
    }
    int64_t stride = 1;
    for (int64_t i = outputMemRef.getRank() - 1; i >= 0; --i) {
      outputDesc.setConstantStride(rewriter, loc, i, stride);
      stride *= outputMemRef.getDimSize(i);
    }

    rewriter.replaceOp(op, Value(outputDesc));
    return success();
  }
};

#include "prime_ir/Dialect/EllipticCurve/Conversions/EllipticCurveToLLVM/EllipticCurveToLLVM.cpp.inc"
} // namespace

void populateEllipticCurveToLLVMTypeConversion(
    LLVMTypeConverter &typeConverter) {
  typeConverter.addConversion(
      [](field::PrimeFieldType type) { return type.getStorageType(); });
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
      ConvertBitcast,
      ConvertFromCoords,
      ConvertToCoords
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
  explicit EllipticCurveToLLVMDialectInterface(mlir::Dialect *dialect)
      : ConvertToLLVMPatternInterface(dialect) {}

  void loadDependentDialects(mlir::MLIRContext *context) const final {
    context->loadDialect<LLVM::LLVMDialect>();
  }

  // Hook for derived dialect interface to provide conversion patterns and mark
  // dialect legal for the conversion target.
  void populateConvertToLLVMConversionPatterns(
      ConversionTarget &target, LLVMTypeConverter &typeConverter,
      RewritePatternSet &patterns) const final {
    // Without this the ops stay legal and the patterns below never fire, which
    // is silent rather than fatal: the op survives to the end of the pipeline
    // still carrying its point and field types.
    target.addIllegalOp<FromCoordsOp, ToCoordsOp>();
    // Only the memref form has a descriptor to rebuild. A tensor-typed bitcast
    // has no LLVM type to convert to, so demanding one here would turn an op
    // this pass has nothing to say about into a hard failure.
    target.addDynamicallyLegalOp<BitcastOp>(
        [](BitcastOp op) { return !isMemRefBitcast(op); });
    populateEllipticCurveToLLVMTypeConversion(typeConverter);
    populateEllipticCurveToLLVMConversionPatterns(typeConverter, patterns);
  }
};

void registerConvertEllipticCurveToLLVMInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, EllipticCurveDialect *dialect) {
    dialect->addInterfaces<EllipticCurveToLLVMDialectInterface>();
  });
}
} // namespace mlir::prime_ir::elliptic_curve
