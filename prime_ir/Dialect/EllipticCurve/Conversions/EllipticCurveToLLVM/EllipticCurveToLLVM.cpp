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

struct ConvertBitcast : public ConvertOpToLLVMPattern<BitcastOp> {
  using ConvertOpToLLVMPattern<BitcastOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(BitcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type convertedOutputType = typeConverter->convertType(op.getType());
    if (!convertedOutputType) {
      return op.emitOpError("failed to convert output type");
    }

    // Memref bitcasts (produced by bufferization) reinterpret one buffer as a
    // tensor of N points <-> N*K coordinates. The element COUNT changes (K
    // coordinates per point), so the descriptor must be rebuilt with
    // sizes/strides/offset in the output element's units. A plain
    // unrealized_conversion_cast preserves the input descriptor — sizes still
    // in input-element units — and any later dealloc/copy then computes the
    // wrong byte count (heap corruption). Mirrors
    // field::ConvertBitcast::convertMemRefBitcast.
    if (isa<MemRefType>(op.getInput().getType()))
      return convertMemRefBitcast(op, adaptor, rewriter, convertedOutputType);

    // Value-level reinterpret: the memory is identical and the input already
    // has the right LLVM struct type. reconcile-unrealized-casts cleans it up.
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        op, convertedOutputType, adaptor.getInput());
    return success();
  }

private:
  // Number of base-field coordinates an element type spans: an EC point has
  // getNumCoords() (affine 2, jacobian 3, xyzz 4); a field coordinate is 1.
  static unsigned getNumCoords(Type elementType) {
    if (auto pt = dyn_cast<PointTypeInterface>(elementType))
      return pt.getNumCoords();
    return 1;
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

    MemRefDescriptor inputDesc(adaptor.getInput());

    // Same-shape reinterpret: descriptor unchanged. Unreachable for EC -- a
    // point has K>=2 coords, so point<->coordinate element counts always
    // differ -- but kept as a defensive mirror of field::convertMemRefBitcast,
    // where the equal-byte-size case (e.g. ef4 <-> i128) is reachable.
    if (inputMemRef.getShape() == outputMemRef.getShape()) {
      rewriter.replaceOp(op, adaptor.getInput());
      return success();
    }

    // Point tensor <-> coordinate tensor: rebuild the descriptor sharing the
    // buffer, adjusting offset by the coordinate-count ratio and resetting
    // sizes/strides to the contiguous output shape.
    auto outputDesc = MemRefDescriptor::poison(rewriter, loc, llvmDescTy);

    // Same underlying memory — pointers are shared.
    outputDesc.setAllocatedPtr(rewriter, loc,
                               inputDesc.allocatedPtr(rewriter, loc));
    outputDesc.setAlignedPtr(rewriter, loc,
                             inputDesc.alignedPtr(rewriter, loc));

    // Adjust offset: offset_out = offset_in × coordsIn / coordsOut, where
    // coords is the base-field coordinate count of the respective element type.
    unsigned coordsIn = getNumCoords(inputMemRef.getElementType());
    unsigned coordsOut = getNumCoords(outputMemRef.getElementType());

    Value offset = inputDesc.offset(rewriter, loc);
    if (coordsIn != coordsOut) {
      Type idxTy = getTypeConverter()->getIndexType();
      if (coordsOut > coordsIn) {
        // coords -> point: divide offset (output counts fewer, larger elems).
        Value ratio =
            createIndexAttrConstant(rewriter, loc, idxTy, coordsOut / coordsIn);
        offset = LLVM::SDivOp::create(rewriter, loc, offset, ratio);
      } else {
        // point -> coords: multiply offset (output counts more, smaller elems).
        Value ratio =
            createIndexAttrConstant(rewriter, loc, idxTy, coordsIn / coordsOut);
        offset = LLVM::MulOp::create(rewriter, loc, offset, ratio);
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
