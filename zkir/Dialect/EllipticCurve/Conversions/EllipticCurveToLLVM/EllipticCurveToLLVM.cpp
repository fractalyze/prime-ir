#include "zkir/Dialect/EllipticCurve/Conversions/EllipticCurveToLLVM/EllipticCurveToLLVM.h"

#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveDialect.h"
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveOps.h"
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveTypes.h"
#include "zkir/Dialect/Field/IR/FieldTypes.h"

namespace mlir::zkir::elliptic_curve {

#define GEN_PASS_DEF_ELLIPTICCURVETOLLVM
#include "zkir/Dialect/EllipticCurve/Conversions/EllipticCurveToLLVM/EllipticCurveToLLVM.h.inc"

using namespace mlir::LLVM;

//===----------------------------------------------------------------------===//
// StructBuilder implementations.
//===----------------------------------------------------------------------===//

namespace {
constexpr unsigned kXPosInPointStruct = 0;
constexpr unsigned kYPosInPointStruct = 1;
constexpr unsigned kZPosInPointStruct = 2;
constexpr unsigned kZzPosInPointStruct = 2;
constexpr unsigned kZzzPosInPointStruct = 3;
} // namespace

// static
AffineStructBuilder AffineStructBuilder::poison(OpBuilder &builder,
                                                Location loc, Type type) {
  Value val = builder.create<LLVM::PoisonOp>(loc, type);
  return AffineStructBuilder(val);
}

void AffineStructBuilder::setX(OpBuilder &builder, Location loc, Value x) {
  setPtr(builder, loc, kXPosInPointStruct, x);
}

Value AffineStructBuilder::x(OpBuilder &builder, Location loc) {
  return extractPtr(builder, loc, kXPosInPointStruct);
}

void AffineStructBuilder::setY(OpBuilder &builder, Location loc, Value y) {
  setPtr(builder, loc, kYPosInPointStruct, y);
}

Value AffineStructBuilder::y(OpBuilder &builder, Location loc) {
  return extractPtr(builder, loc, kYPosInPointStruct);
}

// static
JacobianStructBuilder JacobianStructBuilder::poison(OpBuilder &builder,
                                                    Location loc, Type type) {
  Value val = builder.create<LLVM::PoisonOp>(loc, type);
  return JacobianStructBuilder(val);
}

void JacobianStructBuilder::setX(OpBuilder &builder, Location loc, Value x) {
  setPtr(builder, loc, kXPosInPointStruct, x);
}

Value JacobianStructBuilder::x(OpBuilder &builder, Location loc) {
  return extractPtr(builder, loc, kXPosInPointStruct);
}

void JacobianStructBuilder::setY(OpBuilder &builder, Location loc, Value y) {
  setPtr(builder, loc, kYPosInPointStruct, y);
}

Value JacobianStructBuilder::y(OpBuilder &builder, Location loc) {
  return extractPtr(builder, loc, kYPosInPointStruct);
}

void JacobianStructBuilder::setZ(OpBuilder &builder, Location loc, Value z) {
  setPtr(builder, loc, kZPosInPointStruct, z);
}

Value JacobianStructBuilder::z(OpBuilder &builder, Location loc) {
  return extractPtr(builder, loc, kZPosInPointStruct);
}

// static
XYZZStructBuilder XYZZStructBuilder::poison(OpBuilder &builder, Location loc,
                                            Type type) {
  Value val = builder.create<LLVM::PoisonOp>(loc, type);
  return XYZZStructBuilder(val);
}

void XYZZStructBuilder::setX(OpBuilder &builder, Location loc, Value x) {
  setPtr(builder, loc, kXPosInPointStruct, x);
}

Value XYZZStructBuilder::x(OpBuilder &builder, Location loc) {
  return extractPtr(builder, loc, kXPosInPointStruct);
}

void XYZZStructBuilder::setY(OpBuilder &builder, Location loc, Value y) {
  setPtr(builder, loc, kYPosInPointStruct, y);
}

Value XYZZStructBuilder::y(OpBuilder &builder, Location loc) {
  return extractPtr(builder, loc, kYPosInPointStruct);
}

void XYZZStructBuilder::setZz(OpBuilder &builder, Location loc, Value zz) {
  setPtr(builder, loc, kZzPosInPointStruct, zz);
}

Value XYZZStructBuilder::zz(OpBuilder &builder, Location loc) {
  return extractPtr(builder, loc, kZzPosInPointStruct);
}

void XYZZStructBuilder::setZzz(OpBuilder &builder, Location loc, Value zzz) {
  setPtr(builder, loc, kZzzPosInPointStruct, zzz);
}

Value XYZZStructBuilder::zzz(OpBuilder &builder, Location loc) {
  return extractPtr(builder, loc, kZzzPosInPointStruct);
}

//===----------------------------------------------------------------------===//
// Conversion patterns.
//===----------------------------------------------------------------------===//
namespace {
template <typename T>
Type convertPointType(T type) {
  Type baseFieldType = type.getCurve().getBaseField();
  Type integerType =
      cast<field::PrimeFieldType>(baseFieldType).getModulus().getType();
  if constexpr (std::is_same_v<T, AffineType>) {
    return LLVM::LLVMStructType::getLiteral(type.getContext(),
                                            {integerType, integerType});
  } else if constexpr (std::is_same_v<T, JacobianType>) {
    return LLVM::LLVMStructType::getLiteral(
        type.getContext(), {integerType, integerType, integerType});

  } else if constexpr (std::is_same_v<T, XYZZType>) {
    return LLVM::LLVMStructType::getLiteral(
        type.getContext(),
        {integerType, integerType, integerType, integerType});
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
      auto pointStruct = AffineStructBuilder::poison(rewriter, loc, structType);
      pointStruct.setX(rewriter, loc, adaptor.getCoords()[0]);
      pointStruct.setY(rewriter, loc, adaptor.getCoords()[1]);
      rewriter.replaceOp(op, {pointStruct});
    } else if (isa<JacobianType>(op.getType())) {
      auto pointStruct =
          JacobianStructBuilder::poison(rewriter, loc, structType);
      pointStruct.setX(rewriter, loc, adaptor.getCoords()[0]);
      pointStruct.setY(rewriter, loc, adaptor.getCoords()[1]);
      pointStruct.setZ(rewriter, loc, adaptor.getCoords()[2]);
      rewriter.replaceOp(op, {pointStruct});
    } else if (isa<XYZZType>(op.getType())) {
      auto pointStruct = XYZZStructBuilder::poison(rewriter, loc, structType);
      pointStruct.setX(rewriter, loc, adaptor.getCoords()[0]);
      pointStruct.setY(rewriter, loc, adaptor.getCoords()[1]);
      pointStruct.setZz(rewriter, loc, adaptor.getCoords()[2]);
      pointStruct.setZzz(rewriter, loc, adaptor.getCoords()[3]);
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
      AffineStructBuilder affineStruct(adaptor.getInput());
      Value x = affineStruct.x(rewriter, op.getLoc());
      Value y = affineStruct.y(rewriter, op.getLoc());
      rewriter.replaceOpWithMultiple(op, {x, y});
    } else if (isa<JacobianType>(op.getInput().getType())) {
      JacobianStructBuilder jacobianStruct(adaptor.getInput());
      Value x = jacobianStruct.x(rewriter, op.getLoc());
      Value y = jacobianStruct.y(rewriter, op.getLoc());
      Value z = jacobianStruct.z(rewriter, op.getLoc());
      rewriter.replaceOpWithMultiple(op, {x, y, z});
    } else if (isa<XYZZType>(op.getInput().getType())) {
      XYZZStructBuilder xyzzStruct(adaptor.getInput());
      Value x = xyzzStruct.x(rewriter, op.getLoc());
      Value y = xyzzStruct.y(rewriter, op.getLoc());
      Value zz = xyzzStruct.zz(rewriter, op.getLoc());
      Value zzz = xyzzStruct.zzz(rewriter, op.getLoc());
      rewriter.replaceOpWithMultiple(op, {x, y, zz, zzz});
    }
    return success();
  }
};

#include "zkir/Dialect/EllipticCurve/Conversions/EllipticCurveToLLVM/EllipticCurveToLLVM.cpp.inc"
} // namespace

void populateEllipticCurveToLLVMTypeConversion(
    LLVMTypeConverter &typeConverter) {
  typeConverter.addConversion(
      [](AffineType type) { return convertPointType(type); });
  typeConverter.addConversion(
      [](JacobianType type) { return convertPointType(type); });
  typeConverter.addConversion(
      [](XYZZType type) { return convertPointType(type); });
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
  registry.addExtension(
      +[](MLIRContext *ctx,
          zkir::elliptic_curve::EllipticCurveDialect *dialect) {
        dialect->addInterfaces<EllipticCurveToLLVMDialectInterface>();
      });
}
} // namespace mlir::zkir::elliptic_curve
