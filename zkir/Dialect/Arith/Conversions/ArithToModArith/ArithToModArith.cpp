#include "zkir/Dialect/Arith/Conversions/ArithToModArith/ArithToModArith.h"

#include <cassert>
#include <cstdint>
#include <utility>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "zkir/Dialect/ModArith/IR/ModArithDialect.h"
#include "zkir/Dialect/ModArith/IR/ModArithOps.h"
#include "zkir/Dialect/ModArith/IR/ModArithTypes.h"
#include "zkir/Utils/ConversionUtils.h"

namespace mlir::zkir::arith {

#define GEN_PASS_DEF_ARITHTOMODARITH
#include "zkir/Dialect/Arith/Conversions/ArithToModArith/ArithToModArith.h.inc"

static mod_arith::ModArithType convertArithType(Type type) {
  auto modulusBitSize = static_cast<uint64_t>(type.getIntOrFloatBitWidth());
  auto modulus = (1L << (modulusBitSize - 1L));
  auto newType = mlir::IntegerType::get(type.getContext(), modulusBitSize + 1);

  return mod_arith::ModArithType::get(type.getContext(),
                                      mlir::IntegerAttr::get(newType, modulus));
}

static Type convertArithLikeType(ShapedType type) {
  if (auto arithType = llvm::dyn_cast<IntegerType>(type.getElementType())) {
    return type.cloneWith(type.getShape(), convertArithType(arithType));
  }
  return type;
}

static Value buildLoadOps(OpBuilder &builder, Type resultTypes,
                          ValueRange inputs, Location loc) {
  assert(inputs.size() == 1);
  auto loadOp = inputs[0].getDefiningOp<memref::LoadOp>();

  if (!loadOp) return {};

  auto *globaMemReflOp = loadOp.getMemRef().getDefiningOp();

  if (!globaMemReflOp) return {};

  return builder.create<mod_arith::EncapsulateOp>(
      loc, convertArithType(loadOp.getType()), loadOp.getResult());
}

class ArithToModArithTypeConverter : public TypeConverter {
 public:
  explicit ArithToModArithTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion([](IntegerType type) -> mod_arith::ModArithType {
      return convertArithType(type);
    });
    addConversion(
        [](ShapedType type) -> Type { return convertArithLikeType(type); });
    addTargetMaterialization(buildLoadOps);
  }
};

struct ConvertConstant : public OpConversionPattern<mlir::arith::ConstantOp> {
  explicit ConvertConstant(mlir::MLIRContext *context)
      : OpConversionPattern<mlir::arith::ConstantOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ::mlir::arith::ConstantOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    if (isa<IndexType>(op.getValue().getType())) {
      return failure();
    }

    auto result = b.create<mod_arith::ConstantOp>(
        convertArithType(op.getType()), op.getValue());

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ConvertExtSI : public OpConversionPattern<mlir::arith::ExtSIOp> {
  explicit ConvertExtSI(mlir::MLIRContext *context)
      : OpConversionPattern<mlir::arith::ExtSIOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ::mlir::arith::ExtSIOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto result = b.create<mod_arith::ModSwitchOp>(
        op.getLoc(), convertArithType(op.getType()), adaptor.getIn());
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ConvertExtUI : public OpConversionPattern<mlir::arith::ExtUIOp> {
  explicit ConvertExtUI(mlir::MLIRContext *context)
      : OpConversionPattern<mlir::arith::ExtUIOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ::mlir::arith::ExtUIOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto result = b.create<mod_arith::ModSwitchOp>(
        op.getLoc(), convertArithType(op.getType()), adaptor.getIn());
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ConvertLoadOp : public OpConversionPattern<mlir::memref::LoadOp> {
  explicit ConvertLoadOp(mlir::MLIRContext *context)
      : OpConversionPattern<mlir::memref::LoadOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ::mlir::memref::LoadOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto defineOp = op.getMemRef().getDefiningOp();

    if (op.getMemRef().getDefiningOp()) {
      if (isa<memref::GetGlobalOp>(defineOp)) {
        // skip global memref
        return success();
      }
    }

    auto result = rewriter.create<memref::LoadOp>(
        op.getLoc(), convertArithType(op.getType()), adaptor.getOperands()[0],
        op.getIndices());

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithToModArith : impl::ArithToModArithBase<ArithToModArith> {
  using ArithToModArithBase::ArithToModArithBase;

  void runOnOperation() override;
};

void ArithToModArith::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();
  ArithToModArithTypeConverter typeConverter(context);

  ConversionTarget target(*context);
  target.addLegalDialect<mod_arith::ModArithDialect>();
  target.addIllegalDialect<mlir::arith::ArithDialect>();

  target.addDynamicallyLegalOp<mlir::arith::ConstantOp>(
      [](mlir::arith::ConstantOp op) {
        return isa<IndexType>(op.getValue().getType());
      });

  target.addDynamicallyLegalOp<memref::LoadOp>([&](Operation *op) {
    auto users = cast<memref::LoadOp>(op).getResult().getUsers();
    if (cast<memref::LoadOp>(op).getResult().getDefiningOp()) {
      if (isa<memref::GetGlobalOp>(
              cast<memref::LoadOp>(op).getResult().getDefiningOp())) {
        auto detectable = llvm::any_of(users, [](Operation *user) {
          return isa<mod_arith::EncapsulateOp>(user);
        });
        return detectable;
      }
    }

    return (typeConverter.isLegal(op->getOperandTypes()) &&
            typeConverter.isLegal(op->getResultTypes()));
  });

  target.addDynamicallyLegalOp<tensor::FromElementsOp>([&](Operation *op) {
    return typeConverter.isLegal(op->getOperandTypes()) &&
           typeConverter.isLegal(op->getResultTypes());
  });

  RewritePatternSet patterns(context);
  patterns.add<ConvertConstant, ConvertExtSI, ConvertExtUI,
               ConvertBinOp<mlir::arith::AddIOp, mod_arith::AddOp>,
               ConvertBinOp<mlir::arith::SubIOp, mod_arith::SubOp>,
               ConvertBinOp<mlir::arith::MulIOp, mod_arith::MulOp>,
               ConvertAny<tensor::FromElementsOp>, ConvertLoadOp>(typeConverter,
                                                                  context);

  addStructuralConversionPatterns(typeConverter, patterns, target);

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // namespace mlir::zkir::arith
