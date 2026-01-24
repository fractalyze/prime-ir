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

#include "prime_ir/Dialect/Field/Conversions/FieldVectorize/FieldVectorize.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "prime_ir/Dialect/Field/IR/FieldDialect.h"
#include "prime_ir/Dialect/Field/IR/FieldOps.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"
#include "prime_ir/Utils/ConversionUtils.h"
#include "prime_ir/Utils/ShapedTypeConverter.h"

namespace mlir::prime_ir::field {

#define GEN_PASS_DEF_FIELDVECTORIZE
#include "prime_ir/Dialect/Field/Conversions/FieldVectorize/FieldVectorize.h.inc"

namespace {

/// Type converter that transforms scalar field types to vector field types.
/// - PrimeFieldType -> VectorType<N x PrimeFieldType>
/// - memref<M x !pf> -> memref<M x vector<N x !pf>>
/// - tensor<M x !pf> -> tensor<M x vector<N x !pf>>
class FieldVectorizeTypeConverter : public ShapedTypeConverter {
public:
  explicit FieldVectorizeTypeConverter(MLIRContext *ctx, unsigned vectorWidth)
      : vectorWidth_(vectorWidth) {
    // Identity conversion for non-field types
    addConversion([](Type type) { return type; });

    // Scalar prime field -> vector<N x prime field>
    addConversion([this](PrimeFieldType type) -> Type {
      return VectorType::get({static_cast<int64_t>(vectorWidth_)}, type);
    });

    // Target materialization: scalar field -> vector field via splat
    // This handles cases where we have a scalar field value (e.g., from
    // tensor.extract) and need to use it where a vector is expected.
    addTargetMaterialization([](OpBuilder &builder, Type resultType,
                                ValueRange inputs, Location loc) -> Value {
      if (inputs.size() != 1)
        return nullptr;

      Value input = inputs[0];
      auto vecType = dyn_cast<VectorType>(resultType);
      if (!vecType)
        return nullptr;

      // Check if input is a scalar field and result is a vector of that field
      if (isa<PrimeFieldType>(input.getType()) &&
          isa<PrimeFieldType>(vecType.getElementType())) {
        return builder.create<vector::SplatOp>(loc, vecType, input);
      }
      return nullptr;
    });

    // Handle shaped types (memref only, not tensors)
    // Tensors are kept as scalar because constant tensors use bitcast which
    // doesn't support tensor<vector<field>>. We splat at extraction instead.
    // VectorType with field elements should NOT be converted (would create
    // nested vectors)
    addConversion([this](ShapedType type) -> Type {
      // VectorType with field elements stays as-is - it's already vectorized
      // (nested vectors are not supported)
      if (isa<VectorType>(type)) {
        return type;
      }

      // Only convert memrefs, not tensors
      // Tensors containing field elements stay scalar (for constant tables)
      // The splat happens at tensor.extract
      if (!isa<MemRefType>(type)) {
        return type;
      }

      Type elementType = type.getElementType();

      // Direct field element type in memref
      if (auto pfType = dyn_cast<PrimeFieldType>(elementType)) {
        auto vecType =
            VectorType::get({static_cast<int64_t>(vectorWidth_)}, pfType);
        return convertShapedType(type, type.getShape(), vecType);
      }

      // Already a vector of field elements - don't double-vectorize
      if (auto vecType = dyn_cast<VectorType>(elementType)) {
        if (isa<PrimeFieldType>(vecType.getElementType())) {
          return type; // Already vectorized
        }
      }

      return type;
    });
  }

  unsigned getVectorWidth() const { return vectorWidth_; }

private:
  unsigned vectorWidth_;
};

/// Pattern for BitcastOp that handles type conversion.
/// Tensor bitcasts (tensor<i32> -> tensor<!pf>) are kept as-is since
/// we don't vectorize tensors (constant tables stay scalar).
/// Only scalar/vector bitcasts are converted.
struct ConvertBitcast : public OpConversionPattern<BitcastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BitcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = op.getType();

    // Get the converted result type
    Type convertedResultType = getTypeConverter()->convertType(resultType);
    if (!convertedResultType)
      return failure();

    // If no conversion needed, fail to let other patterns handle it
    if (convertedResultType == resultType) {
      return failure();
    }

    // Create new bitcast with converted type
    rewriter.replaceOpWithNewOp<BitcastOp>(op, convertedResultType,
                                           adaptor.getInput());
    return success();
  }
};

/// Pattern to handle tensor.extract when source is bufferization.to_tensor.
/// Case 1: Adapted tensor has vector elements -> extract vector directly
/// Case 2: Original source is from to_tensor (scalar) -> use memref.load
/// For constant tensor extracts, they are legal and target materialization adds
/// splat.
struct ConvertTensorExtract : public OpConversionPattern<tensor::ExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tensor::ExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = op.getType();

    // Only handle prime field results
    auto pfType = dyn_cast<PrimeFieldType>(resultType);
    if (!pfType) {
      return failure();
    }

    // Check the adapted tensor type
    auto tensorType = dyn_cast<RankedTensorType>(adaptor.getTensor().getType());
    if (!tensorType)
      return failure();

    // Case 1: Adapted tensor has vector elements (from converted to_tensor)
    // Extract gives vector directly
    if (auto vecElemType = dyn_cast<VectorType>(tensorType.getElementType())) {
      if (isa<PrimeFieldType>(vecElemType.getElementType())) {
        rewriter.replaceOpWithNewOp<tensor::ExtractOp>(
            op, vecElemType, adaptor.getTensor(), adaptor.getIndices());
        return success();
      }
    }

    // Case 2: Tensor has scalar elements but source is from to_tensor
    // This shouldn't happen after ConvertToTensor runs, but handle it
    if (isa<PrimeFieldType>(tensorType.getElementType())) {
      auto toTensorOp =
          op.getTensor().getDefiningOp<bufferization::ToTensorOp>();
      if (toTensorOp) {
        Value memref = toTensorOp.getBuffer();

        // Get the converted memref type (should have vector elements)
        Type convertedMemrefType =
            getTypeConverter()->convertType(memref.getType());
        auto memrefType = dyn_cast<MemRefType>(convertedMemrefType);
        if (!memrefType)
          return failure();

        auto memrefVecElem = dyn_cast<VectorType>(memrefType.getElementType());
        if (!memrefVecElem ||
            !isa<PrimeFieldType>(memrefVecElem.getElementType())) {
          return failure();
        }

        // Get the remapped memref value (converted type)
        Value adaptedMemref = rewriter.getRemappedValue(memref);
        if (!adaptedMemref)
          adaptedMemref = memref; // Fall back to original if not remapped

        // Load directly from the vectorized memref
        auto load = rewriter.create<memref::LoadOp>(
            op.getLoc(), memrefVecElem, adaptedMemref, adaptor.getIndices());
        rewriter.replaceOp(op, load);
        return success();
      }
    }

    return failure();
  }
};

/// Pattern to handle bufferization.to_tensor when the memref operand is
/// vectorized. Creates a new to_tensor with tensor<vector<...>> output matching
/// the vectorized memref.
struct ConvertToTensor : public OpConversionPattern<bufferization::ToTensorOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(bufferization::ToTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Check if the buffer operand type changed
    auto adaptedMemrefType =
        dyn_cast<MemRefType>(adaptor.getBuffer().getType());
    if (!adaptedMemrefType)
      return failure();

    auto origMemrefType = dyn_cast<MemRefType>(op.getBuffer().getType());
    if (!origMemrefType)
      return failure();

    // If types are the same, no conversion needed
    if (adaptedMemrefType == origMemrefType)
      return failure();

    // Check if the memref was vectorized
    auto vecElemType = dyn_cast<VectorType>(adaptedMemrefType.getElementType());
    if (!vecElemType || !isa<PrimeFieldType>(vecElemType.getElementType()))
      return failure();

    // Create tensor type matching the vectorized memref
    auto tensorType =
        RankedTensorType::get(adaptedMemrefType.getShape(), vecElemType);

    // Create a new to_tensor with matching types
    rewriter.replaceOpWithNewOp<bufferization::ToTensorOp>(
        op, tensorType, adaptor.getBuffer(), op.getRestrictAttr(),
        op.getWritableAttr());
    return success();
  }
};

/// Pattern to handle bufferization.to_buffer when the target memref is
/// vectorized.
struct ConvertToBuffer : public OpConversionPattern<bufferization::ToBufferOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(bufferization::ToBufferOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto convertedMemrefType = getTypeConverter()->convertType(op.getType());
    if (!convertedMemrefType)
      return failure();

    rewriter.replaceOpWithNewOp<bufferization::ToBufferOp>(
        op, convertedMemrefType, adaptor.getTensor(), op.getReadOnlyAttr());
    return success();
  }
};

/// Pattern to convert field constants by broadcasting them to vectors.
/// field.constant 42 : !pf -> vector.splat (field.constant 42) : vector<N x
/// !pf>
struct ConvertFieldConstant : public OpConversionPattern<ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type origType = op.getType();

    // Only handle scalar prime field constants
    if (!isa<PrimeFieldType>(origType)) {
      return failure();
    }

    Type convertedType = getTypeConverter()->convertType(origType);
    if (!convertedType || convertedType == origType) {
      return failure();
    }

    auto vecType = cast<VectorType>(convertedType);

    // Create a dense attribute for the splatted vector constant
    // We create an arith.constant with i32 vector, then bitcast to field vector
    auto intType =
        IntegerType::get(rewriter.getContext(),
                         cast<PrimeFieldType>(origType).getStorageBitWidth());
    auto vecIntType = VectorType::get(vecType.getShape(), intType);

    auto intAttr = op.getValueAttr();
    auto splatAttr = DenseElementsAttr::get(vecIntType, intAttr);

    auto arithConst =
        rewriter.create<arith::ConstantOp>(op.getLoc(), splatAttr);
    auto bitcast =
        rewriter.create<BitcastOp>(op.getLoc(), convertedType, arithConst);
    rewriter.replaceOp(op, bitcast);
    return success();
  }
};

struct FieldVectorize : impl::FieldVectorizeBase<FieldVectorize> {
  using FieldVectorizeBase::FieldVectorizeBase;

  void runOnOperation() override;
};

void FieldVectorize::runOnOperation() {
  func::FuncOp func = getOperation();
  MLIRContext *context = &getContext();

  FieldVectorizeTypeConverter typeConverter(context, vectorWidth);

  ConversionTarget target(*context);

  // Field dialect operations are legal if their types are converted
  target.addDynamicallyLegalDialect<FieldDialect>(
      [&](Operation *op) { return typeConverter.isLegal(op); });

  // Standard dialects remain legal if their types are converted
  target.addDynamicallyLegalDialect<
      arith::ArithDialect, func::FuncDialect, memref::MemRefDialect,
      tensor::TensorDialect, vector::VectorDialect, affine::AffineDialect,
      scf::SCFDialect, bufferization::BufferizationDialect>(
      [&](Operation *op) { return typeConverter.isLegal(op); });

  RewritePatternSet patterns(context);

  // Add specialized patterns (higher priority)
  patterns.add<ConvertFieldConstant>(typeConverter, context);
  patterns.add<ConvertBitcast>(typeConverter, context);
  patterns.add<ConvertTensorExtract>(typeConverter, context);
  patterns.add<ConvertToTensor>(typeConverter, context);
  patterns.add<ConvertToBuffer>(typeConverter, context);

  // Add generic patterns for field operations
  // These operations support vectors via ElementwiseMappable trait
  patterns.add<
      // clang-format off
      ConvertAny<AddOp>,
      ConvertAny<SubOp>,
      ConvertAny<MulOp>,
      ConvertAny<DoubleOp>,
      ConvertAny<SquareOp>,
      ConvertAny<NegateOp>,
      ConvertAny<InverseOp>,
      ConvertAny<PowUIOp>,
      ConvertAny<ToMontOp>,
      ConvertAny<FromMontOp>,
      ConvertAny<CmpOp>,
      // Memory operations
      ConvertAny<memref::AllocOp>,
      ConvertAny<memref::AllocaOp>,
      ConvertAny<memref::LoadOp>,
      ConvertAny<memref::StoreOp>,
      ConvertAny<memref::SubViewOp>,
      ConvertAny<memref::CastOp>,
      ConvertAny<memref::CopyOp>,
      // Tensor operations (tensor.extract handled by ConvertTensorExtract)
      ConvertAny<tensor::EmptyOp>,
      ConvertAny<tensor::FromElementsOp>,
      ConvertAny<tensor::InsertOp>,
      ConvertAny<tensor::InsertSliceOp>,
      ConvertAny<tensor::ExtractSliceOp>,
      ConvertAny<tensor::CastOp>,
      ConvertAny<tensor::YieldOp>,
      // Vector operations
      ConvertAny<vector::SplatOp>,
      ConvertAny<vector::BroadcastOp>,
      // Affine operations
      ConvertAny<affine::AffineForOp>,
      ConvertAny<affine::AffineLoadOp>,
      ConvertAny<affine::AffineStoreOp>,
      ConvertAny<affine::AffineYieldOp>,
      // SCF operations
      ConvertAny<scf::ForOp>,
      ConvertAny<scf::YieldOp>,
      ConvertAny<scf::IfOp>,
      // Bufferization operations handled by specialized patterns
      // Arith operations that might use field types
      ConvertAny<arith::SelectOp>
      // clang-format on
      >(typeConverter, context);

  // Add structural conversion patterns for func dialect
  addStructuralConversionPatterns(typeConverter, patterns, target);

  // BitcastOp is legal if types are converted
  target.addDynamicallyLegalOp<BitcastOp>(
      [&](BitcastOp op) { return typeConverter.isLegal(op); });

  // tensor.extract with scalar field result needs conversion when:
  // - Source tensor has scalar field elements (will be vectorized) -> need to
  // extract vector
  // - Source is constant tensor -> legal, target materialization adds splat
  // when result is used
  target.addDynamicallyLegalOp<tensor::ExtractOp>([&](tensor::ExtractOp op) {
    Type resultType = op.getType();
    if (!isa<PrimeFieldType>(resultType)) {
      return true; // Non-field types are legal
    }

    auto tensorType = dyn_cast<RankedTensorType>(op.getTensor().getType());
    if (!tensorType) {
      return true;
    }

    // If tensor has scalar field elements
    if (isa<PrimeFieldType>(tensorType.getElementType())) {
      // From to_tensor (will be converted to tensor<vector<...>>) -> needs
      // conversion
      if (op.getTensor().getDefiningOp<bufferization::ToTensorOp>()) {
        return false;
      }
      // Constant tensor - legal, materialization adds splat
      return true;
    }

    // If tensor has vector field elements, needs conversion (extract vector
    // result)
    if (auto vecElem = dyn_cast<VectorType>(tensorType.getElementType())) {
      if (isa<PrimeFieldType>(vecElem.getElementType())) {
        return false; // Need to extract with vector result type
      }
    }

    return true;
  });

  // bufferization.to_tensor needs conversion when the memref operand will be
  // vectorized. The pattern updates the operand type while keeping the scalar
  // tensor output.
  target.addDynamicallyLegalOp<bufferization::ToTensorOp>(
      [&](bufferization::ToTensorOp op) {
        auto memrefType = dyn_cast<MemRefType>(op.getBuffer().getType());
        if (!memrefType)
          return true;
        // If memref has scalar field elements, it will be vectorized -> needs
        // conversion
        if (isa<PrimeFieldType>(memrefType.getElementType())) {
          return false; // Needs pattern to update operand type
        }
        // If memref already has vector field elements, already converted
        if (auto vecElem = dyn_cast<VectorType>(memrefType.getElementType())) {
          if (isa<PrimeFieldType>(vecElem.getElementType())) {
            return true; // Already converted
          }
        }
        return true;
      });

  // bufferization.to_buffer is legal if types are converted
  target.addDynamicallyLegalOp<bufferization::ToBufferOp>(
      [&](bufferization::ToBufferOp op) { return typeConverter.isLegal(op); });

  // Mark operations as dynamically legal
  target.addDynamicallyLegalOp<
      // clang-format off
      AddOp, SubOp, MulOp, DoubleOp, SquareOp, NegateOp,
      InverseOp, PowUIOp, ToMontOp, FromMontOp, CmpOp,
      memref::AllocOp, memref::AllocaOp, memref::LoadOp, memref::StoreOp,
      memref::SubViewOp, memref::CastOp, memref::CopyOp,
      tensor::EmptyOp, tensor::FromElementsOp,
      tensor::InsertOp, tensor::InsertSliceOp, tensor::ExtractSliceOp,
      tensor::CastOp, tensor::YieldOp,
      vector::SplatOp, vector::BroadcastOp,
      affine::AffineForOp, affine::AffineLoadOp, affine::AffineStoreOp,
      affine::AffineYieldOp,
      scf::ForOp, scf::YieldOp, scf::IfOp,
      arith::SelectOp
      // clang-format on
      >([&](auto op) { return typeConverter.isLegal(op); });

  if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

} // namespace

} // namespace mlir::prime_ir::field
