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

#include "zkir/Dialect/Poly/Conversions/PolyToField/PolyToField.h"

#include <utility>

#include "mlir/Dialect/GPU/Transforms/ParallelLoopMapper.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "zkir/Dialect/Field/IR/FieldAttributes.h"
#include "zkir/Dialect/Field/IR/FieldOps.h"
#include "zkir/Dialect/ModArith/IR/ModArithTypes.h"
#include "zkir/Dialect/Poly/IR/PolyAttributes.h"
#include "zkir/Dialect/Poly/IR/PolyDialect.h"
#include "zkir/Dialect/Poly/IR/PolyOps.h"
#include "zkir/Dialect/Poly/IR/PolyTypes.h"
#include "zkir/Dialect/TensorExt/IR/TensorExtOps.h"
#include "zkir/Utils/APIntUtils.h"
#include "zkir/Utils/ConversionUtils.h"

namespace mlir::zkir::poly {

#define GEN_PASS_DEF_POLYTOFIELD
#include "zkir/Dialect/Poly/Conversions/PolyToField/PolyToField.h.inc"

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

      auto padValue = cast<field::FieldTypeInterface>(typeInfo.coefficientType)
                          .createZeroConstant(b);
      coeffValue = b.create<tensor::PadOp>(typeInfo.tensorType, coeffValue, low,
                                           high, padValue,
                                           /*nofold=*/false);
    }

    rewriter.replaceOp(op, coeffValue);
    return success();
  }
};

// Butterfly : Cooley-Tukey
static std::pair<Value, Value> bflyCT(ImplicitLocOpBuilder &b, Value A, Value B,
                                      Value root) {
  Value rootB = b.create<field::MulOp>(B, root);
  auto ctPlus = b.create<field::AddOp>(A, rootB);
  auto ctMinus = b.create<field::SubOp>(A, rootB);
  return {std::move(ctPlus), std::move(ctMinus)};
}

// Butterfly : Gentleman-Sande
static std::pair<Value, Value> bflyGS(ImplicitLocOpBuilder &b, Value A, Value B,
                                      Value root) {
  auto gsPlus = b.create<field::AddOp>(A, B);
  auto gsMinus = b.create<field::SubOp>(A, B);
  Value gsMinusRoot = b.create<field::MulOp>(gsMinus, root);
  return {std::move(gsPlus), std::move(gsMinusRoot)};
}

static Value fastNTT(ImplicitLocOpBuilder &b, NTTOpAdaptor adaptor,
                     Value source, Value dest, Attribute gpuMappingAttr) {
  auto tensorType = cast<RankedTensorType>(adaptor.getDest().getType());
  auto coeffType = cast<field::PrimeFieldType>(tensorType.getElementType());

  auto coeffStorageType = coeffType.getStorageType();
  auto intTensorType =
      RankedTensorType::get(tensorType.getShape(), coeffStorageType);

  bool kInverse = adaptor.getInverse();

  // -------------------------------------------------------------------------
  // Compute basic parameters and precompute the roots of unity.
  // -------------------------------------------------------------------------
  // `degree` is the number of coefficients (assumed to be a power of 2).
  unsigned degree = tensorType.getShape()[0];
  assert(llvm::has_single_bit(degree) &&
         "expected the number of coefficients to be a power of 2");

  // `stages` is the number of iterations required by the NTT.
  unsigned stages = llvm::countr_zero(degree);

  // -------------------------------------------------------------------------
  // Precompute the roots of unity in the given prime field.
  // -------------------------------------------------------------------------
  Value roots;
  if (adaptor.getTwiddles()) {
    roots = adaptor.getTwiddles();
  } else {
    assert(adaptor.getRoot() &&
           "Root of unity is required if no twiddles are provided");
    field::RootOfUnityAttr rootAttr = adaptor.getRoot().value();
    APInt cmod = coeffType.getModulus().getValue();
    APInt root = rootAttr.getRoot().getValue();

    mod_arith::MontgomeryAttr montAttr;
    if (coeffType.isMontgomery()) {
      montAttr = mod_arith::MontgomeryAttr::get(b.getContext(),
                                                coeffType.getModulus());
    }
    auto primitiveRootsAttr =
        PrimitiveRootAttr::get(b.getContext(), rootAttr, montAttr);

    // Create a tensor constant of precomputed roots for fast access during the
    // NTT.
    auto rootsType = intTensorType.clone({degree});
    roots = !kInverse
                ? b.create<arith::ConstantOp>(rootsType,
                                              primitiveRootsAttr.getRoots())
                : b.create<arith::ConstantOp>(rootsType,
                                              primitiveRootsAttr.getInvRoots());

    // Wrap the roots in a field encapsulation for further field operations.
    roots = b.create<field::BitcastOp>(tensorType, roots);
  }

  // -------------------------------------------------------------------------
  // Iterative NTT computation using a modified Cooley-Tukey / Gentleman-Sande
  // approach.
  //
  // We perform the NTT in log2(degree) stages.
  //
  // The algorithm works with the following parameters:
  //  - `batchSize` (or stride) determines the size of each butterfly block.
  //  - `rootExp` determines the step for selecting the twiddle factor from
  //  `roots`.
  //
  // The pseudocode is as follows:
  //    m = kInverse ? degree : 2           (initial batchSize)
  //    r = kInverse ? 1 : degree / 2         (initial root exponent)
  //    for s in 0 .. log2(degree)-1:
  //      for k in 0 .. degree/m - 1:
  //        for j in 0 .. m/2 - 1:
  //          A = coeffs[k*m + j]
  //          B = coeffs[k*m + j + m/2]
  //          // Compute twiddle factor from precomputed roots:
  //          root = roots[ (j * rootExp) ]
  //          (bfA, bfB) = bflyOp(A, B, root)
  //          coeffs[k*m + j] = bfA
  //          coeffs[k*m + j + m/2] = bfB
  //      Update:
  //         batchSize = kInverse ? batchSize/2 : batchSize*2
  //         rootExp   = kInverse ? rootExp*2   : batchSize/2
  // -------------------------------------------------------------------------

  // Initialize the loop control variables:
  // - For the forward NTT, we start with a `batchSize` of 2 and `rootExp` of
  // `degree`/2.
  // - For the inverse NTT, we start with a `batchSize` equal to the full
  // `degree` and `rootExp` of 1.
  Value initialBatchSize =
      b.create<arith::ConstantIndexOp>(kInverse ? degree : 2);
  Value initialRootExp =
      b.create<arith::ConstantIndexOp>(kInverse ? 1 : degree / 2);

  // Define constants for index calculations.
  auto c0 = b.create<arith::ConstantIndexOp>(0);
  auto c1 = b.create<arith::ConstantIndexOp>(1);
  auto c2 = b.create<arith::ConstantIndexOp>(2);
  auto cDegree = b.create<arith::ConstantIndexOp>(degree);
  auto cStages = b.create<arith::ConstantIndexOp>(stages);

  // Create a memref buffer for in-place updates
  auto memrefType = MemRefType::get(intTensorType.getShape(), coeffType);
  Value srcMemref = b.create<bufferization::ToBufferOp>(memrefType, source,
                                                        /*read_only=*/true);
  Value destMemref = b.create<bufferization::ToBufferOp>(memrefType, dest);

  // Begin the outer loop over the stages of the NTT.
  // The iterative loop carries three values:
  //   - The current batchSize,
  //   - The current root exponent (rootExp).
  b.create<scf::ForOp>(
      /*lowerBound=*/c0, /* upperBound=*/cStages,
      /*step=*/c1,
      /*initArgs=*/ValueRange{initialBatchSize, initialRootExp, srcMemref},
      /*bodyBuilder=*/
      [&](OpBuilder &nestedBuilder, Location nestedLoc, Value stageIndex,
          ValueRange args) {
        ImplicitLocOpBuilder b(nestedLoc, nestedBuilder);
        Value batchSize = args[0];
        Value rootExp = args[1];
        Value stageMemref = args[2];

        Value batchNum = b.create<arith::DivUIOp>(cDegree, batchSize);
        Value bflyNum = b.create<arith::DivUIOp>(batchSize, c2);

        Value tileX, tileY;
        // Adaptive tiling if `tileCap` is provided. Otherwise, don't tile.
        if (adaptor.getTileX()) {
          Value cTileXCap = b.create<arith::ConstantIndexOp>(
              adaptor.getTileX().value().getValue().getSExtValue());
          Value cGridSize =
              adaptor.getGridSize()
                  ? b.create<arith::ConstantIndexOp>(
                        adaptor.getGridSize().value().getValue().getSExtValue())
                  : b.create<arith::ConstantIndexOp>(1024);

          // Adaptive tiles.
          tileX = b.create<arith::MinUIOp>(bflyNum, cTileXCap);
          Value tileYCap = b.create<arith::DivUIOp>(cGridSize, tileX);
          tileY = b.create<arith::MinUIOp>(batchNum, tileYCap);
        } else {
          tileX = bflyNum;
          tileY = batchNum;
        }

        // Grid sizes.
        Value gridX = b.create<arith::CeilDivUIOp>(batchNum, tileY);
        Value gridY = b.create<arith::CeilDivUIOp>(bflyNum, tileX);

        // The inner loop processes groups of coefficients defined by the
        // current batchSize with a tile size of (tileX, tileY).
        auto parallelLoop = b.create<scf::ParallelOp>(
            /*lowerBounds=*/ValueRange{c0, c0, c0, c0},
            /*upperBounds=*/ValueRange{gridX, gridY, tileX, tileY},
            /*steps=*/ValueRange{c1, c1, c1, c1});

        // Build the body of the parallel loop.
        {
          Block &parallelBlock = parallelLoop.getRegion().front();
          OpBuilder parallelBuilder = OpBuilder::atBlockBegin(&parallelBlock);
          ImplicitLocOpBuilder pb(parallelBlock.getArgument(0).getLoc(),
                                  parallelBuilder);

          Value bx = parallelBlock.getArgument(0);
          Value by = parallelBlock.getArgument(1);
          Value tx = parallelBlock.getArgument(2);
          Value ty = parallelBlock.getArgument(3);

          // Global indices:
          //   indexK = bx*tileY + ty
          //   indexJ = by*tileX + tx
          Value kOuter = pb.create<arith::MulIOp>(bx, tileY);
          Value jOuter = pb.create<arith::MulIOp>(by, tileX);
          Value indexK = pb.create<arith::AddIOp>(kOuter, ty);
          Value indexJ = pb.create<arith::AddIOp>(jOuter, tx);

          // Tail guards for partial tiles.
          Value kOk = pb.create<arith::CmpIOp>(arith::CmpIPredicate::ult,
                                               indexK, batchNum);
          Value jOk = pb.create<arith::CmpIOp>(arith::CmpIPredicate::ult,
                                               indexJ, bflyNum);
          Value ok = pb.create<arith::AndIOp>(kOk, jOk);

          pb.create<scf::IfOp>(
              ok,
              /*thenBuilder=*/
              [&](OpBuilder &thenB, Location thenLoc) {
                ImplicitLocOpBuilder t(thenLoc, thenB);

                // `indexBfly` calculates the starting index of the current
                // butterfly group.
                Value indexBfly = t.create<arith::MulIOp>(batchSize, indexK);

                // Compute the indices for the butterfly pair:
                //   - `A` is the coefficient from the upper half of the
                //   butterfly.
                //   - `B` is the coefficient from the lower half.
                // ---------------------------------------------------------

                // `indexA` is computed by combining the local index
                // `indexJ` with the base `indexBfly`.
                Value indexA = t.create<arith::AddIOp>(indexJ, indexBfly);
                // `indexB` is calculated by shifting `indexA` by half the
                // batch size.
                Value halfBatch = t.create<arith::DivUIOp>(batchSize, c2);
                Value indexB = t.create<arith::AddIOp>(indexA, halfBatch);

                // Load values from previous stage output.
                Value A =
                    t.create<memref::LoadOp>(stageMemref, ValueRange{indexA});
                Value B =
                    t.create<memref::LoadOp>(stageMemref, ValueRange{indexB});

                // ---------------------------------------------------------
                // Compute the twiddle factor for the butterfly.
                // The appropriate twiddle factor is selected from the
                // precomputed `roots` tensor. The index is computed via an
                // affine map using the current butterfly index `indexJ` and
                // the current `rootExp` value
                // ---------------------------------------------------------
                Value rootIndex = t.create<arith::MulIOp>(indexJ, rootExp);
                Value root = t.create<tensor::ExtractOp>(roots, rootIndex);

                // ---------------------------------------------------------
                // Apply the butterfly operation.
                // Use either the Cooley-Tukey (bflyCT) or Gentleman-Sande
                // (bflyGS) variant, depending on whether we are performing
                // an inverse transform.
                // ---------------------------------------------------------
                auto bflyResult =
                    kInverse ? bflyGS(t, A, B, root) : bflyCT(t, A, B, root);

                t.create<memref::StoreOp>(bflyResult.first, destMemref,
                                          ValueRange{indexA});
                t.create<memref::StoreOp>(bflyResult.second, destMemref,
                                          ValueRange{indexB});

                t.create<scf::YieldOp>(ValueRange{});
              });
        }

        // Forward GPU mapping attribute if present.
        if (gpuMappingAttr) {
          parallelLoop->setAttr(gpu::getMappingAttrName(), gpuMappingAttr);
        }

        // ---------------------------------------------------------------------
        // Update control variables for the next stage:
        // For the forward NTT:
        //    - Increase the batch size by a factor of 2.
        //    - Decrease the root exponent by dividing by 2.
        // For the inverse NTT:
        //    - Decrease the batch size by a factor of 2.
        //    - Increase the root exponent by multiplying by 2.
        // ---------------------------------------------------------------------
        batchSize = kInverse
                        ? b.create<arith::DivUIOp>(batchSize, c2).getResult()
                        : b.create<arith::MulIOp>(batchSize, c2).getResult();

        rootExp = kInverse ? b.create<arith::MulIOp>(rootExp, c2).getResult()
                           : b.create<arith::DivUIOp>(rootExp, c2).getResult();

        // Yield the updated `batchSize`, and `rootExp` for the next
        // stage.
        b.create<scf::YieldOp>(ValueRange{batchSize, rootExp, destMemref});
      });

  // For the inverse NTT, we must scale the output by the multiplicative inverse
  // of the degree.
  if (kInverse) {
    APInt modulus = coeffType.getModulus().getValue();
    APInt invDegree =
        multiplicativeInverse(APInt(modulus.getBitWidth(), degree), modulus);
    auto invDegreeAttr =
        IntegerAttr::get(coeffType.getStorageType(), invDegree);
    if (coeffType.isMontgomery()) {
      invDegreeAttr = mod_arith::getAttrAsMontgomeryForm(coeffType.getModulus(),
                                                         invDegreeAttr);
    }
    // TODO(batzor): Use scalar multiplication directly when it's available.
    auto invDegreeConst = b.create<field::ConstantOp>(coeffType, invDegreeAttr);
    b.create<linalg::MapOp>(
        /*inputs=*/ValueRange{destMemref},
        /*outputs=*/destMemref,
        /*bodyBuilder=*/
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          ImplicitLocOpBuilder b(nestedLoc, nestedBuilder);
          Value prod = b.create<field::MulOp>(args[0], invDegreeConst);
          b.create<linalg::YieldOp>(ValueRange{prod});
        });
  }

  // The final result is the coefficient tensor after all stages.
  Value result = b.create<bufferization::ToTensorOp>(
      intTensorType.cloneWith(std::nullopt, coeffType), destMemref,
      /*restrict=*/true, /*writable=*/true);

  return result;
}

struct ConvertNTT : public OpConversionPattern<NTTOp> {
  explicit ConvertNTT(MLIRContext *context)
      : OpConversionPattern<NTTOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(NTTOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Value nttResult;

    Attribute nttMappingAttr = op->getAttr("ntt_gpu_mapping");
    Attribute bitReverseMappingAttr = op->getAttr("bit_reverse_gpu_mapping");

    // Transform the input tensor to bit-reversed order at first if performing
    // forward NTT.
    if (!adaptor.getInverse() && adaptor.getBitReverse()) {
      auto bitReversed = b.create<tensor_ext::BitReverseOp>(adaptor.getSource(),
                                                            adaptor.getDest());

      // Forward GPU mapping attribute if present.
      if (bitReverseMappingAttr) {
        bitReversed->setAttr(gpu::getMappingAttrName(), bitReverseMappingAttr);
      }

      // NOTE(batzor): We should not use `dest` operand for the destination
      // here. Otherwise, writable `ToBufferOp` will be called twice on the same
      // `dest` SSA Value causing conflict and force memory copy.
      nttResult = fastNTT(b, adaptor, bitReversed.getResult(),
                          bitReversed.getResult(), nttMappingAttr);
    } else {
      nttResult = fastNTT(b, adaptor, adaptor.getSource(), adaptor.getDest(),
                          nttMappingAttr);
    }

    // Transform the input tensor to bit-reversed order at last if performing
    // inverse NTT.
    if (adaptor.getInverse() && adaptor.getBitReverse()) {
      auto nttResultBitReversed =
          b.create<tensor_ext::BitReverseOp>(nttResult, nttResult);

      // Forward GPU mapping attribute if present.
      if (bitReverseMappingAttr) {
        nttResultBitReversed->setAttr(gpu::getMappingAttrName(),
                                      bitReverseMappingAttr);
      }

      rewriter.replaceOp(op, nttResultBitReversed);
    } else {
      rewriter.replaceOp(op, nttResult);
    }

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
      ConvertNTT,
      ConvertToTensor,
      ConvertAny<bufferization::AllocTensorOp>,
      ConvertAny<tensor_ext::BitReverseOp>
      // clang-format on
      >(typeConverter, context);

  addStructuralConversionPatterns(typeConverter, patterns, target);

  target.addDynamicallyLegalOp<
      // clang-format off
      bufferization::AllocTensorOp,
      tensor_ext::BitReverseOp
      // clang-format on
      >([&](auto op) { return typeConverter.isLegal(op); });

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

} // namespace mlir::zkir::poly
