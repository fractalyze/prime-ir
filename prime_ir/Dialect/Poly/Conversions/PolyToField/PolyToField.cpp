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

#include "prime_ir/Dialect/Poly/Conversions/PolyToField/PolyToField.h"

#include <utility>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "prime_ir/Dialect/Field/IR/FieldAttributes.h"
#include "prime_ir/Dialect/Field/IR/FieldOperation.h"
#include "prime_ir/Dialect/Field/IR/FieldOps.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithTypes.h"
#include "prime_ir/Dialect/Poly/IR/PolyAttributes.h"
#include "prime_ir/Dialect/Poly/IR/PolyDialect.h"
#include "prime_ir/Dialect/Poly/IR/PolyOps.h"
#include "prime_ir/Dialect/Poly/IR/PolyTypes.h"
#include "prime_ir/Dialect/TensorExt/IR/TensorExtOps.h"
#include "prime_ir/Utils/ConversionUtils.h"

namespace mlir::prime_ir::poly {

#define GEN_PASS_DEF_POLYTOFIELD
#include "prime_ir/Dialect/Poly/Conversions/PolyToField/PolyToField.h.inc"

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

      Value padValue = field::createFieldZero(typeInfo.coefficientType, b);
      coeffValue = tensor::PadOp::create(b, typeInfo.tensorType, coeffValue,
                                         low, high, padValue,
                                         /*nofold=*/false);
    }

    rewriter.replaceOp(op, coeffValue);
    return success();
  }
};

// Butterfly : Cooley-Tukey
static std::pair<Value, Value> bflyCT(ImplicitLocOpBuilder &b, Value A, Value B,
                                      Value root) {
  Value rootB = field::MulOp::create(b, B, root);
  auto ctPlus = field::AddOp::create(b, A, rootB);
  auto ctMinus = field::SubOp::create(b, A, rootB);
  return {std::move(ctPlus), std::move(ctMinus)};
}

// Butterfly : Gentleman-Sande
static std::pair<Value, Value> bflyGS(ImplicitLocOpBuilder &b, Value A, Value B,
                                      Value root) {
  auto gsPlus = field::AddOp::create(b, A, B);
  auto gsMinus = field::SubOp::create(b, A, B);
  Value gsMinusRoot = field::MulOp::create(b, gsMinus, root);
  return {std::move(gsPlus), std::move(gsMinusRoot)};
}

static Value fastNTT(ImplicitLocOpBuilder &b, NTTOpAdaptor adaptor,
                     Value source, Value dest) {
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
    roots = !kInverse ? arith::ConstantOp::create(b, rootsType,
                                                  primitiveRootsAttr.getRoots())
                      : arith::ConstantOp::create(
                            b, rootsType, primitiveRootsAttr.getInvRoots());

    // Wrap the roots in a field encapsulation for further field operations.
    roots = field::BitcastOp::create(b, tensorType, roots);
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
      arith::ConstantIndexOp::create(b, kInverse ? degree : 2);
  Value initialRootExp =
      arith::ConstantIndexOp::create(b, kInverse ? 1 : degree / 2);

  // Define constants for index calculations.
  auto c0 = arith::ConstantIndexOp::create(b, 0);
  auto c1 = arith::ConstantIndexOp::create(b, 1);
  auto c2 = arith::ConstantIndexOp::create(b, 2);
  auto cDegree = arith::ConstantIndexOp::create(b, degree);
  auto cStages = arith::ConstantIndexOp::create(b, stages);

  // Create a memref buffer for in-place updates
  auto memrefType = MemRefType::get(intTensorType.getShape(), coeffType);
  Value srcMemref = bufferization::ToBufferOp::create(b, memrefType, source,
                                                      /*read_only=*/true);
  Value destMemref = bufferization::ToBufferOp::create(b, memrefType, dest);

  // Begin the outer loop over the stages of the NTT.
  // The iterative loop carries three values:
  //   - The current batchSize,
  //   - The current root exponent (rootExp).
  scf::ForOp::create(
      b, /*lowerBound=*/c0, /* upperBound=*/cStages,
      /*step=*/c1,
      /*initArgs=*/ValueRange{initialBatchSize, initialRootExp, srcMemref},
      /*bodyBuilder=*/
      [&](OpBuilder &nestedBuilder, Location nestedLoc, Value stageIndex,
          ValueRange args) {
        ImplicitLocOpBuilder b(nestedLoc, nestedBuilder);
        Value batchSize = args[0];
        Value rootExp = args[1];
        Value stageMemref = args[2];

        Value batchNum = arith::DivUIOp::create(b, cDegree, batchSize);
        Value bflyNum = arith::DivUIOp::create(b, batchSize, c2);

        Value tileX, tileY;
        // Adaptive tiling if `tileCap` is provided. Otherwise, don't tile.
        if (adaptor.getTileX()) {
          Value cTileXCap = arith::ConstantIndexOp::create(
              b, adaptor.getTileX().value().getValue().getSExtValue());
          Value cGridSize =
              adaptor.getGridSize()
                  ? arith::ConstantIndexOp::create(
                        b,
                        adaptor.getGridSize().value().getValue().getSExtValue())
                  : arith::ConstantIndexOp::create(b, 1024);

          // Adaptive tiles.
          tileX = arith::MinUIOp::create(b, bflyNum, cTileXCap);
          Value tileYCap = arith::DivUIOp::create(b, cGridSize, tileX);
          tileY = arith::MinUIOp::create(b, batchNum, tileYCap);
        } else {
          tileX = bflyNum;
          tileY = batchNum;
        }

        // Grid sizes.
        Value gridX = arith::CeilDivUIOp::create(b, batchNum, tileY);
        Value gridY = arith::CeilDivUIOp::create(b, bflyNum, tileX);

        // The inner loop processes groups of coefficients defined by the
        // current batchSize with a tile size of (tileX, tileY).
        auto parallelLoop = scf::ParallelOp::create(
            b, /*lowerBounds=*/ValueRange{c0, c0, c0, c0},
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
          Value kOuter = arith::MulIOp::create(pb, bx, tileY);
          Value jOuter = arith::MulIOp::create(pb, by, tileX);
          Value indexK = arith::AddIOp::create(pb, kOuter, ty);
          Value indexJ = arith::AddIOp::create(pb, jOuter, tx);

          // Tail guards for partial tiles.
          Value kOk = arith::CmpIOp::create(pb, arith::CmpIPredicate::ult,
                                            indexK, batchNum);
          Value jOk = arith::CmpIOp::create(pb, arith::CmpIPredicate::ult,
                                            indexJ, bflyNum);
          Value ok = arith::AndIOp::create(pb, kOk, jOk);

          scf::IfOp::create(
              pb, ok,
              /*thenBuilder=*/
              [&](OpBuilder &thenB, Location thenLoc) {
                ImplicitLocOpBuilder t(thenLoc, thenB);

                // `indexBfly` calculates the starting index of the current
                // butterfly group.
                Value indexBfly = arith::MulIOp::create(t, batchSize, indexK);

                // Compute the indices for the butterfly pair:
                //   - `A` is the coefficient from the upper half of the
                //   butterfly.
                //   - `B` is the coefficient from the lower half.
                // ---------------------------------------------------------

                // `indexA` is computed by combining the local index
                // `indexJ` with the base `indexBfly`.
                Value indexA = arith::AddIOp::create(t, indexJ, indexBfly);
                // `indexB` is calculated by shifting `indexA` by half the
                // batch size.
                Value halfBatch = arith::DivUIOp::create(t, batchSize, c2);
                Value indexB = arith::AddIOp::create(t, indexA, halfBatch);

                // Load values from previous stage output.
                Value A =
                    memref::LoadOp::create(t, stageMemref, ValueRange{indexA});
                Value B =
                    memref::LoadOp::create(t, stageMemref, ValueRange{indexB});

                // ---------------------------------------------------------
                // Compute the twiddle factor for the butterfly.
                // The appropriate twiddle factor is selected from the
                // precomputed `roots` tensor. The index is computed via an
                // affine map using the current butterfly index `indexJ` and
                // the current `rootExp` value
                // ---------------------------------------------------------
                Value rootIndex = arith::MulIOp::create(t, indexJ, rootExp);
                Value root = tensor::ExtractOp::create(t, roots, rootIndex);

                // ---------------------------------------------------------
                // Apply the butterfly operation.
                // Use either the Cooley-Tukey (bflyCT) or Gentleman-Sande
                // (bflyGS) variant, depending on whether we are performing
                // an inverse transform.
                // ---------------------------------------------------------
                auto bflyResult =
                    kInverse ? bflyGS(t, A, B, root) : bflyCT(t, A, B, root);

                memref::StoreOp::create(t, bflyResult.first, destMemref,
                                        ValueRange{indexA});
                memref::StoreOp::create(t, bflyResult.second, destMemref,
                                        ValueRange{indexB});

                scf::YieldOp::create(t, ValueRange{});
              });
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
                        ? arith::DivUIOp::create(b, batchSize, c2).getResult()
                        : arith::MulIOp::create(b, batchSize, c2).getResult();

        rootExp = kInverse ? arith::MulIOp::create(b, rootExp, c2).getResult()
                           : arith::DivUIOp::create(b, rootExp, c2).getResult();

        // Yield the updated `batchSize`, and `rootExp` for the next
        // stage.
        scf::YieldOp::create(b, ValueRange{batchSize, rootExp, destMemref});
      });

  // For the inverse NTT, we must scale the output by the multiplicative inverse
  // of the degree.
  if (kInverse) {
    APInt modulus = coeffType.getModulus().getValue();
    auto degreeOp = field::PrimeFieldOperation(
        APInt(modulus.getBitWidth(), degree), coeffType);
    IntegerAttr invDegreeAttr = degreeOp.inverse().getIntegerAttr();
    // TODO(batzor): Use scalar multiplication directly when it's available.
    auto invDegreeConst =
        field::ConstantOp::create(b, coeffType, invDegreeAttr);
    linalg::MapOp::create(
        b, /*inputs=*/ValueRange{destMemref},
        /*outputs=*/destMemref,
        /*bodyBuilder=*/
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          ImplicitLocOpBuilder b(nestedLoc, nestedBuilder);
          Value prod = field::MulOp::create(b, args[0], invDegreeConst);
          linalg::YieldOp::create(b, ValueRange{prod});
        });
  }

  // The final result is the coefficient tensor after all stages.
  Value result = bufferization::ToTensorOp::create(
      b, intTensorType.cloneWith(std::nullopt, coeffType), destMemref,
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

    // Transform the input tensor to bit-reversed order at first if performing
    // forward NTT.
    if (!adaptor.getInverse() && adaptor.getBitReverse()) {
      auto bitReversed = tensor_ext::BitReverseOp::create(
          b, adaptor.getSource(), adaptor.getDest(), /*dimension=*/0);

      // NOTE(batzor): We should not use `dest` operand for the destination
      // here. Otherwise, writable `ToBufferOp` will be called twice on the same
      // `dest` SSA Value causing conflict and force memory copy.
      nttResult =
          fastNTT(b, adaptor, bitReversed.getResult(), bitReversed.getResult());
    } else {
      nttResult = fastNTT(b, adaptor, adaptor.getSource(), adaptor.getDest());
    }

    // Transform the input tensor to bit-reversed order at last if performing
    // inverse NTT.
    if (adaptor.getInverse() && adaptor.getBitReverse()) {
      auto nttResultBitReversed = tensor_ext::BitReverseOp::create(
          b, nttResult, nttResult, /*dimension=*/0);

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

} // namespace mlir::prime_ir::poly
