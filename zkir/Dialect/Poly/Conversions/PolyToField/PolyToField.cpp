#include "zkir/Dialect/Poly/Conversions/PolyToField/PolyToField.h"

#include <utility>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "zkir/Dialect/Field/IR/FieldAttributes.h"
#include "zkir/Dialect/Field/IR/FieldOps.h"
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

static FailureOr<CommonConversionInfo> getCommonConversionInfo(
    Operation *op, const TypeConverter *typeConverter) {
  // Most ops have a single result type that is a polynomial
  PolyType polyTy = dyn_cast<PolyType>(op->getResult(0).getType());

  if (!polyTy) {
    op->emitError(
        "Can't directly lower for a tensor of polynomials. "
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
  info.coefficientStorageType = info.coefficientType.getModulus().getType();
  return std::move(info);
}

template <typename SourceOp, typename TargetFieldOp>
struct ConvertPolyBinOp : public OpConversionPattern<SourceOp> {
  explicit ConvertPolyBinOp(MLIRContext *context)
      : OpConversionPattern<SourceOp>(context) {}

  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SourceOp op, typename SourceOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (PolyType poly_ty = dyn_cast<PolyType>(op.getResult().getType())) {
      ImplicitLocOpBuilder b(op.getLoc(), rewriter);
      auto result = b.create<TargetFieldOp>(adaptor.getLhs(), adaptor.getRhs());
      rewriter.replaceOp(op, result);
      return success();
    }
    return failure();
  }
};

struct ConvertToTensor : public OpConversionPattern<ToTensorOp> {
  explicit ConvertToTensor(MLIRContext *context)
      : OpConversionPattern<ToTensorOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ToTensorOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};

struct ConvertFromTensor : public OpConversionPattern<FromTensorOp> {
  explicit ConvertFromTensor(MLIRContext *context)
      : OpConversionPattern<FromTensorOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      FromTensorOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto res = getCommonConversionInfo(op, typeConverter);
    if (failed(res)) return failure();
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

      auto padValue = b.create<field::ConstantOp>(typeInfo.coefficientType, 0);
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
                     Value source, Value dest) {
  auto tensorType = cast<RankedTensorType>(adaptor.getDest().getType());
  auto coeffType = cast<field::PrimeFieldType>(tensorType.getElementType());

  auto coeffStorageType = coeffType.getModulus().getType();
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
    APInt root = rootAttr.getRoot().getValue().getValue();

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
    roots = !kInverse ? b.create<arith::ConstantOp>(
                            rootsType, primitiveRootsAttr.getRoots())
                      : b.create<arith::ConstantOp>(
                            rootsType, primitiveRootsAttr.getInvRoots());

    // Wrap the roots in a field encapsulation for further field operations.
    roots = b.create<field::EncapsulateOp>(tensorType, roots);
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
  Value two = b.create<arith::ConstantIndexOp>(2);
  Value n = b.create<arith::ConstantIndexOp>(degree);

  // Create a memref buffer for in-place updates
  auto memrefType = MemRefType::get(intTensorType.getShape(), coeffType);
  Value srcMemref = b.create<bufferization::ToMemrefOp>(memrefType, source,
                                                        /*read_only=*/true);
  Value destMemref = b.create<bufferization::ToMemrefOp>(memrefType, dest);

  // Define affine expressions for index calculations.
  // `x` and `y` will be used in affine maps to compute proper indices.
  AffineExpr x, y;
  bindDims(b.getContext(), x, y);

  // Begin the outer loop over the stages of the NTT.
  // The iterative loop carries three values:
  //   - The current batchSize,
  //   - The current root exponent (rootExp).
  b.create<affine::AffineForOp>(
      /*lowerBound=*/0, /* upperBound=*/stages, /*step=*/1,
      /*iterArgs=*/ValueRange{initialBatchSize, initialRootExp},
      /*bodyBuilder=*/
      [&](OpBuilder &nestedBuilder, Location nestedLoc, Value index,
          ValueRange args) {
        ImplicitLocOpBuilder b(nestedLoc, nestedBuilder);
        Value batchSize = args[0];
        Value rootExp = args[1];

        // The inner loop processes groups of coefficients defined by the
        // current batchSize.
        auto parallelLoop = b.create<affine::AffineParallelOp>(
            /*resultTypes=*/TypeRange{},
            /*reductions=*/ArrayRef<arith::AtomicRMWKind>{},
            /*lbMaps=*/
            ArrayRef<AffineMap>{b.getConstantAffineMap(0),
                                b.getConstantAffineMap(0)},
            /*lbArgs=*/ValueRange{},
            /*ubMaps=*/
            ArrayRef<AffineMap>{AffineMap::get(2, 0, x.floorDiv(y)),
                                AffineMap::get(2, 0, y.floorDiv(2))},
            /*ubArgs=*/ValueRange{n, batchSize},
            /*steps=*/ArrayRef<int64_t>{1, 1});

        // Build the body of the parallel loop.
        {
          Block &parallelBlock = parallelLoop.getRegion().front();
          OpBuilder parallelBuilder = OpBuilder::atBlockBegin(&parallelBlock);
          ImplicitLocOpBuilder pb(parallelBlock.getArgument(0).getLoc(),
                                  parallelBuilder);

          Value indexK = parallelBlock.getArgument(0);
          Value indexJ = parallelBlock.getArgument(1);

          // `indexBfly` calculates the starting index of the current butterfly
          // group.
          Value indexBfly = pb.create<affine::AffineApplyOp>(
              x * y, ValueRange{batchSize, indexK});

          // ---------------------------------------------------------
          // Compute the indices for the butterfly pair:
          //   - `A` is the coefficient from the upper half of the
          //   butterfly.
          //   - `B` is the coefficient from the lower half.
          // ---------------------------------------------------------

          // `indexA` is computed by combining the local index
          // `indexJ` with the base `indexBfly`.
          Value indexA = pb.create<affine::AffineApplyOp>(
              x + y, ValueRange{indexJ, indexBfly});
          // `indexB` is calculated by shifting `indexA` by half the
          // batch size.
          Value indexB = pb.create<affine::AffineApplyOp>(
              x + y.floorDiv(2), ValueRange{indexA, batchSize});

          // Load values from memref
          Value A =
              pb.create<affine::AffineLoadOp>(srcMemref, ValueRange{indexA});
          Value B =
              pb.create<affine::AffineLoadOp>(srcMemref, ValueRange{indexB});

          // ---------------------------------------------------------
          // Compute the twiddle factor for the butterfly.
          // The appropriate twiddle factor is selected from the
          // precomputed `roots` tensor. The index is computed via an
          // affine map using the current butterfly index `indexJ` and
          // the current `rootExp` value
          // ---------------------------------------------------------
          Value rootIndex = pb.create<affine::AffineApplyOp>(
              x * y, ValueRange{indexJ, rootExp});
          Value root = pb.create<tensor::ExtractOp>(roots, rootIndex);

          // ---------------------------------------------------------
          // Apply the butterfly operation.
          // Use either the Cooley-Tukey (bflyCT) or Gentleman-Sande
          // (bflyGS) variant, depending on whether we are performing
          // an inverse transform.
          // ---------------------------------------------------------
          auto bflyResult =
              kInverse ? bflyGS(pb, A, B, root) : bflyCT(pb, A, B, root);

          // Write the results back into the coefficient array.
          // Insert the "plus" result into `indexA` and the "minus"
          // result into `indexB`.
          pb.create<affine::AffineStoreOp>(bflyResult.first, destMemref,
                                           ValueRange{indexA});
          pb.create<affine::AffineStoreOp>(bflyResult.second, destMemref,
                                           ValueRange{indexB});

          // Empty yield is implicitly added here.
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
                        ? b.create<arith::DivUIOp>(batchSize, two).getResult()
                        : b.create<arith::MulIOp>(batchSize, two).getResult();

        rootExp = kInverse ? b.create<arith::MulIOp>(rootExp, two).getResult()
                           : b.create<arith::DivUIOp>(rootExp, two).getResult();

        // Yield the updated `batchSize`, and `rootExp` for the next
        // stage.
        b.create<affine::AffineYieldOp>(ValueRange{batchSize, rootExp});
      });

  // For the inverse NTT, we must scale the output by the multiplicative inverse
  // of the degree.
  if (kInverse) {
    APInt modulus = coeffType.getModulus().getValue();
    APInt invDegree =
        multiplicativeInverse(APInt(modulus.getBitWidth(), degree), modulus);
    auto invDegreeAttr = field::PrimeFieldAttr::get(
        cast<field::PrimeFieldType>(getStandardFormType(coeffType)), invDegree);
    if (coeffType.isMontgomery()) {
      invDegreeAttr = getAttrAsMontgomeryForm(invDegreeAttr);
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

  LogicalResult matchAndRewrite(
      NTTOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Value nttResult;

    // Transform the input tensor to bit-reversed order at first if performing
    // forward NTT.
    if (!adaptor.getInverse() && adaptor.getBitReverse()) {
      Value bitReversed = b.create<tensor_ext::BitReverseOp>(
          adaptor.getSource(), adaptor.getDest());

      // NOTE(batzor): We should not use `dest` operand for the destination
      // here. Otherwise, writable `ToMemrefOp` will be called twice on the same
      // `dest` SSA Value causing conflict and force memory copy.
      nttResult = fastNTT(b, adaptor, bitReversed, bitReversed);
    } else {
      nttResult = fastNTT(b, adaptor, adaptor.getSource(), adaptor.getDest());
    }

    // Transform the input tensor to bit-reversed order at last if performing
    // inverse NTT.
    auto nttResultBitReversed =
        !adaptor.getInverse() || !adaptor.getBitReverse()
            ? nttResult
            : b.create<tensor_ext::BitReverseOp>(nttResult, nttResult);
    rewriter.replaceOp(op, nttResultBitReversed);
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
      ConvertPolyBinOp<AddOp, field::AddOp>,
      ConvertPolyBinOp<SubOp, field::SubOp>,
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

}  // namespace mlir::zkir::poly
