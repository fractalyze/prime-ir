#include "zkir/Dialect/Poly/Conversions/PolyToField/PolyToField.h"

#include <utility>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "zkir/Dialect/Field/IR/FieldAttributes.h"
#include "zkir/Dialect/Field/IR/FieldOps.h"
#include "zkir/Dialect/Poly/IR/PolyOps.h"
#include "zkir/Dialect/Poly/IR/PolyTypes.h"
#include "zkir/Utils/ConversionUtils.h"

namespace mlir::zkir::poly {

#define GEN_PASS_DEF_POLYTOFIELD
#include "zkir/Dialect/Poly/Conversions/PolyToField/PolyToField.h.inc"

RankedTensorType convertPolyType(PolyType type) {
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

FailureOr<CommonConversionInfo> getCommonConversionInfo(
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
  info.coefficientType =
      llvm::dyn_cast<field::PrimeFieldType>(polyTy.getBaseField());
  if (!info.coefficientType) {
    op->emitError("Polynomial base field must be of field type");
    return failure();
  }
  info.tensorType = cast<RankedTensorType>(typeConverter->convertType(polyTy));
  info.coefficientStorageType = info.coefficientType.getModulus().getType();
  return std::move(info);
}

struct ConvertConstant : public OpConversionPattern<ConstantOp> {
  explicit ConvertConstant(mlir::MLIRContext *context)
      : OpConversionPattern<ConstantOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ConstantOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto res = getCommonConversionInfo(op, typeConverter);
    if (failed(res)) return failure();
    auto typeInfo = res.value();

    auto uniPolyAttr = dyn_cast<UnivariatePolyAttr>(op.getValue());
    if (!uniPolyAttr) return failure();
    SmallVector<Attribute> coeffs;
    Type eltStorageType = typeInfo.coefficientStorageType;

    // Create all the attributes as arith types since mod_arith.constant
    // doesn't support tensor attribute inputs. Instead we
    // mod_arith.encapsulate them.
    //
    // This is inefficient for large-degree polys, but as of this writing we
    // don't have a lowering that uses a sparse representation.
    unsigned numTerms = typeInfo.tensorType.getShape()[0];
    coeffs.reserve(numTerms);
    for (size_t i = 0; i < numTerms; ++i) {
      coeffs.push_back(IntegerAttr::get(eltStorageType, 0));
    }

    // WARNING: if you don't store the IntPolynomial as an intermediate value
    // before iterating over the terms, you will get a use-after-free bug. See
    // the "Temporary range expression" section in
    // https://en.cppreference.com/w/cpp/language/range-for
    const polynomial::IntPolynomial &poly =
        uniPolyAttr.getValue().getPolynomial();
    for (const auto &term : poly.getTerms()) {
      int64_t idx = term.getExponent().getSExtValue();
      APInt coeff = term.getCoefficient();
      APInt modulus = typeInfo.polyType.getBaseField().getModulus().getValue();
      // APInt `srem` gives remainder with sign matching the sign of the
      // coefficient
      coeff = coeff.sextOrTrunc(modulus.getBitWidth()).srem(modulus);
      if (coeff.isNegative()) {
        // We need to add the modulus to get the positive remainder.
        coeff += modulus;
      }
      assert(coeff.sge(0));
      coeffs[idx] = IntegerAttr::get(eltStorageType, coeff.getZExtValue());
    }

    auto intTensorType =
        RankedTensorType::get(typeInfo.tensorType.getShape(),
                              typeInfo.coefficientType.getModulus().getType());
    auto constOp = b.create<arith::ConstantOp>(
        DenseElementsAttr::get(intTensorType, coeffs));
    rewriter.replaceOpWithNewOp<field::EncapsulateOp>(op, typeInfo.tensorType,
                                                      constOp.getResult());
    return success();
  }
};

template <typename SourceOp, typename TargetFieldOp>
struct ConvertPolyBinOp : public OpConversionPattern<SourceOp> {
  explicit ConvertPolyBinOp(mlir::MLIRContext *context)
      : OpConversionPattern<SourceOp>(context) {}

  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SourceOp op, typename SourceOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (PolyType poly_ty = llvm::dyn_cast<PolyType>(op.getResult().getType())) {
      ImplicitLocOpBuilder b(op.getLoc(), rewriter);
      auto result = b.create<TargetFieldOp>(adaptor.getLhs(), adaptor.getRhs());
      rewriter.replaceOp(op, result);
      return success();
    }
    return failure();
  }
};

struct ConvertToTensor : public OpConversionPattern<ToTensorOp> {
  explicit ConvertToTensor(mlir::MLIRContext *context)
      : OpConversionPattern<ToTensorOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ToTensorOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getOperands()[0].getDefiningOp());
    return success();
  }
};

struct ConvertFromTensor : public OpConversionPattern<FromTensorOp> {
  explicit ConvertFromTensor(mlir::MLIRContext *context)
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
                                      Value root,
                                      mod_arith::MontgomeryAttr montAttr) {
  Value rootB;
  if (montAttr != mod_arith::MontgomeryAttr()) {
    rootB = b.create<field::MontMulOp>(B, root, montAttr);
  } else {
    rootB = b.create<field::MulOp>(B, root);
  }
  auto ctPlus = b.create<field::AddOp>(A, rootB);
  auto ctMinus = b.create<field::SubOp>(A, rootB);
  return {std::move(ctPlus), std::move(ctMinus)};
}

// Butterfly : Gentleman-Sande
static std::pair<Value, Value> bflyGS(ImplicitLocOpBuilder &b, Value A, Value B,
                                      Value root,
                                      mod_arith::MontgomeryAttr montAttr) {
  auto gsPlus = b.create<field::AddOp>(A, B);
  auto gsMinus = b.create<field::SubOp>(A, B);
  Value gsMinusRoot;
  if (montAttr != mod_arith::MontgomeryAttr()) {
    gsMinusRoot = b.create<field::MontMulOp>(gsMinus, root, montAttr);
  } else {
    gsMinusRoot = b.create<field::MulOp>(gsMinus, root);
  }
  return {std::move(gsPlus), std::move(gsMinusRoot)};
}

static Value computeReverseBitOrder(ImplicitLocOpBuilder &b,
                                    RankedTensorType tensorType, Value tensor) {
  unsigned numCoeffs = tensorType.getShape()[0];
  assert(llvm::has_single_bit(numCoeffs) &&
         "expected the number of coefficients to be a power of 2");

  unsigned indexBitWidth = llvm::countr_zero(numCoeffs);

  auto indicesType = RankedTensorType::get(tensorType.getShape(),
                                           IndexType::get(b.getContext()));

  auto coeffType = llvm::cast<IntegerType>(tensorType.getElementType());

  SmallVector<APInt> _indices;
  _indices.reserve(numCoeffs);

  for (unsigned index = 0; index < numCoeffs; index++) {
    _indices.push_back(APInt(indexBitWidth, index).reverseBits());
  }
  auto indices = b.create<arith::ConstantOp>(
      indicesType, DenseElementsAttr::get(indicesType, _indices));

  SmallVector<utils::IteratorType> iteratorTypes(1,
                                                 utils::IteratorType::parallel);
  AffineExpr d0;
  bindDims(b.getContext(), d0);
  SmallVector<AffineMap> indexingMaps = {AffineMap::get(1, 0, {d0}),
                                         AffineMap::get(1, 0, {d0})};

  // construct 0 tensor of field elements
  auto out = b.create<arith::ConstantOp>(
      tensorType,
      DenseElementsAttr::get(tensorType, APInt(coeffType.getWidth(), 0)));
  auto bfOut = b.create<field::EncapsulateOp>(tensor.getType(), out);
  auto shuffleOp = b.create<linalg::GenericOp>(
      /*resultTypes=*/TypeRange{tensor.getType()},
      /*inputs=*/ValueRange{indices.getResult()},
      /*outputs=*/ValueRange{bfOut.getResult()},
      /*indexingMaps=*/indexingMaps,
      /*iteratorTypes=*/iteratorTypes,
      /*bodyBuilder=*/
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        ImplicitLocOpBuilder b(nestedLoc, nestedBuilder);
        auto idx = args[0];
        auto elem = b.create<tensor::ExtractOp>(tensor, ValueRange{idx});
        b.create<linalg::YieldOp>(elem.getResult());
      });
  return shuffleOp.getResult(0);
}

template <bool kInverse>
static Value fastNTT(ImplicitLocOpBuilder &b, PrimitiveRootAttr rootAttr,
                     RankedTensorType tensorType, Type modType, Value input) {
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
  auto baseFieldType =
      dyn_cast<field::PrimeFieldType>(rootAttr.getRoot().getType());
  APInt cmod = baseFieldType.getModulus().getValue();
  APInt root = rootAttr.getRoot().getValue().getValue();

  // Create a tensor constant of precomputed roots for fast access during the
  // NTT.
  auto rootsType = tensorType.clone({degree});
  Value roots =
      !kInverse
          ? b.create<arith::ConstantOp>(rootsType, rootAttr.getRoots())
          : b.create<arith::ConstantOp>(rootsType, rootAttr.getInvRoots());

  // Wrap the roots in a field encapsulation for further field operations.
  roots = b.create<field::EncapsulateOp>(modType, roots);

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
  //          (A, B) = bflyOp(A, B, root)
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
  Value zero = b.create<arith::ConstantIndexOp>(0);
  Value two = b.create<arith::ConstantIndexOp>(2);
  Value n = b.create<arith::ConstantIndexOp>(degree);

  // Define affine expressions for index calculations.
  // `x` and `y` will be used in affine maps to compute proper indices.
  AffineExpr x, y;
  bindDims(b.getContext(), x, y);

  // Begin the outer loop over the stages of the NTT.
  // The iterative loop carries three values:
  //   - The current state of the coefficients (input tensor),
  //   - The current batchSize,
  //   - The current root exponent (rootExp).
  auto stagesLoop = b.create<affine::AffineForOp>(
      /*lowerBound=*/0, /* upperBound=*/stages, /*step=*/1,
      /*iterArgs=*/ValueRange{input, initialBatchSize, initialRootExp},
      /*bodyBuilder=*/
      [&](OpBuilder &nestedBuilder, Location nestedLoc, Value index,
          ValueRange args) {
        ImplicitLocOpBuilder b(nestedLoc, nestedBuilder);
        Value batchSize = args[1];
        Value rootExp = args[2];

        // The inner loop processes groups of coefficients defined by the
        // current batchSize.
        auto innerLoop = b.create<affine::AffineForOp>(
            /*lbOperands=*/zero, /*lbMap=*/AffineMap::get(1, 0, x),
            /*ubOperands=*/ValueRange{n, batchSize},
            /*ubMap=*/AffineMap::get(2, 0, x.floorDiv(y)),
            /*step=*/1,
            // Pass the current coefficient tensor as the iteration argument.
            /*iterArgs=*/args[0],
            /*bodyBuilder=*/
            [&](OpBuilder &nestedBuilder, Location nestedLoc, Value index,
                ValueRange args) {
              ImplicitLocOpBuilder b(nestedLoc, nestedBuilder);

              // `indexK` calculates the starting index of the current butterfly
              // group.
              Value indexK = b.create<affine::AffineApplyOp>(
                  x * y, ValueRange{batchSize, index});

              // Process each butterfly pair within the current group.
              auto arithLoop = b.create<affine::AffineForOp>(
                  /*lbOperands=*/zero, /*lbMap=*/AffineMap::get(1, 0, x),
                  /*ubOperands=*/batchSize,
                  /*ubMap=*/AffineMap::get(1, 0, x.floorDiv(2)),
                  /*step=*/1, /*iterArgs=*/args[0],
                  /*bodyBuilder=*/
                  [&](OpBuilder &nestedBuilder, Location nestedLoc,
                      Value indexJ, ValueRange args) {
                    ImplicitLocOpBuilder b(nestedLoc, nestedBuilder);

                    Value target = args[0];

                    // ---------------------------------------------------------
                    // Compute the indices for the butterfly pair:
                    //   - `A` is the coefficient from the upper half of the
                    //   butterfly.
                    //   - `B` is the coefficient from the lower half.
                    // ---------------------------------------------------------

                    // `indexA` is computed by combining the local index
                    // `indexJ` with the base `indexK`.
                    Value indexA = b.create<affine::AffineApplyOp>(
                        x + y, ValueRange{indexJ, indexK});
                    Value A = b.create<tensor::ExtractOp>(target, indexA);

                    // `indexB` is calculated by shifting `indexA` by half the
                    // batch size.
                    Value indexB = b.create<affine::AffineApplyOp>(
                        x + y.floorDiv(2), ValueRange{indexA, batchSize});
                    Value B = b.create<tensor::ExtractOp>(target, indexB);

                    // ---------------------------------------------------------
                    // Compute the twiddle factor for the butterfly.
                    // The appropriate twiddle factor is selected from the
                    // precomputed `roots` tensor. The index is computed via an
                    // affine map using the current butterfly index `indexJ` and
                    // the current `rootExp` value
                    // ---------------------------------------------------------
                    Value rootIndex = b.create<affine::AffineApplyOp>(
                        x * y, ValueRange{indexJ, rootExp});
                    Value root = b.create<tensor::ExtractOp>(roots, rootIndex);

                    // ---------------------------------------------------------
                    // Apply the butterfly operation.
                    // Use either the Cooley-Tukey (bflyCT) or Gentleman-Sande
                    // (bflyGS) variant, depending on whether we are performing
                    // an inverse transform.
                    // ---------------------------------------------------------
                    auto bflyResult =
                        kInverse
                            ? bflyGS(b, A, B, root, rootAttr.getMontgomery())
                            : bflyCT(b, A, B, root, rootAttr.getMontgomery());

                    // Write the results back into the coefficient array.
                    // Insert the "plus" result into `indexA` and the "minus"
                    // result into `indexB`.
                    auto insertPlus = b.create<tensor::InsertOp>(
                        bflyResult.first, target, indexA);
                    auto insertMinus = b.create<tensor::InsertOp>(
                        bflyResult.second, insertPlus, indexB);

                    b.create<affine::AffineYieldOp>(insertMinus.getResult());
                  });

              // Yield the updated coefficient tensor from the inner loop.
              b.create<affine::AffineYieldOp>(arithLoop.getResult(0));
            });

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

        // Yield the updated tensor, `batchSize`, and `rootExp` for the next
        // stage.
        b.create<affine::AffineYieldOp>(
            ValueRange{innerLoop.getResult(0), batchSize, rootExp});
      });

  // The final result is the coefficient tensor after all stages.
  Value result = stagesLoop.getResult(0);

  // For the inverse NTT, we must scale the output by the multiplicative inverse
  // of the degree.
  if constexpr (kInverse) {
    // TODO(batzor): Use scalar multiplication directly when it's available.
    auto invDegreeConst =
        b.create<arith::ConstantOp>(rootAttr.getInvDegree().getValue());
    auto degreeInvTensor = b.create<tensor::SplatOp>(invDegreeConst, rootsType);
    auto fieldTensor = b.create<field::EncapsulateOp>(modType, degreeInvTensor);
    result = b.create<field::MulOp>(result, fieldTensor);
  }

  return result;
}

struct ConvertNTT : public OpConversionPattern<NTTOp> {
  explicit ConvertNTT(mlir::MLIRContext *context)
      : OpConversionPattern<NTTOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      NTTOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto polyTy = dyn_cast<PolyType>(op.getInput().getType());
    if (!polyTy) {
      op.emitError(
          "Can't directly lower for a tensor of polynomials. "
          "First run --convert-elementwise-to-affine.");
      return failure();
    }

    auto coeffType = cast<field::PrimeFieldType>(polyTy.getBaseField());
    auto coeffStorageType = coeffType.getModulus().getType();
    auto inputType = dyn_cast<RankedTensorType>(adaptor.getInput().getType());
    auto intTensorType =
        RankedTensorType::get(inputType.getShape(), coeffStorageType);

    // Compute the ntt and extract the values
    Value nttResult = fastNTT<false>(
        b, op.getRoot(), intTensorType, inputType,
        computeReverseBitOrder(b, intTensorType, adaptor.getInput()));

    rewriter.replaceOp(op, nttResult);
    return success();
  }
};

struct ConvertINTT : public OpConversionPattern<INTTOp> {
  explicit ConvertINTT(mlir::MLIRContext *context)
      : OpConversionPattern<INTTOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      INTTOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto res = getCommonConversionInfo(op, typeConverter);
    if (failed(res)) return failure();
    auto typeInfo = res.value();

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto coeffType = cast<field::PrimeFieldType>(typeInfo.coefficientType);
    auto coeffStorageType = coeffType.getModulus().getType();
    auto inputType = cast<RankedTensorType>(adaptor.getInput().getType());
    auto intTensorType =
        RankedTensorType::get(inputType.getShape(), coeffStorageType);

    auto tensorType = typeConverter->convertType(op.getOutput().getType());
    auto inttResult = fastNTT<true>(b, op.getRoot(), intTensorType, tensorType,
                                    adaptor.getInput());
    auto reversedBitOrder =
        computeReverseBitOrder(b, intTensorType, inttResult);
    rewriter.replaceOp(op, reversedBitOrder);

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

  patterns.add<ConvertPolyBinOp<AddOp, field::AddOp>,
               ConvertPolyBinOp<SubOp, field::SubOp>, ConvertConstant,
               ConvertFromTensor, ConvertToTensor, ConvertNTT, ConvertINTT>(
      typeConverter, context);
  addStructuralConversionPatterns(typeConverter, patterns, target);

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // namespace mlir::zkir::poly
