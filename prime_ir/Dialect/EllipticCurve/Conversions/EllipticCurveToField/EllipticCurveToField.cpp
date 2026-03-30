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

#include "prime_ir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/EllipticCurveToField.h"

#include <utility>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "prime_ir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/ConversionUtils.h"
#include "prime_ir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/MSM/Pippengers/Generic.h"
#include "prime_ir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/PairingOperations/PairingCodeGen.h"
#include "prime_ir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/PointOperations/PointCodeGen.h"
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveOps.h"
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveTypes.h"
#include "prime_ir/Dialect/EllipticCurve/IR/KnownCurves.h"
#include "prime_ir/Dialect/Field/IR/FieldOps.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"
#include "prime_ir/Utils/BitSerialAlgorithm.h"
#include "prime_ir/Utils/ConversionUtils.h"
#include "prime_ir/Utils/LoweringMode.h"

namespace mlir::prime_ir::elliptic_curve {

#define GEN_PASS_DEF_ELLIPTICCURVETOFIELD
#include "prime_ir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/EllipticCurveToField.h.inc"

// ---- AOT Runtime helpers ----

static llvm::StringRef pointKindToString(PointKind kind) {
  switch (kind) {
  case PointKind::kAffine:
    return "affine";
  case PointKind::kJacobian:
    return "jacobian";
  case PointKind::kXYZZ:
    return "xyzz";
  }
  llvm_unreachable("unknown PointKind");
}

// Return "_mont" suffix if the field uses Montgomery form, else "".
static std::string getMontSuffix(Type baseFieldType) {
  return field::isMontgomery(baseFieldType) ? "_mont" : "";
}

// Build AOT runtime function name from curve alias and point kind.
// Example: "ec_add_bn254_g1_xyzz", "ec_add_bn254_g2_xyzz_mont"
static std::optional<std::string> getAOTRuntimeFuncName(llvm::StringRef op,
                                                        Type pointType) {
  auto pti = dyn_cast<PointTypeInterface>(pointType);
  if (!pti)
    return std::nullopt;

  auto curveAttr = dyn_cast<ShortWeierstrassAttr>(pti.getCurveAttr());
  if (!curveAttr)
    return std::nullopt;
  auto alias = getKnownCurveAlias(curveAttr);
  if (!alias)
    return std::nullopt;

  return ("ec_" + op + "_" + *alias + "_" +
          pointKindToString(pti.getPointKind()) +
          getMontSuffix(pti.getBaseFieldType()))
      .str();
}

// Build AOT runtime name for point type conversions.
// Example: "ec_jacobian_to_affine_bn254_g1", "ec_xyzz_to_affine_bn254_g2_mont"
static std::optional<std::string>
getConvertFuncName(PointTypeInterface inputPti, PointTypeInterface outputPti) {
  auto curveAttr = dyn_cast<ShortWeierstrassAttr>(inputPti.getCurveAttr());
  if (!curveAttr)
    return std::nullopt;
  auto alias = getKnownCurveAlias(curveAttr);
  if (!alias)
    return std::nullopt;

  return ("ec_" + pointKindToString(inputPti.getPointKind()) + "_to_" +
          pointKindToString(outputPti.getPointKind()) + "_" + *alias +
          getMontSuffix(inputPti.getBaseFieldType()))
      .str();
}

// Check if AOTRuntime should be used for this operation.
static bool shouldUseAOTRuntime(Operation *op, Type pointType,
                                LoweringMode mode) {
  if (mode == LoweringMode::AOTRuntime)
    return true;
  if (mode != LoweringMode::Auto)
    return false;
  // Auto: AOT for extension field curves (G2), inline for prime field (G1)
  auto pti = dyn_cast<PointTypeInterface>(pointType);
  if (!pti)
    return false;
  return isa<field::ExtensionFieldType>(pti.getBaseFieldType());
}

struct ConvertConstant : public OpConversionPattern<ConstantOp> {
  explicit ConvertConstant(MLIRContext *context)
      : OpConversionPattern<ConstantOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Type resultType = op.getType();
    Type pointType = getElementTypeOrSelf(resultType);
    auto pointTypeInterface = cast<PointTypeInterface>(pointType);
    Type baseFieldType =
        cast<ShortWeierstrassAttr>(pointTypeInterface.getCurveAttr())
            .getBaseField();

    // Convert coordinate attributes to field constants
    // Unified structure: ArrayAttr<ArrayAttr<Attr>> where outer array contains
    // points and inner array contains coordinates per point
    ArrayAttr coordsAttr = op.getCoords();

    // Helper to create a single point from its coordinate attributes
    auto createPointFromCoords =
        [&](ArrayAttr pointCoords) -> FailureOr<Value> {
      SmallVector<Value> coordValues;
      for (auto coordAttr : pointCoords) {
        Value coordValue;
        if (auto intAttr = dyn_cast<IntegerAttr>(coordAttr)) {
          // Prime field coordinate
          coordValue = b.create<field::ConstantOp>(baseFieldType, intAttr);
        } else if (auto denseAttr = dyn_cast<DenseIntElementsAttr>(coordAttr)) {
          // Extension field coordinate
          coordValue = b.create<field::ConstantOp>(baseFieldType, denseAttr);
        } else {
          return failure();
        }
        coordValues.push_back(coordValue);
      }
      return fromCoords(b, pointType, coordValues);
    };

    if (auto shapedType = dyn_cast<ShapedType>(resultType)) {
      // Tensor constant: create flat integer tensor, bitcast to field tensor,
      // then bitcast to point tensor
      unsigned numCoords = pointTypeInterface.getNumCoords();
      int64_t numPoints = shapedType.getNumElements();

      // Determine extension field degree and get base prime field
      unsigned extDegree = field::getExtensionDegree(baseFieldType);
      auto primeFieldType = field::getBasePrimeField(baseFieldType);

      // Collect all coordinate values into a flat vector
      SmallVector<APInt> flatValues;
      flatValues.reserve(numPoints * numCoords * extDegree);

      for (auto pointAttr : coordsAttr) {
        auto pointCoords = cast<ArrayAttr>(pointAttr);
        for (auto coordAttr : pointCoords) {
          if (auto intAttr = dyn_cast<IntegerAttr>(coordAttr)) {
            // Prime field coordinate
            flatValues.push_back(intAttr.getValue());
          } else if (auto denseAttr =
                         dyn_cast<DenseIntElementsAttr>(coordAttr)) {
            // Extension field coordinate - flatten all coefficients
            for (const APInt &coeff : denseAttr.getValues<APInt>())
              flatValues.push_back(coeff);
          } else {
            return op.emitError() << "unsupported coordinate attribute type";
          }
        }
      }

      // Create flat integer tensor constant using arith.constant
      int64_t flatSize = numPoints * numCoords * extDegree;
      auto intTensorType =
          RankedTensorType::get({flatSize}, primeFieldType.getStorageType());
      auto flatDenseAttr = DenseIntElementsAttr::get(intTensorType, flatValues);
      auto intConstant =
          b.create<arith::ConstantOp>(intTensorType, flatDenseAttr);

      // Bitcast integer tensor to prime field tensor
      auto pfTensorType = RankedTensorType::get({flatSize}, primeFieldType);
      auto pfTensor =
          b.create<field::BitcastOp>(pfTensorType, intConstant.getResult());

      // Use elliptic_curve.bitcast to convert prime field tensor to point
      // tensor. This can be properly bufferized unlike
      // UnrealizedConversionCastOp.
      Value fieldTensor = pfTensor.getResult();
      Value tensorResult =
          b.create<BitcastOp>(shapedType, fieldTensor).getResult();
      rewriter.replaceOp(op, tensorResult);
    } else {
      // Scalar constant: create single point
      ArrayAttr pointCoords = cast<ArrayAttr>(coordsAttr[0]);
      auto pointOrFailure = createPointFromCoords(pointCoords);
      if (failed(pointOrFailure))
        return op.emitError() << "unsupported coordinate attribute type";
      rewriter.replaceOp(op, *pointOrFailure);
    }

    return success();
  }
};

struct ConvertIsZero : public OpConversionPattern<IsZeroOp> {
  explicit ConvertIsZero(MLIRContext *context)
      : OpConversionPattern<IsZeroOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IsZeroOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Operation::result_range coords = toCoords(b, op.getInput());
    Type baseFieldType =
        cast<PointTypeInterface>(op.getInput().getType()).getBaseFieldType();
    Value zeroBF = field::createFieldZero(baseFieldType, b);

    Value isZero;
    if (isa<AffineType>(op.getInput().getType())) {
      Value xIsZero =
          b.create<field::CmpOp>(arith::CmpIPredicate::eq, coords[0], zeroBF);
      Value yIsZero =
          b.create<field::CmpOp>(arith::CmpIPredicate::eq, coords[1], zeroBF);
      isZero = b.create<arith::AndIOp>(xIsZero, yIsZero);
    } else {
      isZero =
          b.create<field::CmpOp>(arith::CmpIPredicate::eq, coords[2], zeroBF);
    }
    rewriter.replaceOp(op, isZero);
    return success();
  }
};

struct ConvertConvertPointType
    : public OpConversionPattern<ConvertPointTypeOp> {
  ConvertConvertPointType(MLIRContext *context,
                          LoweringMode mode = LoweringMode::Inline)
      : OpConversionPattern<ConvertPointTypeOp>(context), mode(mode) {}

  LogicalResult
  matchAndRewrite(ConvertPointTypeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type inputType = op.getInput().getType();
    Type outputType = getElementTypeOrSelf(op.getType());

    // AOT runtime path for point type conversions.
    if (shouldUseAOTRuntime(op, outputType, mode)) {
      auto inputPti = cast<PointTypeInterface>(inputType);
      auto outputPti = cast<PointTypeInterface>(outputType);
      auto funcName = getConvertFuncName(inputPti, outputPti);
      if (funcName) {
        rewriter.replaceOp(op, emitAOTFuncCall(op, *funcName, op.getType(),
                                               {op.getInput()}, rewriter));
        return success();
      }
    }

    // Inline path.
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    ScopedBuilderContext scopedBuilderContext(&b);

    PointCodeGen inputCodeGen(inputType, adaptor.getInput());
    PointKind outKind = cast<PointTypeInterface>(outputType).getPointKind();
    rewriter.replaceOp(op, {inputCodeGen.convert(outKind)});
    return success();
  }

  LoweringMode mode;
};

///////////// POINT ARITHMETIC OPERATIONS //////////////

struct ConvertAdd : public OpConversionPattern<AddOp> {
  explicit ConvertAdd(MLIRContext *context, LoweringMode mode = LoweringMode::Inline)
      : OpConversionPattern<AddOp>(context), mode(mode) {}

  LogicalResult
  matchAndRewrite(AddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type outputType = getElementTypeOrSelf(op.getType());

    // AOT runtime path: emit func.call for known curves.
    if (shouldUseAOTRuntime(op, outputType, mode)) {
      Type lhsType = getElementTypeOrSelf(op->getOperandTypes()[0]);
      Type rhsType = getElementTypeOrSelf(op->getOperandTypes()[1]);
      // Determine function name based on types.
      std::string opName;
      if (lhsType == rhsType)
        opName = "add";
      else
        opName = "mixed_add";
      auto funcName = getAOTRuntimeFuncName(opName, outputType);
      if (funcName) {
        rewriter.replaceOp(op, emitAOTFuncCall(op, *funcName, op.getType(),
                                               {op.getLhs(), op.getRhs()},
                                               rewriter));
        return success();
      }
    }

    // Inline path (original).
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    ScopedBuilderContext scopedBuilderContext(&b);

    Type lhsPointType = getElementTypeOrSelf(op->getOperandTypes()[0]);
    PointCodeGen lhsCodeGen(lhsPointType, adaptor.getLhs());
    Type rhsPointType = getElementTypeOrSelf(op->getOperandTypes()[1]);
    PointCodeGen rhsCodeGen(rhsPointType, adaptor.getRhs());
    PointKind outputKind = cast<PointTypeInterface>(outputType).getPointKind();
    rewriter.replaceOp(op, {lhsCodeGen.add(rhsCodeGen, outputKind)});
    return success();
  }

  LoweringMode mode;
};

struct ConvertDouble : public OpConversionPattern<DoubleOp> {
  explicit ConvertDouble(MLIRContext *context, LoweringMode mode = LoweringMode::Inline)
      : OpConversionPattern<DoubleOp>(context), mode(mode) {}

  LogicalResult
  matchAndRewrite(DoubleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type outputType = getElementTypeOrSelf(op.getType());

    if (shouldUseAOTRuntime(op, outputType, mode)) {
      auto funcName = getAOTRuntimeFuncName("double", outputType);
      if (funcName) {
        rewriter.replaceOp(op, emitAOTFuncCall(op, *funcName, op.getType(),
                                               {op.getInput()}, rewriter));
        return success();
      }
    }

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    ScopedBuilderContext scopedBuilderContext(&b);

    Type inputType = op.getInput().getType();
    PointCodeGen inputCodeGen(inputType, adaptor.getInput());
    PointKind outputKind = cast<PointTypeInterface>(outputType).getPointKind();
    rewriter.replaceOp(op, {inputCodeGen.dbl(outputKind)});
    return success();
  }

  LoweringMode mode;
};

struct ConvertNegate : public OpConversionPattern<NegateOp> {
  explicit ConvertNegate(MLIRContext *context)
      : OpConversionPattern<NegateOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(NegateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Operation::result_range coords = toCoords(b, op.getInput());

    auto negatedY = b.create<field::NegateOp>(coords[1]);
    SmallVector<Value> outputCoords(coords);
    outputCoords[1] = negatedY;

    rewriter.replaceOp(op, fromCoords(b, op.getType(), outputCoords));
    return success();
  }
};

struct ConvertSub : public OpConversionPattern<SubOp> {
  explicit ConvertSub(MLIRContext *context)
      : OpConversionPattern<SubOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SubOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Value negP2 = b.create<NegateOp>(op.getRhs());
    Value result = b.create<AddOp>(op.getType(), op.getLhs(), negP2);

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ConvertCmp : public OpConversionPattern<CmpOp> {
  explicit ConvertCmp(MLIRContext *context)
      : OpConversionPattern<CmpOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CmpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Operation::result_range lhsCoords = toCoords(b, lhs);
    Operation::result_range rhsCoords = toCoords(b, rhs);
    llvm::SmallVector<Value, 4> cmps;
    for (auto [lhsCoord, rhsCoord] : llvm::zip(lhsCoords, rhsCoords)) {
      cmps.push_back(
          b.create<field::CmpOp>(op.getPredicate(), lhsCoord, rhsCoord));
    }
    Value result;
    if (op.getPredicate() == arith::CmpIPredicate::eq) {
      result = combineCmps<arith::AndIOp>(b, cmps);
    } else if (op.getPredicate() == arith::CmpIPredicate::ne) {
      result = combineCmps<arith::OrIOp>(b, cmps);
    } else {
      llvm_unreachable(
          "Unsupported comparison predicate for EllipticCurve point type");
    }
    rewriter.replaceOp(op, result);
    return success();
  }

  template <typename Op>
  Value combineCmps(ImplicitLocOpBuilder &b, ValueRange cmps) const {
    Op result = b.create<Op>(cmps[0], cmps[1]);
    if (cmps.size() == 3) {
      result = b.create<Op>(result, cmps[2]);
    } else if (cmps.size() == 4) {
      Op result2 = b.create<Op>(cmps[2], cmps[3]);
      result = b.create<Op>(result, result2);
    }
    return result;
  }
};

namespace {
/// Converts a prime field value to its integer representation.
/// For field::ConstantOp: extracts the mathematical value via maybeToStandard.
/// For dynamic values: emits FromMontOp (if Montgomery) + BitcastOp.
Value fieldPrimeToInteger(ImplicitLocOpBuilder &b, Value fieldVal) {
  auto pfType = cast<field::PrimeFieldType>(fieldVal.getType());
  auto intType = pfType.getStorageType();

  if (auto constOp = fieldVal.getDefiningOp<field::ConstantOp>()) {
    Attribute stdAttr =
        field::maybeToStandard(constOp.getType(), constOp.getValue());
    return b.create<arith::ConstantIntOp>(
        intType, cast<IntegerAttr>(stdAttr).getValue());
  }

  Value std = pfType.isMontgomery()
                  ? b.create<field::FromMontOp>(
                        field::getStandardFormType(pfType), fieldVal)
                  : fieldVal;
  return b.create<field::BitcastOp>(TypeRange{intType}, std);
}
} // namespace

// Currently implements Double-and-Add algorithm
// TODO(ashjeong): implement GLV
struct ConvertScalarMul : public OpConversionPattern<ScalarMulOp> {
  ConvertScalarMul(MLIRContext *context,
                   LoweringMode mode = LoweringMode::Inline)
      : OpConversionPattern<ScalarMulOp>(context), mode(mode) {}

  LogicalResult
  matchAndRewrite(ScalarMulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Value point = op.getPoint();
    Value scalarPF = op.getScalar();

    Type pointType = op.getPoint().getType();
    Type outputType = op.getType();

    // AOT runtime path: emit func.call for scalar multiply.
    // Use "scalar_mul" for affine input, "scalar_mul_jac" for jacobian input.
    if (shouldUseAOTRuntime(op, outputType, mode)) {
      std::string opName =
          isa<AffineType>(pointType) ? "scalar_mul" : "scalar_mul_jac";
      auto funcName = getAOTRuntimeFuncName(opName, outputType);
      if (funcName) {
        rewriter.replaceOp(op, emitAOTFuncCall(op, *funcName, op.getType(),
                                               {scalarPF, point}, rewriter));
        return success();
      }
    }

    // Inline implementation: Double-and-Add algorithm
    Value zeroPoint = createZeroPoint(b, outputType);
    Value initialPoint = isa<AffineType>(pointType)
                             ? b.create<ConvertPointTypeOp>(outputType, point)
                             : point;

    Value scalarInt = fieldPrimeToInteger(b, scalarPF);
    Value result = generateBitSerialLoop(
        b, scalarInt, initialPoint, zeroPoint,
        [outputType](ImplicitLocOpBuilder &b, Value v) {
          return b.create<DoubleOp>(outputType, v);
        },
        [outputType](ImplicitLocOpBuilder &b, Value acc, Value v) {
          return b.create<AddOp>(outputType, acc, v);
        });
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  LoweringMode mode;
};

// Currently implements Pippenger's
struct ConvertMSM : public OpConversionPattern<MSMOp> {
  explicit ConvertMSM(MLIRContext *context)
      : OpConversionPattern<MSMOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MSMOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Value scalars = op.getScalars();
    Value points = op.getPoints();

    Type outputType = op.getType();

    PippengersGeneric pippengers(scalars, points, outputType, b,
                                 adaptor.getParallel(), adaptor.getDegree(),
                                 adaptor.getWindowBits());

    rewriter.replaceOp(op, pippengers.generate());
    return success();
  }
};

struct ConvertPairingCheck : public OpConversionPattern<PairingCheckOp> {
  explicit ConvertPairingCheck(MLIRContext *context)
      : OpConversionPattern<PairingCheckOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(PairingCheckOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Value g1Points = op.getG1Points();
    Value g2Points = op.getG2Points();

    // Get pairing curve family from G1/G2 attributes.
    auto g1PointType = cast<PointTypeInterface>(
        cast<RankedTensorType>(g1Points.getType()).getElementType());
    auto g2PointType = cast<PointTypeInterface>(
        cast<RankedTensorType>(g2Points.getType()).getElementType());
    auto g1CurveAttr = cast<ShortWeierstrassAttr>(g1PointType.getCurveAttr());
    auto g2CurveAttr = cast<ShortWeierstrassAttr>(g2PointType.getCurveAttr());

    auto family = getKnownPairingCurveFamily(g1CurveAttr, g2CurveAttr);
    if (!family)
      return op.emitError() << "unsupported pairing curve family";

    bool isMontgomery =
        cast<field::PrimeFieldType>(g1PointType.getBaseFieldType())
            .isMontgomery();

    Value result =
        emitBN254PairingCheck(b, *family, g1Points, g2Points, isMontgomery);
    rewriter.replaceOp(op, result);
    return success();
  }
};

namespace rewrites {
// In an inner namespace to avoid conflicts with canonicalization patterns
#include "prime_ir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/EllipticCurveToField.cpp.inc"
} // namespace rewrites

struct EllipticCurveToField
    : impl::EllipticCurveToFieldBase<EllipticCurveToField> {
  using EllipticCurveToFieldBase::EllipticCurveToFieldBase;

  void runOnOperation() override;
};

void EllipticCurveToField::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  // Parse lowering mode option
  LoweringMode mode = mlir::prime_ir::parseLoweringMode(loweringMode);

  ConversionTarget target(*context);
  target.addIllegalOp<
      // clang-format off
      AddOp,
      ConstantOp,
      CmpOp,
      ConvertPointTypeOp,
      DoubleOp,
      IsZeroOp,
      MSMOp,
      NegateOp,
      PairingCheckOp,
      ScalarMulOp,
      SubOp
      // clang-format on
      >();

  target.addLegalDialect<
      // clang-format off
      arith::ArithDialect,
      bufferization::BufferizationDialect,
      field::FieldDialect,
      func::FuncDialect,
      linalg::LinalgDialect,
      memref::MemRefDialect,
      scf::SCFDialect,
      tensor::TensorDialect
      // clang-format on
      >();

  target.addLegalOp<
      // clang-format off
      BitcastOp,
      FromCoordsOp,
      ToCoordsOp,
      func::FuncOp,
      func::CallOp,
      func::ReturnOp,
      UnrealizedConversionCastOp
      // clang-format on
      >();

  RewritePatternSet patterns(context);
  linalg::populateLinalgNamedOpsGeneralizationPatterns(patterns);
  rewrites::populateWithGenerated(patterns);

  // Register patterns with mode-aware lowering (inline vs AOT runtime).
  patterns.add<ConvertScalarMul>(context, mode);
  patterns.add<ConvertAdd>(context, mode);
  patterns.add<ConvertDouble>(context, mode);
  patterns.add<ConvertConvertPointType>(context, mode);
  // NOTE: We intentionally do NOT add function signature conversion patterns.
  // Converting function signatures would require converting all operations that
  // use EC tensor types (e.g., linalg.reduce), which is beyond the scope of
  // EC-to-Field. The EC tensor types in function signatures will remain and be
  // handled by later passes or explicit bitcasts.

  patterns.add<
      // clang-format off
      ConvertConstant,
      ConvertCmp,
      ConvertIsZero,
      ConvertMSM,
      ConvertNegate,
      ConvertPairingCheck,
      ConvertSub
      // clang-format on
      >(context);

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

} // namespace mlir::prime_ir::elliptic_curve
