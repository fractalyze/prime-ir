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
#include "prime_ir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/IntrinsicFunctionGenerator.h"
#include "prime_ir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/MSM/Pippengers/Generic.h"
#include "prime_ir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/PointOperations/PointCodeGen.h"
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveOps.h"
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveTypes.h"
#include "prime_ir/Dialect/Field/IR/FieldOps.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"
#include "prime_ir/Utils/BitSerialAlgorithm.h"

namespace mlir::prime_ir::elliptic_curve {

#define GEN_PASS_DEF_ELLIPTICCURVETOFIELD
#include "prime_ir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/EllipticCurveToField.h.inc"

struct ConvertConstant : public OpConversionPattern<ConstantOp> {
  explicit ConvertConstant(MLIRContext *context)
      : OpConversionPattern<ConstantOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Type pointType = getElementTypeOrSelf(op.getType());
    auto pointTypeInterface = cast<PointTypeInterface>(pointType);
    Type baseFieldType =
        cast<ShortWeierstrassAttr>(pointTypeInterface.getCurveAttr())
            .getBaseField();

    // Convert coordinate attributes to field constants
    SmallVector<Value> coordValues;
    for (auto coordAttr : op.getCoords()) {
      Value coordValue;
      if (auto intAttr = dyn_cast<IntegerAttr>(coordAttr)) {
        // Prime field coordinate
        coordValue = b.create<field::ConstantOp>(baseFieldType, intAttr);
      } else if (auto denseAttr = dyn_cast<DenseIntElementsAttr>(coordAttr)) {
        // Extension field coordinate
        coordValue = b.create<field::ConstantOp>(baseFieldType, denseAttr);
      } else {
        return op.emitError() << "unsupported coordinate attribute type";
      }
      coordValues.push_back(coordValue);
    }

    // Create point from coordinates
    Value point = fromCoords(b, op.getType(), coordValues);
    rewriter.replaceOp(op, point);
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
  explicit ConvertConvertPointType(MLIRContext *context)
      : OpConversionPattern<ConvertPointTypeOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConvertPointTypeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    ScopedBuilderContext scopedBuilderContext(&b);

    Type inputType = op.getInput().getType();
    PointCodeGen inputCodeGen(inputType, adaptor.getInput());
    Type outputType = getElementTypeOrSelf(op.getType());
    PointKind outputKind = cast<PointTypeInterface>(outputType).getPointKind();
    rewriter.replaceOp(op, {inputCodeGen.convert(outputKind)});
    return success();
  }
};

///////////// POINT ARITHMETIC OPERATIONS //////////////

struct ConvertAdd : public OpConversionPattern<AddOp> {
  explicit ConvertAdd(MLIRContext *context)
      : OpConversionPattern<AddOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    ScopedBuilderContext scopedBuilderContext(&b);

    Type lhsPointType = getElementTypeOrSelf(op->getOperandTypes()[0]);
    PointCodeGen lhsCodeGen(lhsPointType, adaptor.getLhs());
    Type rhsPointType = getElementTypeOrSelf(op->getOperandTypes()[1]);
    PointCodeGen rhsCodeGen(rhsPointType, adaptor.getRhs());
    Type outputType = getElementTypeOrSelf(op.getType());
    PointKind outputKind = cast<PointTypeInterface>(outputType).getPointKind();
    rewriter.replaceOp(op, {lhsCodeGen.add(rhsCodeGen, outputKind)});
    return success();
  }
};

struct ConvertDouble : public OpConversionPattern<DoubleOp> {
  explicit ConvertDouble(MLIRContext *context)
      : OpConversionPattern<DoubleOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DoubleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    ScopedBuilderContext scopedBuilderContext(&b);

    Type inputType = op.getInput().getType();
    PointCodeGen inputCodeGen(inputType, adaptor.getInput());
    Type outputType = getElementTypeOrSelf(op.getType());
    PointKind outputKind = cast<PointTypeInterface>(outputType).getPointKind();
    rewriter.replaceOp(op, {inputCodeGen.dbl(outputKind)});
    return success();
  }
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

// Currently implements Double-and-Add algorithm
// TODO(ashjeong): implement GLV
struct ConvertScalarMul : public OpConversionPattern<ScalarMulOp> {
  ConvertScalarMul(MLIRContext *context,
                   IntrinsicFunctionGenerator *generator = nullptr,
                   LoweringMode mode = LoweringMode::Inline)
      : OpConversionPattern<ScalarMulOp>(context), generator(generator),
        mode(mode) {}

  LogicalResult
  matchAndRewrite(ScalarMulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Value point = op.getPoint();
    Value scalarPF = op.getScalar();

    Type pointType = op.getPoint().getType();
    Type outputType = op.getType();

    // Use intrinsic for complex cases (G2 points, etc.)
    if (generator &&
        IntrinsicFunctionGenerator::shouldUseIntrinsic(pointType, mode)) {
      Value result = generator->emitScalarMulCall(
          rewriter, op.getLoc(), pointType, outputType, point, scalarPF);
      rewriter.replaceOp(op, result);
      return success();
    }

    // Inline implementation: Double-and-Add algorithm
    auto scalarFieldType = cast<field::PrimeFieldType>(scalarPF.getType());
    auto scalarIntType = scalarFieldType.getStorageType();
    Value scalarReduced =
        scalarFieldType.isMontgomery()
            ? b.create<field::FromMontOp>(
                  field::getStandardFormType(scalarFieldType), scalarPF)
            : scalarPF;
    Value scalarInt =
        b.create<field::BitcastOp>(TypeRange{scalarIntType}, scalarReduced);

    Value zeroPoint = createZeroPoint(b, outputType);

    Value initialPoint = isa<AffineType>(pointType)
                             ? b.create<ConvertPointTypeOp>(outputType, point)
                             : point;

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
  IntrinsicFunctionGenerator *generator;
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

struct ConvertScalarDecomp : public OpConversionPattern<ScalarDecompOp> {
  explicit ConvertScalarDecomp(MLIRContext *context)
      : OpConversionPattern<ScalarDecompOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ScalarDecompOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Value scalars =
        cast<field::PrimeFieldType>(op.getScalars().getType().getElementType())
                .isMontgomery()
            ? b.create<field::FromMontOp>(
                   field::getStandardFormType(op.getScalars().getType()),
                   op.getScalars())
                  .getResult()
            : op.getScalars();
    RankedTensorType scalarsType = cast<RankedTensorType>(scalars.getType());
    field::PrimeFieldType scalarFieldType =
        cast<field::PrimeFieldType>(scalarsType.getElementType());
    int32_t bitsPerWindow = op.getBitsPerWindow();

    unsigned scalarBitWidth = scalarFieldType.getStorageBitWidth();
    IntegerType scalarIntType = b.getIntegerType(scalarBitWidth);
    scalars =
        b.create<field::BitcastOp>(scalarsType.clone(scalarIntType), scalars)
            .getResult();

    // Use scalar_max_bits if specified, otherwise use the arithmetic bit size
    // of scalar modulus
    if (auto scalarMaxBitsAttr = op.getScalarMaxBitsAttr()) {
      scalarBitWidth = scalarMaxBitsAttr.getValue().getSExtValue();
      scalarIntType = b.getIntegerType(scalarBitWidth);
      scalars =
          b.create<arith::TruncIOp>(scalarsType.clone(scalarIntType), scalars);
    }

    int32_t numWindows = (scalarBitWidth + bitsPerWindow - 1) / bitsPerWindow;
    int64_t totalSize = scalarsType.getNumElements() * numWindows;

    RankedTensorType resultType =
        scalarsType.clone({totalSize}, b.getIndexType());
    IntegerType windowBitIntType = b.getIntegerType(bitsPerWindow);
    Value zero = b.create<arith::ConstantIndexOp>(0);
    Value numWindowsIndex = b.create<arith::ConstantIndexOp>(numWindows);
    Value zeroSplitScalar = b.create<arith::ConstantIntOp>(windowBitIntType, 0);
    Value bitsPerWindowIndex = b.create<arith::ConstantIndexOp>(bitsPerWindow);
    Value bucketIndices = b.create<tensor::GenerateOp>(
        resultType, SmallVector<Value>{},
        [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
          ImplicitLocOpBuilder nb(loc, nestedBuilder);
          Value splitScalarIdx = ivs[0];

          Value scalarIdx =
              nb.create<arith::DivUIOp>(splitScalarIdx, numWindowsIndex);
          Value windowIdx =
              nb.create<arith::RemUIOp>(splitScalarIdx, numWindowsIndex);

          Value scalar =
              nb.create<tensor::ExtractOp>(scalars, ValueRange{scalarIdx});
          Value windowOffset =
              nb.create<arith::MulIOp>(windowIdx, bitsPerWindowIndex);
          Value windowOffsetInt =
              nb.create<arith::IndexCastOp>(scalarIntType, windowOffset);
          Value shiftedScalar =
              nb.create<arith::ShRUIOp>(scalar, windowOffsetInt);
          Value splitScalar =
              nb.create<arith::TruncIOp>(windowBitIntType, shiftedScalar);

          Value isZeroSplitScalar = nb.create<arith::CmpIOp>(
              arith::CmpIPredicate::eq, splitScalar, zeroSplitScalar);

          // Calculate bucket index: if splitScalar is 0, bucketIdx = 0;
          // otherwise bucketIdx = (window << bitsPerWindow) | splitScalar
          Value windowShift =
              nb.create<arith::ShLIOp>(windowIdx, bitsPerWindowIndex);
          Value splitScalarIndex =
              nb.create<arith::IndexCastUIOp>(nb.getIndexType(), splitScalar);
          Value bucketIdx =
              nb.create<arith::OrIOp>(windowShift, splitScalarIndex);
          bucketIdx =
              nb.create<arith::SelectOp>(isZeroSplitScalar, zero, bucketIdx);
          nb.create<tensor::YieldOp>(bucketIdx);
        });

    Value pointIndices = b.create<tensor::GenerateOp>(
        resultType, SmallVector<Value>{},
        [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
          ImplicitLocOpBuilder nb(loc, nestedBuilder);
          Value splitScalarIdx = ivs[0];

          Value scalarIdx =
              nb.create<arith::DivUIOp>(splitScalarIdx, numWindowsIndex);
          nb.create<tensor::YieldOp>(scalarIdx);
        });

    rewriter.replaceOp(op, {bucketIndices, pointIndices});
    return success();
  }
};

struct ConvertBucketAcc : public OpConversionPattern<BucketAccOp> {
  explicit ConvertBucketAcc(MLIRContext *context)
      : OpConversionPattern<BucketAccOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BucketAccOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Value zero = b.create<arith::ConstantIndexOp>(0);
    Value one = b.create<arith::ConstantIndexOp>(1);

    Value points = op.getPoints();
    Value sortedPointIndices = op.getSortedPointIndices();
    Value sortedUniqueBucketIndices = op.getSortedUniqueBucketIndices();
    Value bucketOffsets = op.getBucketOffsets();
    TensorType bucketResultsType = op.getBucketResults().getType();
    Type outputType = bucketResultsType.getElementType();

    // Create buckets and initialize all buckets to zeroPoint
    MemRefType memrefBucketResultsType =
        MemRefType::get(bucketResultsType.getShape(), outputType);
    Value bucketResults = b.create<memref::AllocOp>(memrefBucketResultsType);
    Value zeroPoint = createZeroPoint(b, outputType);
    b.create<linalg::FillOp>(zeroPoint, bucketResults);

    // Compute bucket accumulation across all buckets
    Value nofBucketsToCompute = b.create<arith::ConstantIndexOp>(
        cast<TensorType>(sortedUniqueBucketIndices.getType()).getNumElements());
    b.create<scf::ParallelOp>(
        zero, nofBucketsToCompute, one,
        [&](OpBuilder &builder, Location loc, ValueRange ivs) {
          ImplicitLocOpBuilder b0(loc, builder);
          Value i = ivs[0];

          Value bucketOffsetStart =
              b0.create<tensor::ExtractOp>(bucketOffsets, i);
          Value indexPlusOne = b0.create<arith::AddIOp>(i, one);
          Value bucketOffsetEnd =
              b0.create<tensor::ExtractOp>(bucketOffsets, indexPlusOne);

          // TODO(ashjeong): Replace with linalg::ReduceOp once supported on the
          // EC level
          // Aggregate all points per bucket
          auto pointLoop = b0.create<scf::ForOp>(
              /*lowerBound=*/bucketOffsetStart, /*upperBound=*/bucketOffsetEnd,
              /*step=*/one, zeroPoint,
              [&](OpBuilder &builder, Location loc, Value j, ValueRange args) {
                ImplicitLocOpBuilder b1(loc, builder);
                Value bucketResult = args[0];

                Value pointIndex =
                    b1.create<tensor::ExtractOp>(sortedPointIndices, j);
                Value point = b1.create<tensor::ExtractOp>(points, pointIndex);
                bucketResult =
                    b1.create<AddOp>(outputType, bucketResult, point);
                b1.create<scf::YieldOp>(bucketResult);
              });

          Value bucketIndex =
              b0.create<tensor::ExtractOp>(sortedUniqueBucketIndices, i);
          b0.create<memref::StoreOp>(pointLoop.getResult(0), bucketResults,
                                     bucketIndex);
          b0.create<scf::ReduceOp>();
        });

    // Convert bucket results to tensor
    Value bucketResultsTensor =
        b.create<bufferization::ToTensorOp>(bucketResultsType, bucketResults,
                                            /*restrict=*/true);
    rewriter.replaceOp(op, bucketResultsTensor);
    return success();
  }
};

struct ConvertBucketReduce : public OpConversionPattern<BucketReduceOp> {
  explicit ConvertBucketReduce(MLIRContext *context)
      : OpConversionPattern<BucketReduceOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BucketReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Value buckets = op.getBuckets();
    RankedTensorType bucketsType = cast<RankedTensorType>(buckets.getType());
    Type pointType = bucketsType.getElementType();
    Type standardScalarType = field::getStandardFormType(op.getScalarType());
    Type scalarIntType = op.getScalarType().getStorageType();

    // Create bucket weights vector
    int64_t numBucketsPerWindow = bucketsType.getShape()[1];
    RankedTensorType arithWeightsType =
        bucketsType.clone({numBucketsPerWindow}, scalarIntType);
    SmallVector<Value> weightDims =
        numBucketsPerWindow == ShapedType::kDynamic
            ? SmallVector<Value>{b.create<tensor::DimOp>(buckets, 1)}
            : SmallVector<Value>{};
    Value arithWeights = b.create<tensor::GenerateOp>(
        arithWeightsType, weightDims,
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          ImplicitLocOpBuilder b0(loc, builder);
          Value arithScalar =
              b0.create<arith::IndexCastOp>(scalarIntType, args[0]);
          b0.create<tensor::YieldOp>(arithScalar);
        });
    RankedTensorType weightsType = arithWeightsType.clone(standardScalarType);
    Value fieldWeights = b.create<field::BitcastOp>(weightsType, arithWeights);

    // Create output windows tensor
    int64_t numWindows = bucketsType.getShape()[0];
    RankedTensorType windowsType = bucketsType.clone({numWindows}, pointType);
    SmallVector<Value> windowDims =
        numWindows == ShapedType::kDynamic
            ? SmallVector<Value>{b.create<tensor::DimOp>(buckets, 0)}
            : SmallVector<Value>{};
    Value zeroPoint = createZeroPoint(b, pointType);
    Value windows = b.create<tensor::GenerateOp>(
        windowsType, windowDims,
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          builder.create<tensor::YieldOp>(loc, zeroPoint);
        });

    // Calculate windows
    auto res = b.create<linalg::MatvecOp>(ValueRange{buckets, fieldWeights},
                                          ValueRange{windows});

    rewriter.replaceOp(op, res.getResult(0));
    return success();
  }
};

struct ConvertWindowReduce : public OpConversionPattern<WindowReduceOp> {
  explicit ConvertWindowReduce(MLIRContext *context)
      : OpConversionPattern<WindowReduceOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(WindowReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Value windows = op.getWindows();
    Type pointType = cast<RankedTensorType>(windows.getType()).getElementType();
    Type standardScalarType = getStandardFormType(op.getScalarType());
    Type scalarIntType = op.getScalarType().getStorageType();

    // Create output 0D tensor
    Value zeroPoint = createZeroPoint(b, pointType);
    Value outputTensor = b.create<tensor::FromElementsOp>(
        RankedTensorType::get({}, pointType), zeroPoint);

    // Calculate weighted windows reduction
    SmallVector<int64_t, 1> reductionDims = {0};
    Value c =
        b.create<arith::ConstantIntOp>(scalarIntType, op.getBitsPerWindow());
    Value base = b.create<field::ConstantOp>(standardScalarType, 2);
    // TODO(ashjeong): Try benchmarking against creating a separate weights
    // tensor & dot product. We want to test whether calculating  2ᶜⁱ in
    // parallel and reducing is faster than two loops of calculating 2ᶜⁱ
    // iteratively then reducing.
    auto total = b.create<linalg::ReduceOp>(
        windows, outputTensor, reductionDims,
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          ImplicitLocOpBuilder b0(loc, builder);
          // Current point (args[0]) and accumulator (args[1]).
          Value i = b0.create<linalg::IndexOp>(0);
          Value arithI = b0.create<arith::IndexCastOp>(scalarIntType, i);
          Value exp = b0.create<arith::MulIOp>(c, arithI);
          Value weight = b0.create<field::PowUIOp>(base, exp);
          Value weightedValue =
              b0.create<ScalarMulOp>(pointType, weight, args[0]);

          Value sum = b0.create<AddOp>(pointType, args[1], weightedValue);
          b0.create<linalg::YieldOp>(sum);
        });

    auto result = b.create<tensor::ExtractOp>(total.getResult(0));
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

  // Create intrinsic function generator if needed
  std::unique_ptr<IntrinsicFunctionGenerator> intrinsicGenerator;
  if (mode != LoweringMode::Inline) {
    intrinsicGenerator = std::make_unique<IntrinsicFunctionGenerator>(module);
  }

  ConversionTarget target(*context);
  target.addIllegalOp<
      // clang-format off
      AddOp,
      BucketAccOp,
      BucketReduceOp,
      ConstantOp,
      CmpOp,
      ConvertPointTypeOp,
      DoubleOp,
      IsZeroOp,
      MSMOp,
      NegateOp,
      ScalarDecompOp,
      ScalarMulOp,
      SubOp,
      WindowReduceOp,
      linalg::MatvecOp
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
      FromCoordsOp,
      ToCoordsOp,
      func::FuncOp,
      func::CallOp,
      func::ReturnOp
      // clang-format on
      >();

  RewritePatternSet patterns(context);
  linalg::populateLinalgNamedOpsGeneralizationPatterns(patterns);
  rewrites::populateWithGenerated(patterns);

  // Register ConvertScalarMul with intrinsic mode support
  patterns.add<ConvertScalarMul>(context, intrinsicGenerator.get(), mode);

  patterns.add<
      // clang-format off
      ConvertAdd,
      ConvertBucketAcc,
      ConvertBucketReduce,
      ConvertConstant,
      ConvertCmp,
      ConvertConvertPointType,
      ConvertDouble,
      ConvertIsZero,
      ConvertMSM,
      ConvertNegate,
      ConvertScalarDecomp,
      ConvertSub,
      ConvertWindowReduce
      // clang-format on
      >(context);

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

} // namespace mlir::prime_ir::elliptic_curve
