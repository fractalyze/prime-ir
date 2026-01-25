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

#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveOps.h"

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveAttributes.h"
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveDialect.h"
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveTypes.h"
#include "prime_ir/Dialect/Field/IR/FieldDialect.h"
#include "prime_ir/Dialect/Field/IR/FieldOps.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithTypes.h"

// IWYU pragma: begin_keep
// Headers needed for EllipticCurveCanonicalization.cpp.inc
#include "mlir/IR/Matchers.h"
// IWYU pragma: end_keep

namespace mlir::prime_ir::elliptic_curve {
namespace {
template <typename OpType>
LogicalResult verifyMSMPointTypes(OpType op, Type inputType, Type outputType) {
  if (isa<JacobianType>(inputType) && isa<AffineType>(outputType)) {
    return op.emitError()
           << "jacobian input points require a jacobian or xyzz output type";
  } else if (isa<XYZZType>(inputType) &&
             (isa<AffineType>(outputType) || isa<JacobianType>(outputType))) {
    return op.emitError() << "xyzz input points require an xyzz output type";
  }
  return success();
}

template <typename OpType>
LogicalResult verifyPointCoordTypes(OpType op, Type pointType, Type coordType) {
  Type ptBaseField = cast<PointTypeInterface>(pointType).getBaseFieldType();
  if (isa<field::FieldDialect>(coordType.getDialect())) {
    if (ptBaseField != coordType) {
      return op.emitError() << "coord must be base field of point; but got "
                            << coordType << " expected " << ptBaseField;
    }
    return success();
  }
  if (auto pfType = dyn_cast<field::PrimeFieldType>(ptBaseField)) {
    if (auto modArithType = dyn_cast<mod_arith::ModArithType>(coordType)) {
      if (pfType.getModulus() != modArithType.getModulus())
        return op.emitError()
               << "output must have the same modulus as the base "
                  "field of input";
      if (pfType.isMontgomery() != modArithType.isMontgomery())
        return op.emitError()
               << "output must have the same montgomery form as the "
                  "base field of input";
    } else if (auto intType = dyn_cast<IntegerType>(coordType)) {
      if (intType.getWidth() != pfType.getStorageBitWidth())
        return op.emitError()
               << "output must have the same bitwidth as the base "
                  "field of input";
    } else {
      return op.emitError()
             << "output must be a mod_arith type or an integer "
                "type if the base field is a prime field; but got "
             << coordType;
    }
  } else {
    auto efType = cast<field::ExtensionFieldType>(ptBaseField);
    if (auto structType = dyn_cast<LLVM::LLVMStructType>(coordType)) {
      unsigned degree = efType.getDegree();
      if (structType.getBody().size() != degree)
        return op.emitError()
               << "output struct must have " << degree
               << " elements for extension field of degree " << degree;
      // NOTE: In case of extension fields, the types are not lowered to modular
      // types since struct cannot contain modular types so we can ignore that
      // case.
      auto baseFieldStorageType = efType.getBaseField().getStorageType();
      for (unsigned i = 0; i < degree; ++i) {
        if (structType.getBody()[i] != baseFieldStorageType)
          return op.emitError() << "output struct element must have the same "
                                   "bitwidth as the base field of input";
      }
    } else {
      return op.emitError()
             << "output must be a struct type for extension field; but got "
             << coordType;
    }
  }
  return success();
}
} // namespace

// WARNING: Assumes Jacobian or XYZZ point types
Value createZeroPoint(ImplicitLocOpBuilder &b, Type pointType) {
  auto baseFieldType = cast<PointTypeInterface>(pointType).getBaseFieldType();
  auto zeroBF =
      cast<field::FieldTypeInterface>(baseFieldType).createZeroConstant(b);
  Value oneBF =
      cast<field::FieldTypeInterface>(baseFieldType).createOneConstant(b);
  return isa<XYZZType>(pointType)
             ? b.create<FromCoordsOp>(pointType,
                                      ValueRange{oneBF, oneBF, zeroBF, zeroBF})
             : b.create<FromCoordsOp>(pointType,
                                      ValueRange{oneBF, oneBF, zeroBF});
}

/////////////// VERIFY OPS /////////////////

LogicalResult FromCoordsOp::verify() {
  Type outputType = getType();
  unsigned numCoords = cast<PointTypeInterface>(outputType).getNumCoords();
  if (numCoords != getCoords().size()) {
    return emitError() << outputType << " should have " << numCoords
                       << " coordinates";
  }
  return verifyPointCoordTypes(*this, outputType, getCoords()[0].getType());
}

LogicalResult ToCoordsOp::verify() {
  Type inputType = getInput().getType();
  unsigned numCoords =
      cast<PointTypeInterface>(getElementTypeOrSelf(inputType)).getNumCoords();
  if (numCoords != getOutput().size()) {
    return emitError() << inputType << " should have " << numCoords
                       << " coordinates";
  }
  return verifyPointCoordTypes(*this, inputType, getType(0));
}

LogicalResult IsZeroOp::verify() {
  Type inputType = getInput().getType();
  if (isa<AffineType>(getElementTypeOrSelf(inputType)) ||
      isa<JacobianType>(getElementTypeOrSelf(inputType)) ||
      isa<XYZZType>(getElementTypeOrSelf(inputType)))
    return success();
  return emitError() << "invalid input type";
}

namespace {
template <typename OpType>
LogicalResult verifyBinaryOp(OpType op) {
  Type lhsType = getElementTypeOrSelf(op.getLhs().getType());
  Type rhsType = getElementTypeOrSelf(op.getRhs().getType());
  Type outputType = getElementTypeOrSelf(op.getType());
  if (isa<AffineType>(lhsType) || isa<AffineType>(rhsType)) {
    if (lhsType == rhsType &&
        (isa<JacobianType>(outputType) || isa<XYZZType>(outputType))) {
      // affine, affine -> jacobian
      // affine, affine -> xyzz
      return success();
    } else if (!isa<AffineType>(outputType) &&
               (lhsType == outputType || rhsType == outputType)) {
      // affine, jacobian -> jacobian
      // affine, xyzz -> xyzz
      return success();
    }
  } else if (lhsType == rhsType && rhsType == outputType) {
    // jacobian, jacobian -> jacobian
    // xyzz, xyzz -> xyzz
    return success();
  }
  // TODO(ashjeong): check the curves of given types are the same
  return op->emitError() << "input or output types are wrong";
}
} // namespace

LogicalResult AddOp::verify() { return verifyBinaryOp(*this); }

LogicalResult SubOp::verify() { return verifyBinaryOp(*this); }

LogicalResult DoubleOp::verify() {
  Type inputType = getElementTypeOrSelf(getInput());
  Type outputType = getElementTypeOrSelf(getType());
  if ((isa<AffineType>(inputType) &&
       (isa<JacobianType>(outputType) || isa<XYZZType>(outputType))) ||
      inputType == outputType)
    return success();
  // TODO(ashjeong): check curves/fields are the same
  return emitError() << "wrong output type given input type";
}

LogicalResult ScalarMulOp::verify() {
  Type pointType = getElementTypeOrSelf(getPoint());
  Type outputType = getElementTypeOrSelf(getType());
  if ((isa<AffineType>(pointType) &&
       (isa<JacobianType>(outputType) || isa<XYZZType>(outputType))) ||
      pointType == outputType)
    return success();
  // TODO(ashjeong): check curves/fields are the same
  return emitError() << "wrong output type given point type";
}

LogicalResult CmpOp::verify() {
  arith::CmpIPredicate predicate = getPredicate();
  if (predicate == arith::CmpIPredicate::eq ||
      predicate == arith::CmpIPredicate::ne) {
    return success();
  } else {
    return emitOpError() << "only 'eq' and 'ne' comparisons are supported for "
                            "elliptic curve points";
  }
}

LogicalResult MSMOp::verify() {
  TensorType scalarsType = getScalars().getType();
  TensorType pointsType = getPoints().getType();
  if (scalarsType.getRank() != pointsType.getRank()) {
    return emitError() << "scalars and points must have the same rank";
  }
  Type inputType = pointsType.getElementType();
  Type outputType = getType();
  if (failed(verifyMSMPointTypes(*this, inputType, outputType))) {
    return failure();
  }

  int32_t degree = getDegree();
  if (degree <= 0) {
    return emitError() << "degree must be greater than 0";
  }

  if (scalarsType.hasStaticShape() && pointsType.hasStaticShape()) {
    if (scalarsType.getNumElements() != pointsType.getNumElements()) {
      return emitError()
             << "scalars and points must have the same number of elements";
    }
    if (scalarsType.getNumElements() > (int64_t{1} << degree)) {
      return emitError() << "scalars must have at least 2^degree elements";
    }
  } else if (scalarsType.hasStaticShape() && !pointsType.hasStaticShape()) {
    return emitError() << "scalars has static shape and points does not";
  } else if (!scalarsType.hasStaticShape() && pointsType.hasStaticShape()) {
    return emitError() << "points has static shape and scalars does not";
  }

  int32_t windowBits = getWindowBits();
  if (windowBits < 0) {
    return emitError() << "window bits must be greater than or equal to 0";
  }

  return success();
  // TODO(ashjeong): check curves/fields are the same
}

LogicalResult ScalarDecompOp::verify() {
  TensorType scalarsType = getScalars().getType();

  if (!scalarsType.hasStaticShape() ||
      !getBucketIndices().getType().hasStaticShape() ||
      !getPointIndices().getType().hasStaticShape()) {
    return emitOpError("requires statically shaped scalars, bucket_indices, "
                       "and point_indices tensors");
  }

  size_t numScalars = scalarsType.getNumElements();

  int16_t defaultScalarMaxBits =
      cast<field::PrimeFieldType>(scalarsType.getElementType())
          .getStorageBitWidth();
  int16_t scalarMaxBits = getScalarMaxBits().value_or(defaultScalarMaxBits);
  int16_t bitsPerWindow = getBitsPerWindow();

  int32_t numWindows = (scalarMaxBits + bitsPerWindow - 1) / bitsPerWindow;
  if (getBucketIndices().getType().getNumElements() !=
      numScalars * numWindows) {
    return emitError() << "bucket_indices size should be #scalars * #windows ("
                       << (numScalars * numWindows) << "), but got "
                       << getBucketIndices().getType().getNumElements();
  }
  return success();
}

LogicalResult BucketAccOp::verify() {
  TensorType sortedUniqueBucketIndices =
      getSortedUniqueBucketIndices().getType();
  TensorType bucketOffsets = getBucketOffsets().getType();
  TensorType bucketResults = getBucketResults().getType();
  TensorType points = getPoints().getType();

  if (sortedUniqueBucketIndices.getNumElements() !=
      bucketOffsets.getNumElements() - 1) {
    return emitError() << "bucket_offsets must have one more element than "
                          "sorted_unique_bucket_indices";
  }

  Type inputType = points.getElementType();
  Type outputType = bucketResults.getElementType();
  if (failed(verifyMSMPointTypes(*this, inputType, outputType))) {
    return failure();
  }

  // TODO(ashjeong): check summed result of all bucket sizes equals to
  // points.size()
  // TODO(ashjeong): verify output numElements = number of buckets

  return success();
}

LogicalResult BucketReduceOp::verify() {
  TensorType bucketsType = getBuckets().getType();
  TensorType windowsType = getWindows().getType();
  if (bucketsType.getShape()[0] != windowsType.getShape()[0]) {
    return emitError() << "dimension 0 of buckets and windows must be the same";
  }
  return success();
}

LogicalResult WindowReduceOp::verify() {
  unsigned scalarBitWidth = getScalarType().getStorageBitWidth();
  int16_t bitsPerWindow = getBitsPerWindow();
  TensorType windowsType = getWindows().getType();

  unsigned numWindows = (scalarBitWidth + bitsPerWindow - 1) / bitsPerWindow;
  if (numWindows != windowsType.getNumElements()) {
    return emitError() << "number of calculated windows (" << numWindows
                       << ") must be the same as the number of windows in the "
                          "windows tensor ("
                       << windowsType.getNumElements() << ")";
  }
  return success();
}

//////////////// VERIFY POINT CONVERSIONS ////////////////

LogicalResult ConvertPointTypeOp::verify() {
  Type inputType = getInput().getType();
  Type outputType = getType();
  if (inputType == outputType)
    return emitError() << "Converting on same types";
  // TODO(ashjeong): check curves are the same
  return success();
}

//////////////// PARSE AND PRINT CONSTANT ////////////////

ParseResult parseEllipticCurveConstant(OpAsmParser &parser,
                                       OperationState &result) {
  // TODO(chokobole): Support nested towers of extension fields.
  SmallVector<SmallVector<APInt>> parsedCoordArrays;

  auto coordinateParser = [&]() -> ParseResult {
    SmallVector<APInt> coordValues;

    // Check if the coordinate is an extension field element [coeff0, coeff1,
    // ...] or a single base field integer.
    if (succeeded(parser.parseOptionalLSquare())) {
      auto elementParser = [&]() -> ParseResult {
        APInt val;
        if (failed(parser.parseInteger(val)))
          return failure();
        coordValues.push_back(val);
        return success();
      };

      if (failed(parser.parseCommaSeparatedList(elementParser)) ||
          failed(parser.parseRSquare())) {
        return failure();
      }
    } else {
      // Parse as a single base field integer literal.
      APInt val;
      if (failed(parser.parseInteger(val)))
        return failure();
      coordValues.push_back(val);
    }

    parsedCoordArrays.push_back(std::move(coordValues));
    return success();
  };

  if (failed(parser.parseCommaSeparatedList(OpAsmParser::Delimiter::None,
                                            coordinateParser))) {
    return parser.emitError(parser.getNameLoc(),
                            "expected comma-separated coordinates (e.g., x, y "
                            "or [x0, x1], [y0, y1])");
  }
  // Parse type
  Type parsedType;
  if (failed(parser.parseColon()) || failed(parser.parseType(parsedType))) {
    return failure();
  }

  Type pointType = getElementTypeOrSelf(parsedType);
  auto pointTypeInterface = cast<PointTypeInterface>(pointType);
  if (!pointTypeInterface) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected point type, but got ")
           << parsedType;
  }

  // Get base field from curve
  ShortWeierstrassAttr curveAttr =
      cast<ShortWeierstrassAttr>(pointTypeInterface.getCurveAttr());
  Type baseFieldType = curveAttr.getBaseField();

  unsigned numCoords = pointTypeInterface.getNumCoords();

  if (parsedCoordArrays.size() != numCoords) {
    return parser.emitError(parser.getCurrentLocation())
           << "expected " << numCoords << " coordinates for " << parsedType
           << " but got " << parsedCoordArrays.size();
  }

  // Determine if base field is an extension field
  bool isExtensionField = isa<field::ExtensionFieldType>(baseFieldType);
  unsigned extensionDegree = 1;
  if (isExtensionField) {
    extensionDegree =
        cast<field::ExtensionFieldType>(baseFieldType).getDegreeOverPrime();
  }

  // Validate and create field attributes
  SmallVector<Attribute> coordAttrs;
  for (unsigned i = 0; i < numCoords; ++i) {
    const auto &coordValues = parsedCoordArrays[i];

    // Validate coordinate format matches base field type
    if (isExtensionField) {
      if (coordValues.size() != extensionDegree) {
        return parser.emitError(parser.getCurrentLocation())
               << "coordinate " << i << " must have " << extensionDegree
               << " values for extension field, but got " << coordValues.size();
      }
    } else {
      if (coordValues.size() != 1) {
        return parser.emitError(parser.getCurrentLocation())
               << "coordinate " << i
               << " must be a single integer for prime field, but got "
               << coordValues.size() << " values";
      }
    }

    auto attrOrErr = field::validateAndCreateFieldAttribute(
        parser, baseFieldType, coordValues);

    if (failed(attrOrErr)) {
      return failure();
    }
    coordAttrs.push_back(*attrOrErr);
  }

  result.addAttribute("coords",
                      ArrayAttr::get(parser.getContext(), coordAttrs));
  result.addTypes(parsedType);
  return success();
}

ParseResult ConstantOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseEllipticCurveConstant(parser, result);
}

void ConstantOp::print(OpAsmPrinter &p) {
  p << " ";
  auto coords = getCoords();
  for (size_t i = 0; i < coords.size(); ++i) {
    if (i > 0)
      p << ", ";
    p.printAttributeWithoutType(coords[i]);
  }
  p << " : ";
  p.printType(getType());
}

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
  return adaptor.getCoords();
}

// static
ConstantOp ConstantOp::materialize(OpBuilder &builder, Attribute value,
                                   Type type, Location loc) {
  if (!isa<PointTypeInterface>(getElementTypeOrSelf(type))) {
    return nullptr;
  }

  if (auto arrayAttr = dyn_cast<ArrayAttr>(value)) {
    return ConstantOp::create(builder, loc, type, arrayAttr);
  }
  return nullptr;
}

Operation *EllipticCurveDialect::materializeConstant(OpBuilder &builder,
                                                     Attribute value, Type type,
                                                     Location loc) {
  return ConstantOp::materialize(builder, value, type, loc);
}

//////////////// CANONICALIZATION PATTERNS ////////////////

namespace {

struct FromCoordsOfToCoords : public mlir::OpRewritePattern<FromCoordsOp> {
  using OpRewritePattern<FromCoordsOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(FromCoordsOp op,
                                PatternRewriter &rewriter) const override {
    // Match: elliptic_curve.from_coords(elliptic_curve.to_coords(arg))
    if (op.getOperands().empty())
      return failure();

    auto extToCoordsOp = op.getOperands().front().getDefiningOp<ToCoordsOp>();
    if (!extToCoordsOp)
      return failure();

    // The operands must be exactly the results of the ToCoordsOp, in order.
    if (op.getOperands() != extToCoordsOp->getResults())
      return failure();

    rewriter.replaceOp(op, extToCoordsOp->getOperands());
    return success();
  }
};

struct ToCoordsOfFromCoords : public mlir::OpRewritePattern<ToCoordsOp> {
  using OpRewritePattern<ToCoordsOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ToCoordsOp op,
                                PatternRewriter &rewriter) const override {
    // Match:
    // elliptic_curve.to_coords(elliptic_curve.from_coords(arg...))
    auto extFromCoordOp = op.getOperand().getDefiningOp<FromCoordsOp>();
    if (!extFromCoordOp)
      return failure();

    rewriter.replaceOp(op, extFromCoordOp->getOperands());
    return success();
  }
};

} // namespace

void FromCoordsOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                               MLIRContext *context) {
  patterns.add<FromCoordsOfToCoords>(context);
}

void ToCoordsOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                             MLIRContext *context) {
  patterns.add<ToCoordsOfFromCoords>(context);
}

namespace {
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveCanonicalization.cpp.inc"
} // namespace

#include "prime_ir/Utils/CanonicalizationPatterns.inc"

void AddOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                        MLIRContext *context) {
#define PRIME_IR_ADD_PATTERN(Name) patterns.add<EllipticCurve##Name>(context);
  PRIME_IR_GROUP_ADD_PATTERN_LIST(PRIME_IR_ADD_PATTERN)
#undef PRIME_IR_ADD_PATTERN
}

void SubOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                        MLIRContext *context) {
#define PRIME_IR_SUB_PATTERN(Name) patterns.add<EllipticCurve##Name>(context);
  PRIME_IR_GROUP_SUB_PATTERN_LIST(PRIME_IR_SUB_PATTERN)
#undef PRIME_IR_SUB_PATTERN
}

} // namespace mlir::prime_ir::elliptic_curve
