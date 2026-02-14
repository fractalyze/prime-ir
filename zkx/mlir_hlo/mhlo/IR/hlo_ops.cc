/* Copyright 2019 The OpenXLA Authors.
Copyright 2025 The ZKX Authors.

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

// This file defines the operations used in the MHLO dialect.

#include "zkx/mlir_hlo/mhlo/IR/hlo_ops.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <functional>
#include <optional>
#include <utility>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"
#include "stablehlo/dialect/AssemblyFormat.h"
#include "stablehlo/dialect/TypeInference.h"

#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveTypes.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"
#include "prime_ir/IR/Attributes.h"
#include "zkx/mlir_hlo/mhlo/IR/hlo_ops.h.inc"
#include "zkx/mlir_hlo/utils/convert_op_folder.h"
#include "zkx/mlir_hlo/utils/hlo_utils.h" // IWYU pragma: keep

namespace mlir {
#include "zkx/mlir_hlo/mhlo/IR/hlo_patterns.cc.inc"
} // namespace mlir

using mlir::hlo::parseDimSizes;
using mlir::hlo::printDimSizes;

#include "zkx/mlir_hlo/mhlo/IR/hlo_ops_enums.cc.inc"
#define GET_ATTRDEF_CLASSES
#include "zkx/mlir_hlo/mhlo/IR/hlo_ops_attrs.cc.inc"
#define GET_TYPEDEF_CLASSES
// TODO(chokobole): Uncomment this. Dependency: hlo_ops_typedefs.td
// #include "zkx/mlir_hlo/mhlo/IR/hlo_ops_typedefs.cc.inc"

namespace mlir::mhlo {
namespace {

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

hlo::HloDialectInterface *getMhloDialect(MLIRContext *context) {
  MhloDialect *dialect = context->getLoadedDialect<MhloDialect>();
  return dialect->getRegisteredInterface<hlo::HloDialectInterface>();
}

// Replaces the given op with the contents of the given single-block region,
// using the operands of the block terminator to replace operation results.
void replaceOpWithRegion(PatternRewriter &rewriter, Operation *op,
                         Region &region, ValueRange blockArgs = {}) {
  assert(llvm::hasSingleElement(region) && "expected single-block region");
  Block *block = &region.front();
  Operation *terminator = block->getTerminator();
  ValueRange results = terminator->getOperands();
  rewriter.inlineBlockBefore(block, op, blockArgs);
  rewriter.replaceOp(op, results);
  rewriter.eraseOp(terminator);
}

// Returns a new scalar integer value having type `type`. Here `type` must be
// an integer or index type.
Value maybeCastTo(OpBuilder &b, Location loc, Value value, Type type) {
  if (type == value.getType())
    return value;
  assert(type.isIndex() || value.getType().isIndex());
  return b.create<arith::IndexCastOp>(loc, type, value);
}

DenseElementsAttr reshape(DenseElementsAttr attr, ShapedType newType) {
  // TODO(b/232866626): DenseElementsAttr::reshape is broken for bool splats.
  // Once that ticket is fixed, we can remove this conditional.
  if (attr.isSplat() && newType.getElementType().isInteger(/*width=*/1)) {
    auto splatValue = attr.getValues<bool>()[0];
    return DenseElementsAttr::get(newType, {splatValue});
  }
  return attr.reshape(newType);
}

Value castToIndexTensor(OpBuilder &builder, Location loc, Value shapeOp) {
  ShapedType resultTy = shape::getExtentTensorType(
      builder.getContext(), cast<ShapedType>(shapeOp.getType()).getDimSize(0));
  if (shapeOp.getType() == resultTy)
    return shapeOp; // Nothing to do.
  return builder.create<arith::IndexCastOp>(loc, resultTy, shapeOp);
}

template <typename T>
struct Max {
  T operator()(const T &a, const T &b) const { return std::max<T>(a, b); }
};

template <typename T>
struct Min {
  T operator()(const T &a, const T &b) const { return std::min<T>(a, b); }
};

//===----------------------------------------------------------------------===//
// Utilities for the canonicalize patterns
//===----------------------------------------------------------------------===//

// This is an upper limit on how many elements can be folded by an op folder.
// This limit doesn't apply to some special cases like adding a zero,
// multiplying by one, doing many operations with splats.
constexpr int64_t kFoldOpEltLimit = 65536;

APSInt addSign(const APInt &v, Type t) {
  // Add signedness information to the value, treating signless as signed,
  // unless it's i1.
  return APSInt(v, t.isUnsignedInteger() || t.isSignlessInteger(1));
}

template <typename Op, typename ElementType = Type, typename ValType,
          typename Convert>
Attribute BinaryFolder(Op *op, ArrayRef<Attribute> attrs) {
  if (!attrs[0] || !attrs[1])
    return {};

  auto lhs = dyn_cast<DenseElementsAttr>(attrs[0]);
  auto rhs = dyn_cast<DenseElementsAttr>(attrs[1]);
  if (!lhs || !rhs)
    return {};

  auto type = cast<ShapedType>(op->getType());
  if (!type.hasStaticShape()) {
    return {};
  }

  Type etype = type.getElementType();

  // Evaluate for integer values.
  if (!isa<ElementType>(etype)) {
    return {};
  }

  // Special case for folding splats no matter how large.
  // Only covers the case of both attrs being splats; operation-specific cases
  // like adding a zero or multiplying by one are handled elsewhere.
  auto splatLhs = dyn_cast<SplatElementsAttr>(lhs);
  auto splatRhs = dyn_cast<SplatElementsAttr>(rhs);
  if (splatLhs && splatRhs) {
    APSInt signedLhs = addSign(splatLhs.getSplatValue<ValType>(), etype);
    APSInt signedRhs = addSign(splatRhs.getSplatValue<ValType>(), etype);
    FailureOr<decltype(signedLhs)> result(Convert()(signedLhs, signedRhs));
    return succeeded(result) ? SplatElementsAttr::get(type, *result)
                             : Attribute();
  }

  // Prevent folding if the result is too large.
  if (lhs.getNumElements() > kFoldOpEltLimit)
    return {};

  SmallVector<ValType, 6> values;
  values.reserve(lhs.getNumElements());
  for (const auto zip :
       llvm::zip(lhs.getValues<ValType>(), rhs.getValues<ValType>())) {
    APSInt signedLhs = addSign(std::get<0>(zip), etype);
    APSInt signedRhs = addSign(std::get<1>(zip), etype);
    FailureOr<decltype(signedLhs)> result(Convert()(signedLhs, signedRhs));
    if (failed(result)) {
      return {};
    }
    values.push_back(std::move(*result));
  }

  return DenseElementsAttr::get(type, values);
}

// Clamps value to the range [lower, upper]. Requires lower <= upper.
template <typename T>
T clamp(const T &value, const T &lower, const T &upper) {
  assert(lower <= upper);
  return std::max(lower, std::min(value, upper));
}

// Verifies that dimension attribute for the op correctly indexes in operand or
// result shape.
template <typename OpT>
LogicalResult verifyDimAttr(OpT op) {
  int64_t rank = -1;
  if (auto ty = dyn_cast<RankedTensorType>(op.getOperand().getType())) {
    rank = ty.getRank();
  } else if (auto ty = dyn_cast<RankedTensorType>(op.getType())) {
    rank = ty.getRank();
  } else {
    return success();
  }

  int64_t dim = op.getDimension();
  if (dim < 0 || dim >= rank)
    return op.emitOpError() << "requires dimension attribute in range [0, "
                            << rank << "); found (" << dim << ")";
  return success();
}

} // namespace

#include "zkx/mlir_hlo/mhlo/IR/mhlo_canonicalize.inc"

//===----------------------------------------------------------------------===//
// Utilities for verifiers
//===----------------------------------------------------------------------===//

namespace {

LogicalResult verify1dTensor(std::optional<Location> loc,
                             DenseIntElementsAttr attr,
                             std::string_view attrName) {
  int64_t rank = attr.getType().getRank();
  if (rank != 1) {
    return emitOptionalError(loc, attrName, " has rank ", rank,
                             " instead of required rank 1.");
  }
  return success();
}

} // namespace

LogicalResult TypeExtensionsAttr::verifyEncoding(
    ArrayRef<int64_t> shape, mlir::Type elementType,
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError) const {
  return hlo::verifyBounds(
      getBounds(), RankedTensorType::get(shape, elementType), emitError);
}

//===----------------------------------------------------------------------===//
// CompatibleOperandsAndResultType
//===----------------------------------------------------------------------===//

// TODO(b/231358795): Review the use of InferTypeOpInterface for ops that
// support sparsity.
#define INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(Op)                         \
  LogicalResult Op::inferReturnTypeComponents(                                 \
      MLIRContext *context, std::optional<Location> location,                  \
      ValueShapeRange operands, DictionaryAttr attributes,                     \
      OpaqueProperties properties, RegionRange regions,                        \
      SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {           \
    return inferReturnTypeComponentsFromOperands(                              \
        context, location, operands, attributes, properties, regions,          \
        inferredReturnShapes);                                                 \
  }

INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(AddOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(AndOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(BitReverseOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(ClzOp)
// TODO(chokobole): uncomment this. Dependency: CollectiveBroadcastOp
// INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(CollectiveBroadcastOp)
// TODO(chokobole): uncomment this. Dependency: CollectivePermuteOp
// INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(CollectivePermuteOp)
// TODO(chokobole): uncomment this. Dependency: CopyOp
// INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(CopyOp)
// TODO(chokobole): uncomment this. Dependency: CrossReplicaSumOp
// INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(CrossReplicaSumOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(DivOp)
// TODO(chokobole): uncomment this. Dependency: DomainOp
// INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(DomainOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(InverseOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(MaxOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(MinOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(MulOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(NegOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(NotOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(OrOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(PopulationCountOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(PowOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(RemOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(ReverseOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(ShiftLeftOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(ShiftRightArithmeticOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(ShiftRightLogicalOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(SignOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(SubtractOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(XorOp)

//===----------------------------------------------------------------------===//
// AbsOp
//===----------------------------------------------------------------------===//

LogicalResult
AbsOp::inferReturnTypes(MLIRContext *, std::optional<Location> location,
                        ValueRange operands, DictionaryAttr attributes,
                        OpaqueProperties properties, RegionRange regions,
                        SmallVectorImpl<Type> &inferredReturnTypes) {
  AbsOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferAbsOp(location, adaptor.getOperand(), inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// BitcastOp
//===----------------------------------------------------------------------===//

OpFoldResult BitcastOp::fold(FoldAdaptor) {
  if (getResult().getType() != getOperand().getType()) {
    return {};
  }

  auto sourceLayout =
      getOperation()->getAttrOfType<DenseIntElementsAttr>("source_layout");
  auto resultLayout =
      getOperation()->getAttrOfType<DenseIntElementsAttr>("result_layout");

  if (sourceLayout == resultLayout) {
    return getOperand();
  }

  return {};
}

//===----------------------------------------------------------------------===//
// BitcastConvertOp
//===----------------------------------------------------------------------===//

LogicalResult BitcastConvertOp::reifyReturnTypeShapes(
    OpBuilder &builder, ValueRange operands,
    SmallVectorImpl<Value> &reifiedReturnShapes) {
  auto operandType = dyn_cast<RankedTensorType>(operands[0].getType());
  auto resultType = dyn_cast<RankedTensorType>(getType());

  // Only ranked tensors are supported.
  if (!operandType || !resultType)
    return failure();

  // Shape-changing bitcast convert is not implemented.
  // TODO(kramerb): This could be done by adjusting the last dimension.
  DataLayout dataLayout = DataLayout::closest(*this);
  unsigned operandElementSize =
      dataLayout.getTypeSizeInBits(operandType.getElementType());
  unsigned resultElementSize =
      dataLayout.getTypeSizeInBits(resultType.getElementType());
  if (operandElementSize != resultElementSize)
    return failure();

  return hlo::deriveShapeFromOperand(&builder, getOperation(), operands.front(),
                                     &reifiedReturnShapes);
}

LogicalResult BitcastConvertOp::verify() {
  return hlo::verifyBitcastConvertOp(getLoc(), getOperand(), getResult());
}

//===----------------------------------------------------------------------===//
// BroadcastOp
//===----------------------------------------------------------------------===//

OpFoldResult BroadcastOp::fold(FoldAdaptor adaptor) {
  auto attrs = adaptor.getOperands();
  auto type = cast<ShapedType>(getType());
  auto sizesType = getBroadcastSizes().getType();
  if (sizesType.getNumElements() == 0) {
    return getOperand();
  }

  // Constant fold when an operand is a splat tensor attribute.
  if (!attrs[0] || !type.hasStaticShape())
    return {};
  auto splatOperandAttr = dyn_cast<SplatElementsAttr>(attrs[0]);
  if (!splatOperandAttr)
    return {};

  type = prime_ir::maybeConvertPrimeIRToBuiltinType(type);
  return SplatElementsAttr::get(
      type, splatOperandAttr.getSplatValue<mlir::Attribute>());
}

LogicalResult BroadcastOp::inferReturnTypeComponents(
    MLIRContext *, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  BroadcastOp::Adaptor adaptor(operands, attributes, properties, regions);
  if (failed(verify1dTensor(location, adaptor.getBroadcastSizes(),
                            "broadcast_sizes")))
    return failure();
  return hlo::inferBroadcastOp(
      location, adaptor.getOperand(),
      llvm::to_vector(adaptor.getBroadcastSizes().getValues<int64_t>()),
      inferredReturnShapes);
}

LogicalResult BroadcastOp::reifyReturnTypeShapes(
    OpBuilder &builder, ValueRange operands,
    SmallVectorImpl<Value> &reifiedReturnShapes) {
  BroadcastOp::Adaptor adaptor(operands);
  Value operand = adaptor.getOperand();

  auto operandType = dyn_cast<RankedTensorType>(operand.getType());
  // Unranked tensors are not supported.
  if (!operandType)
    return failure();

  Location loc = getLoc();
  SmallVector<Value, 4> shapeValues;

  // Collect the broadcast sizes.
  for (const auto &size : getBroadcastSizes()) {
    shapeValues.push_back(
        builder.create<arith::ConstantIndexOp>(loc, size.getZExtValue()));
  }

  // Collect the operand sizes.
  for (auto index : llvm::seq<int64_t>(0, operandType.getRank())) {
    shapeValues.push_back(
        builder.createOrFold<tensor::DimOp>(loc, operand, index));
  }

  reifiedReturnShapes.push_back(builder.create<tensor::FromElementsOp>(
      loc,
      RankedTensorType::get({static_cast<int64_t>(shapeValues.size())},
                            builder.getIndexType()),
      shapeValues));

  return success();
}

//===----------------------------------------------------------------------===//
// BroadcastInDimOp
//===----------------------------------------------------------------------===//

LogicalResult BroadcastInDimOp::verify() {
  return hlo::verifyBroadcastInDimOp(
      getLoc(), getOperand(),
      llvm::to_vector(getBroadcastDimensions().getValues<int64_t>()),
      getResult());
}

OpFoldResult BroadcastInDimOp::fold(FoldAdaptor adaptor) {
  auto attrs = adaptor.getOperands();
  auto type = cast<RankedTensorType>(getType());
  if (type == getOperand().getType()) {
    auto broadcastValues = getBroadcastDimensions().getValues<int64_t>();
    if (!std::equal(broadcastValues.begin(), broadcastValues.end(),
                    llvm::seq<int64_t>(0, type.getRank()).begin())) {
      return {};
    }
    return getOperand();
  }

  // Constant fold when an operand is a splat tensor attribute.
  if (!attrs[0] || !type.hasStaticShape())
    return {};
  auto splatOperandAttr = dyn_cast<SplatElementsAttr>(attrs[0]);
  if (!splatOperandAttr)
    return {};

  type =
      cast<RankedTensorType>(prime_ir::maybeConvertPrimeIRToBuiltinType(type));
  return SplatElementsAttr::get(
      type, splatOperandAttr.getSplatValue<mlir::Attribute>());
}

namespace {

// Simplify BroadcastInDim has the following behaviors: replace BroadcastInDim
// with Reshape or Transpose if they are equivalent or replace
// BroadcastInDim(BroadcastInDim(X)) with BroadcastInDim(X)
class BroadcastInDimSimplifier : public OpRewritePattern<BroadcastInDimOp> {
public:
  using OpRewritePattern<BroadcastInDimOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(BroadcastInDimOp op,
                                PatternRewriter &rewriter) const override {
    auto operandType = dyn_cast<RankedTensorType>(op.getOperand().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!operandType || !resultType) {
      return failure();
    }
    auto bsDimIndices = op.getBroadcastDimensions().getValues<int64_t>();
    if (operandType.hasStaticShape() && resultType.hasStaticShape()) {
      bool sameTotalElements =
          operandType.getNumElements() == resultType.getNumElements();
      // BroadcastInDim equivalent to reshape
      if (llvm::is_sorted(bsDimIndices) && sameTotalElements) {
        rewriter.replaceOpWithNewOp<ReshapeOp>(op, op.getType(),
                                               op.getOperand());
        return success();
      }
      // BroadcastInDim equivalent to transpose
      if (operandType.getRank() == resultType.getRank() && sameTotalElements) {
        rewriter.replaceOpWithNewOp<TransposeOp>(
            op, op.getType(), op.getOperand(), op.getBroadcastDimensions());
        return success();
      }
    }
    // eliminate redundant BroadcastInDim
    if (auto broadcastInDimOp = llvm::dyn_cast_or_null<BroadcastInDimOp>(
            op.getOperand().getDefiningOp())) {
      auto newIndices = cast<DenseIntElementsAttr>(
          broadcastInDimOp.getBroadcastDimensions().mapValues(
              op.getBroadcastDimensions().getElementType(),
              [&bsDimIndices](const APInt &dim) -> APInt {
                return APInt(dim.getBitWidth(),
                             bsDimIndices[dim.getSExtValue()], true);
              }));
      rewriter.replaceOpWithNewOp<BroadcastInDimOp>(
          op, op.getType(), broadcastInDimOp.getOperand(), newIndices);
      return success();
    }
    return failure();
  }
};

} // namespace

void BroadcastInDimOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                   MLIRContext *context) {
  results.add<BroadcastInDimSimplifier>(context);
}

//===----------------------------------------------------------------------===//
// Case Op
//===----------------------------------------------------------------------===//

LogicalResult
CaseOp::inferReturnTypes(MLIRContext *context, std::optional<Location> location,
                         ValueRange operands, DictionaryAttr attributes,
                         OpaqueProperties properties, RegionRange regions,
                         SmallVectorImpl<Type> &inferredReturnTypes) {
  CaseOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferCaseOp(location, adaptor.getIndex(), adaptor.getRegions(),
                          inferredReturnTypes);
}

namespace {

LogicalResult inlineCaseConstantCondition(CaseOp caseOp,
                                          PatternRewriter &rewriter) {
  DenseIntElementsAttr indexAttr;
  if (!matchPattern(caseOp.getIndex(), m_Constant(&indexAttr))) {
    return failure();
  }
  int64_t index =
      indexAttr.getSplatValue<IntegerAttr>().getValue().getSExtValue();
  // For an OOB index, the last branch is executed as the default branch:
  // https://www.tensorflow.org/xla/operation_semantics#conditional
  if (index < 0 || index >= caseOp.getNumRegions())
    index = caseOp.getNumRegions() - 1;

  Region &region = caseOp.getRegion(index);
  if (!llvm::hasSingleElement(region))
    return failure();
  replaceOpWithRegion(rewriter, caseOp, region);
  return success();
}

} // namespace

void CaseOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.add(&inlineCaseConstantCondition);
}

//===----------------------------------------------------------------------===//
// ClampOp
//===----------------------------------------------------------------------===//

OpFoldResult ClampOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  auto operand = dyn_cast_or_null<ElementsAttr>(operands[1]);
  auto min = dyn_cast_or_null<ElementsAttr>(operands[0]);
  auto max = dyn_cast_or_null<ElementsAttr>(operands[2]);
  if (!operand || !min || !max) {
    return {};
  }
  if (min.getShapedType().getRank() == 0) {
    min = DenseElementsAttr::get(operand.getShapedType(),
                                 min.getValues<Attribute>()[0]);
  }
  if (max.getShapedType().getRank() == 0) {
    max = DenseElementsAttr::get(operand.getShapedType(),
                                 max.getValues<Attribute>()[0]);
  }
  Attribute result = {};
  if (isa<IntegerType>(operand.getShapedType().getElementType())) {
    result = BinaryFolder<ClampOp, IntegerType, APInt, Max<APSInt>>(
        this, ArrayRef<Attribute>{min, operand});
    result = BinaryFolder<ClampOp, IntegerType, APInt, Min<APSInt>>(
        this, ArrayRef<Attribute>{max, result});
  }
  return result;
}

LogicalResult ClampOp::inferReturnTypeComponents(
    MLIRContext *, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ClampOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferClampOp(location, adaptor.getMin(), adaptor.getOperand(),
                           adaptor.getMax(), inferredReturnShapes);
}

LogicalResult
ClampOp::reifyReturnTypeShapes(OpBuilder &builder, ValueRange operands,
                               SmallVectorImpl<Value> &reifiedReturnShapes) {
  // For `mhlo.clamp`, the first operand may be a scalar.
  return hlo::deriveShapeFromOperand(&builder, getOperation(), operands[1],
                                     &reifiedReturnShapes);
}

//===----------------------------------------------------------------------===//
// CompareOp
//===----------------------------------------------------------------------===//

LogicalResult CompareOp::inferReturnTypeComponents(
    MLIRContext *context, std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  CompareOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferCompareOp(context, location, adaptor.getLhs(),
                             inferredReturnShapes);
}

LogicalResult
CompareOp::reifyReturnTypeShapes(OpBuilder &builder, ValueRange operands,
                                 SmallVectorImpl<Value> &reifiedReturnShapes) {
  return hlo::deriveShapeFromOperand(&builder, getOperation(), operands.front(),
                                     &reifiedReturnShapes);
}

namespace {

template <typename Op, typename ElementType, typename SrcType, typename Convert>
Attribute CompareFolder(CompareOp op, ArrayRef<Attribute> attrs) {
  if (!attrs[0] || !attrs[1])
    return {};

  DenseElementsAttr lhs = dyn_cast<DenseElementsAttr>(attrs[0]);
  DenseElementsAttr rhs = dyn_cast<DenseElementsAttr>(attrs[1]);
  if (!lhs || !rhs)
    return {};

  ShapedType operandType = cast<ShapedType>(op.getOperand(0).getType());
  if (!operandType.hasStaticShape()) {
    return {};
  }

  auto etype = operandType.getElementType();
  if (!isa<ElementType>(etype)) {
    return {};
  }

  // Prevent folding if the result is too large.
  if (lhs.getNumElements() > kFoldOpEltLimit)
    return {};

  SmallVector<bool, 6> values;
  values.reserve(lhs.getNumElements());
  for (const auto zip :
       llvm::zip(lhs.getValues<SrcType>(), rhs.getValues<SrcType>())) {
    values.push_back(
        Convert()(addSign(std::get<0>(zip), lhs.getElementType()),
                  addSign(std::get<1>(zip), rhs.getElementType())));
  }

  auto resultTy = cast<ShapedType>(op.getType());
  return DenseElementsAttr::get(resultTy, values);
}

} // namespace

OpFoldResult CompareOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  auto resultTy = cast<ShapedType>(getType());
  if (!resultTy.hasStaticShape())
    return {};

  auto direction = getComparisonDirection();
  if (getLhs() == getRhs()) {
    if (direction == ComparisonDirection::LE ||
        direction == ComparisonDirection::EQ ||
        direction == ComparisonDirection::GE) {
      return DenseIntElementsAttr::get(resultTy, {true});
    }
    return DenseIntElementsAttr::get(resultTy, {false});
  }

  auto opElType = cast<ShapedType>(getLhs().getType()).getElementType();
  // Fold tensor<*xi1> != false to just return tensor<*xi1>
  if (direction == ComparisonDirection::NE && opElType.isInteger(1)) {
    DenseIntElementsAttr cstAttr;
    if (matchPattern(getLhs(), m_Constant(&cstAttr))) {
      if (cstAttr.isSplat() && !cstAttr.getSplatValue<bool>()) {
        return getRhs();
      }
    }

    if (matchPattern(getRhs(), m_Constant(&cstAttr))) {
      if (cstAttr.isSplat() && !cstAttr.getSplatValue<bool>()) {
        return getLhs();
      }
    }
  }

  // Fold tensor<*xi1> == True to just return tensor<*xi1>
  if (direction == ComparisonDirection::EQ && opElType.isInteger(1)) {
    DenseIntElementsAttr cstAttr;
    if (matchPattern(getLhs(), m_Constant(&cstAttr))) {
      if (cstAttr.isSplat() && cstAttr.getSplatValue<bool>()) {
        return getRhs();
      }
    }

    if (matchPattern(getRhs(), m_Constant(&cstAttr))) {
      if (cstAttr.isSplat() && cstAttr.getSplatValue<bool>()) {
        return getLhs();
      }
    }
  }

  if (!operands[0] || !operands[1]) {
    return {};
  }

#define COMPARE_FOLDER(Op, comparison, Func)                                   \
  if (direction == comparison) {                                               \
    if (auto folded = CompareFolder<Op, IntegerType, APInt, Func<APSInt>>(     \
            *this, operands))                                                  \
      return folded;                                                           \
  }

  COMPARE_FOLDER(CompareOp, ComparisonDirection::EQ, std::equal_to);
  COMPARE_FOLDER(CompareOp, ComparisonDirection::NE, std::not_equal_to);
  COMPARE_FOLDER(CompareOp, ComparisonDirection::LT, std::less);
  COMPARE_FOLDER(CompareOp, ComparisonDirection::LE, std::less_equal);
  COMPARE_FOLDER(CompareOp, ComparisonDirection::GT, std::greater);
  COMPARE_FOLDER(CompareOp, ComparisonDirection::GE, std::greater_equal);
#undef COMPARE_FOLDER

  return {};
}

//===----------------------------------------------------------------------===//
// ConcatenateOp
//===----------------------------------------------------------------------===//

namespace {

class SingleOperandConcatenateToCast : public OpRewritePattern<ConcatenateOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ConcatenateOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getVal().size() != 1)
      return failure();

    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, op.getType(),
                                                op.getVal().front());
    return success();
  }
};

class ConcatenateOperandRemoval : public OpRewritePattern<ConcatenateOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ConcatenateOp op,
                                PatternRewriter &rewriter) const override {
    auto axis = op.getDimension();
    llvm::SmallVector<Value, 6> newOperands;
    for (auto operand : op.getOperands()) {
      auto ty = cast<ShapedType>(operand.getType());
      if (!ty.hasRank() || ty.getDimSize(axis) != 0) {
        newOperands.push_back(operand);
      }
    }

    if (!newOperands.empty() && newOperands.size() < op.getNumOperands()) {
      rewriter.replaceOpWithNewOp<ConcatenateOp>(
          op, op.getResult().getType(), newOperands, op.getDimension());
      return success();
    }

    return failure();
  }
};

class ConcatenateForwarding : public OpRewritePattern<ConcatenateOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ConcatenateOp op,
                                PatternRewriter &rewriter) const override {
    auto getFlattenedOperands = [&](const Value &val) -> ValueRange {
      auto definingOp = dyn_cast_or_null<ConcatenateOp>(val.getDefiningOp());
      // To avoid inflate the memory footprint, only flatten the ConcatenateOp
      // when it has only one use.
      if (definingOp && definingOp->hasOneUse() &&
          definingOp.getDimension() == op.getDimension())
        return definingOp.getVal();
      return val;
    };

    bool needToFlatten = false;
    int operandCount = 0;
    llvm::for_each(op.getVal(), [&](Value val) {
      auto result = getFlattenedOperands(val);
      if (result.size() != 1 || result[0] != val)
        needToFlatten = true;
      operandCount += result.size();
    });

    if (!needToFlatten)
      return failure();

    llvm::SmallVector<Value, 6> newOperands;
    newOperands.reserve(operandCount);

    for (auto operand : op.getVal()) {
      auto flattenedOperands = getFlattenedOperands(operand);
      newOperands.append(flattenedOperands.begin(), flattenedOperands.end());
    }

    rewriter.replaceOpWithNewOp<ConcatenateOp>(op, op.getResult().getType(),
                                               newOperands, op.getDimension());
    return success();
  }
};

} // namespace

LogicalResult ConcatenateOp::inferReturnTypes(
    MLIRContext *, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  ConcatenateOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferConcatenateOp(location, adaptor.getVal().getTypes(),
                                 adaptor.getDimension(), inferredReturnTypes);
}

void ConcatenateOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                MLIRContext *context) {
  results.add<ConcatenateOperandRemoval, ConcatenateForwarding,
              SingleOperandConcatenateToCast>(context);
}

namespace {

template <typename T>
Attribute foldConcatenateHelper(ConcatenateOp *op,
                                ArrayRef<Attribute> operands) {
  auto axis = op->getDimension();
  auto type = cast<ShapedType>(op->getType());
  auto shape = type.getShape();

  size_t topSize = 1;
  for (int i = 0, e = axis; i < e; i++) {
    topSize = topSize * shape[i];
  }

  // Prevent folding if the result is too large.
  if (type.getNumElements() > kFoldOpEltLimit)
    return {};

  SmallVector<T, 6> values;
  for (size_t i = 0; i < topSize; i++) {
    for (auto operand : operands) {
      DenseElementsAttr attr = cast<DenseElementsAttr>(operand);
      size_t bottomSize = attr.getNumElements() / topSize;
      auto iter = attr.getValues<T>().begin() + i * bottomSize;
      values.append(iter, iter + bottomSize);
    }
  }

  type = prime_ir::maybeConvertPrimeIRToBuiltinType(type);
  return DenseElementsAttr::get(type, values);
}

Attribute foldConcatenate(ConcatenateOp *op, ArrayRef<Attribute> operands) {
  for (auto operand : operands) {
    if (!operand)
      return {};
  }

  auto type = cast<ShapedType>(op->getResult().getType());
  auto etype = type.getElementType();
  if (isa<IntegerType>(etype) || isa<prime_ir::field::PrimeFieldType>(etype)) {
    return foldConcatenateHelper<APInt>(op, operands);
  }

  return {};
}

} // namespace

OpFoldResult ConcatenateOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  if (getNumOperands() == 1 && getOperand(0).getType() == getType())
    return getOperand(0);

  ShapedType type = cast<ShapedType>(getResult().getType());
  if (!type.hasStaticShape())
    return {};

  auto axis = getDimension();
  if (auto attr = foldConcatenate(this, operands)) {
    return attr;
  }

  for (auto operand : getOperands()) {
    auto ty = cast<ShapedType>(operand.getType());
    if (ty.getDimSize(axis) != 0) {
      return {};
    }
  }

  type = prime_ir::maybeConvertPrimeIRToBuiltinType(type);
  return DenseElementsAttr::get(type, ArrayRef<Attribute>());
}

LogicalResult ConcatenateOp::reifyReturnTypeShapes(
    OpBuilder &builder, ValueRange operands,
    SmallVectorImpl<Value> &reifiedReturnShapes) {
  ConcatenateOp::Adaptor adaptor(operands);
  auto inputs = adaptor.getVal();

  auto operandType = dyn_cast<RankedTensorType>(inputs[0].getType());
  // Not support unranked type a.t.m.
  if (!operandType)
    return failure();

  Location loc = this->getLoc();
  Type shapeScalarType = builder.getIndexType();
  auto toShapeScalarType = [&](Value v) {
    return maybeCastTo(builder, loc, v, shapeScalarType);
  };

  SmallVector<SmallVector<Value, 4>, 4> allShapeValues;
  for (size_t inputId = 0; inputId < inputs.size(); ++inputId) {
    Value operand = inputs[inputId];
    auto operandType = dyn_cast<RankedTensorType>(operand.getType());
    if (!operandType)
      return failure();

    SmallVector<Value, 4> shapeVals;
    for (const auto &element : llvm::enumerate(operandType.getShape())) {
      Value valueDim = toShapeScalarType(
          builder.create<tensor::DimOp>(loc, operand, element.index()));
      shapeVals.push_back(valueDim);
    }
    allShapeValues.emplace_back(std::move(shapeVals));
  }

  int axis = this->getDimension();
  auto &shapeValues = allShapeValues[0];
  for (size_t vecId = 1; vecId < allShapeValues.size(); ++vecId) {
    auto &otherShapeValues = allShapeValues[vecId];
    if (otherShapeValues.size() != shapeValues.size()) {
      this->emitOpError()
          << "Concatenate expects all operands must be of the same rank";
      return failure();
    }
    shapeValues[axis] = builder.create<arith::AddIOp>(loc, shapeValues[axis],
                                                      otherShapeValues[axis]);
  }

  Value outputShape = builder.create<tensor::FromElementsOp>(
      loc,
      RankedTensorType::get({static_cast<int64_t>(shapeValues.size())},
                            shapeScalarType),
      shapeValues);
  reifiedReturnShapes.push_back(outputShape);

  return success();
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
  assert(adaptor.getOperands().empty() && "constant has no operands");

  // Return the held attribute value.
  return getValue();
}

// static
// Builds a constant op with the specified attribute `value`.
void ConstantOp::build(OpBuilder & /*builder*/, OperationState &result,
                       Attribute value) {
  Properties &properties = result.getOrAddProperties<Properties>();
  Type type;
  if (auto elemAttr = dyn_cast<ElementsAttr>(value)) {
    type = elemAttr.getType();
    properties.value = elemAttr;
  } else if (isa<BoolAttr, IntegerAttr>(value)) {
    // All ZKX types must be tensor types. In the build() method, we want to
    // provide more flexibility by allowing attributes of scalar types. But we
    // need to wrap it up with ElementsAttr to construct valid ZKX constants.
    type =
        RankedTensorType::get(/*shape=*/{}, cast<TypedAttr>(value).getType());
    properties.value = DenseElementsAttr::get(cast<TensorType>(type), value);
  }

  // TODO: support other ZKX specific types.
  assert(type && "unsupported attribute type for building mhlo.constant");
  result.types.push_back(type);
}

LogicalResult
ConstantOp::inferReturnTypes(MLIRContext *, std::optional<Location> location,
                             ValueRange operands, DictionaryAttr attributes,
                             OpaqueProperties properties, RegionRange regions,
                             SmallVectorImpl<Type> &inferredReturnTypes) {
  ConstantOpAdaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferConstantOp(location, adaptor.getValue(),
                              inferredReturnTypes);
}

bool ConstantOp::isCompatibleReturnTypes(TypeRange l, TypeRange r) {
  if (l.size() != r.size() || l.size() != 1)
    return false;
  auto lhsTy = cast<ShapedType>(l.front());
  auto rhsTy = cast<ShapedType>(r.front());
  if (!lhsTy || !rhsTy)
    return false;

  if (lhsTy == rhsTy)
    return true;

  Type lhsElementType = getElementTypeOrSelf(lhsTy);
  Type rhsElementType = getElementTypeOrSelf(rhsTy);
  // NOTE(chokobole): This allows us to create constants of prime field from
  // integer constants.
  if (isa<IntegerType>(lhsElementType) &&
      isa<prime_ir::field::PrimeFieldType>(rhsElementType)) {
    return lhsTy.clone(rhsElementType) == rhsTy;
  }
  return false;
}

ParseResult ConstantOp::parse(OpAsmParser &parser, OperationState &result) {
  return hlo::parseConstantOp(parser, result);
}

void ConstantOp::print(OpAsmPrinter &p) {
  hlo::printConstantOp(p, getOperation(), getValue());
}

//===----------------------------------------------------------------------===//
// ConvertOp
//===----------------------------------------------------------------------===//

// static
void ConvertOp::build(OpBuilder &builder, OperationState &result, Value operand,
                      Type resultElementTy) {
  auto rankedTy = cast<RankedTensorType>(operand.getType());
  auto resultTy = RankedTensorType::get(rankedTy.getShape(), resultElementTy);
  build(builder, result, resultTy, operand);
}

OpFoldResult ConvertOp::fold(FoldAdaptor adaptor) {
  ArrayRef<Attribute> operands = adaptor.getOperands();
  auto operandTy = cast<TensorType>(getOperand().getType());
  auto resultTy = cast<TensorType>(getResult().getType());
  if (operandTy == resultTy)
    return getOperand();

  // If the result has non-static shape, a convert op is necessary to go from
  // static shape to non-static shape.
  if (!resultTy.hasStaticShape())
    return {};

  // If the operand is constant, we can do the conversion now.
  auto elementsAttr = dyn_cast_or_null<ElementsAttr>(operands.front());
  if (!elementsAttr)
    return {};

  // Prevent folding if the result is too large.
  if (elementsAttr.getNumElements() > kFoldOpEltLimit)
    return {};
  return hlo::convertElementsAttr(elementsAttr,
                                  getElementTypeOrSelf(getResult()));
}

namespace {

struct EliminateRedundantConvert : public OpRewritePattern<ConvertOp> {
  using OpRewritePattern<ConvertOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ConvertOp op,
                                PatternRewriter &rewriter) const override {
    auto convertOp = op.getOperand().getDefiningOp<ConvertOp>();
    if (!convertOp) {
      return failure();
    }
    auto firstType =
        cast<TensorType>(convertOp.getOperand().getType()).getElementType();
    auto secondType =
        cast<TensorType>(op.getOperand().getType()).getElementType();
    auto thirdType =
        cast<TensorType>(op.getResult().getType()).getElementType();
    Location loc = rewriter.getFusedLoc({convertOp->getLoc(), op->getLoc()});
    if (isa<IntegerType>(firstType) && isa<IntegerType>(secondType) &&
        isa<IntegerType>(thirdType)) {
      // fold when the second integer type's width is longer than first,
      // like i16 -> i32 -> i64, u16 -> i32 -> u32
      if (cast<IntegerType>(secondType).getWidth() >
          cast<IntegerType>(firstType).getWidth()) {
        Value result = rewriter.create<ConvertOp>(loc, op.getResult().getType(),
                                                  convertOp.getOperand());
        rewriter.replaceOp(op, result);
        return success();
      }
    }
    return failure();
  }
};

} // namespace

void ConvertOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add<EliminateIdentityConvert>(context);
  results.add<EliminateRedundantConvert>(context);
}

//===----------------------------------------------------------------------===//
// CreateTokenOp
//===----------------------------------------------------------------------===//

LogicalResult
CreateTokenOp::inferReturnTypes(MLIRContext *context,
                                std::optional<Location> location, ValueRange,
                                DictionaryAttr, OpaqueProperties, RegionRange,
                                SmallVectorImpl<Type> &inferredReturnTypes) {
  return hlo::inferCreateTokenOp(getMhloDialect(context), location,
                                 inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// DotGeneralOp
//===----------------------------------------------------------------------===//

LogicalResult DotGeneralOp::verify() {
  return hlo::verifyDotGeneralOp(
      getLoc(), getLhs(), getRhs(),
      getDotDimensionNumbersAttr().getLhsBatchingDimensions(),
      getDotDimensionNumbersAttr().getRhsBatchingDimensions(),
      getDotDimensionNumbersAttr().getLhsContractingDimensions(),
      getDotDimensionNumbersAttr().getRhsContractingDimensions(), getResult());
}

LogicalResult DotGeneralOp::reifyReturnTypeShapes(
    OpBuilder &builder, ValueRange operands,
    SmallVectorImpl<Value> &reifiedReturnShapes) {
  auto lhsRankedType = dyn_cast<RankedTensorType>(getLhs().getType());
  auto rhsRankedType = dyn_cast<RankedTensorType>(getRhs().getType());
  if (!lhsRankedType || !rhsRankedType)
    return failure();

  Adaptor adaptor(operands);
  auto dimNumbers = getDotDimensionNumbers();
  SmallVector<Value> dimensions;

  for (const int64_t lhsDim : dimNumbers.getLhsBatchingDimensions())
    dimensions.push_back(
        builder.create<tensor::DimOp>(getLoc(), adaptor.getLhs(), lhsDim));

  for (int64_t i = 0; i < lhsRankedType.getRank(); i++)
    if (!llvm::is_contained(dimNumbers.getLhsContractingDimensions(), i) &&
        !llvm::is_contained(dimNumbers.getLhsBatchingDimensions(), i))
      dimensions.push_back(
          builder.create<tensor::DimOp>(getLoc(), adaptor.getLhs(), i));

  for (int64_t i = 0; i < rhsRankedType.getRank(); i++)
    if (!llvm::is_contained(dimNumbers.getRhsContractingDimensions(), i) &&
        !llvm::is_contained(dimNumbers.getRhsBatchingDimensions(), i))
      dimensions.push_back(
          builder.create<tensor::DimOp>(getLoc(), adaptor.getRhs(), i));

  reifiedReturnShapes.push_back(
      builder.create<tensor::FromElementsOp>(getLoc(), dimensions));
  return success();
}

Speculation::Speculatability DotGeneralOp::getSpeculatability() {
  // Batching and contracting dims must be static, otherwise they could disagree
  // at runtime.
  // Other dims follow SpeculatableIfStaticDimInOutputIsStaticInInput.

  auto lhsType = getLhs().getType();
  auto rhsType = getRhs().getType();

  auto dimensionsAttr = getDotDimensionNumbersAttr();
  auto lhsBatchingDimensions = dimensionsAttr.getLhsBatchingDimensions();
  auto lhsContractingDimensions = dimensionsAttr.getLhsContractingDimensions();
  auto rhsBatchingDimensions = dimensionsAttr.getRhsBatchingDimensions();
  auto rhsContractingDimensions = dimensionsAttr.getRhsContractingDimensions();

  auto lhsSpecialDimensions = llvm::concat<const int64_t>(
      lhsBatchingDimensions, lhsContractingDimensions);
  auto rhsSpecialDimensions = llvm::concat<const int64_t>(
      rhsBatchingDimensions, rhsContractingDimensions);

  for (auto i : lhsSpecialDimensions)
    if (lhsType.isDynamicDim(i))
      return Speculation::NotSpeculatable;
  for (auto i : rhsSpecialDimensions)
    if (rhsType.isDynamicDim(i))
      return Speculation::NotSpeculatable;

  auto resultType = getType();
  int64_t resultIndex = lhsBatchingDimensions.size();
  for (int64_t i = 0; i < lhsType.getRank(); i++) {
    if (llvm::is_contained(lhsSpecialDimensions, i))
      continue;
    if (!resultType.isDynamicDim(resultIndex) && lhsType.isDynamicDim(i))
      return Speculation::NotSpeculatable;
    resultIndex++;
  }
  for (int64_t i = 0; i < rhsType.getRank(); i++) {
    if (llvm::is_contained(rhsSpecialDimensions, i))
      continue;
    if (!resultType.isDynamicDim(resultIndex) && rhsType.isDynamicDim(i))
      return Speculation::NotSpeculatable;
    resultIndex++;
  }

  return Speculation::Speculatable;
}

//===----------------------------------------------------------------------===//
// DynamicBroadcastInDimOp
//===----------------------------------------------------------------------===//

LogicalResult DynamicBroadcastInDimOp::verify() {
  // Check for unranked dynamism. Unranked dynamism is not supported by
  // StableHLO (hlo::verifyReshapeOp will fail) and we can't verify
  // anything statically in that case anyway.
  auto outputdimensionsType = cast<ShapedType>(getOutputDimensions().getType());
  auto resultType = cast<ShapedType>(getResult().getType());
  if (!outputdimensionsType.hasRank() || !resultType.hasRank()) {
    return success();
  }

  return hlo::verifyDynamicBroadcastInDimOp(
      getLoc(), getOperand(), getOutputDimensions(),
      llvm::to_vector(getBroadcastDimensions().getValues<int64_t>()),
      getKnownExpandingDimensionsAttr()
          ? std::optional<SmallVector<int64_t>>(llvm::to_vector(
                getKnownExpandingDimensions()->getValues<int64_t>()))
          : std::nullopt,
      getKnownNonexpandingDimensions()
          ? std::optional<SmallVector<int64_t>>(llvm::to_vector(
                getKnownNonexpandingDimensions()->getValues<int64_t>()))
          : std::nullopt,
      getResult());
}

namespace {

// Does the same as PatternRewriter::replaceOpWithNewOp, but with a twist.
//
// Sometimes, we want to replace an op with a new op and simultaneously refine
// the result type from a dynamically-shaped type to a statically-shaped type.
// (Search for usages of this function for examples).
//
// Oftentimes, this works just fine because MHLO is designed to accommodate
// this kind of type refinements. But sometimes, this doesn't work - when
// the op is used outside of the MHLO dialect (e.g. in func.return). In these
// cases, we insert a tensor.cast to smooth things out.
template <typename OpTy, typename... Args>
OpTy refineOpWithNewOp(PatternRewriter &rewriter, Operation *op,
                       Args &&...args) {
  auto newOp = rewriter.create<OpTy>(op->getLoc(), std::forward<Args>(args)...);

  llvm::SmallVector<Value> replacementResults;
  assert(op->getNumResults() == newOp->getNumResults() &&
         "replacement op doesn't match results of original op");
  for (auto [opResult, newOpResult] :
       llvm::zip(op->getResults(), newOp->getResults())) {
    Value replacementResult = newOpResult;
    if (llvm::any_of(opResult.getUsers(), [&](Operation *user) {
          return user->getDialect() != op->getDialect();
        })) {
      replacementResult = rewriter.create<tensor::CastOp>(
          op->getLoc(), opResult.getType(), newOpResult);
    }
    replacementResults.push_back(replacementResult);
  }

  rewriter.replaceOp(op, replacementResults);
  return newOp;
}

// If a DynamicBroadCastInDimOp is not actually dynamic, use an ordinary
// BroadcastInDimOp.
class DynamicBroadcastInDimOpNotActuallyDynamic
    : public OpRewritePattern<DynamicBroadcastInDimOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DynamicBroadcastInDimOp op,
                                PatternRewriter &rewriter) const override {
    auto type = dyn_cast<RankedTensorType>(op.getType());
    auto operandType = dyn_cast<RankedTensorType>(op.getOperand().getType());
    auto *outputDimOp = op.getOutputDimensions().getDefiningOp();
    if (!type || !operandType || !operandType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op, "requires operand static shape");
    }
    // output has static shape, replace with broadcast_in_dim
    if (type.hasStaticShape()) {
      rewriter.replaceOpWithNewOp<BroadcastInDimOp>(
          op, type, op.getOperand(), op.getBroadcastDimensions());
      return success();
    }
    // output_dimensions are constant, set output shape with output_dimensions,
    // then replace with broadcast_in_dim
    if (outputDimOp && outputDimOp->hasTrait<mlir::OpTrait::ConstantLike>()) {
      DenseIntElementsAttr shapeAttr;
      if (matchPattern(outputDimOp, m_Constant(&shapeAttr))) {
        SmallVector<int64_t> outputShape;
        for (APInt shape : shapeAttr.getValues<APInt>()) {
          outputShape.push_back(shape.getZExtValue());
        }
        refineOpWithNewOp<BroadcastInDimOp>(
            rewriter, op,
            RankedTensorType::get(outputShape, type.getElementType()),
            op.getOperand(), op.getBroadcastDimensions());
        return success();
      }
    }
    return rewriter.notifyMatchFailure(
        op, "requires output static shape or constant broadcast dimensions");
  }
};

class ChainedDynamicBroadcastInDimCanonicalization
    : public OpRewritePattern<DynamicBroadcastInDimOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DynamicBroadcastInDimOp bcast,
                                PatternRewriter &rewriter) const override {
    auto precedingBcast =
        bcast.getOperand().getDefiningOp<DynamicBroadcastInDimOp>();
    if (!precedingBcast)
      return failure();

    // Compose broadcast dimensions.
    DenseIntElementsAttr precedingBcastDims =
        precedingBcast.getBroadcastDimensions();
    DenseIntElementsAttr bcastDims = bcast.getBroadcastDimensions();
    SmallVector<APInt, 4> composition;
    for (APInt precedingDim : precedingBcastDims) {
      composition.push_back(
          bcastDims.getValues<APInt>()[precedingDim.getZExtValue()]);
    }
    auto composedBcastDims =
        DenseIntElementsAttr::get(precedingBcastDims.getType(), composition);

    rewriter.replaceOpWithNewOp<DynamicBroadcastInDimOp>(
        bcast, bcast.getType(), precedingBcast.getOperand(),
        bcast.getOutputDimensions(), composedBcastDims);
    return success();
  }
};

// If all dimensions are known to be nonexpanding from the attribute, replace
// the dynamic broadcast with a cast.
class DynamicBroadcastInDimAllDimsNonExpanding
    : public OpRewritePattern<DynamicBroadcastInDimOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DynamicBroadcastInDimOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "requires ranked result type");

    if (!op.getKnownNonexpandingDimensions().has_value() ||
        op.getKnownNonexpandingDimensions()->size() != resultType.getRank()) {
      return rewriter.notifyMatchFailure(
          op, "known_nonexpanding_dimensions don't cover all output dims");
    }

    auto cast = rewriter.createOrFold<tensor::CastOp>(op.getLoc(), resultType,
                                                      op.getOperand());
    rewriter.replaceOp(op, cast);
    return success();
  }
};
} // namespace

void DynamicBroadcastInDimOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add<ChainedDynamicBroadcastInDimCanonicalization,
              DynamicBroadcastInDimOpNotActuallyDynamic,
              DynamicBroadcastInDimAllDimsNonExpanding,
              DynamicBroadcastToOwnShape_1, DynamicBroadcastToOwnShape_2,
              DynamicBroadcastToOwnShape_3, DynamicBroadcastToOwnShape_4>(
      context);
}

LogicalResult DynamicBroadcastInDimOp::reifyReturnTypeShapes(
    OpBuilder &builder, ValueRange operands,
    SmallVectorImpl<Value> &reifiedReturnShapes) {
  DynamicBroadcastInDimOp::Adaptor adaptor(operands);
  reifiedReturnShapes.push_back(
      castToIndexTensor(builder, getLoc(), adaptor.getOutputDimensions()));
  return success();
}

//===----------------------------------------------------------------------===//
// DynamicIotaOp
//===----------------------------------------------------------------------===//

namespace {

struct DynamicIotaIsStatic : public OpRewritePattern<DynamicIotaOp> {
  using OpRewritePattern<DynamicIotaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DynamicIotaOp iota,
                                PatternRewriter &rewriter) const override {
    // Result type has static shape, replace with iota.
    auto resultTy = cast<ShapedType>(iota.getType());
    if (resultTy.hasStaticShape()) {
      rewriter.replaceOpWithNewOp<IotaOp>(iota, resultTy,
                                          iota.getIotaDimension());
      return success();
    }

    return rewriter.notifyMatchFailure(iota, "requires output static shape");
  }
};

// Dynamic Iota operations across multiple dimensions can be reduced to an iota
// and a ranked broadcast.
struct DynamicIotaBroadcast : public OpRewritePattern<DynamicIotaOp> {
  using OpRewritePattern<DynamicIotaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DynamicIotaOp iota,
                                PatternRewriter &rewriter) const override {
    auto resultTy = cast<ShapedType>(iota.getType());
    if (!resultTy.hasRank() || resultTy.getRank() < 2) {
      return failure();
    }

    auto iotaDimension = iota.getIotaDimension();
    auto iotaDimensionInt = iotaDimension;

    auto convertedShape = rewriter.create<arith::IndexCastOp>(
        iota.getLoc(),
        RankedTensorType::get(
            cast<ShapedType>(iota.getOutputShape().getType()).getShape(),
            rewriter.getI64Type()),
        iota.getOutputShape());

    auto slicedShape = rewriter.create<SliceOp>(
        iota.getLoc(), convertedShape,
        rewriter.getI64TensorAttr(iotaDimensionInt),
        rewriter.getI64TensorAttr(iotaDimensionInt + 1),
        rewriter.getI64TensorAttr(1));

    auto convertedSlicedShape = rewriter.create<arith::IndexCastOp>(
        iota.getLoc(),
        RankedTensorType::get(
            {1},
            cast<ShapedType>(iota.getOutputShape().getType()).getElementType()),
        slicedShape);

    auto iotaType = RankedTensorType::get(
        {resultTy.getDimSize(iotaDimensionInt)}, resultTy.getElementType());

    auto newIota = rewriter.create<DynamicIotaOp>(
        iota.getLoc(), iotaType, convertedSlicedShape,
        rewriter.getI64IntegerAttr(0));

    auto broadcastAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({1}, rewriter.getIntegerType(64)),
        {iotaDimension});
    rewriter.replaceOpWithNewOp<DynamicBroadcastInDimOp>(
        iota, resultTy, newIota, iota.getOutputShape(), broadcastAttr);
    return success();
  }
};

} // namespace

void DynamicIotaOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                MLIRContext *context) {
  results.add<DynamicIotaIsStatic>(context);
  results.add<DynamicIotaBroadcast>(context);
}

LogicalResult DynamicIotaOp::reifyReturnTypeShapes(
    OpBuilder &builder, ValueRange operands,
    SmallVectorImpl<Value> &reifiedReturnShapes) {
  DynamicIotaOp::Adaptor adaptor(operands);
  reifiedReturnShapes.push_back(
      castToIndexTensor(builder, getLoc(), adaptor.getOutputShape()));
  return success();
}

//===----------------------------------------------------------------------===//
// DynamicPadOp
//===----------------------------------------------------------------------===//

namespace {

// If the input tensor has a dimension of length-0, the input tensor is
// irrelevant. Instead we can broadcast the pad value to the output size rather
// than pad the input tensor.
struct DynamicPadEmptyTensor : public OpRewritePattern<DynamicPadOp> {
  using OpRewritePattern<DynamicPadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DynamicPadOp op,
                                PatternRewriter &rewriter) const override {
    // auto loc = op.getLoc();
    auto operand = op.getOperand();
    auto padVal = op.getPaddingValue();

    auto operandTy = cast<RankedTensorType>(operand.getType());

    if (llvm::all_of(operandTy.getShape(), [](int64_t d) { return d != 0; })) {
      return failure();
    }

    llvm::SmallVector<Value> reifiedShapes;
    if (failed(op.reifyReturnTypeShapes(rewriter, op->getOperands(),
                                        reifiedShapes))) {
      return failure();
    }

    auto dimsType = RankedTensorType::get({0}, rewriter.getIntegerType(64));
    auto broadcastDims =
        DenseIntElementsAttr::get(dimsType, SmallVector<int64_t, 1>{});
    rewriter.replaceOpWithNewOp<DynamicBroadcastInDimOp>(
        op, op.getType(), padVal, reifiedShapes.front(), broadcastDims);
    return success();
  }
};

} // namespace

void DynamicPadOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.add<DPadToPad, DynamicPadEmptyTensor>(context);
}

// TODO(chokobole): Do we need this? Dependency: interior_padding
LogicalResult DynamicPadOp::verify() {
  return hlo::verifyDynamicPadOp(getLoc(), getOperand(), getPaddingValue(),
                                 getEdgePaddingLow(), getEdgePaddingHigh(),
                                 getResult());
}

// TODO(chokobole): Do we need this? Dependency: interior_padding
LogicalResult DynamicPadOp::reifyReturnTypeShapes(
    OpBuilder &builder, ValueRange operands,
    SmallVectorImpl<Value> &reifiedReturnShapes) {
  DynamicPadOp::Adaptor adaptor(operands);
  Value operand = adaptor.getOperand();
  Value edgePaddingLow = adaptor.getEdgePaddingLow();
  Value edgePaddingHigh = adaptor.getEdgePaddingHigh();

  auto operandType = dyn_cast<RankedTensorType>(operand.getType());
  // Not support unranked pad a.t.m.
  if (!operandType)
    return failure();

  auto loc = this->getLoc();
  SmallVector<Value, 4> shapeValues;
  shapeValues.reserve(operandType.getRank());
  Type shapeScalarType =
      cast<ShapedType>(edgePaddingLow.getType()).getElementType();

  auto toShapeScalarType = [&](Value v) {
    return maybeCastTo(builder, loc, v, shapeScalarType);
  };

  for (int idx : llvm::seq<int>(0, operandType.getShape().size())) {
    Value valueDim =
        toShapeScalarType(builder.create<tensor::DimOp>(loc, operand, idx));
    Value offset = builder.create<arith::ConstantIndexOp>(loc, idx);
    Value valueLow =
        builder.create<tensor::ExtractOp>(loc, edgePaddingLow, offset);
    Value valueHigh =
        builder.create<tensor::ExtractOp>(loc, edgePaddingHigh, offset);
    // output_size = input_size + padding_low + padding_high
    shapeValues.push_back(builder.create<arith::AddIOp>(
        loc, builder.create<arith::AddIOp>(loc, valueDim, valueLow),
        valueHigh));
  }

  reifiedReturnShapes.push_back(builder.create<tensor::FromElementsOp>(
      loc,
      RankedTensorType::get({static_cast<int64_t>(shapeValues.size())},
                            shapeScalarType),
      shapeValues));

  return success();
}

//===----------------------------------------------------------------------===//
// DynamicReshapeOp
//===----------------------------------------------------------------------===//

LogicalResult DynamicReshapeOp::verify() {
  // Check for unranked dynamism. Unranked dynamism is not supported by
  // StableHLO (hlo::verifyDynamicReshapeOp will fail) and we can't verify
  // anything statically in that case anyway.
  auto operandType = cast<ShapedType>(getOperand().getType());
  auto resultType = cast<ShapedType>(getResult().getType());
  auto outputShapeType = cast<ShapedType>(getOutputShape().getType());
  if (!operandType.hasRank() || !resultType.hasRank() ||
      !outputShapeType.hasStaticShape())
    return success();

  return hlo::verifyDynamicReshapeOp(getLoc(), getOperand(), getOutputShape(),
                                     getResult());
}

LogicalResult DynamicReshapeOp::reifyReturnTypeShapes(
    OpBuilder &builder, ValueRange operands,
    SmallVectorImpl<Value> &reifiedReturnShapes) {
  DynamicReshapeOp::Adaptor adaptor(operands);
  reifiedReturnShapes.push_back(
      castToIndexTensor(builder, getLoc(), adaptor.getOutputShape()));
  return success();
}

namespace {

class DynamicReshapeOpNotActuallyDynamic
    : public OpRewritePattern<DynamicReshapeOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DynamicReshapeOp op,
                                PatternRewriter &rewriter) const override {
    auto type = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!type || !type.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op, "requires static shape tensor");
    }
    rewriter.replaceOpWithNewOp<ReshapeOp>(op, op.getType(), op.getOperand());
    return success();
  }
};

// Canonicalizes
// %0 = some_op(%tensor)
// %1 = "mhlo.dynamic_reshape"(%0, %shape)
//      (tensor<?xT>, tensor<1xindex>) -> tensor<?xT>
// ... uses of %1.
//
// into
//
// ... uses of %0.
// This canonicalization is only correct if the input is correct!
// TODO(b/178779691): Use a more sophisticated canonicalization that preserves
// errors in input, and still allows us to get rid of redundant reshapes.
class RemoveRedundantRank1DynamicReshape
    : public OpRewritePattern<DynamicReshapeOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DynamicReshapeOp op,
                                PatternRewriter &rewriter) const override {
    auto type = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!type || type.getRank() != 1 || type.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op, "requires rank 1 shape tensor with dynamic dimension");
    }
    auto operandType = dyn_cast<RankedTensorType>(op.getOperand().getType());
    if (!operandType || operandType.getRank() != 1 ||
        operandType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op, "requires rank 1 shape tensor with dynamic dimension");
    }
    rewriter.replaceOp(op, {op.getOperand()});
    return success();
  }
};

// Canonicalizes
// %0 = "mhlo.dynamic_reshape"(%tensor, %shape)
// %1 = same_operands_and_result_shape_op(%tensor)
// %2 = "mhlo.dynamic_reshape"(%1, %shape)
// ... uses of %2.
//
// into
//
// %0 = "mhlo.dynamic_reshape"(%tensor, %shape)
// %1 = same_operands_and_result_shape_op(%tensor)
// ... uses of %1.
class DynamicReshapeOpSameShapeOpResult
    : public OpRewritePattern<DynamicReshapeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DynamicReshapeOp op,
                                PatternRewriter &rewriter) const override {
    Operation *defOp = op.getOperand().getDefiningOp();
    if (!defOp ||
        !defOp->hasTrait<mlir::OpTrait::SameOperandsAndResultShape>()) {
      return failure();
    }
    Operation *inputDefOp = defOp->getOperand(0).getDefiningOp();
    if (!inputDefOp) {
      return failure();
    }
    auto reshape = dyn_cast<DynamicReshapeOp>(*inputDefOp);
    if (reshape && reshape.getOutputShape() == op.getOutputShape()) {
      rewriter.replaceOp(op, {defOp->getResult(0)});
      return success();
    }
    return failure();
  }
};

} // namespace

void DynamicReshapeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                   MLIRContext *context) {
  // clang-format off
  results.add<
      DynamicReshapeOpNotActuallyDynamic,
      DynamicReshapeOpSameShapeOpResult,
      RemoveRedundantDynamicBroadcast,
      RemoveRedundantDynamicReshape,
      RemoveRedundantRank1DynamicReshape,
      ShapeOfDynamicReshape
    >(context);
  // clang-format on
}

//===----------------------------------------------------------------------===//
// DynamicSliceOp
//===----------------------------------------------------------------------===//

namespace {

// Given the start indices and slice sizes for a dynamic-slice that can be
// converted to a static slice, returns the limits for the static slice.
DenseIntElementsAttr buildSliceLimits(DenseIntElementsAttr startIndices,
                                      DenseIntElementsAttr sliceSizes,
                                      Builder *builder) {
  SmallVector<int64_t, 4> sliceLimits;
  for (int64_t i = 0; i < sliceSizes.getNumElements(); ++i) {
    int64_t startIndex = startIndices.getValues<IntegerAttr>()[i].getInt();
    int64_t sliceSize = sliceSizes.getValues<IntegerAttr>()[i].getInt();
    sliceLimits.push_back(startIndex + sliceSize);
  }
  return builder->getI64TensorAttr(sliceLimits);
}

// Canonicalizes DynamicSlice ops that can be replaced instead with Slice ops.
// This canonicalization is applied the case when the `begin` input values are
// compile time constants and thus can be made into a tensor.
struct DynamicSliceToSlice : public OpRewritePattern<DynamicSliceOp> {
  using OpRewritePattern<DynamicSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DynamicSliceOp dynamicSlice,
                                PatternRewriter &rewriter) const override {
    Value input = dynamicSlice.getOperand();
    auto inputTensor = dyn_cast<RankedTensorType>(input.getType());
    if (!inputTensor || !inputTensor.hasStaticShape())
      return failure();

    auto sliceSizes = dynamicSlice.getSliceSizes().getValues<int64_t>();
    SmallVector<int64_t, 4> tempStartIndices;
    for (const auto &indexAndSliceStart :
         llvm::enumerate(dynamicSlice.getStartIndices())) {
      APInt val;
      Value start = indexAndSliceStart.value();
      int64_t index = indexAndSliceStart.index();
      if (!matchPattern(start, m_ConstantInt(&val))) {
        return failure();
      }
      // Clamp the indices within bounds to faithfully mirror dynamic slice
      // semantics.
      int64_t clampedStart =
          clamp(val.getSExtValue(), static_cast<int64_t>(0),
                inputTensor.getDimSize(index) - sliceSizes[index]);
      tempStartIndices.push_back(clampedStart);
    }

    // At this point we've determined that the start indices are all constants;
    // pack them into a single tensor.
    auto loc = dynamicSlice.getLoc();
    int64_t inputRank = inputTensor.getRank();
    auto sliceStartIndices = rewriter.getI64TensorAttr(tempStartIndices);
    DenseIntElementsAttr sliceLimits = buildSliceLimits(
        sliceStartIndices, dynamicSlice.getSliceSizes(), &rewriter);
    DenseIntElementsAttr sliceStrides =
        rewriter.getI64TensorAttr(SmallVector<int64_t, 4>(inputRank, 1));
    auto result = rewriter.create<SliceOp>(loc, input, sliceStartIndices,
                                           sliceLimits, sliceStrides);
    rewriter.replaceOp(dynamicSlice, result);
    return success();
  }
};

} // namespace

void DynamicSliceOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                 MLIRContext *context) {
  results.add<DynamicSliceToSlice>(context);
}

LogicalResult DynamicSliceOp::inferReturnTypeComponents(
    MLIRContext *, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  DynamicSliceOp::Adaptor adaptor(operands, attributes, properties, regions);
  if (failed(verify1dTensor(location, adaptor.getSliceSizes(), "slice_sizes")))
    return failure();
  return hlo::inferDynamicSliceOp(
      location, adaptor.getOperand().getType(),
      adaptor.getStartIndices().getTypes(),
      llvm::to_vector(adaptor.getSliceSizes().getValues<int64_t>()),
      inferredReturnShapes);
}

//===----------------------------------------------------------------------===//
// DynamicUpdateSliceOp
//===----------------------------------------------------------------------===//

LogicalResult DynamicUpdateSliceOp::inferReturnTypeComponents(
    MLIRContext *, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  DynamicUpdateSliceOp::Adaptor adaptor(operands, attributes, properties,
                                        regions);
  return hlo::inferDynamicUpdateSliceOp(
      location, adaptor.getOperand(), adaptor.getUpdate(),
      adaptor.getStartIndices(), inferredReturnShapes);
}

OpFoldResult DynamicUpdateSliceOp::fold(FoldAdaptor /*adaptor*/) {
  auto operandShape = cast<RankedTensorType>(this->getOperand().getType());
  auto updateShape = cast<RankedTensorType>(this->getUpdate().getType());

  // If any of the dimensions are length-0, the update does nothing.
  for (auto dim : updateShape.getShape()) {
    if (dim == 0) {
      return this->getOperand();
    }
  }

  if (operandShape != updateShape || !operandShape.hasStaticShape()) {
    return {};
  }

  // Ensure that indices are 0 constants. The 0 check mostly ensures
  // correctness. For non-constants, the pattern does not fold to avoid hiding
  // the behavior of incorrect user input.
  for (Value index : this->getStartIndices()) {
    DenseIntElementsAttr deAttr;
    if (!matchPattern(index, m_Constant(&deAttr)))
      return {};
    if (!deAttr.getSplatValue<IntegerAttr>().getValue().isZero())
      return {};
  }
  return this->getUpdate();
}

//===----------------------------------------------------------------------===//
// GetDimensionSizeOp
//===----------------------------------------------------------------------===//

LogicalResult GetDimensionSizeOp::inferReturnTypeComponents(
    MLIRContext *, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  GetDimensionSizeOp::Adaptor adaptor(operands, attributes, properties,
                                      regions);
  return hlo::inferGetDimensionSizeOp(location, adaptor.getOperand().getType(),
                                      adaptor.getDimension(),
                                      inferredReturnShapes);
}

// Fold get_dimension_size when the said shape dimension is a constant.
OpFoldResult GetDimensionSizeOp::fold(FoldAdaptor) {
  RankedTensorType type = dyn_cast<RankedTensorType>(getOperand().getType());
  if (!type)
    return {};

  int32_t dim = getDimension();
  if (type.isDynamicDim(dim))
    return {};
  // The result type is always is a 0-d i32 tensor.
  return DenseIntElementsAttr::get<int32_t>(
      cast<RankedTensorType>(getResult().getType()), type.getDimSize(dim));
}

//===----------------------------------------------------------------------===//
// GetTupleElementOp
//===----------------------------------------------------------------------===//

OpFoldResult GetTupleElementOp::fold(FoldAdaptor /*adaptor*/) {
  if (auto tupleOp = getOperand().getDefiningOp<TupleOp>()) {
    return tupleOp.getOperand(getIndex());
  }

  return {};
}

LogicalResult GetTupleElementOp::inferReturnTypes(
    MLIRContext *, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  GetTupleElementOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferGetTupleElementOp(location, adaptor.getOperand(),
                                     adaptor.getIndex(), inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// If Op
//===----------------------------------------------------------------------===//

LogicalResult
IfOp::inferReturnTypes(MLIRContext *context, std::optional<Location> location,
                       ValueRange operands, DictionaryAttr attributes,
                       OpaqueProperties properties, RegionRange regions,
                       SmallVectorImpl<Type> &inferredReturnTypes) {
  IfOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferIfOp(location, adaptor.getPred(), adaptor.getRegions(),
                        inferredReturnTypes);
}

static LogicalResult inlineIfConstantCondition(IfOp ifOp,
                                               PatternRewriter &rewriter) {
  DenseIntElementsAttr predAttr;
  if (!matchPattern(ifOp.getPred(), m_Constant(&predAttr)))
    return failure();

  if (predAttr.getSplatValue<BoolAttr>().getValue()) {
    replaceOpWithRegion(rewriter, ifOp, ifOp.getTrueBranch());
  } else {
    replaceOpWithRegion(rewriter, ifOp, ifOp.getFalseBranch());
  }
  return success();
}

void IfOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                       MLIRContext *context) {
  results.add(&inlineIfConstantCondition);
}

//===----------------------------------------------------------------------===//
// IotaOp
//===----------------------------------------------------------------------===//

LogicalResult IotaOp::verify() {
  return hlo::verifyIotaOp(getLoc(), getIotaDimension(), getResult());
}

namespace {

// Iota operations across multiple dimensions can be reduced to an iota and a
// ranked broadcast.
struct IotaBroadcast : public OpRewritePattern<IotaOp> {
  using OpRewritePattern<IotaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IotaOp iota,
                                PatternRewriter &rewriter) const override {
    auto resultTy = cast<ShapedType>(iota.getType());
    if (!resultTy.hasRank() || resultTy.getRank() < 2) {
      return failure();
    }

    auto iotaDimension = iota.getIotaDimension();

    auto iotaType = RankedTensorType::get({resultTy.getDimSize(iotaDimension)},
                                          resultTy.getElementType());

    auto newIota = rewriter.create<IotaOp>(iota.getLoc(), iotaType,
                                           rewriter.getI64IntegerAttr(0));

    auto broadcastAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({1}, rewriter.getIntegerType(64)),
        {iotaDimension});
    rewriter.replaceOpWithNewOp<BroadcastInDimOp>(iota, resultTy, newIota,
                                                  broadcastAttr);
    return success();
  }
};

} // namespace

void IotaOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.add<IotaBroadcast>(context);
}

OpFoldResult IotaOp::fold(FoldAdaptor /*adaptor*/) {
  auto dimension = getIotaDimension();
  auto resultTy = cast<ShapedType>(getResult().getType());
  if (resultTy.hasRank() && resultTy.getDimSize(dimension) == 1) {
    Builder builder(getContext());
    return builder.getZeroAttr(resultTy);
  }

  return {};
}

//===----------------------------------------------------------------------===//
// MapOp
//===----------------------------------------------------------------------===//

LogicalResult MapOp::inferReturnTypeComponents(
    MLIRContext *, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  MapOp::Adaptor adaptor(operands, attributes, properties, regions);
  if (failed(verify1dTensor(location, adaptor.getDimensions(), "dimensions")))
    return failure();
  return hlo::inferMapOp(
      location, adaptor.getInputs(),
      llvm::to_vector(adaptor.getDimensions().getValues<int64_t>()),
      adaptor.getComputation(), inferredReturnShapes);
}

OpFoldResult MapOp::fold(FoldAdaptor) {
  mlir::Block &bb = getComputation().front();
  mlir::Operation &frontOp = bb.front();

  auto retOp = mlir::dyn_cast<ReturnOp>(frontOp);
  if (!retOp)
    return nullptr;
  if (retOp.getResults().size() != 1)
    return nullptr;

  for (mlir::BlockArgument barg : bb.getArguments()) {
    if (barg == retOp.getResults()[0])
      return getOperands()[barg.getArgNumber()];
  }
  return nullptr;
}

LogicalResult
MapOp::reifyReturnTypeShapes(OpBuilder &builder, ValueRange operands,
                             SmallVectorImpl<Value> &reifiedReturnShapes) {
  return hlo::deriveShapeFromOperand(&builder, getOperation(), operands.front(),
                                     &reifiedReturnShapes);
}

//===----------------------------------------------------------------------===//
// PadOp
//===----------------------------------------------------------------------===//

// TODO(chokobole): Do we need this? Dependency: interior_padding
LogicalResult
PadOp::inferReturnTypes(MLIRContext *, std::optional<Location> location,
                        ValueRange operands, DictionaryAttr attributes,
                        OpaqueProperties properties, RegionRange regions,
                        SmallVectorImpl<Type> &inferredReturnTypes) {
  PadOp::Adaptor adaptor(operands, attributes, properties, regions);
  if (failed(verify1dTensor(location, adaptor.getEdgePaddingLow(),
                            "edge_padding_low")) ||
      failed(verify1dTensor(location, adaptor.getEdgePaddingHigh(),
                            "edge_padding_high")))
    return failure();
  return hlo::inferPadOp(
      location, adaptor.getOperand().getType(),
      adaptor.getPaddingValue().getType(),
      llvm::to_vector(adaptor.getEdgePaddingLow().getValues<int64_t>()),
      llvm::to_vector(adaptor.getEdgePaddingHigh().getValues<int64_t>()),
      inferredReturnTypes);
}

namespace {

template <typename T>
OpFoldResult padOpFoldHelper(DenseElementsAttr input, DenseElementsAttr padding,
                             RankedTensorType returnType,
                             DenseIntElementsAttr edgePaddingLow,
                             DenseIntElementsAttr /*edgePaddingHigh*/) {
  // Prevent folding if the result is too large.
  if (returnType.getNumElements() > kFoldOpEltLimit)
    return {};

  // Fill the full result tensor with the padding value.
  llvm::SmallVector<T, 4> result(returnType.getNumElements(),
                                 padding.getValues<T>()[0]);

  auto nextIndex = [](llvm::SmallVector<uint64_t, 8> &index,
                      llvm::ArrayRef<int64_t> shape) {
    for (int64_t i = index.size() - 1; i >= 0; --i) {
      ++index[i];
      if (static_cast<int64_t>(index[i]) < shape[i])
        return;
      index[i] = 0;
    }
  };

  // Iterate over all elements of the input tensor and copy it to the correct
  // location in the output tensor.
  llvm::SmallVector<uint64_t, 8> index(input.getType().getRank(), 0);
  uint64_t numElements = input.getNumElements();
  for (uint64_t operandIdx = 0; operandIdx < numElements; operandIdx++) {
    uint64_t resultIdx = 0;
    uint64_t idxMultiplyer = 1;
    for (int64_t i = index.size() - 1; i >= 0; --i) {
      resultIdx +=
          (edgePaddingLow.getValues<int64_t>()[i] + index[i]) * idxMultiplyer;
      idxMultiplyer *= returnType.getDimSize(i);
    }
    result[resultIdx] = input.getValues<T>()[index];
    nextIndex(index, input.getType().getShape());
  }
  return DenseElementsAttr::get(returnType, result);
}

} // namespace

// TODO(chokobole): Do we need this? Dependency: interior_padding
OpFoldResult PadOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  // If all padding is zero then it is an identity pad.
  auto isZero = [](const APInt &i) { return i == 0; };

  if (llvm::all_of(getEdgePaddingLow().getValues<APInt>(), isZero) &&
      llvm::all_of(getEdgePaddingHigh().getValues<APInt>(), isZero))
    return getOperand();

  // If any padding is negative then it isn't supported by the folder (yet).
  auto isNegative = [](const APInt &i) { return i.slt(0); };
  if (llvm::any_of(getEdgePaddingLow().getValues<APInt>(), isNegative) ||
      llvm::any_of(getEdgePaddingHigh().getValues<APInt>(), isNegative))
    return {};

  DenseElementsAttr input = dyn_cast_or_null<DenseElementsAttr>(operands[0]);
  DenseElementsAttr padding = dyn_cast_or_null<DenseElementsAttr>(operands[1]);
  RankedTensorType returnType = dyn_cast_or_null<RankedTensorType>(getType());
  if (!input || !input.getType().hasRank() || !padding || !returnType ||
      !returnType.hasStaticShape())
    return {};

  Type returnElemType = returnType.getElementType();
  if (isa<IntegerType>(returnElemType) ||
      isa<prime_ir::field::PrimeFieldType>(returnElemType))
    returnType = cast<RankedTensorType>(
        prime_ir::maybeConvertPrimeIRToBuiltinType(returnType));
  return padOpFoldHelper<APInt>(input, padding, returnType, getEdgePaddingLow(),
                                getEdgePaddingHigh());
  return {};
}

// TODO(chokobole): Do we need this? Dependency: interior_padding
LogicalResult
PadOp::reifyReturnTypeShapes(OpBuilder &builder, ValueRange operands,
                             SmallVectorImpl<Value> &reifiedReturnShapes) {
  PadOp::Adaptor adaptor(operands, this->getOperation()->getAttrDictionary(),
                         this->getOperation()->getPropertiesStorage());
  auto loc = this->getLoc();
  Value operand = adaptor.getOperand();
  auto operandTy = cast<RankedTensorType>(operand.getType());

  llvm::SmallVector<int32_t> padHigh;
  llvm::SmallVector<int32_t> padLow;
  llvm::SmallVector<int32_t> padInterior;

  auto padHighAttr = adaptor.getEdgePaddingHigh();
  auto padLowAttr = adaptor.getEdgePaddingLow();

  padHigh.reserve(padHighAttr.getNumElements());
  padLow.reserve(padLowAttr.getNumElements());

  for (const APInt &val : padHighAttr.getValues<APInt>())
    padHigh.push_back(val.getSExtValue());

  for (const APInt &val : padLowAttr.getValues<APInt>())
    padLow.push_back(val.getSExtValue());

  llvm::SmallVector<Value> dimensions;
  dimensions.reserve(operandTy.getRank());
  for (int i = 0, s = operandTy.getRank(); i < s; ++i) {
    Value padEdge =
        builder.create<arith::ConstantIndexOp>(loc, padHigh[i] + padLow[i]);

    // First we grab the initial interior size.
    Value dim = builder.create<tensor::DimOp>(loc, operand, i).getResult();

    // Then we add the padding on the edge of the tensor.
    dim = builder.create<arith::AddIOp>(loc, dim, padEdge).getResult();
    dimensions.push_back(dim);
  }

  Value dimensionTensor =
      builder.create<tensor::FromElementsOp>(loc, dimensions).getResult();
  reifiedReturnShapes.push_back(dimensionTensor);
  return success();
}

namespace {

// If the input tensor has a dimension of length-0, the input tensor is
// irrelevant. Instead we can broadcast the pad value to the output size rather
// than pad the input tensor.
struct PadEmptyTensor : public OpRewritePattern<PadOp> {
  using OpRewritePattern<PadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PadOp op,
                                PatternRewriter &rewriter) const override {
    auto operand = op.getOperand();
    auto padVal = op.getPaddingValue();

    auto operandTy = cast<RankedTensorType>(operand.getType());
    auto resultTy = cast<RankedTensorType>(op.getType());

    if (llvm::all_of(operandTy.getShape(), [](int64_t d) { return d != 0; })) {
      return failure();
    }

    if (resultTy.hasStaticShape()) {
      auto dimsType = RankedTensorType::get({0}, rewriter.getIntegerType(64));
      auto dims =
          DenseIntElementsAttr::get(dimsType, SmallVector<int64_t, 1>{});
      rewriter.replaceOpWithNewOp<BroadcastInDimOp>(op, resultTy, padVal, dims);
      return success();
    }

    llvm::SmallVector<Value> reifiedShapes;
    if (failed(op.reifyReturnTypeShapes(rewriter, op.getOperands(),
                                        reifiedShapes))) {
      return failure();
    }

    auto dimsType = RankedTensorType::get({0}, rewriter.getIntegerType(64));
    auto broadcastDims =
        DenseIntElementsAttr::get(dimsType, SmallVector<int64_t, 1>{});
    rewriter.replaceOpWithNewOp<DynamicBroadcastInDimOp>(
        op, op.getType(), padVal, reifiedShapes.front(), broadcastDims);
    return success();
  }
};

} // namespace

void PadOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<PadEmptyTensor>(context);
}

//===----------------------------------------------------------------------===//
// RealDynamicSliceOp
//===----------------------------------------------------------------------===//
// Verifies that operand rank matches start_indices/limit_indices/strides size
LogicalResult RealDynamicSliceOp::verify() {
  return hlo::verifyRealDynamicSliceOp(getLoc(), getOperand(),
                                       getStartIndices(), getLimitIndices(),
                                       getStrides());
}

namespace {

struct RealDSliceToDSlice : public OpRewritePattern<RealDynamicSliceOp> {
  using OpRewritePattern<RealDynamicSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(RealDynamicSliceOp op,
                                PatternRewriter &rewriter) const override {
    // This rewrite only works for unit strides because DynamicSliceOp
    // doesn't support strides (i.e. it implicitly has unit strides).
    DenseIntElementsAttr stridesAttr;
    if (!matchPattern(op.getStrides(), m_Constant(&stridesAttr)))
      return rewriter.notifyMatchFailure(op, "requires constant strides");
    if (!llvm::all_of(stridesAttr.getValues<APInt>(),
                      [&](APInt stride) { return stride == 1; }))
      return rewriter.notifyMatchFailure(op, "requires unit strides");

    // Check that slice sizes are fully static (DynamicSliceOp style).
    // To detect that, we check whether `limit_indices` is defined as
    // `start_indices + constant` or `constant + start_indices`.
    DenseIntElementsAttr sliceSizesAttr;
    auto m_startIndices = matchers::m_Val(op.getStartIndices());
    if (!matchPattern(
            op.getLimitIndices(),
            m_Op<AddOp>(m_startIndices, m_Constant(&sliceSizesAttr))) &&
        !matchPattern(op.getLimitIndices(),
                      m_Op<AddOp>(m_Constant(&sliceSizesAttr), m_startIndices)))
      return rewriter.notifyMatchFailure(
          op, "requires limit indices equal to start indices plus constant");

    // RealDynamicSliceOp can take tensors of integer or index element types.
    // DynamicSliceOp::slice_sizes only supports i64 element type.
    // Adapt accordingly in order to be compatible with DynamicSliceOp.
    SmallVector<int64_t> sliceSizes;
    for (auto element : sliceSizesAttr.getValues<APInt>()) {
      sliceSizes.push_back(element.getSExtValue());
    }

    // RealDynamicSliceOp::start_indices is a 1-dimensional tensor.
    // DynamicSliceOp::start_indices is a vararg of 0-dimensional tensors.
    // Adapt accordingly in order to be compatible with DynamicSliceOp.
    SmallVector<Value> startIndices;
    for (auto i = 0; i < static_cast<int64_t>(sliceSizes.size()); ++i) {
      auto startIndex1D = rewriter.create<SliceOp>(
          op.getLoc(), op.getStartIndices(), rewriter.getI64TensorAttr(i),
          rewriter.getI64TensorAttr(i + 1), rewriter.getI64TensorAttr(1));
      auto startIndex0DType = RankedTensorType::get(
          {},
          cast<ShapedType>(op.getStartIndices().getType()).getElementType());
      auto startIndex0D = rewriter.create<ReshapeOp>(
          op.getLoc(), startIndex0DType, startIndex1D);
      startIndices.push_back(startIndex0D);
    }

    rewriter.replaceOpWithNewOp<DynamicSliceOp>(
        op, op.getOperand(), startIndices,
        rewriter.getI64TensorAttr(sliceSizes));
    return success();
  }
};

} // namespace

void RealDynamicSliceOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                     MLIRContext *context) {
  results.add<RealDSliceToSlice, RealDSliceToDSlice>(context);
}

LogicalResult RealDynamicSliceOp::reifyReturnTypeShapes(
    OpBuilder &builder, ValueRange operands,
    SmallVectorImpl<Value> &reifiedReturnShapes) {
  RealDynamicSliceOp::Adaptor adaptor(operands);
  Value operand = adaptor.getOperand();
  Value startIndices = adaptor.getStartIndices();
  Value limitIndices = adaptor.getLimitIndices();
  Value strides = adaptor.getStrides();

  auto operandType = dyn_cast<RankedTensorType>(operand.getType());
  // Not support unranked type a.t.m.
  if (!operandType)
    return failure();

  Location loc = this->getLoc();
  SmallVector<Value, 4> shapeValues;
  shapeValues.reserve(operandType.getRank());
  Type shapeScalarType =
      cast<ShapedType>(startIndices.getType()).getElementType();
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  one = maybeCastTo(builder, loc, one, shapeScalarType);
  for (const auto &element : llvm::enumerate(operandType.getShape())) {
    Value offset = builder.create<arith::ConstantIndexOp>(loc, element.index());
    Value valueStart =
        builder.create<tensor::ExtractOp>(loc, startIndices, offset);
    Value valueLimit =
        builder.create<tensor::ExtractOp>(loc, limitIndices, offset);
    Value valueStride = builder.create<tensor::ExtractOp>(loc, strides, offset);
    // size = (limit - start + stride - 1) / stride
    shapeValues.push_back(builder.create<arith::DivSIOp>(
        loc,
        builder.create<arith::SubIOp>(
            loc,
            builder.create<arith::AddIOp>(
                loc, valueStride,
                builder.create<arith::SubIOp>(loc, valueLimit, valueStart)),
            one),
        valueStride));
  }

  reifiedReturnShapes.push_back(builder.create<tensor::FromElementsOp>(
      loc,
      RankedTensorType::get({static_cast<int64_t>(shapeValues.size())},
                            shapeScalarType),
      shapeValues));
  return success();
}

//===----------------------------------------------------------------------===//
// ReduceOp
//===----------------------------------------------------------------------===//

namespace {

LogicalResult tryFoldZeroDimReduction(ReduceOp reduceOp,
                                      SmallVectorImpl<OpFoldResult> &results) {
  if (reduceOp.getDimensions().getNumElements() != 0)
    return failure();
  // No dimensions to reduce.
  for (auto [operand, opResult] :
       llvm::zip_equal(reduceOp.getInputs(), reduceOp.getResults())) {
    if (operand.getType() != opResult.getType()) {
      results.clear();
      return failure();
    }
    results.push_back(operand);
  }
  return success();
}

LogicalResult
tryFoldOutsideValuesReduction(ReduceOp reduceOp,
                              SmallVectorImpl<OpFoldResult> &results) {
  // If all returned values in the ReduceOp region exists outside
  // the region replace the ReduceOp with those values.
  mlir::Block &bb = reduceOp.getBody().front();
  auto retOp = mlir::dyn_cast<ReturnOp>(bb.back());
  if (!retOp)
    return failure();
  for (auto [result, opResult] :
       llvm::zip_equal(retOp.getResults(), reduceOp.getResults())) {
    if (result.getParentRegion() == retOp->getParentRegion() ||
        result.getType() != opResult.getType()) {
      results.clear();
      return failure();
    }
    results.push_back(result);
  }
  return success();
}

} // namespace

LogicalResult ReduceOp::fold(FoldAdaptor /*adaptor*/,
                             SmallVectorImpl<OpFoldResult> &results) {
  if (succeeded(tryFoldZeroDimReduction(*this, results)))
    return success();
  if (succeeded(tryFoldOutsideValuesReduction(*this, results)))
    return success();
  return failure();
}

void ReduceOp::print(OpAsmPrinter &p) {
  auto dimensions = llvm::to_vector(getDimensions().getValues<int64_t>());
  hlo::printReduceOp(p, getOperation(), getInputs(), dimensions, getBody());
}

ParseResult ReduceOp::parse(OpAsmParser &parser, OperationState &result) {
  auto parseDenseElements = [](OpBuilder &b,
                               ArrayRef<int64_t> dims) -> Attribute {
    return b.getI64TensorAttr(dims);
  };
  return hlo::parseReduceOp(parser, result, parseDenseElements);
}

LogicalResult ReduceOp::inferReturnTypeComponents(
    MLIRContext *, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ReduceOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferReduceOp(
      location, adaptor.getInputs().getTypes(),
      llvm::to_vector(adaptor.getDimensions().getValues<int64_t>()),
      adaptor.getBody(), inferredReturnShapes);
}

void ReduceOp::build(OpBuilder &, OperationState &odsState, ValueRange inputs,
                     ValueRange initValues, DenseIntElementsAttr dimensions,
                     TypeRange elementTypes) {
  odsState.addOperands(inputs);
  odsState.addOperands(initValues);
  Properties &properties = odsState.getOrAddProperties<Properties>();
  properties.dimensions = dimensions;
  (void)odsState.addRegion();

  SmallVector<int64_t> newDimensions;
  Attribute encoding;
  ReduceOp::Adaptor adaptor(
      odsState.operands,
      odsState.attributes.getDictionary(odsState.getContext()), {},
      odsState.regions);

  SmallVector<ShapedType> inputArgTensorTypes{
      llvm::map_range(adaptor.getInputs().getTypes(),
                      [](Type t) { return cast<ShapedType>(t); })};
  SmallVector<ShapedType> initValueTensorTypes{
      llvm::map_range(adaptor.getInitValues().getTypes(),
                      [](Type t) { return cast<ShapedType>(t); })};

  if (succeeded(hlo::verifyReduceOpInputsAndInferShape(
          odsState.location, inputArgTensorTypes,
          llvm::to_vector(dimensions.getValues<int64_t>()), newDimensions,
          encoding))) {
    SmallVector<Type> inferredReturnTypes;
    for (uint64_t inputIdx = 0; inputIdx < inputArgTensorTypes.size();
         ++inputIdx) {
      Type elementTy = elementTypes[inputIdx];
      ShapedType inputType = inputArgTensorTypes[inputIdx];
      if (inputType.hasRank()) {
        inferredReturnTypes.push_back(
            RankedTensorType::get(newDimensions, elementTy, encoding));
      } else {
        assert(encoding == nullptr && "attribute not supported");
        inferredReturnTypes.push_back(UnrankedTensorType::get(elementTy));
      }
    }
    odsState.addTypes(inferredReturnTypes);
  } else {
    llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

LogicalResult ReduceOp::verify() {
  if (failed(verify1dTensor(getLoc(), getDimensions(), "dimensions")))
    return failure();
  return hlo::verifyReduceOp(
      getLoc(), getInputs(), getInitValues(),
      llvm::to_vector(getDimensions().getValues<int64_t>()), getBody());
}

//===----------------------------------------------------------------------===//
// ReduceWindowOp
//===----------------------------------------------------------------------===//

LogicalResult ReduceWindowOp::inferReturnTypeComponents(
    MLIRContext *, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ReduceWindowOp::Adaptor adaptor(operands, attributes, properties, regions);
  auto optToVec = [](std::optional<DenseIntElementsAttr> attr)
      -> std::optional<SmallVector<int64_t>> {
    if (!attr)
      return std::nullopt;
    return llvm::to_vector(attr->getValues<int64_t>());
  };
  return hlo::inferReduceWindowOp(
      location, adaptor.getInputs(), adaptor.getInitValues(),
      llvm::to_vector(adaptor.getWindowDimensions().getValues<int64_t>()),
      optToVec(adaptor.getWindowStrides()),
      optToVec(adaptor.getBaseDilations()),
      optToVec(adaptor.getWindowDilations()), adaptor.getPadding(),
      adaptor.getBody(), inferredReturnShapes);
}

LogicalResult ReduceWindowOp::verify() {
  auto optToVec = [](std::optional<DenseIntElementsAttr> attr)
      -> std::optional<SmallVector<int64_t>> {
    if (!attr)
      return std::nullopt;
    return llvm::to_vector(attr->getValues<int64_t>());
  };
  return hlo::verifyReduceWindowOp(
      getLoc(), getInputs(), getInitValues(),
      llvm::to_vector(getWindowDimensions().getValues<int64_t>()),
      optToVec(getWindowStrides()), optToVec(getBaseDilations()),
      optToVec(getWindowDilations()), getPadding(), getBody());
}

namespace {

// Enable constant folding to occur within the region of the ReduceOp
// by replacing block argument uses with constants if:
//  1. All the ReduceOp operands are splat constants.
//  2. The ReduceOp region consists of a single logical AND or logical OR.
// The pattern leverages the idempotent property of the AND and OR operators
// to determine the value of a reduction on splat constants. Other boolean
// operators do not have this property, and need separate patterns to resolve
// reductions of their splat constants.
struct LowerBoolSplatConstantsIntoRegion : public OpRewritePattern<ReduceOp> {
  using OpRewritePattern<ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReduceOp op,
                                PatternRewriter &rewriter) const override {
    mlir::Block &bb = op.getBody().front();

    // Ensure only a compute op and return op exist and the
    // compute op is an AND or OR op.
    if (bb.getOperations().size() != 2)
      return failure();
    if (!mlir::isa<AndOp, OrOp>(bb.front()))
      return failure();

    // Ensure all operands are splat constants.
    SmallVector<DenseElementsAttr, 4> bargCstAttrs;
    for (auto inpAndBarg : llvm::zip(op.getOperands(), bb.getArguments())) {
      Value inp = std::get<0>(inpAndBarg);
      BlockArgument barg = std::get<1>(inpAndBarg);
      ConstantOp cst = inp.getDefiningOp<ConstantOp>();
      if (!cst)
        return failure();

      auto cstAttr = dyn_cast_or_null<DenseElementsAttr>(cst.getValue());
      if (!cstAttr.isSplat()) {
        return rewriter.notifyMatchFailure(op, "Must be splat constant.");
      }

      auto bargShapedType = dyn_cast<ShapedType>(barg.getType());
      if (!bargShapedType)
        return failure();

      auto bargCstAttr = DenseElementsAttr::get(
          bargShapedType, cstAttr.getSplatValue<mlir::Attribute>());
      bargCstAttrs.push_back(bargCstAttr);
    }

    // Create new splat constants to replace block arguments.
    for (BlockArgument barg : bb.getArguments()) {
      int argIdx = barg.getArgNumber();
      ConstantOp newCst = rewriter.create<mhlo::ConstantOp>(
          bb.front().getLoc(), barg.getType(), bargCstAttrs[argIdx]);
      barg.replaceAllUsesWith(newCst);
    }
    return success();
  }
};

LogicalResult convertEmptyReduces(ReduceOp op, PatternRewriter &rewriter) {
  // We require all reduce shapes to be the same, up to the element types, so we
  // can just the first operand and the first result as a representative.
  RankedTensorType t =
      dyn_cast<RankedTensorType>(op.getInputs().getType().front());
  if (!t)
    return rewriter.notifyMatchFailure(op.getLoc(),
                                       "unranked input unsupported");
  bool zeroExtent = any_of(t.getShape(), [](int64_t d) { return d == 0; });
  if (zeroExtent) {
    auto empty = rewriter.getI64TensorAttr({});
    if (t.hasStaticShape()) {
      for (auto [init, out] : llvm::zip(op.getInitValues(), op.getResults())) {
        out.replaceAllUsesWith(rewriter.create<BroadcastInDimOp>(
            op.getLoc(), out.getType(), init, empty));
      }
      return success();
    }

    SmallVector<Value, 4> shapes;
    if (failed(op.reifyReturnTypeShapes(rewriter, op.getOperands(), shapes)))
      return failure();
    for (auto [init, shape, out] :
         llvm::zip(op.getInitValues(), shapes, op.getResults())) {
      out.replaceAllUsesWith(rewriter.create<DynamicBroadcastInDimOp>(
          op.getLoc(), out.getType(), init, shape, empty));
    }
    return success();
  }
  return rewriter.notifyMatchFailure(op.getLoc(), "non-empty input");
}

} // namespace

void ReduceOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.add<LowerBoolSplatConstantsIntoRegion>(context);
  results.add(convertEmptyReduces);
}

LogicalResult
ReduceOp::reifyReturnTypeShapes(OpBuilder &builder, ValueRange operands,
                                SmallVectorImpl<Value> &reifiedReturnShapes) {
  ReduceOp::Adaptor adaptor(operands);
  auto inputs = adaptor.getInputs();

  auto operandType = dyn_cast<RankedTensorType>(inputs[0].getType());
  // Not support unranked type a.t.m.
  if (!operandType)
    return failure();

  Location loc = this->getLoc();
  SmallVector<Value, 4> shapeValues;
  SmallVector<int64_t, 4> dimensions(
      this->getDimensions().getValues<int64_t>());
  shapeValues.reserve(operandType.getRank());
  Type shapeScalarType = builder.getIndexType();
  auto toShapeScalarType = [&](Value v) {
    return maybeCastTo(builder, loc, v, shapeScalarType);
  };

  for (const auto &element : llvm::enumerate(operandType.getShape())) {
    int64_t idx = element.index();
    auto *it = std::find(dimensions.begin(), dimensions.end(), idx);
    if (it != dimensions.end()) {
      continue;
    }
    Value valueDim = toShapeScalarType(
        builder.create<tensor::DimOp>(loc, inputs[0], element.index()));
    shapeValues.push_back(valueDim);
  }

  Value outputShape = builder.create<tensor::FromElementsOp>(
      loc,
      RankedTensorType::get({static_cast<int64_t>(shapeValues.size())},
                            shapeScalarType),
      shapeValues);
  for (size_t i = 0; i < inputs.size(); ++i) {
    reifiedReturnShapes.push_back(outputShape);
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

LogicalResult ReshapeOp::verify() {
  // Check for unranked dynamism. Unranked dynamism is not supported by
  // StableHLO (hlo::verifyReshapeOp will fail) and we can't verify
  // anything statically in that case anyway.
  auto operandType = cast<ShapedType>(getOperand().getType());
  auto resultType = cast<ShapedType>(getResult().getType());
  if (!operandType.hasRank() || !resultType.hasRank()) {
    return success();
  }
  return hlo::verifyReshapeOp(getLoc(), getOperand(), getResult());
}

OpFoldResult ReshapeOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  if (getOperand().getType() == getType()) {
    return getOperand();
  }

  if (auto prevOp = getOperand().getDefiningOp<ReshapeOp>()) {
    setOperand(prevOp.getOperand());
    return getResult();
  }

  if (auto elements = dyn_cast_or_null<DenseElementsAttr>(operands.front())) {
    ShapedType resultType =
        prime_ir::maybeConvertPrimeIRToBuiltinType(getResult().getType());
    return reshape(elements, resultType);
  }

  return {};
}

void ReshapeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add<IdentityBroadcastReshape, IdentityBroadcastInDimReshape,
              EliminateRedundantReshape, EliminateIdentityReshape>(context);
}

//===----------------------------------------------------------------------===//
// BitReverseOp
//===----------------------------------------------------------------------===//

LogicalResult BitReverseOp::verify() {
  if (failed(verify1dTensor(getLoc(), getDimensions(), "dimensions")))
    return failure();
  return hlo::verifyBitReverseOp(
      getLoc(), getOperand(),
      llvm::to_vector(getDimensions().getValues<int64_t>()));
}

OpFoldResult BitReverseOp::fold(FoldAdaptor adaptor) {
  // No dimensions to bit-reverse.
  DenseIntElementsAttr dims = getDimensions();
  if (dims.getNumElements() == 0)
    return getOperand();

  // Involution: bit_reverse(bit_reverse(x, dims), dims) → x
  auto input = getOperand().getDefiningOp<BitReverseOp>();
  if (input && input.getDimensions() == dims)
    return input.getOperand();

  return {};
}

//===----------------------------------------------------------------------===//
// ReverseOp
//===----------------------------------------------------------------------===//

LogicalResult ReverseOp::verify() {
  if (failed(verify1dTensor(getLoc(), getDimensions(), "dimensions")))
    return failure();
  return hlo::verifyReverseOp(
      getLoc(), getOperand(),
      llvm::to_vector(getDimensions().getValues<int64_t>()));
}

template <typename T>
static Attribute foldReverseHelper(DenseElementsAttr &attr, ShapedType &type,
                                   DenseIntElementsAttr &dims) {
  int64_t numElements = attr.getNumElements();
  // No-op if the tensor has 0 elements.
  // No-op if the result of folding is too large.
  if (numElements == 0 || numElements > kFoldOpEltLimit)
    return {};

  SmallVector<T> result(attr.getValues<T>().begin(), attr.getValues<T>().end());

  size_t rank = type.getRank();
  SmallVector<int64_t> stride(rank + 1, numElements);
  for (size_t i = 0; i < rank; i++) {
    if (type.getDimSize(i) == 0)
      return {};
    stride[i + 1] = stride[i] / type.getDimSize(i);
  }

  for (auto dim : dims.getValues<int64_t>()) {
    // For example, given:
    //   * tensor: tensor<2x3x2xi32>
    //     [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9,10], [11, 12]]]
    //   * dim: [1]
    //
    // We're going to reverse the tensor with respect to dim as follows:
    //   1) Split the tensor into blocks, i.e. smaller tensors whose type is
    //   derived from the tensor by dropping the first `dim` dimensions, i.e.
    //   tensor<3x2xi32> for the running example.
    //   2) Split each block into windows, i.e. even smaller tensors whose type
    //   is derived from the block by dropping the first dimension of the
    //   block, i.e. tensor<2xi32> for the running example.
    //   3) Within each block, swap windows but don't change the order of
    //   elements within the windows: 0th window goes to N-1st spot, 1st window
    //   goes to N-2nd spot etc.
    //
    // For the running example, the result will be:
    //   [[[5, 6], [3, 4], [1, 2]], [[11, 12], [9, 10], [7, 8]]].
    //
    // Note how elements within windows haven't changed their order with respect
    // to each other and how blocks haven't changed their order with respect to
    // each other.
    int64_t numWindows = type.getDimSize(dim);
    int64_t windowSize = stride[dim] / numWindows;

    for (int64_t index = 0; index < numElements; index++) {
      int64_t blockNumber = index / stride[dim];
      int64_t windowNumber = (index % stride[dim]) / windowSize;
      int64_t reversedWindowNumber = numWindows - windowNumber - 1;
      if (windowNumber >= reversedWindowNumber)
        continue;
      int64_t reversedIndex = blockNumber * stride[dim] +
                              reversedWindowNumber * windowSize +
                              index % windowSize;
      std::swap(result[index], result[reversedIndex]);
    }
  }
  return DenseElementsAttr::get(type, result);
}

OpFoldResult ReverseOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  Value input = getOperand();

  // No dimensions to reverse.
  DenseIntElementsAttr dims = getDimensions();
  if (dims.getNumElements() == 0)
    return input;

  // If size of all dimensions to reverse equals 1, then the reverse is a no-op.
  // Eg. Reverse dimensions {0,1} of a 1x1x2 tensor
  auto shapedType = cast<ShapedType>(input.getType());
  if (llvm::all_of(dims.getValues<int64_t>(), [&](int64_t dim) {
        return shapedType.getDimSize(dim) == 1;
      }))
    return input;

  // If the operand is a static shaped tensor of constants, return reversed
  // tensor
  DenseElementsAttr inputAttr =
      mlir::dyn_cast_or_null<DenseElementsAttr>(*operands.begin());
  if (inputAttr && shapedType.hasStaticShape()) {
    auto etype = shapedType.getElementType();
    if (isa<IntegerType>(etype) ||
        isa<prime_ir::field::PrimeFieldType>(etype)) {
      shapedType = prime_ir::maybeConvertPrimeIRToBuiltinType(shapedType);
      return foldReverseHelper<APInt>(inputAttr, shapedType, dims);
    }
  }

  return {};
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

OpFoldResult SelectOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  if (getOnTrue() == getOnFalse()) {
    return getOnTrue();
  }

  auto predicate = dyn_cast_or_null<DenseIntElementsAttr>(operands[0]);
  if (!predicate) {
    return {};
  }

  auto predicateTy = cast<ShapedType>(predicate.getType());
  if (!predicateTy.getElementType().isInteger(1)) {
    return {};
  }

  if (predicate.isSplat()) {
    return predicate.getSplatValue<APInt>().getBoolValue() ? getOnTrue()
                                                           : getOnFalse();
  }

  return {};
}

void SelectOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.add<FusePredNegIntoSelect, FuseBroadcastedPredNegIntoSelect>(context);
}

// Makes it such that a SelectOp that is a non-root operation in a DRR infers
// the return type based on operand type.
LogicalResult SelectOp::inferReturnTypeComponents(
    MLIRContext *, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  SelectOp::Adaptor op(operands, attributes, properties, regions);
  return hlo::inferSelectOp(location, op.getPred(), op.getOnTrue(),
                            op.getOnFalse(), inferredReturnShapes);
}

LogicalResult
SelectOp::reifyReturnTypeShapes(OpBuilder &builder, ValueRange operands,
                                SmallVectorImpl<Value> &reifiedReturnShapes) {
  // For `hlo.select`, the first operand may be a scalar.
  return hlo::deriveShapeFromOperand(&builder, getOperation(), operands[1],
                                     &reifiedReturnShapes);
}

//===----------------------------------------------------------------------===//
// SetDimensionSizeOp
//===----------------------------------------------------------------------===//

OpFoldResult SetDimensionSizeOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();

  // Even if all operands are constants, we can't fold SetDimensionSize to a
  // constant, since mhlo.constant doesn't support dynamic dimensions. We can,
  // however, replace the op with its operand, in the case where the (constant)
  // bound of a dimension is the same as the full extent of said dimension.
  DenseElementsAttr size = dyn_cast_or_null<DenseElementsAttr>(operands[1]);
  if (!size || !size.isSplat())
    return {};

  // TODO(b/377537099): This is the result type, which is always dynamic in the
  // dimension we're looking at. So the code below doesn't do anything.
  auto ty = dyn_cast<RankedTensorType>(getType());
  if (!ty)
    return {};

  int64_t dimSize = ty.getDimSize(getDimension());
  if (dimSize == size.getSplatValue<IntegerAttr>().getInt())
    return getOperand();
  return {};
}

LogicalResult SetDimensionSizeOp::inferReturnTypeComponents(
    MLIRContext *context, std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  SetDimensionSizeOp::Adaptor adaptor(operands, attributes, properties,
                                      regions);
  return hlo::inferSetDimensionSizeOp(
      getMhloDialect(context), location, adaptor.getOperand().getType(),
      adaptor.getSize(), adaptor.getDimension(), inferredReturnShapes);
}

//===----------------------------------------------------------------------===//
// SliceOp
//===----------------------------------------------------------------------===//

LogicalResult SliceOp::inferReturnTypes(
    MLIRContext * /*context*/, std::optional<Location> location,
    ValueRange operands, DictionaryAttr attributes, OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  SliceOpAdaptor adaptor(operands, attributes, properties, regions);
  if (failed(verify1dTensor(location, adaptor.getStartIndices(),
                            "start_indices")) ||
      failed(verify1dTensor(location, adaptor.getLimitIndices(),
                            "limit_indices")) ||
      failed(verify1dTensor(location, adaptor.getStrides(), "strides")))
    return failure();
  return hlo::inferSliceOp(
      location, adaptor.getOperand().getType(),
      llvm::to_vector(adaptor.getStartIndices().getValues<int64_t>()),
      llvm::to_vector(adaptor.getLimitIndices().getValues<int64_t>()),
      llvm::to_vector(adaptor.getStrides().getValues<int64_t>()),
      inferredReturnTypes);
}

namespace {

template <typename I, typename E>
void sliceElements(I values, ArrayRef<int64_t> sizes, ArrayRef<int64_t> starts,
                   ArrayRef<int64_t> limits, ArrayRef<int64_t> strides,
                   llvm::SmallVectorImpl<E> *outValues) {
  assert(starts.size() == limits.size());
  assert(starts.size() == strides.size());
  if (starts.empty())
    return;

  int64_t start = starts.front();
  int64_t limit = limits.front();
  int64_t stride = strides.front();
  if (starts.size() == 1) {
    for (int i = start; i < limit; i += stride) {
      outValues->push_back(*(values + i));
    }
    return;
  }

  for (; start < limit; start += stride) {
    auto begin = values + start * sizes.front();
    sliceElements<I, E>(begin, sizes.drop_front(), starts.drop_front(),
                        limits.drop_front(), strides.drop_front(), outValues);
  }
}

template <typename I, typename E>
Attribute foldSlice(SliceOp *op, I values) {
  auto start = llvm::to_vector<6>(op->getStartIndices().getValues<int64_t>());
  auto limit = llvm::to_vector<6>(op->getLimitIndices().getValues<int64_t>());
  auto stride = llvm::to_vector<6>(op->getStrides().getValues<int64_t>());

  // TODO(b/235903849): This should be op->getType().case<ShapedType>().
  auto resultType = cast<ShapedType>(op->getOperand().getType());
  if (!resultType.hasStaticShape())
    return {};

  ArrayRef<int64_t> shape = resultType.getShape();
  int64_t count = resultType.getNumElements();
  if (count == 0) {
    return DenseElementsAttr::get<E>(
        cast<ShapedType>(op->getResult().getType()),
        /*list=*/{});
  }

  // Compute the striding for each dimension.
  llvm::SmallVector<int64_t, 6> sizes;
  sizes.reserve(shape.size());
  for (auto v : shape) {
    count = count / v;
    sizes.push_back(count);
  }

  // Prevent folding if the result is too large.
  if (resultType.getNumElements() > kFoldOpEltLimit)
    return {};

  llvm::SmallVector<E, 6> outValues;
  outValues.reserve(resultType.getNumElements());
  sliceElements<I, E>(values, sizes, start, limit, stride, &outValues);

  ShapedType shapedType =
      prime_ir::maybeConvertPrimeIRToBuiltinType(op->getResult().getType());
  return DenseElementsAttr::get(shapedType, outValues);
}

} // namespace

OpFoldResult SliceOp::fold(FoldAdaptor adaptor) {
  ArrayRef<Attribute> operands = adaptor.getOperands();
  // Check if the SliceOp is a NoOp operation.
  auto operandType = cast<ShapedType>(getOperand().getType());
  auto resultType = cast<ShapedType>(getResult().getType());

  if (operandType.hasStaticShape() && resultType.hasStaticShape() &&
      (operandType.getShape() == resultType.getShape())) {
    return getOperand();
  }

  if (operands.empty() || !operands.front())
    return {};

  // Evaluate for statically valued inputs.
  auto elements = dyn_cast<DenseElementsAttr>(operands.front());
  if (!elements)
    return {};

  auto etype = elements.getType().getElementType();
  if (isa<IntegerType>(etype)) {
    return foldSlice<DenseElementsAttr::IntElementIterator, APInt>(
        this, elements.value_begin<APInt>());
  }

  return {};
}

//===----------------------------------------------------------------------===//
// SortOp
//===----------------------------------------------------------------------===//

void SortOp::build(OpBuilder &builder, OperationState &state,
                   ValueRange operands, int64_t dimension, bool isStable) {
  state.addOperands(operands);
  Properties &properties = state.getOrAddProperties<Properties>();
  properties.dimension = builder.getI64IntegerAttr(dimension);
  properties.is_stable = builder.getBoolAttr(isStable);

  for (Value operand : operands)
    state.addTypes(operand.getType());

  state.addRegion();
}

LogicalResult SortOp::inferReturnTypeComponents(
    MLIRContext *, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  SortOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferSortOp(location, adaptor.getInputs(), inferredReturnShapes);
}

LogicalResult SortOp::verify() {
  return hlo::verifySortOp(getLoc(), getInputs(), getDimension(),
                           getComparator());
}

namespace {

// Drops the operands if the results are not used and they are not used in
// op.comparator().
LogicalResult sortDropEmptyUseArgs(SortOp op, PatternRewriter &rewriter) {
  DenseSet<unsigned> erasedArgs;
  unsigned numOperands = op.getNumOperands();
  for (unsigned i = 0; i < numOperands; ++i) {
    if (!op.getResult(i).use_empty())
      continue;
    Block &block = op.getComparator().front();
    if (!block.getArgument(i * 2).use_empty())
      continue;
    if (!block.getArgument(i * 2 + 1).use_empty())
      continue;
    erasedArgs.insert(i);
  }
  if (erasedArgs.empty())
    return failure();

  SmallVector<Value> newOperands;
  BitVector erasedBlockArgs(op.getNumOperands() * 2);
  for (const auto &en : llvm::enumerate(op.getInputs())) {
    if (erasedArgs.contains(en.index())) {
      erasedBlockArgs.set(en.index() * 2);
      erasedBlockArgs.set(en.index() * 2 + 1);
    } else {
      newOperands.push_back(en.value());
    }
  }

  auto newOp = rewriter.create<SortOp>(op.getLoc(), newOperands,
                                       op.getDimension(), op.getIsStable());
  Region &region = newOp.getComparator();
  rewriter.inlineRegionBefore(op.getComparator(), region, region.end());
  region.front().eraseArguments(erasedBlockArgs);

  SmallVector<Value> results;
  for (unsigned i = 0, j = 0; i < numOperands; ++i) {
    if (erasedArgs.contains(i)) {
      results.push_back({});
    } else {
      results.push_back(newOp.getResult(j++));
    }
  }
  rewriter.replaceOp(op, results);

  return success();
}

// Set the sorting dimension to the last dimension if it's not set and the rank
// is known.
LogicalResult sortOpInferDefaultDimension(SortOp op,
                                          PatternRewriter &rewriter) {
  auto ty = dyn_cast<ShapedType>(op.getResultTypes()[0]);
  if (!ty) {
    return failure();
  }
  if (static_cast<int64_t>(op.getDimension()) != -1) {
    return failure();
  }

  IntegerAttr dim = rewriter.getI64IntegerAttr(ty.getRank() - 1);
  auto newOp =
      rewriter.create<SortOp>(op.getLoc(), op.getResultTypes(), op.getInputs(),
                              dim, op.getIsStableAttr());
  Region &region = newOp.getComparator();
  rewriter.inlineRegionBefore(op.getComparator(), region, region.end());
  rewriter.replaceOp(op, newOp.getResults());

  return success();
}

} // namespace

void SortOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext * /*context*/) {
  results.add(sortDropEmptyUseArgs);
  results.add(sortOpInferDefaultDimension);
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

OpFoldResult TransposeOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  if (auto elements = dyn_cast_or_null<SplatElementsAttr>(operands.front())) {
    ShapedType type =
        prime_ir::maybeConvertPrimeIRToBuiltinType(getResult().getType());
    return reshape(elements, type);
  }
  for (const auto &it : llvm::enumerate(getPermutation().getValues<APInt>())) {
    if (it.index() != it.value()) {
      return {};
    }
  }
  if (getOperand().getType() == getType())
    return getOperand();
  return {};
}

namespace {

// transpose(transpose(X)) => transpose(X)
class EliminateRedundantTranspose : public OpRewritePattern<TransposeOp> {
public:
  using OpRewritePattern<TransposeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto tranposeOperand = op.getOperand().getDefiningOp<TransposeOp>();
    if (!tranposeOperand) {
      return failure();
    }
    auto operandPermutation =
        tranposeOperand.getPermutation().getValues<APInt>();
    auto newPermutation =
        cast<DenseIntElementsAttr>(op.getPermutation().mapValues(
            op.getPermutation().getElementType(),
            [&operandPermutation](const APInt &index) -> APInt {
              return operandPermutation[index.getSExtValue()];
            }));
    rewriter.replaceOpWithNewOp<TransposeOp>(op, op.getResult().getType(),
                                             tranposeOperand.getOperand(),
                                             newPermutation);
    return success();
  }
};

// BroadcastInDim(BroadcastInDim(X)) => BroadcastInDim(X)
class EliminateBroadcastInDimTranspose : public OpRewritePattern<TransposeOp> {
public:
  using OpRewritePattern<TransposeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto broadcastInDimOp = op.getOperand().getDefiningOp<BroadcastInDimOp>();
    if (!broadcastInDimOp) {
      return failure();
    }
    DenseIntElementsAttr broadcastDimensions =
        broadcastInDimOp.getBroadcastDimensions();
    DenseIntElementsAttr permutation = op.getPermutation();
    SmallVector<int64_t> newBroadcastDimensions;
    for (auto dimension : broadcastDimensions.getValues<int64_t>()) {
      int64_t index = 0;
      for (auto p : permutation.getValues<int64_t>()) {
        if (p == dimension) {
          newBroadcastDimensions.push_back(index);
          break;
        }
        index++;
      }
    }
    rewriter.replaceOpWithNewOp<BroadcastInDimOp>(
        op, op->getResultTypes(), broadcastInDimOp.getOperand(),
        rewriter.getI64TensorAttr(newBroadcastDimensions));
    return success();
  }
};

// simplify Transpose: replace Transpose with Reshape if they are equivalent
class SimplifyTranspose : public OpRewritePattern<TransposeOp> {
public:
  using OpRewritePattern<TransposeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto operandType = dyn_cast<RankedTensorType>(op.getOperand().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!operandType || !resultType) {
      return failure();
    }
    // Not support dynamic shape a.t.m. BTW, when it's dynamic shape,
    // maybe Transpose should be replaced by DynamicReshape.
    if (!operandType.hasStaticShape() || !resultType.hasStaticShape()) {
      return failure();
    }
    auto permutation = op.getPermutation().getValues<int64_t>();
    llvm::SmallVector<int64_t> sortedPermutation;
    for (int64_t i = 0, e = resultType.getRank(); i < e; i++) {
      if (resultType.getDimSize(i) != 1) {
        sortedPermutation.push_back(permutation[i]);
      }
    }
    if (llvm::is_sorted(sortedPermutation)) {
      rewriter.replaceOpWithNewOp<ReshapeOp>(op, op.getType(), op.getOperand());
      return success();
    }
    return failure();
  }
};

} // namespace

void TransposeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.add<EliminateRedundantTranspose>(context);
  results.add<EliminateBroadcastInDimTranspose>(context);
  results.add<SimplifyTranspose>(context);
}

LogicalResult TransposeOp::reifyReturnTypeShapes(
    OpBuilder &builder, ValueRange operands,
    SmallVectorImpl<Value> &reifiedReturnShapes) {
  TransposeOp::Adaptor adaptor(operands);
  Value operand = adaptor.getOperand();

  auto operandType = dyn_cast<RankedTensorType>(operand.getType());
  // Not support unranked type a.t.m.
  if (!operandType)
    return failure();

  Location loc = this->getLoc();
  SmallVector<int64_t, 4> permutation(
      this->getPermutation().getValues<int64_t>());
  SmallVector<Value, 4> shapeValues(permutation.size());

  Type shapeScalarType = builder.getIndexType();
  auto toShapeScalarType = [&](Value v) {
    return maybeCastTo(builder, loc, v, shapeScalarType);
  };

  for (const auto &element : llvm::enumerate(operandType.getShape())) {
    int64_t idx = element.index();
    auto *it = std::find(permutation.begin(), permutation.end(), idx);
    Value valueDim = toShapeScalarType(
        builder.createOrFold<tensor::DimOp>(loc, operand, element.index()));
    shapeValues[std::distance(permutation.begin(), it)] = valueDim;
  }

  Value outputShape = builder.create<tensor::FromElementsOp>(
      loc,
      RankedTensorType::get({static_cast<int64_t>(shapeValues.size())},
                            shapeScalarType),
      shapeValues);
  reifiedReturnShapes.push_back(outputShape);

  return success();
}

LogicalResult
TransposeOp::inferReturnTypes(MLIRContext *, std::optional<Location> loc,
                              ValueRange operands, DictionaryAttr attributes,
                              OpaqueProperties properties, RegionRange regions,
                              SmallVectorImpl<Type> &inferredReturnTypes) {
  TransposeOp::Adaptor adaptor(operands, attributes, properties, regions);
  if (failed(verify1dTensor(loc, adaptor.getPermutation(), "permutation")))
    return failure();
  return hlo::inferTransposeOp(
      loc, adaptor.getOperand(),
      llvm::to_vector(adaptor.getPermutation().getValues<int64_t>()),
      inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// TupleOp
//===----------------------------------------------------------------------===//

namespace {

// Pattern for unpacking and repacking the same tuple.
struct UnpackRepackSameTuple : public OpRewritePattern<TupleOp> {
  using OpRewritePattern<TupleOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TupleOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getVal().empty())
      return failure();

    Value firstElement = op.getVal().front();
    auto firstElementOp = firstElement.getDefiningOp<GetTupleElementOp>();
    if (!firstElementOp || firstElementOp.getIndexAttr().getInt() != 0)
      return failure();

    Value tuplePredecessor = firstElementOp.getOperand();
    if (tuplePredecessor.getType() != op.getType())
      return failure();

    for (const auto &elementAndIdx :
         llvm::enumerate(op.getVal().drop_front(1))) {
      auto elementOp = elementAndIdx.value().getDefiningOp<GetTupleElementOp>();
      if (!elementOp ||
          elementOp.getIndexAttr().getInt() !=
              static_cast<int64_t>(elementAndIdx.index() + 1) ||
          elementOp.getOperand() != tuplePredecessor)
        return failure();
    }

    rewriter.replaceOp(op, tuplePredecessor);
    return success();
  }
};

} // namespace

void TupleOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.add<UnpackRepackSameTuple>(context);
}

LogicalResult TupleOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  TupleOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferTupleOp(context, location, adaptor.getVal(),
                           inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// GatherOp
//===----------------------------------------------------------------------===//

// Converts gather ops to slice ops in case we have a single set of constant
// indices.
struct GatherSlice : public OpRewritePattern<GatherOp> {
  using OpRewritePattern<GatherOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GatherOp gather,
                                PatternRewriter &rewriter) const override {
    DenseIntElementsAttr index;
    if (!matchPattern(gather.getStartIndices(), m_Constant(&index)))
      return failure();

    const auto &dnums = gather.getDimensionNumbers();
    if (dnums.getIndexVectorDim() != 0 || index.getType().getRank() > 1)
      return failure();

    // TODO(tberghammer): Remove when the verifier catches this case what is
    // invalid if all previous condition holds.
    if (index.getNumElements() !=
        static_cast<int64_t>(dnums.getStartIndexMap().size()))
      return failure();

    RankedTensorType operandType =
        dyn_cast<RankedTensorType>(gather->getOperand(0).getType());
    if (!operandType || !operandType.hasStaticShape())
      return failure();

    auto sliceEnd =
        llvm::to_vector<8>(gather.getSliceSizes().getValues<int64_t>());
    llvm::SmallVector<int64_t, 8> sliceStart(sliceEnd.size(), 0);
    for (auto it :
         llvm::zip(dnums.getStartIndexMap(), index.getValues<APInt>())) {
      int64_t mapIndex = std::get<0>(it);
      // Clamp the indices within bounds to faithfully mirror gather semantics.
      int64_t offset =
          clamp(std::get<1>(it).getSExtValue(), static_cast<int64_t>(0),
                operandType.getDimSize(mapIndex) - sliceEnd[mapIndex]);
      sliceStart[mapIndex] += offset;
      sliceEnd[mapIndex] += offset;
    }

    llvm::SmallVector<int64_t, 8> sliceStride(sliceEnd.size(), 1);
    llvm::SmallVector<int64_t, 8> sliceShape(sliceEnd.size());
    for (size_t i = 0; i < sliceEnd.size(); ++i) {
      sliceShape[i] = sliceEnd[i] - sliceStart[i];
    }
    Type elementType = cast<TensorType>(gather.getType()).getElementType();
    auto sliceType = RankedTensorType::get(sliceShape, elementType);
    Value result = rewriter.create<SliceOp>(
        gather.getLoc(), sliceType, gather.getOperand(),
        rewriter.getI64TensorAttr(sliceStart),
        rewriter.getI64TensorAttr(sliceEnd),
        rewriter.getI64TensorAttr(sliceStride));

    auto collapsedSliceDims = dnums.getCollapsedSliceDims();
    if (!collapsedSliceDims.empty()) {
      llvm::SmallVector<int64_t, 8> reshapeShape;
      for (size_t i = 0; i < sliceShape.size(); ++i) {
        if (llvm::count(collapsedSliceDims, i) == 0) {
          reshapeShape.push_back(sliceShape[i]);
        }
      }
      auto reshapeType = RankedTensorType::get(reshapeShape, elementType);
      result = rewriter.create<ReshapeOp>(gather.getLoc(), reshapeType, result);
    }

    result.setType(gather.getType());
    rewriter.replaceOp(gather, result);
    return success();
  }
};

void GatherOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.add<GatherSlice>(context);
}

namespace {

// following https://www.tensorflow.org/xla/operation_semantics#gather
// The bounds for the output array along dimension i is computed as follows:
// (1) If i is present in batch_dims (i.e. is equal to batch_dims[k] for some k)
// then we pick
// the corresponding dimension bounds out of start_indices.shape, skipping
// index_vector_dim
// (i.e. pick start_indices.shape.dims[k] if k < index_vector_dim and
// start_indices.shape.dims[k+1] otherwise).
// (2) If i is present in offset_dims (i.e. equal to offset_dims[k] for some k)
// then we pick
// the corresponding bound out of slice_sizes after accounting for
// collapsed_slice_dims
// (i.e. we pick adjusted_slice_sizes[k] where adjusted_slice_sizes is
// slice_sizes with the bounds at indices collapsed_slice_dims removed).

void getSliceSizeValues(GatherOp *gather, OpBuilder &builder, Location loc,
                        ValueRange operands,
                        SmallVectorImpl<Value> &sliceSizes) {
  for (int64_t val : gather->getSliceSizes().getValues<int64_t>()) {
    sliceSizes.push_back(builder.create<arith::ConstantIndexOp>(loc, val));
  }
}

template <typename Op>
LogicalResult reifyGatherShape(Op *op, OpBuilder &builder, ValueRange operands,
                               SmallVectorImpl<Value> &reifiedReturnShapes) {
  // No support for unranked gather output shape a.t.m.
  auto resultTy = mlir::dyn_cast<RankedTensorType>(op->getResult().getType());
  if (!resultTy)
    return failure();

  typename Op::Adaptor adaptor(operands);
  Value startIndices = adaptor.getStartIndices();

  Location loc = op->getLoc();
  int resultRank = resultTy.getRank();
  Type shapeElTy = builder.getIndexType();
  auto toShapeElType = [&](Value v) {
    return maybeCastTo(builder, loc, v, shapeElTy);
  };

  SmallVector<Value, 4> sliceSizes;
  getSliceSizeValues(op, builder, loc, operands, sliceSizes);
  llvm::transform(sliceSizes, sliceSizes.begin(),
                  [&](Value v) { return toShapeElType(v); });

  auto getStartIndicesDim = [&](int64_t index) {
    return toShapeElType(
        builder.create<tensor::DimOp>(loc, startIndices, index));
  };
  SmallVector<Value, 4> shapeValues;
  auto getSliceDim = [&sliceSizes](int64_t index) -> Value {
    auto ret = sliceSizes[index];
    return ret;
  };
  hlo::reifyGatherDimSizes(resultRank, getStartIndicesDim, getSliceDim,
                           op->getDimensionNumbers().getOffsetDims(),
                           op->getDimensionNumbers().getCollapsedSliceDims(),
                           op->getDimensionNumbers().getOperandBatchingDims(),
                           op->getDimensionNumbers().getIndexVectorDim(),
                           shapeValues);

  Value outputShape = builder.create<tensor::FromElementsOp>(
      loc, RankedTensorType::get({resultRank}, shapeElTy), shapeValues);
  reifiedReturnShapes.push_back(outputShape);

  return success();
}

} // namespace

LogicalResult
GatherOp::reifyReturnTypeShapes(OpBuilder &builder, ValueRange operands,
                                SmallVectorImpl<Value> &reifiedReturnShapes) {
  return reifyGatherShape(this, builder, operands, reifiedReturnShapes);
}

LogicalResult GatherOp::inferReturnTypeComponents(
    MLIRContext *context, std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  GatherOp::Adaptor adaptor(operands, attributes, properties, regions);
  if (failed(verify1dTensor(location, adaptor.getSliceSizes(), "slice_sizes")))
    return failure();
  return hlo::inferGatherOp(
      location, adaptor.getOperand(), adaptor.getStartIndices(),
      adaptor.getDimensionNumbers().getOffsetDims(),
      adaptor.getDimensionNumbers().getCollapsedSliceDims(),
      adaptor.getDimensionNumbers().getOperandBatchingDims(),
      adaptor.getDimensionNumbers().getStartIndicesBatchingDims(),
      adaptor.getDimensionNumbers().getStartIndexMap(),
      adaptor.getDimensionNumbers().getIndexVectorDim(),
      llvm::to_vector(adaptor.getSliceSizes().getValues<int64_t>()),
      inferredReturnShapes);
}

//===----------------------------------------------------------------------===//
// ScatterOp
//===----------------------------------------------------------------------===//

LogicalResult
ScatterOp::inferReturnTypes(MLIRContext *, std::optional<Location> location,
                            ValueRange operands, DictionaryAttr attributes,
                            OpaqueProperties properties, RegionRange regions,
                            SmallVectorImpl<Type> &inferredReturnTypes) {
  ScatterOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferScatterOp(location, adaptor.getInputs(),
                             adaptor.getUpdateComputation(),
                             inferredReturnTypes);
}

LogicalResult ScatterOp::verify() {
  return hlo::verifyScatterOp(
      getLoc(), getInputs(), getScatterIndices(), getUpdates(),
      getScatterDimensionNumbers().getUpdateWindowDims(),
      getScatterDimensionNumbers().getInsertedWindowDims(),
      getScatterDimensionNumbers().getInputBatchingDims(),
      getScatterDimensionNumbers().getScatterIndicesBatchingDims(),
      getScatterDimensionNumbers().getScatterDimsToOperandDims(),
      getScatterDimensionNumbers().getIndexVectorDim(), getUpdateComputation());
}

namespace {

llvm::SmallVector<Attribute, 4> evaluateMhloRegion(Region &region,
                                                   ArrayRef<Attribute> inputs) {
  if (region.getNumArguments() != inputs.size())
    return {};

  llvm::DenseMap<Value, Attribute> values;
  values.reserve(region.getNumArguments());
  for (auto it : llvm::zip(region.getArguments(), inputs)) {
    values.try_emplace(std::get<0>(it), std::get<1>(it));
  }

  for (auto &op : region.getOps()) {
    llvm::SmallVector<Attribute, 4> opInputs;
    for (auto &operand : op.getOpOperands()) {
      opInputs.push_back(values.lookup(operand.get()));
    }
    if (isa<ReturnOp>(op))
      return opInputs;

    llvm::SmallVector<OpFoldResult, 4> results;
    if (failed(op.fold(opInputs, results)))
      return {};
    for (auto it : llvm::zip(op.getResults(), results)) {
      if (!std::get<1>(it).is<Attribute>())
        return {};
      values.insert({std::get<0>(it), std::get<1>(it).get<Attribute>()});
    }
  }
  return {};
}

} // namespace

LogicalResult
ScatterOp::fold(FoldAdaptor adaptor,
                llvm::SmallVectorImpl<OpFoldResult> &foldResults) {
  auto args = adaptor.getOperands();
  // Variadic Scatter not yet implemented
  if (getInputs().size() != 1 || getUpdates().size() != 1)
    return failure();
  auto index = dyn_cast_or_null<DenseIntElementsAttr>(args[1]);
  if (!index)
    return failure();

  auto baseType = dyn_cast<RankedTensorType>(getInputs().getTypes()[0]);
  auto updateType = dyn_cast<RankedTensorType>(getUpdates().getTypes()[0]);
  auto indexType = cast<RankedTensorType>(index.getType());
  if (!baseType || !indexType || !updateType)
    return failure();

  // Catch a trivial full replacement of base with update, this does not require
  // these to be constant: just that we know the type.
  // The update computation must return the update value (second block argument)
  // for this to be a valid full replacement.
  auto &computationBlock = getUpdateComputation().front();
  if (updateType == baseType && updateType.hasStaticShape() &&
      baseType.hasStaticShape() && index.isSplat() &&
      index.getSplatValue<APInt>().isZero() &&
      llvm::hasSingleElement(computationBlock)) {
    auto returnOp = dyn_cast<ReturnOp>(computationBlock.getTerminator());
    if (returnOp && returnOp.getNumOperands() == 1 &&
        returnOp.getOperand(0) == computationBlock.getArgument(1)) {
      foldResults.push_back(getUpdates()[0]);
      return success();
    }
  }
  auto base = dyn_cast_or_null<DenseElementsAttr>(args[0]);
  auto update = dyn_cast_or_null<DenseElementsAttr>(args[2]);
  if (!base || !update)
    return failure();

  // Add the virtual trailing dimension of size 1 if indexVectorDim equals to
  // indexType.rank.
  const int64_t indexVectorDim =
      getScatterDimensionNumbers().getIndexVectorDim();
  if (indexVectorDim == indexType.getRank()) {
    auto indexShape = indexType.getShape().vec();
    indexShape.push_back(1);
    indexType = RankedTensorType::get(indexShape, indexType.getElementType());
    index = cast<DenseIntElementsAttr>(reshape(index, indexType));
  }

  // Increment the multi-dimensional index vector based on the limits for each
  // dimension specified by shape and returns false if the index rolled around
  // with true otherwise.
  auto nextIndex = [](llvm::SmallVector<uint64_t, 8> &index,
                      llvm::ArrayRef<int64_t> shape) {
    for (int64_t i = index.size() - 1; i >= 0; --i) {
      ++index[i];
      if (index[i] < static_cast<unsigned long>(shape[i]))
        return true;
      index[i] = 0;
    }
    return false;
  };

  // Prevent folding if the result is too large.
  if (base.getNumElements() > kFoldOpEltLimit)
    return failure();

  // Iterate over all elements of the update tensor, then find the corresponding
  // value in the indices tensor to determine which location we have to update
  // in the base/result tensor.
  llvm::SmallVector<Attribute, 8> results(base.getValues<Attribute>());
  llvm::SmallVector<uint64_t, 8> updateIndex(updateType.getRank(), 0);
  llvm::SmallVector<uint64_t, 8> indexIndex;
  indexIndex.reserve(indexType.getRank());
  llvm::SmallVector<int64_t, 8> baseIndex;
  baseIndex.reserve(baseType.getRank());
  do {
    // Compute the index for the slice of the indices tensor for this update
    // value.
    indexIndex.clear();
    if (indexVectorDim == 0)
      indexIndex.push_back(0);
    auto updateWindowDims = getScatterDimensionNumbers().getUpdateWindowDims();
    for (int64_t i = 0; i < static_cast<int64_t>(updateIndex.size()); ++i) {
      if (!llvm::is_contained(updateWindowDims, i))
        indexIndex.push_back(updateIndex[i]);
      if (static_cast<int64_t>(indexIndex.size()) == indexVectorDim)
        indexIndex.push_back(0);
    }

    // Compute the index for the given update value in the base tensor.
    baseIndex.assign(baseType.getRank(), 0);
    auto inputBatchingDims =
        getScatterDimensionNumbers().getInputBatchingDims();
    auto scatterIndicesBatchingDims =
        getScatterDimensionNumbers().getScatterIndicesBatchingDims();
    for (auto [operandDim, indicesDim] :
         llvm::zip_equal(inputBatchingDims, scatterIndicesBatchingDims)) {
      baseIndex[operandDim] = indexIndex[indicesDim];
    }
    uint64_t indexCount = indexType.getShape()[indexVectorDim];
    for (uint64_t i = 0; i < indexCount; ++i) {
      uint64_t operandDim =
          getScatterDimensionNumbers().getScatterDimsToOperandDims()[i];
      indexIndex[indexVectorDim] = i;
      baseIndex[operandDim] +=
          index.getValues<APInt>()[indexIndex].getSExtValue();
    }
    uint64_t updateWindowDimIndex = 0;
    auto insertedWindowDims =
        getScatterDimensionNumbers().getInsertedWindowDims();
    for (uint64_t i = 0; i < baseIndex.size(); ++i) {
      if (llvm::is_contained(insertedWindowDims, i) ||
          llvm::is_contained(inputBatchingDims, i))
        continue;
      baseIndex[i] += updateIndex[updateWindowDims[updateWindowDimIndex]];
      updateWindowDimIndex++;
    }

    // Compute the linear index for the index into the base tensor.
    int64_t linearBaseIndex = 0;
    int64_t linearBaseIndexMultiplyer = 1;
    for (int64_t i = baseIndex.size() - 1; i >= 0; --i) {
      // Out of bound index have backend specific behaviour so avoid folding it.
      if (baseIndex[i] < 0 || baseIndex[i] >= baseType.getShape()[i])
        return failure();
      linearBaseIndex += baseIndex[i] * linearBaseIndexMultiplyer;
      linearBaseIndexMultiplyer *= baseType.getShape()[i];
    }

    // Evaluate update computation and update the value with the newly computed
    // attribute in the base tensor.
    auto lhs = DenseElementsAttr::get(
        prime_ir::maybeConvertPrimeIRToBuiltinType(
            RankedTensorType::get({}, baseType.getElementType())),
        results[linearBaseIndex]);
    auto rhs = DenseElementsAttr::get(
        prime_ir::maybeConvertPrimeIRToBuiltinType(
            RankedTensorType::get({}, baseType.getElementType())),
        update.getValues<Attribute>()[updateIndex]);
    auto newValue = evaluateMhloRegion(getUpdateComputation(), {lhs, rhs});
    if (newValue.size() != 1 || !newValue[0])
      return failure();
    results[linearBaseIndex] =
        cast<DenseElementsAttr>(newValue[0]).getValues<Attribute>()[0];
  } while (nextIndex(updateIndex, updateType.getShape()));

  baseType = cast<RankedTensorType>(
      prime_ir::maybeConvertPrimeIRToBuiltinType(baseType));
  foldResults.push_back(DenseElementsAttr::get(baseType, results));
  return success();
}

namespace {

// Replace mhlo.scatter overwriting the entire input with mhlo.map.
struct ScatterFullReplace : public OpRewritePattern<ScatterOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ScatterOp scatter,
                                PatternRewriter &rewriter) const override {
    // Variadic Scatter not yet implemented
    if (scatter.getInputs().size() != 1 || scatter.getUpdates().size() != 1)
      return failure();

    auto baseType =
        dyn_cast<RankedTensorType>(scatter.getInputs().getTypes()[0]);
    auto updateType =
        dyn_cast<RankedTensorType>(scatter.getUpdates().getTypes()[0]);
    auto indexType =
        dyn_cast<RankedTensorType>(scatter.getScatterIndices().getType());
    if (!baseType || !indexType || !updateType)
      return failure();

    // If scatter_indices has zero elements, the scatter is a no-op.
    // Per StableHLO spec, return the input tensor unchanged.
    if (!indexType.hasStaticShape() || indexType.getNumElements() > 0)
      return failure();

    rewriter.replaceOp(scatter, scatter.getInputs());
    return success();
  }
};

} // namespace

void ScatterOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add<ScatterFullReplace>(context);
}

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//

LogicalResult WhileOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  WhileOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferWhileOp(location, adaptor.getOperand(), inferredReturnTypes);
}

LogicalResult WhileOp::verify() {
  return hlo::verifyWhileOp(getLoc(), getOperand(), getCond(), getBody());
}

void WhileOp::print(OpAsmPrinter &p) {
  hlo::printWhileOp(p, getOperation(), getCond(), getBody());
}

ParseResult WhileOp::parse(OpAsmParser &parser, OperationState &result) {
  return hlo::parseWhileOp(parser, result);
}

LogicalResult WhileOp::fold(FoldAdaptor /*adaptor*/,
                            SmallVectorImpl<OpFoldResult> &results) {
  DenseIntElementsAttr condValue;
  // TODO: This folder is executed on invalid mhlo.while ops during
  // LegalizeMhlo, mlir_hlo/tosa/tests/unary.mlir. Broken pattern?
  auto condReturnOp = dyn_cast<ReturnOp>(getCond().front().back());
  if (!condReturnOp)
    return failure();
  if (!matchPattern(condReturnOp.getOperand(0), m_Constant(&condValue)))
    return failure();
  if (condValue.getSplatValue<BoolAttr>().getValue())
    return failure(); // TODO(mhlo): this is an infinite loop, should we fold?

  results.append(getOperands().begin(), getOperands().end());
  return success(!results.empty());
}

static LogicalResult whileCanonicalization(WhileOp whileOp,
                                           PatternRewriter &rewriter) {
  // Turn loop invariant values into implicit capture.
  // Check if there is at least one value is forwarded from one iteration to the
  // next, or one of the yielded value is an implicit capture already. Otherwise
  // there is nothing to do here.
  Block *cond = whileOp.SingleBlock::getBody(0);
  Block *body = whileOp.SingleBlock::getBody(1);
  auto bodyReturnOp = cast<ReturnOp>(body->getTerminator());
  if (!llvm::any_of(llvm::zip(whileOp->getOperands(), body->getArguments(),
                              bodyReturnOp->getOperands()),
                    [&](auto zip) {
                      return (std::get<0>(zip) == std::get<2>(zip) ||
                              std::get<1>(zip) == std::get<2>(zip));
                    }))
    return rewriter.notifyMatchFailure(whileOp, "no loop invariant found");

  SmallVector<Value> newOperands, resultsToReplace;
  SmallVector<unsigned> invariantArgIdxs;
  BitVector invariantArgIdxBitVector(cond->getNumArguments());
  for (const auto &enumeratedOperands : llvm::enumerate(llvm::zip(
           whileOp.getOperands(), cond->getArguments(), body->getArguments(),
           bodyReturnOp->getOperands(), whileOp->getResults()))) {
    const auto &operands = enumeratedOperands.value();
    Value whileOperand = std::get<0>(operands);
    BlockArgument condBlockArg = std::get<1>(operands);
    BlockArgument bodyBlockArg = std::get<2>(operands);
    Value bodyReturnOperand = std::get<3>(operands);
    Value whileResult = std::get<4>(operands);

    bool forwarded = (whileOperand == bodyReturnOperand ||
                      bodyBlockArg == bodyReturnOperand);
    if (forwarded) {
      invariantArgIdxs.push_back(enumeratedOperands.index());
      invariantArgIdxBitVector.set(enumeratedOperands.index());
      condBlockArg.replaceAllUsesWith(whileOperand);
      bodyBlockArg.replaceAllUsesWith(whileOperand);
      whileResult.replaceAllUsesWith(whileOperand);
      continue;
    }
    newOperands.push_back(whileOperand);
    resultsToReplace.push_back(whileResult);
  }
  cond->eraseArguments(invariantArgIdxBitVector);
  body->eraseArguments(invariantArgIdxBitVector);
  for (int idx : llvm::reverse(invariantArgIdxs))
    bodyReturnOp->eraseOperand(idx);

  WhileOp newWhileOp = rewriter.create<WhileOp>(
      whileOp.getLoc(), bodyReturnOp->getOperandTypes(), newOperands);
  newWhileOp.getBodyRegion(0).takeBody(whileOp.getBodyRegion(0));
  newWhileOp.getBodyRegion(1).takeBody(whileOp.getBodyRegion(1));
  for (auto results : llvm::zip(resultsToReplace, newWhileOp->getResults()))
    std::get<0>(results).replaceAllUsesWith(std::get<1>(results));
  rewriter.eraseOp(whileOp);
  return success();
}

void WhileOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.add(&whileCanonicalization);
}

//===----------------------------------------------------------------------===//
// UnaryOps
//===----------------------------------------------------------------------===//

namespace {

template <typename ValType>
struct AnyValue {
  bool operator()(const ValType &) { return true; }
};

template <typename ValType>
struct NonNegativeValue {
  bool operator()(const ValType &v) { return !v.isNegative(); }
};

template <typename ValType>
struct PositiveValue {
  bool operator()(const ValType &v) { return !v.isNegative() && !v.isZero(); }
};

template <typename Op, typename ElementType, typename ValType, typename Convert,
          typename Validate = AnyValue<ValType>>
Attribute UnaryFolder(Op *op, ArrayRef<Attribute> attrs) {
  if (!attrs[0])
    return {};

  auto val = dyn_cast<DenseElementsAttr>(attrs[0]);
  if (!val)
    return {};

  auto type = cast<ShapedType>(op->getType());
  if (!type.hasStaticShape()) {
    return {};
  }

  Type etype = type.getElementType();

  // Evaluate for integer values.
  if (!isa<ElementType>(etype)) {
    return {};
  }

  // Prevent folding if the result is too large.
  if (val.getNumElements() > kFoldOpEltLimit)
    return {};

  SmallVector<ValType, 6> values;
  values.reserve(val.getNumElements());
  for (const auto v : val.getValues<ValType>()) {
    if (!Validate()(v))
      return {};
    std::optional<ValType> r = Convert()(addSign(v, type));
    if (!r)
      return {};
    values.push_back(r.value());
  }

  return DenseElementsAttr::get(type, values);
}

template <typename T>
struct Sign;

template <>
struct Sign<APInt> {
  APInt operator()(const APInt &i) const {
    APInt r = i;
    if (r == 0)
      return r;
    if (r.isNegative()) {
      return APInt(r.getBitWidth(), -1, /*isSigned=*/true);
    }
    return APInt(r.getBitWidth(), 1, /*isSigned=*/true);
  }
};

template <typename T>
struct Abs;

template <>
struct Abs<APInt> {
  APInt operator()(const APInt &i) const { return i.abs(); }
};

} // namespace

// NOLINTBEGIN(bugprone-macro-parentheses)
#define UNARY_FOLDER(Op, Func)                                                 \
  OpFoldResult Op::fold(FoldAdaptor adaptor) {                                 \
    auto attrs = adaptor.getOperands();                                        \
    if (isa<IntegerType>(getElementTypeOrSelf(getType())))                     \
      return UnaryFolder<Op, IntegerType, APInt, Func<APInt>>(this, attrs);    \
    return {};                                                                 \
  }

UNARY_FOLDER(NegOp, std::negate)
UNARY_FOLDER(SignOp, Sign)
UNARY_FOLDER(AbsOp, Abs)
UNARY_FOLDER(NotOp, std::bit_not)

#undef UNARY_FOLDER

//===----------------------------------------------------------------------===//
// BinaryOps
//===----------------------------------------------------------------------===//

namespace {

template <typename T>
struct Divide : std::divides<T> {};

template <>
struct Divide<APSInt> {
  FailureOr<APSInt> operator()(const APSInt &a, const APSInt &b) const {
    if (b.isZero())
      return failure();
    return a / b;
  }
};

template <typename T>
struct Remainder : std::modulus<T> {};

template <>
struct Remainder<APSInt> {
  FailureOr<APSInt> operator()(const APSInt &a, const APSInt &b) const {
    if (b.isZero())
      return failure();
    return a % b;
  }
};

} // namespace

#define BINARY_FOLDER_INTERNAL(Op, Func)                                       \
  if (isa<IntegerType>(getElementTypeOrSelf(getType())))                       \
    return BinaryFolder<Op, IntegerType, APInt, Func<APSInt>>(this, attrs);    \
  return {};

#define BINARY_FOLDER(Op, Func)                                                \
  OpFoldResult Op::fold(FoldAdaptor adaptor) {                                 \
    auto attrs = adaptor.getOperands();                                        \
    BINARY_FOLDER_INTERNAL(Op, Func)                                           \
  }

// Addition, subtraction and multiplication use the std:: versions of the ops.
// Due to the other ops behaving differently in signed vs unsigned integers,
// APInts need a special implementation. Currently, it replicates signed int
// op behavior.
BINARY_FOLDER(SubtractOp, std::minus)
BINARY_FOLDER(DivOp, Divide)
BINARY_FOLDER(RemOp, Remainder)
BINARY_FOLDER(MaxOp, Max)
BINARY_FOLDER(MinOp, Min)

namespace {

bool isSplatZero(SplatElementsAttr attr) {
  if (!attr)
    return false;
  if (isa<IntegerType>(attr.getElementType())) {
    return attr.getSplatValue<APInt>().isZero();
  }
  return false;
}

} // namespace

OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  auto attrs = adaptor.getOperands();
  // Handle special case where one operand is 0:  x + 0 => x
  if (attrs[0] || attrs[1]) {
    auto splatLhs = dyn_cast_or_null<SplatElementsAttr>(attrs[0]);
    auto splatRhs = dyn_cast_or_null<SplatElementsAttr>(attrs[1]);
    if (isSplatZero(splatLhs))
      return splatRhs ? static_cast<OpFoldResult>(splatRhs) : getRhs();
    if (isSplatZero(splatRhs))
      return splatLhs ? static_cast<OpFoldResult>(splatLhs) : getLhs();
  }
  if (attrs[0] && attrs[1]) {
    BINARY_FOLDER_INTERNAL(AddOp, std::plus)
  }
  return {};
}

namespace {

bool isSplatOne(SplatElementsAttr attr) {
  if (!attr)
    return false;
  if (isa<IntegerType>(attr.getElementType())) {
    return attr.getSplatValue<APInt>().getSExtValue() == 1;
  }
  return false;
}

} // namespace

OpFoldResult MulOp::fold(FoldAdaptor adaptor) {
  auto attrs = adaptor.getOperands();
  // Handle special case where one operand is 1: x * 1 => x
  if (attrs[0] || attrs[1]) {
    auto splatLhs = dyn_cast_or_null<SplatElementsAttr>(attrs[0]);
    auto splatRhs = dyn_cast_or_null<SplatElementsAttr>(attrs[1]);
    if (isSplatOne(splatLhs))
      return splatRhs ? static_cast<OpFoldResult>(splatRhs) : getRhs();
    if (isSplatOne(splatRhs))
      return splatLhs ? static_cast<OpFoldResult>(splatLhs) : getLhs();
  }
  if (attrs[0] && attrs[1]) {
    BINARY_FOLDER_INTERNAL(MulOp, std::multiplies);
  }
  return {};
}

//===----------------------------------------------------------------------===//
// Logical Ops
//===----------------------------------------------------------------------===//

OpFoldResult AndOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  if (getLhs() == getRhs())
    return getLhs();

  auto lhsVal = dyn_cast_or_null<DenseElementsAttr>(operands[0]);
  auto rhsVal = dyn_cast_or_null<DenseElementsAttr>(operands[1]);

  if (lhsVal && lhsVal.isSplat()) {
    if (lhsVal.getSplatValue<IntegerAttr>().getValue().isAllOnes()) {
      return getRhs();
    }

    if (lhsVal.getSplatValue<IntegerAttr>().getValue().isZero()) {
      return lhsVal;
    }
  }

  if (rhsVal && rhsVal.isSplat()) {
    if (rhsVal.getSplatValue<IntegerAttr>().getValue().isAllOnes()) {
      return getLhs();
    }

    if (rhsVal.getSplatValue<IntegerAttr>().getValue().isZero()) {
      return rhsVal;
    }
  }

  if (!rhsVal || !lhsVal)
    return {};
  return BinaryFolder<AndOp, IntegerType, APInt, std::bit_and<APSInt>>(
      this, operands);
}

OpFoldResult OrOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  if (getLhs() == getRhs())
    return getLhs();

  auto lhsVal = dyn_cast_or_null<DenseElementsAttr>(operands[0]);
  auto rhsVal = dyn_cast_or_null<DenseElementsAttr>(operands[1]);

  if (lhsVal && lhsVal.isSplat()) {
    if (lhsVal.getSplatValue<IntegerAttr>().getValue().isAllOnes()) {
      return lhsVal;
    }

    if (lhsVal.getSplatValue<IntegerAttr>().getValue().isZero()) {
      return getRhs();
    }
  }

  if (rhsVal && rhsVal.isSplat()) {
    if (rhsVal.getSplatValue<IntegerAttr>().getValue().isAllOnes()) {
      return rhsVal;
    }

    if (rhsVal.getSplatValue<IntegerAttr>().getValue().isZero()) {
      return getLhs();
    }
  }

  if (!rhsVal || !lhsVal)
    return {};
  return BinaryFolder<OrOp, IntegerType, APInt, std::bit_or<APSInt>>(this,
                                                                     operands);
}

OpFoldResult XorOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  // Fold x^x to 0. Attributes only support static shapes.
  auto rType = cast<ShapedType>(getType());
  if (getLhs() == getRhs() && rType.hasStaticShape()) {
    Builder builder(getContext());
    return builder.getZeroAttr(rType);
  }

  auto lhsVal = dyn_cast_or_null<DenseElementsAttr>(operands[0]);
  auto rhsVal = dyn_cast_or_null<DenseElementsAttr>(operands[1]);

  if (lhsVal && lhsVal.isSplat()) {
    if (lhsVal.getSplatValue<IntegerAttr>().getValue().isZero()) {
      return getRhs();
    }
  }

  if (rhsVal && rhsVal.isSplat()) {
    if (rhsVal.getSplatValue<IntegerAttr>().getValue().isZero()) {
      return getLhs();
    }
  }

  if (!rhsVal || !lhsVal)
    return {};
  return BinaryFolder<XorOp, IntegerType, APInt, std::bit_xor<APSInt>>(
      this, operands);
}

#undef BINARY_FOLDER_INTERNAL
#undef BINARY_FOLDER

} // namespace mlir::mhlo

using mlir::hlo::parsePairwiseOpType;
using mlir::hlo::parseSameOperandsAndResultType;
using mlir::hlo::parseSelectOpType;
using mlir::hlo::parseTupleOpType;
using mlir::hlo::parseVariadicSameOperandsAndResultType;
using mlir::hlo::printPairwiseOpType;
using mlir::hlo::printSameOperandsAndResultType;
using mlir::hlo::printSelectOpType;
using mlir::hlo::printTupleOpType;
using mlir::hlo::printVariadicSameOperandsAndResultType;

#define GET_OP_CLASSES
#include "zkx/mlir_hlo/mhlo/IR/hlo_ops.cc.inc"

namespace mlir::mhlo {

//===----------------------------------------------------------------------===//
// mhlo Dialect Interfaces
//===----------------------------------------------------------------------===//

namespace {

struct MhloDialectInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  // Allow all call operations to be inlined.
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }
  // We don't have any special restrictions on what can be inlined into
  // destination regions (e.g. while/conditional bodies). Always allow it.
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    return true;
  }
  // Operations in mhlo dialect are always legal to inline since they are
  // pure.
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }
};

struct MhloHloDialectInterface : public hlo::HloDialectInterface {
  using HloDialectInterface::HloDialectInterface;

  Type createTokenType() const override {
    return TokenType::get(getDialect()->getContext());
  }

  bool isTokenType(Type type) const override { return isa<TokenType>(type); }

  Attribute createTypeExtensions(ArrayRef<int64_t> bounds) const override {
    return TypeExtensionsAttr::get(getDialect()->getContext(), bounds);
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// mhlo Dialect Constructor
//===----------------------------------------------------------------------===//

MhloDialect::MhloDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<MhloDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "zkx/mlir_hlo/mhlo/IR/hlo_ops.cc.inc" // NOLINT(build/include)
      >();
  addInterfaces<MhloHloDialectInterface>();
  addInterfaces<MhloDialectInlinerInterface>();
  // TODO(chokobole): Uncomment this. Dependency: MhloBytecodeInterface
  // addBytecodeInterface(this);
  // TODO(chokobole): Uncomment this. Dependency: AsyncBundleType
  // addTypes<TokenType, AsyncBundleType>();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "zkx/mlir_hlo/mhlo/IR/hlo_ops_attrs.cc.inc" // NOLINT(build/include)
      >();
}

// Entry point for Attribute parsing, TableGen generated code will handle the
// dispatch to the individual classes.
Attribute MhloDialect::parseAttribute(DialectAsmParser &parser,
                                      Type type) const {
  StringRef attrTag;
  Attribute attr;
  OptionalParseResult parseResult =
      generatedAttributeParser(parser, &attrTag, type, attr);
  if (parseResult.has_value())
    return attr;
  parser.emitError(parser.getNameLoc(), "unknown mhlo attribute");
  return Attribute();
}

// Entry point for Attribute printing, TableGen generated code will handle the
// dispatch to the individual classes.
void MhloDialect::printAttribute(Attribute attr, DialectAsmPrinter &os) const {
  LogicalResult result = generatedAttributePrinter(attr, os);
  std::ignore = result;
  assert(succeeded(result));
}

namespace {

// Helpers for attributes parsing.
ParseResult parseDims(AsmParser &parser, SmallVector<int64_t> &dimSizes) {
  dimSizes.clear();
  auto failOrDims = parseDimSizes(parser);
  if (failed(failOrDims)) {
    return failure();
  }
  dimSizes = std::move(*failOrDims);
  return success();
}

// Parse a custom attribute that resembles a struct of the form
// <
//   foo = something_parsed_by_custom_parser,
//   bar = something_parsed_by_different_custom_parser,
//   baz something_parsed_by_another_custom_parser
// >
// The optional argument `parse_equal` array can be used to denote if
// '=' follows the keyword (see baz in the example above) for a field. If
// not provided, all fields must be followed by a '='.
ParseResult parseStruct(AsmParser &parser, ArrayRef<StringRef> keywords,
                        ArrayRef<llvm::function_ref<ParseResult()>> parseFuncs,
                        ArrayRef<bool> parseEqual = {}) {
  assert(keywords.size() == parseFuncs.size());
  assert(parseEqual.empty() || parseEqual.size() == keywords.size());
  SmallVector<bool> seen(keywords.size(), false);
  while (failed(parser.parseOptionalGreater())) {
    bool foundOne = false;
    for (const auto &it : llvm::enumerate(keywords)) {
      size_t index = it.index();
      StringRef keyword = it.value();
      if (succeeded(parser.parseOptionalKeyword(keyword))) {
        if (seen[index]) {
          return parser.emitError(parser.getCurrentLocation())
                 << "duplicated `" << keyword << "` entry";
        }
        if (parseEqual.empty() || parseEqual[index]) {
          if (failed(parser.parseEqual()))
            return failure();
        }
        if (failed(parseFuncs[index]()))
          return failure();
        if (failed(parser.parseOptionalComma()))
          return parser.parseGreater();
        seen[index] = true;
        foundOne = true;
        break;
      }
    }
    if (!foundOne) {
      auto parseError = parser.emitError(parser.getCurrentLocation())
                        << "expected one of: ";
      llvm::interleaveComma(keywords, parseError, [&](StringRef kw) {
        parseError << '`' << kw << '`';
      });
      return parseError;
    }
  }
  return success();
}

// Helpers to print an optional array or integer field, to simplify writing
// attribute printers.
template <typename T>
void printField(AsmPrinter &printer, StringRef name, T field,
                StringRef &separator) {
  if (field != 0) {
    printer << separator << name << " = " << field;
    separator = ", ";
  }
}
template <typename T>
void printField(AsmPrinter &printer, StringRef name, ArrayRef<T> field,
                StringRef &separator) {
  if (!field.empty()) {
    printer << separator << name << " = [";
    llvm::interleaveComma(field, printer);
    printer << "]";
    separator = ", ";
  }
}

template <typename... Ts>
void printStruct(AsmPrinter &printer, StringRef name, Ts... printFields) {
  printer << "<";
  StringRef separator = "";
  // Fold expression to print each entry in the parameter pack.
  // TODO(mhlo-team): this can be simplified when TF moves to C++17.
  using unused = int[];
  (void)unused{0, (printField(printer, std::get<0>(printFields),
                              std::get<1>(printFields), separator),
                   0)...};
  printer << ">";
}

} // namespace

// Custom printer and parser for ScatterDimensionNumbersAttr.
void ScatterDimensionNumbersAttr::print(AsmPrinter &printer) const {
  printStruct(printer, "scatter",
              std::make_pair("update_window_dims", getUpdateWindowDims()),
              std::make_pair("inserted_window_dims", getInsertedWindowDims()),
              std::make_pair("input_batching_dims", getInputBatchingDims()),
              std::make_pair("scatter_indices_batching_dims",
                             getScatterIndicesBatchingDims()),
              std::make_pair("scatter_dims_to_operand_dims",
                             getScatterDimsToOperandDims()),
              std::make_pair("index_vector_dim", getIndexVectorDim()));
}

Attribute ScatterDimensionNumbersAttr::parse(AsmParser &parser, Type type) {
  if (failed(parser.parseLess()))
    return {};
  SmallVector<int64_t> updateWindowDims;
  SmallVector<int64_t> insertedWindowDims;
  SmallVector<int64_t> inputBatchingDims;
  SmallVector<int64_t> scatterIndicesBatchingDims;
  SmallVector<int64_t> scatterDimsToOperandDims;
  int64_t indexVectorDim = 0;

  if (failed(parseStruct(
          parser,
          {"update_window_dims", "inserted_window_dims", "input_batching_dims",
           "scatter_indices_batching_dims", "scatter_dims_to_operand_dims",
           "index_vector_dim"},
          {[&]() { return parseDims(parser, updateWindowDims); },
           [&]() { return parseDims(parser, insertedWindowDims); },
           [&]() { return parseDims(parser, inputBatchingDims); },
           [&]() { return parseDims(parser, scatterIndicesBatchingDims); },
           [&]() { return parseDims(parser, scatterDimsToOperandDims); },
           [&]() { return parser.parseInteger(indexVectorDim); }}))) {
    parser.emitError(parser.getCurrentLocation())
        << "failed parsing scatter dimension numbers attribute";
    return {};
  }

  return ScatterDimensionNumbersAttr::get(
      parser.getContext(), updateWindowDims, insertedWindowDims,
      inputBatchingDims, scatterIndicesBatchingDims, scatterDimsToOperandDims,
      indexVectorDim);
}

// Custom printer and parser for GatherDimensionNumbersAttr.
void GatherDimensionNumbersAttr::print(AsmPrinter &printer) const {
  printStruct(printer, "gather", std::make_pair("offset_dims", getOffsetDims()),
              std::make_pair("collapsed_slice_dims", getCollapsedSliceDims()),
              std::make_pair("operand_batching_dims", getOperandBatchingDims()),
              std::make_pair("start_indices_batching_dims",
                             getStartIndicesBatchingDims()),
              std::make_pair("start_index_map", getStartIndexMap()),
              std::make_pair("index_vector_dim", getIndexVectorDim()));
}

Attribute GatherDimensionNumbersAttr::parse(AsmParser &parser, Type type) {
  if (failed(parser.parseLess()))
    return {};
  SmallVector<int64_t> offsetDims;
  SmallVector<int64_t> collapsedSliceDims;
  SmallVector<int64_t> operandBatchingDims;
  SmallVector<int64_t> startIndicesBatchingDims;
  SmallVector<int64_t> startIndexMap;
  int64_t indexVectorDim = 0;

  if (failed(parseStruct(
          parser,
          {"offset_dims", "collapsed_slice_dims", "operand_batching_dims",
           "start_indices_batching_dims", "start_index_map",
           "index_vector_dim"},
          {[&]() { return parseDims(parser, offsetDims); },
           [&]() { return parseDims(parser, collapsedSliceDims); },
           [&]() { return parseDims(parser, operandBatchingDims); },
           [&]() { return parseDims(parser, startIndicesBatchingDims); },
           [&]() { return parseDims(parser, startIndexMap); },
           [&]() { return parser.parseInteger(indexVectorDim); }}))) {
    parser.emitError(parser.getCurrentLocation())
        << "failed parsing gather dimension numbers attribute";
    return {};
  }

  return GatherDimensionNumbersAttr::get(
      parser.getContext(), offsetDims, collapsedSliceDims, operandBatchingDims,
      startIndicesBatchingDims, startIndexMap, indexVectorDim);
}

// Custom printer and parser for DotDimensionNumbersAttr.
void DotDimensionNumbersAttr::print(AsmPrinter &printer) const {
  printStruct(
      printer, "dot",
      std::make_pair("lhs_batching_dimensions", getLhsBatchingDimensions()),
      std::make_pair("rhs_batching_dimensions", getRhsBatchingDimensions()),
      std::make_pair("lhs_contracting_dimensions",
                     getLhsContractingDimensions()),
      std::make_pair("rhs_contracting_dimensions",
                     getRhsContractingDimensions()));
}

Attribute DotDimensionNumbersAttr::parse(AsmParser &parser, Type type) {
  if (failed(parser.parseLess()))
    return {};
  SmallVector<int64_t> lhsBatchingDimensions;
  SmallVector<int64_t> rhsBatchingDimensions;
  SmallVector<int64_t> lhsContractingDimensions;
  SmallVector<int64_t> rhsContractingDimensions;

  if (failed(parseStruct(
          parser,
          {"lhs_batching_dimensions", "rhs_batching_dimensions",
           "lhs_contracting_dimensions", "rhs_contracting_dimensions"},
          {[&]() { return parseDims(parser, lhsBatchingDimensions); },
           [&]() { return parseDims(parser, rhsBatchingDimensions); },
           [&]() { return parseDims(parser, lhsContractingDimensions); },
           [&]() { return parseDims(parser, rhsContractingDimensions); }}))) {
    parser.emitError(parser.getCurrentLocation())
        << "failed parsing dot dimension numbers attribute";
    return {};
  }

  return DotDimensionNumbersAttr::get(
      parser.getContext(), lhsBatchingDimensions, rhsBatchingDimensions,
      lhsContractingDimensions, rhsContractingDimensions);
}

//===----------------------------------------------------------------------===//
// Builder utilities
//===----------------------------------------------------------------------===//

namespace {

// Builds the region `body` for mhlo.sort's comparator: for each type in
// `element_types`, create two block arguments, one for lhs and one for rhs, and
// generates mhlo.compare op to compare them with the given `direction`.
//
// Note that this right now only does comparison on the first pair of block
// arguments.
void buildSortComparisonBody(llvm::ArrayRef<Type> elementTypes,
                             ComparisonDirection direction, Region *body,
                             OpBuilder *builder) {
  OpBuilder::InsertionGuard insertionPointGurad(*builder);

  Location loc = body->getLoc();
  Block *block = builder->createBlock(body);
  // Add two arguments for each element type.
  for (Type elementType : elementTypes) {
    TensorType tensorType = RankedTensorType::get({}, elementType);
    block->addArguments({tensorType, tensorType},
                        SmallVector<Location, 2>(2, loc));
  }

  Value compare = builder->create<CompareOp>(loc, block->getArgument(0),
                                             block->getArgument(1), direction);

  builder->create<ReturnOp>(loc, compare);
}

} // namespace

SortOp createSortOp(PatternRewriter *rewriter, const Location &loc,
                    const llvm::ArrayRef<Value> &operands,
                    const llvm::ArrayRef<Type> &elementTypes, int64_t dimension,
                    bool isStable, ComparisonDirection direction) {
  assert(!operands.empty() && "No operands to sort");
  // Create the sort op.
  auto sortOp = rewriter->create<SortOp>(loc, operands, dimension, isStable);
  buildSortComparisonBody(elementTypes, direction, &sortOp.getComparator(),
                          rewriter);
  return sortOp;
}

//===----------------------------------------------------------------------===//
// MHLO Dialect Hooks
//===----------------------------------------------------------------------===//

Operation *MhloDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                            Type type, Location loc) {
  auto elementsAttr = dyn_cast<ElementsAttr>(value);
  // HLO dialect constants only support ElementsAttr unlike standard dialect
  // constant which supports all attributes.
  if (!elementsAttr)
    return nullptr;
  auto resultShapedType = dyn_cast<ShapedType>(type);
  auto attrShapedType = dyn_cast<ShapedType>(elementsAttr.getType());
  if (resultShapedType && attrShapedType) {
    return builder.create<mhlo::ConstantOp>(loc, type, elementsAttr);
  }
  // HLO dialect constants require the type of value and result to match
  if (type != elementsAttr.getType())
    return nullptr;

  return builder.create<mhlo::ConstantOp>(loc, type, elementsAttr);
}

LogicalResult MhloDialect::verifyRegionArgAttribute(Operation *op,
                                                    unsigned /*regionIndex*/,
                                                    unsigned argIndex,
                                                    NamedAttribute attr) {
  // TODO(chokobole): Uncomment this. Dependency: ArgResultAliasAttr
  // if (auto aliasAttr = dyn_cast<ArgResultAliasAttr>(attr.getValue())) {
  //   if (failed(
  //           verifyArgResultAliasAttr(attr.getName(), aliasAttr, argIndex,
  //           op)))
  //     return failure();
  // }
  // TODO(chokobole): Uncomment this. Dependency: ParameterReplicationAttr
  // if (attr.getName() == "mhlo.parameter_replication") {
  //   auto arrayAttr = dyn_cast<ArrayAttr>(attr.getValue());
  //   if (!arrayAttr)
  //     return op->emitOpError() << "parameter_replication: must be an array";
  //   auto func = dyn_cast<FunctionOpInterface>(op);
  //   if (!func) {
  //     return op->emitOpError()
  //            << "has parameter_replication but is not a function";
  //   }
  //   // parameter_replication = [] or [false] is equivalent to
  //   // [false,...,false] and parameter_replication = [true] means
  //   // [true,...,true]
  //   if (arrayAttr.empty() || arrayAttr.size() == 1) return success();
  //   auto num_leaf_buffers =
  //       getNumLeafBuffers(func.getArgumentTypes()[argIndex]);
  //   if ((size_t)num_leaf_buffers != arrayAttr.size())
  //     return op->emitOpError()
  //            << "parameter_replication: arg " << argIndex << " has "
  //            << num_leaf_buffers << " leaf_buffers, but
  //            parameter_replication"
  //            << " expects " << arrayAttr.size();
  // }
  return success();
}

LogicalResult MhloDialect::verifyOperationAttribute(Operation *op,
                                                    NamedAttribute attr) {
  // TODO(chokobole): Uncomment this. Dependency: ArgResultAliasAttr
  // if (auto aliasAttr = dyn_cast<ArgResultAliasAttr>(attr.getValue())) {
  //   if (!isa<FunctionOpInterface>(op))
  //     return op->emitOpError()
  //            << "attribute " << attr.getName()
  //            << " can only be used on function-like operations";
  // }
  // TODO(chokobole): Uncomment this. Dependency: CrossProgramPrefetchAttr
  // if (attr.getName() == "mhlo.cross_program_prefetches") {
  //   auto arrayAttr = dyn_cast<ArrayAttr>(attr.getValue());
  //   if (!arrayAttr)
  //     return op->emitOpError() << "cross_program_prefetches must be an
  //     array";
  //   for (auto attrElt : arrayAttr) {
  //     auto prefetchAttr = dyn_cast<CrossProgramPrefetchAttr>(attrElt);
  //     if (!prefetchAttr)
  //       return op->emitOpError() << "cross_program_prefetches must be an
  //       array "
  //                                   "of cross_program_prefetch attrs";
  //     auto module = dyn_cast<ModuleOp>(op);
  //     if (!module)
  //       return op->emitOpError()
  //              << "has cross_program_prefetches but is not a module";
  //     auto res = verifyCrossProgramPrefetchAttr(prefetchAttr, module);
  //     if (failed(res)) return res;
  //   }
  // }
  if (attr.getName() == "mhlo.spmd_parameters_sharding") {
    auto arrayAttr = dyn_cast<ArrayAttr>(attr.getValue());
    if (!arrayAttr)
      return op->emitOpError() << "spmd_parameters_sharding: must be an array";
    auto module = dyn_cast<ModuleOp>(op);
    if (!module)
      return op->emitOpError()
             << "has spmd_parameters_sharding but is not a module";
    // Check that the "main" function exists:
    auto main = module.lookupSymbol<func::FuncOp>("main");
    if (!main)
      return module.emitOpError() << "spmd_parameters_sharding: main not found";
    if (main.getNumArguments() != arrayAttr.size())
      return module.emitOpError()
             << "spmd_parameters_sharding: main has " << main.getNumArguments()
             << " arguments, but spmd_parameters_sharding expects "
             << arrayAttr.size();
  }
  return success();
}

} // namespace mlir::mhlo
