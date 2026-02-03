/* Copyright 2024 The OpenXLA Authors.
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
#include "zkx/codegen/emitters/elemental_hlo_to_mlir.h"

#include <stddef.h>

#include <iterator>
#include <optional>
#include <queue>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_map.h"
#include "absl/debugging/leak_check.h"
#include "absl/functional/bind_front.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveTypes.h"
#include "xla/tsl/platform/statusor.h"
#include "zkx/codegen/emitters/ir/zkx_ops.h"
#include "zkx/comparison_util.h"
#include "zkx/hlo/analysis/indexing_analysis.h"
#include "zkx/hlo/translate/hlo_to_mhlo/hlo_utils.h"
#include "zkx/mlir/mlir_utils.h"
#include "zkx/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "zkx/mlir_hlo/mhlo/transforms/map_mhlo_to_scalar_op.h"
#include "zkx/primitive_util.h"
#include "zkx/shape_util.h"
#include "zkx/status_macros.h"

namespace zkx::emitters {
namespace {

using llvm::SmallVector;
using llvm::SmallVectorImpl;
using mlir::Block;
using mlir::ImplicitLocOpBuilder;
using mlir::IntegerType;
using mlir::Location;
using mlir::OpBuilder;
using mlir::Value;
using mlir::ValueRange;
using mlir::arith::AndIOp;
using mlir::arith::CmpIOp;
using mlir::arith::CmpIPredicate;
using mlir::arith::ConstantIndexOp;
using mlir::arith::ConstantOp;

namespace arith = ::mlir::arith;
namespace func = ::mlir::func;
namespace mhlo = ::mlir::mhlo;
namespace tensor = ::mlir::tensor;

// HLO opcodes that we never support.
auto& kUnsupportedOps = *absl::IgnoreLeak(new llvm::DenseSet<HloOpcode>{
    HloOpcode::kAddDependency, HloOpcode::kAfterAll, HloOpcode::kAllGather,
    HloOpcode::kAllGatherDone, HloOpcode::kAllGatherStart,
    HloOpcode::kAllReduce, HloOpcode::kAllReduceDone,
    HloOpcode::kAllReduceStart, HloOpcode::kAllToAll, HloOpcode::kAsyncDone,
    HloOpcode::kAsyncStart, HloOpcode::kAsyncUpdate,
    HloOpcode::kCollectivePermute, HloOpcode::kCollectivePermuteDone,
    HloOpcode::kCollectivePermuteStart, HloOpcode::kCopyDone,
    HloOpcode::kCopyStart, HloOpcode::kCustomCall, HloOpcode::kDomain,
    HloOpcode::kDynamicReshape, HloOpcode::kFft, HloOpcode::kFusion,
    HloOpcode::kGetDimensionSize, HloOpcode::kOptimizationBarrier,
    HloOpcode::kInfeed, HloOpcode::kOutfeed, HloOpcode::kPartitionId,
    HloOpcode::kRecv, HloOpcode::kRecvDone, HloOpcode::kReduceScatter,
    HloOpcode::kReplicaId,
    // TODO(chokobole): Uncomment this. Dependency: HloOpcode::kRng...
    //  HloOpcode::kRng,
    //  HloOpcode::kRngBitGenerator,
    //  HloOpcode::kRngGetAndUpdateState,
    HloOpcode::kScatter, HloOpcode::kSend, HloOpcode::kSendDone,
    HloOpcode::kSetDimensionSize, HloOpcode::kSort, HloOpcode::kWhile,
    HloOpcode::kConditional, HloOpcode::kCall});

absl::StatusOr<Value> GetSingleOperandValue(
    const OperandProvider& operand_provider, const HloInstruction* instr,
    int operand_index, ValueRange indices) {
  TF_ASSIGN_OR_RETURN(auto operand,
                      operand_provider(instr, operand_index, indices));
  TF_RET_CHECK(operand.size() == 1) << "Expected operand to be a single value.";
  return operand.front();
}

// For a given instruction, deduces the indices of each parameter that are
// needed for a given output index.
SmallVector<SmallVector<Value, 3>, 2> GetInputIndices(
    const HloInstructionIndexing& indexing, ValueRange output_indices,
    ImplicitLocOpBuilder& b) {
  SmallVector<SmallVector<Value, 3>, 2> indices;
  for (const IndexingMapSet& maps : indexing.indexing_maps) {
    CHECK_EQ(maps.size(), 1);
    CHECK(!maps.begin()->IsUndefined());
    indices.push_back(ApplyIndexing(*maps.begin(), output_indices, {}, b));
  }
  return indices;
}

absl::StatusOr<SmallVector<Value, 1>> EmitParameter(const HloInstruction* instr,
                                                    func::FuncOp this_fn,
                                                    ValueRange indices,
                                                    ImplicitLocOpBuilder& b) {
  Value value = this_fn.getArgument(instr->parameter_number());
  if (mlir::isa<mlir::TensorType>(value.getType())) {
    value = b.create<tensor::ExtractOp>(value, indices);
  } else {
    TF_RET_CHECK(indices.empty());
  }
  return {{value}};
}

template <typename MhloOp, typename... ExtraArgs>
SmallVector<Value, 1> MapHloOp(mlir::Type result_type,
                               llvm::ArrayRef<mlir::Type> arg_types,
                               llvm::ArrayRef<Value> args,
                               llvm::ArrayRef<mlir::NamedAttribute> attributes,
                               ImplicitLocOpBuilder& b,
                               ExtraArgs&&... extra_args) {
  if constexpr (std::is_same_v<MhloOp, mhlo::AddOp> ||
                std::is_same_v<MhloOp, mhlo::SubtractOp> ||
                std::is_same_v<MhloOp, mhlo::MulOp>) {
    // In case of affine points, we convert the result type to Jacobian points.
    // Affine + Affine -> Jacobian
    // Affine - Affine -> Jacobian
    // Affine * ScalarField -> Jacobian
    if (auto affine_type =
            mlir::dyn_cast<mlir::prime_ir::elliptic_curve::AffineType>(
                result_type)) {
      result_type = mlir::prime_ir::elliptic_curve::JacobianType::get(
          b.getContext(), affine_type.getCurve());
    }
  }
  Value result = mhlo::MhloOpToStdScalarOp::mapOpOfType<MhloOp>(
      b.getLoc(), result_type, arg_types,
      typename MhloOp::Adaptor(args, std::forward<ExtraArgs>(extra_args)...),
      attributes, &b);
  if (result.getType().isInteger(1)) {
    result = b.create<arith::ExtUIOp>(b.getI8Type(), result);
  }
  return {result};
}

template <typename MhloOp>
SmallVector<Value, 1> MapElementwiseOp(
    llvm::ArrayRef<mlir::Type> arg_types, llvm::ArrayRef<Value> args,
    ImplicitLocOpBuilder& b,
    llvm::ArrayRef<mlir::NamedAttribute> attributes = std::nullopt) {
  // We use the last argument's type because of select.
  return MapHloOp<MhloOp>(args.back().getType(), arg_types, args, attributes,
                          b);
}

absl::StatusOr<SmallVector<Value, 1>> EmitCompare(
    const HloInstruction* instr, llvm::ArrayRef<mlir::Type> arg_types,
    ValueRange operands, ImplicitLocOpBuilder& builder) {
  auto* context = builder.getContext();
  PrimitiveType operand_type = instr->operand(0)->shape().element_type();
  if (primitive_util::IsExtensionFieldType(operand_type) &&
      instr->comparison_direction() != ComparisonDirection::kEq &&
      instr->comparison_direction() != ComparisonDirection::kNe) {
    return absl::InvalidArgumentError(
        "Extension fields only support EQ and NE comparisons");
  }
  auto direction = mhlo::symbolizeComparisonDirection(
      ComparisonDirectionToString(instr->comparison_direction()));
  mhlo::CompareOp::Properties properties;
  properties.comparison_direction =
      mhlo::ComparisonDirectionAttr::get(context, direction.value());
  SmallVector<Value> args(operands.begin(), operands.end());
  auto result_types = llvm::to_vector(mlir::TypeRange{builder.getI1Type()});
  return {{mhlo::MhloOpToStdScalarOp::mapOpOfType<mhlo::CompareOp>(
      builder.getLoc(), result_types, arg_types,
      mhlo::CompareOp::Adaptor(args, nullptr, properties),
      /*attributes=*/std::nullopt, &builder)}};
}

}  // namespace

SmallVector<Value, 3> ApplyIndexing(const IndexingMap& map_in, ValueRange dims,
                                    ValueRange symbols,
                                    ImplicitLocOpBuilder& b) {
  IndexingMap map = map_in;
  map.ClearConstraints();
  SmallVector<Value, 3> results;
  for (unsigned int i = 0; i < map.GetNumResults(); ++i) {
    SmallVector<Value, 1> result;
    b.createOrFold<ApplyIndexingOp>(result, dims, symbols, map.GetSubMap(i));
    results.append(result);
  }
  return results;
}

namespace {

Value CheckConstraint(Value constrained_value, Interval range,
                      ImplicitLocOpBuilder& b) {
  auto lb = b.create<ConstantOp>(b.getIndexAttr(range.lower));
  if (range.IsPoint()) {
    return b.create<CmpIOp>(CmpIPredicate::eq, constrained_value, lb);
  }
  auto ub = b.create<ConstantOp>(b.getIndexAttr(range.upper));
  return b.create<AndIOp>(
      b.create<CmpIOp>(CmpIPredicate::sge, constrained_value, lb),
      b.create<CmpIOp>(CmpIPredicate::sle, constrained_value, ub));
}

}  // namespace

Value CheckConstraints(const IndexingMap& map, ValueRange dims,
                       ValueRange symbols, ImplicitLocOpBuilder& b) {
  SmallVector<mlir::AffineExpr, 1> expressions;
  for (auto&& [expression, _] : map.GetConstraints()) {
    expressions.push_back(expression);
  }

  // Construct an indexing for the constraints, so we can use `apply_indexing`.
  mlir::AffineMap input_map = map.GetAffineMap();
  IndexingMap constraints_map{
      mlir::AffineMap::get(input_map.getNumDims(), input_map.getNumSymbols(),
                           expressions, input_map.getContext()),
      map.GetDimVars(), map.GetRangeVars(), map.GetRTVars()};
  SmallVector<Value, 1> constraints_values =
      ApplyIndexing(constraints_map, dims, symbols, b);

  Value ret = b.create<ConstantOp>(b.getIntegerAttr(b.getI1Type(), 1));
  for (auto&& [value, expression_and_range] :
       llvm::zip(constraints_values, map.GetConstraints())) {
    ret = b.create<AndIOp>(
        ret, CheckConstraint(value, expression_and_range.second, b));
  }
  for (auto&& [index, bound] : llvm::enumerate(map.GetDimensionBounds())) {
    ret = b.create<AndIOp>(ret, CheckConstraint(dims[index], bound, b));
  }
  return ret;
}

namespace {

absl::StatusOr<SmallVector<Value, 1>> EmitTuple(
    const HloInstruction* instr, ValueRange indices,
    const OperandProvider& operand_provider, ImplicitLocOpBuilder& builder) {
  const Shape* first_shape = &instr->shape().tuple_shapes(0);
  while (first_shape->IsTuple()) {
    first_shape = &first_shape->tuple_shapes(0);
  }
  CHECK_EQ(first_shape->rank(), indices.size())
      << "Indices for tuple must be for the first tuple element";
  SmallVector<Value, 1> operands;
  for (int i = 0; i < instr->operand_count(); ++i) {
    SmallVector<Value> operand_indices;
    // The tuple shapes only need to be bitcast compatible, so insert
    // bitcasts where necessary.
    const HloInstruction* operand = instr->operand(i);
    const Shape* operand_index_shape = &operand->shape();
    while (operand_index_shape->IsTuple()) {
      operand_index_shape = &operand_index_shape->tuple_shapes(0);
    }
    if (i > 0 && !ShapeUtil::EqualIgnoringElementType(*first_shape,
                                                      *operand_index_shape)) {
      IndexingMap operand_map = GetBitcastMap(
          *first_shape, *operand_index_shape, builder.getContext());
      operand_indices = ApplyIndexing(operand_map, indices, {}, builder);
    } else {
      operand_indices = indices;
    }
    TF_ASSIGN_OR_RETURN(auto values,
                        operand_provider(instr, i, operand_indices));
    operands.append(values);
  }
  return operands;
}

absl::StatusOr<SmallVector<Value, 1>> EmitConstant(
    const HloInstruction* instr, ValueRange indices,
    ImplicitLocOpBuilder& builder) {
  mlir::Type result_element_type = mlir_utils::PrimitiveTypeToMlirType(
      instr->shape().element_type(), builder.getContext());
  TF_ASSIGN_OR_RETURN(auto value_attr, CreateDenseElementsAttrFromLiteral(
                                           instr->literal(), builder));
  // Convert the constant element type if needed.
  if (primitive_util::IsUnsignedIntegralType(instr->shape().element_type())) {
    value_attr = value_attr.mapValues(result_element_type,
                                      [](const llvm::APInt& i) { return i; });
  } else if (instr->shape().element_type() == PrimitiveType::PRED) {
    value_attr = value_attr.mapValues(
        result_element_type, [](const llvm::APInt& i) { return i.zext(8); });
  }

  if (ShapeUtil::IsEffectiveScalar(instr->shape())) {
    auto val =
        mlir::cast<mlir::TypedAttr>(value_attr.getValues<mlir::Attribute>()[0]);
    return {{builder.create<ConstantOp>(val).getResult()}};
  }
  auto constant = builder.create<ConstantOp>(value_attr).getResult();
  return {{builder.create<tensor::ExtractOp>(constant, indices)}};
}

absl::StatusOr<SmallVector<Value, 2>> GetOperands(
    const HloInstruction* instr, ValueRange indices,
    const OperandProvider& operand_provider, ImplicitLocOpBuilder& builder) {
  SmallVector<Value, 2> operands;
  bool is_elementwise = HloInstruction::IsOpElementwise(instr->opcode()) ||
                        instr->opcode() == HloOpcode::kMap;
  if (is_elementwise && instr->shape().IsArray()) {
    // Check if the instruction is really elementwise. There may be some
    // broadcasting.
    int64_t rank = instr->shape().rank();
    is_elementwise &=
        absl::c_all_of(instr->operands(), [&](const HloInstruction* operand) {
          return operand->shape().rank() == rank;
        });
  }

  if (is_elementwise) {
    // Avoid materializing the input indices for elementwise ops.
    for (int64_t operand_number = 0; operand_number < instr->operand_count();
         ++operand_number) {
      TF_ASSIGN_OR_RETURN(operands.emplace_back(),
                          GetSingleOperandValue(operand_provider, instr,
                                                operand_number, indices));
    }
  } else {
    auto input_indices = GetInputIndices(
        ComputeOutputToInputIndexing(instr, 0, builder.getContext()), indices,
        builder);
    for (auto&& [operand_number, operand_indices] :
         llvm::enumerate(input_indices)) {
      TF_ASSIGN_OR_RETURN(
          operands.emplace_back(),
          GetSingleOperandValue(operand_provider, instr, operand_number,
                                operand_indices));
    }
  }
  CHECK_NE(operands.size(), 0);
  for (auto [index, operand] : llvm::enumerate(operands)) {
    // Nulls can be pretty hard to debug, so guard against them here. The MHLO
    // conversion functions like to return nullptr for errors.
    TF_RET_CHECK(operand != nullptr) << "null operand at index " << index
                                     << " for " << instr->ToShortString();
  }
  return operands;
}

absl::StatusOr<SmallVector<Value, 1>> EmitConvert(
    const HloInstruction* instr, llvm::ArrayRef<mlir::Type> arg_types,
    ValueRange operands, ImplicitLocOpBuilder& builder) {
  PrimitiveType element_type = instr->shape().element_type();
  auto result_type_with_sign = mlir_utils::PrimitiveTypeToMlirTypeWithSign(
      instr->shape().element_type(), builder.getContext());
  auto result_element_type = mlir_utils::PrimitiveTypeToMlirType(
      instr->shape().element_type(), builder.getContext());
  if (element_type == PRED) {
    if (mlir::isa<IntegerType>(operands[0].getType())) {
      Value i1 = builder.create<arith::CmpIOp>(
          arith::CmpIPredicate::ne, operands[0],
          builder.create<arith::ConstantIntOp>(operands[0].getType(), 0));
      return {{builder.create<arith::ExtUIOp>(builder.getI8Type(), i1)
                   .getResult()}};
    }
  }
  Value out = mhlo::MhloOpToStdScalarOp::mapConvertOpToStdScalarOp(
      builder.getLoc(), result_type_with_sign, result_element_type, arg_types,
      operands, /*attributes=*/std::nullopt, &builder);
  return {{out}};
}

absl::StatusOr<SmallVector<Value, 1>> HloToMlir(
    const HloInstruction* instr, func::FuncOp this_fn, ValueRange indices,
    const OperandProvider& operand_provider,
    const CallTargetProvider& call_target_provider,
    ImplicitLocOpBuilder& builder) {
  CHECK(!kUnsupportedOps.contains(instr->opcode())) << instr->ToShortString();

  PrimitiveType element_type = instr->shape().element_type();

  // Handle ops that aren't elementwise and aren't just indexing
  // transformations.
  switch (instr->opcode()) {
    case HloOpcode::kConcatenate:
      // TODO(chokobole): Implement this. Dependency: EmitConcat
      // return EmitConcat(instr, indices, operand_provider, builder);
      return absl::UnimplementedError(
          "HloToMlir not implemented for HloOpcode::kConcatenate");
    case HloOpcode::kConstant:
      return EmitConstant(instr, indices, builder);
    case HloOpcode::kDynamicSlice:
      // TODO(chokobole): Implement this. Dependency: EmitDynamicSlice
      // return EmitDynamicSlice(instr, indices, operand_provider, builder);
      return absl::UnimplementedError(
          "HloToMlir not implemented for HloOpcode::kDynamicSlice");
    case HloOpcode::kDynamicUpdateSlice:
      // TODO(chokobole): Implement this. Dependency: EmitDynamicUpdateSlice
      // return EmitDynamicUpdateSlice(instr, indices, operand_provider,
      // builder);
      return absl::UnimplementedError(
          "HloToMlir not implemented for HloOpcode::kDynamicUpdateSlice");
    case HloOpcode::kGather:
      // clang-format off
      // TODO(chokobole): Implement this. Dependency: HloInstruction::gather_slice_sizes
      // clang-format on
      return absl::UnimplementedError(
          "HloToMlir not implemented for HloOpcode::kGather");
    case HloOpcode::kIota:
      // TODO(chokobole): Uncomment this. Dependency: EmitIota
      // return EmitIota(instr, indices, builder);
      return absl::UnimplementedError(
          "HloToMlir not implemented for HloOpcode::kIota");
    case HloOpcode::kPad:
      // TODO(chokobole): Uncomment this. Dependency: EmitPad
      // return EmitPad(instr, indices, operand_provider, builder);
      return absl::UnimplementedError(
          "HloToMlir not implemented for HloOpcode::kPad");
    case HloOpcode::kDot:
      // TODO(chokobole): Implement this. Dependency: EmitDot
      // return EmitDot(instr, indices, operand_provider, builder);
      return absl::UnimplementedError(
          "HloToMlir not implemented for HloOpcode::kDot");
    case HloOpcode::kParameter:
      return EmitParameter(instr, this_fn, indices, builder);
    case HloOpcode::kReduce:
      // TODO(chokobole): Implement this. Dependency: EmitReduce
      // return EmitReduce(instr, indices, operand_provider,
      // call_target_provider,
      //                   builder);
      return absl::UnimplementedError(
          "HloToMlir not implemented for HloOpcode::kReduce");
    case HloOpcode::kTuple:
      return EmitTuple(instr, indices, operand_provider, builder);
    case HloOpcode::kGetTupleElement: {
      // We have to generate the entire tuple, but since we don't support
      // internal tuple operations (only root tuples), this will always be
      // cached and computed together anyway (e.g. it'll be a variadic
      // reduce).
      TF_ASSIGN_OR_RETURN(auto tuple, operand_provider(instr, 0, indices));
      return {{tuple[instr->tuple_index()]}};
    }
    default:
      break;
  }

  SmallVector<mlir::Type, 2> arg_types;
  arg_types.reserve(instr->operands().size());
  for (auto operand : instr->operands()) {
    auto operand_element_type = mlir_utils::PrimitiveTypeToMlirTypeWithSign(
        operand->shape().element_type(), builder.getContext());
    arg_types.push_back(operand_element_type);
  }

  TF_ASSIGN_OR_RETURN(auto operands,
                      GetOperands(instr, indices, operand_provider, builder));

  SmallVector<mlir::NamedAttribute> attributes;
  switch (instr->opcode()) {
    case HloOpcode::kAbs:
      // TODO(chokobole): Uncomment this. Dependency: mhlo::AbsOp
      // return {MapHloOp<mhlo::AbsOp>(
      //     PrimitiveTypeToMlirType(element_type, builder), arg_types,
      //     operands,
      //     /*attributes=*/std::nullopt, builder)};
      return absl::UnimplementedError(
          "HloToMlir not implemented for HloOpcode::kAbs");
    case HloOpcode::kAdd:
      if (element_type == PRED) {
        // TODO(chokobole): Uncomment this. Dependency: mhlo::OrOp
        // return MapElementwiseOp<mhlo::OrOp>(arg_types, operands, builder);
        return absl::UnimplementedError(
            "HloToMlir not implemented for HloOpcode::kAdd with PRED element "
            "type");
      }
      return MapElementwiseOp<mhlo::AddOp>(arg_types, operands, builder);
    case HloOpcode::kAnd:
      // TODO(chokobole): Uncomment this. Dependency: mhlo::AndOp
      // return MapElementwiseOp<mhlo::AndOp>(arg_types, operands, builder);
      return absl::UnimplementedError(
          "HloToMlir not implemented for HloOpcode::kAnd");
    case HloOpcode::kClamp:
      // TODO(chokobole): Uncomment this. Dependency: mhlo::ClampOp
      // return MapElementwiseOp<mhlo::ClampOp>(arg_types, operands, builder);
      return absl::UnimplementedError(
          "HloToMlir not implemented for HloOpcode::kClamp");
    case HloOpcode::kClz:
      // TODO(chokobole): Uncomment this. Dependency: mhlo::ClzOp
      // return MapElementwiseOp<mhlo::ClzOp>(arg_types, operands, builder);
      return absl::UnimplementedError(
          "HloToMlir not implemented for HloOpcode::kClz");
    case HloOpcode::kCompare:
      return EmitCompare(instr, arg_types, operands, builder);
    case HloOpcode::kDivide:
      return MapElementwiseOp<mhlo::DivOp>(arg_types, operands, builder);
    case HloOpcode::kMap: {
      auto mapper = call_target_provider(
          instr->called_computations().front()->root_instruction());
      return builder.create<PureCallOp>(mapper, operands).getResults();
    }
    case HloOpcode::kMaximum:
      if (element_type == PRED) {
        // TODO(chokobole): Uncomment this. Dependency: mhlo::OrOp
        // return MapElementwiseOp<mhlo::OrOp>(arg_types, operands, builder);
        return absl::UnimplementedError(
            "HloToMlir not implemented for HloOpcode::kMaximum with PRED "
            "element type");
      }
      // TODO(chokobole): Uncomment this. Dependency: mhlo::MaxOp
      // return MapElementwiseOp<mhlo::MaxOp>(arg_types, operands, builder);
      return absl::UnimplementedError(
          "HloToMlir not implemented for HloOpcode::kMaximum");
    case HloOpcode::kMinimum:
      if (element_type == PRED) {
        // TODO(chokobole): Uncomment this. Dependency: mhlo::AndOp
        // return MapElementwiseOp<mhlo::AndOp>(arg_types, operands, builder);
        return absl::UnimplementedError(
            "HloToMlir not implemented for HloOpcode::kMinimum with PRED "
            "element type");
      }
      // TODO(chokobole): Uncomment this. Dependency: mhlo::MinOp
      // return MapElementwiseOp<mhlo::MinOp>(arg_types, operands, builder);
      return absl::UnimplementedError(
          "HloToMlir not implemented for HloOpcode::kMinimum");
    case HloOpcode::kMultiply:
      if (element_type == PRED) {
        // TODO(chokobole): Uncomment this. Dependency: mhlo::AndOp
        // return MapElementwiseOp<mhlo::AndOp>(arg_types, operands, builder);
        return absl::UnimplementedError(
            "HloToMlir not implemented for HloOpcode::kMultiply with PRED "
            "element type");
      }
      return MapElementwiseOp<mhlo::MulOp>(arg_types, operands, builder);
    case HloOpcode::kNegate:
      return MapElementwiseOp<mhlo::NegOp>(arg_types, operands, builder);
    case HloOpcode::kNot: {
      if (element_type == PRED) {
        auto zero =
            builder.create<arith::ConstantIntOp>(builder.getI8Type(), 0);
        Value result = builder.create<arith::ExtUIOp>(
            builder.getI8Type(),
            builder.create<arith::CmpIOp>(arith::CmpIPredicate::eq, operands[0],
                                          zero));
        return {{result}};
      }
      // TODO(chokobole): Uncomment this. Dependency: mhlo::NotOp
      // return MapElementwiseOp<mhlo::NotOp>(arg_types, operands, builder);
      return absl::UnimplementedError(
          "HloToMlir not implemented for HloOpcode::kNot");
    }
    case HloOpcode::kOr:
      // TODO(chokobole): Uncomment this. Dependency: mhlo::OrOp
      // return MapElementwiseOp<mhlo::OrOp>(arg_types, operands, builder);
      return absl::UnimplementedError(
          "HloToMlir not implemented for HloOpcode::kOr");
    case HloOpcode::kPopulationCount:
      // TODO(chokobole): Uncomment this. Dependency: mhlo::PopulationCountOp
      // return MapHloOp<mhlo::PopulationCountOp>(
      //     PrimitiveTypeToMlirType(element_type, builder), arg_types,
      //     operands,
      //     /*attributes=*/std::nullopt, builder);
      return absl::UnimplementedError(
          "HloToMlir not implemented for HloOpcode::kPopulationCount");
    case HloOpcode::kPower:
      return MapElementwiseOp<mhlo::PowOp>(arg_types, operands, builder);
    case HloOpcode::kRemainder:
      // TODO(chokobole): Uncomment this. Dependency: mhlo::RemOp
      // return MapElementwiseOp<mhlo::RemOp>(arg_types, operands, builder);
      return absl::UnimplementedError(
          "HloToMlir not implemented for HloOpcode::kRemainder");
    case HloOpcode::kSelect: {
      operands[0] = builder.createOrFold<arith::TruncIOp>(builder.getI1Type(),
                                                          operands[0]);
      return MapElementwiseOp<mhlo::SelectOp>(arg_types, operands, builder);
    }
    case HloOpcode::kShiftLeft:
      // TODO(chokobole): Uncomment this. Dependency: mhlo::ShiftLeftOp
      // return MapElementwiseOp<mhlo::ShiftLeftOp>(arg_types, operands,
      // builder);
      return absl::UnimplementedError(
          "HloToMlir not implemented for HloOpcode::kShiftLeft");
    case HloOpcode::kShiftRightArithmetic:
      // clang-format off
      // TODO(chokobole): Uncomment this. Dependency: mhlo::ShiftRightArithmeticOp
      // clang-format on
      // return MapElementwiseOp<mhlo::ShiftRightArithmeticOp>(arg_types,
      // operands,
      //                                                       builder);
      return absl::UnimplementedError(
          "HloToMlir not implemented for HloOpcode::kShiftRightArithmetic");
    case HloOpcode::kShiftRightLogical:
      // TODO(chokobole): Uncomment this. Dependency: mhlo::ShiftRightLogicalOp
      // return MapElementwiseOp<mhlo::ShiftRightLogicalOp>(arg_types, operands,
      //                                                    builder);
      return absl::UnimplementedError(
          "HloToMlir not implemented for HloOpcode::kShiftRightLogical");
    case HloOpcode::kSign:
      // TODO(chokobole): Uncomment this. Dependency: mhlo::SignOp
      // return MapElementwiseOp<mhlo::SignOp>(arg_types, operands, builder);
      return absl::UnimplementedError(
          "HloToMlir not implemented for HloOpcode::kSign");
    case HloOpcode::kSubtract:
      return MapElementwiseOp<mhlo::SubtractOp>(arg_types, operands, builder);
    case HloOpcode::kXor:
      // TODO(chokobole): Uncomment this. Dependency: mhlo::XorOp
      // return MapElementwiseOp<mhlo::XorOp>(arg_types, operands, builder);
      return absl::UnimplementedError(
          "HloToMlir not implemented for HloOpcode::kXor");
    case HloOpcode::kBitcastConvert:
      // TODO(chokobole): Uncomment this. Dependency: mhlo::BitcastConvertOp
      // return MapHloOp<mhlo::BitcastConvertOp>(
      //     mlir_utils::PrimitiveTypeToMlirType(element_type,
      //                                         builder.getContext()),
      //     arg_types, operands,
      //     /*attributes=*/std::nullopt, builder);
      return absl::UnimplementedError(
          "HloToMlir not implemented for HloOpcode::kBitcastConvert");
    case HloOpcode::kConvert:
      return EmitConvert(instr, arg_types, operands, builder);
    case HloOpcode::kBitcast:
    case HloOpcode::kCopy:
    case HloOpcode::kSlice:
    case HloOpcode::kBroadcast:
    case HloOpcode::kReshape:
    case HloOpcode::kReverse:
    case HloOpcode::kTranspose:
      return operands;
    default:
      break;
  }

  return absl::UnimplementedError(absl::StrCat("Unsupported: ", instr->name()));
}

}  // namespace

ValueRange ProvideParameter(const PartitionedComputation& computation,
                            const HloInstruction* instr, int operand_index,
                            ValueRange indices,
                            const CallTargetProvider& call_target_provider,
                            func::FuncOp this_fn, ImplicitLocOpBuilder& builder,
                            const PartitionedComputation::Subgraph* caller) {
  const HloInstruction* operand = instr->operand(operand_index);

  if (!caller) {
    caller = &computation.FindSubgraph(instr);
  }
  const absl::flat_hash_map<const HloInstruction*, int>& injected_value_starts =
      caller->injected_value_starts;
  if (auto it = injected_value_starts.find(operand);
      it != injected_value_starts.end()) {
    return ValueRange(this_fn.getArguments())
        .take_back(caller->num_injected_values)
        .slice(it->second, 1);
  }

  func::FuncOp callee = call_target_provider(operand);
  SmallVector<Value> operands;
  if (std::optional<BackendKind> backend_kind = GetBackendKind(this_fn);
      backend_kind == BackendKind::kCpu && this_fn->getAttr("zkx.entry")) {
    operands =
        SmallVector<Value>{this_fn.getArguments().drop_front().take_front(
            instr->parent()->num_parameters())};
  } else {
    operands = SmallVector<Value>{
        this_fn.getArguments().take_front(instr->parent()->num_parameters())};
  }
  absl::c_copy(indices, std::back_inserter(operands));
  auto results = builder.create<PureCallOp>(callee, operands).getResults();
  const PartitionedComputation::Subgraph& callee_subgraph =
      computation.FindSubgraph(operand);
  if (callee_subgraph.roots.size() == 1) {
    CHECK_EQ(callee_subgraph.roots.front(), operand)
        << "Expected " << operand->ToString() << " to be the root of "
        << callee_subgraph.ToString();
    return results;
  }

  int offset = 0;
  for (const HloInstruction* root : callee_subgraph.roots) {
    int root_arity =
        root->shape().IsTuple() ? root->shape().tuple_shapes_size() : 1;
    if (root == operand) {
      return results.slice(offset, root_arity);
    }
    offset += root_arity;
  }
  LOG(FATAL) << "Did not find operand " << operand->name() << " in roots of "
             << callee_subgraph.ToString();
}

SmallVector<Value, 2> ProvideParameterRange(
    const PartitionedComputation& computation, const HloInstruction* instr,
    int start, int num, ValueRange indices,
    const CallTargetProvider& call_target_provider, func::FuncOp this_fn,
    ImplicitLocOpBuilder& builder) {
  SmallVector<Value, 2> scalars;
  scalars.reserve(num);
  for (int i = 0; i < num; ++i) {
    ValueRange parameter_value =
        ProvideParameter(computation, instr, i + start, indices,
                         call_target_provider, this_fn, builder);
    scalars.append(parameter_value.begin(), parameter_value.end());
  }
  return scalars;
}

namespace {

class SubgraphConverter {
 public:
  SubgraphConverter(const PartitionedComputation& computation,
                    const PartitionedComputation::Subgraph& subgraph,
                    func::FuncOp this_fn,
                    const CallTargetProvider& call_target_provider,
                    ValueRange parameters, ValueRange indices,
                    ImplicitLocOpBuilder& builder)
      : computation_(computation),
        subgraph_(subgraph),
        this_fn_(this_fn),
        call_target_provider_(call_target_provider),
        parameters_(parameters),
        indices_(indices),
        builder_(builder),
        provide_operand_fn_(
            absl::bind_front(&SubgraphConverter::ProvideOperand, this)) {}

  absl::StatusOr<SmallVector<Value>> Convert();
  absl::StatusOr<SmallVector<Value>> ProvideOperand(const HloInstruction* instr,
                                                    int index,
                                                    ValueRange operand_indices);
  absl::StatusOr<SmallVector<Value>> EmitInstruction(
      const HloInstruction* instr, ValueRange indices);
  absl::StatusOr<SmallVector<Value>> EmitElementwiseInstruction(
      const HloInstruction* root, ValueRange indices);

 private:
  const PartitionedComputation& computation_;
  const PartitionedComputation::Subgraph& subgraph_;
  func::FuncOp this_fn_;
  const CallTargetProvider& call_target_provider_;
  ValueRange parameters_;
  ValueRange indices_;
  ImplicitLocOpBuilder& builder_;
  absl::node_hash_map<std::pair<const HloInstruction*, std::vector<void*>>,
                      SmallVector<Value>>
      cached_instructions_;
  OperandProvider provide_operand_fn_;
};

absl::StatusOr<SmallVector<Value>> SubgraphConverter::Convert() {
  SmallVector<Value> results;
  TF_RET_CHECK(subgraph_.roots.size() == subgraph_.root_indexing.size())
      << "roots and root_indexing must have the same size in "
      << subgraph_.ToString();
  for (const auto [root, indexing] :
       llvm::zip(subgraph_.roots, subgraph_.root_indexing)) {
    if (auto it = subgraph_.injected_value_starts.find(root);
        it != subgraph_.injected_value_starts.end()) {
      auto injected =
          this_fn_.getArguments().take_back(subgraph_.num_injected_values);
      int arity =
          root->shape().IsTuple() ? root->shape().tuple_shapes_size() : 1;
      absl::c_copy(injected.slice(it->second, arity),
                   std::back_inserter(results));
      continue;
    }
    int num_dims = indexing.GetAffineMap().getNumDims();
    auto root_indices =
        ApplyIndexing(indexing, /*dims=*/indices_.take_front(num_dims),
                      /*symbols=*/indices_.drop_front(num_dims), builder_);
    TF_ASSIGN_OR_RETURN(auto root_results, EmitInstruction(root, root_indices));
    results.append(root_results.begin(), root_results.end());
  }
  return results;
}

absl::StatusOr<SmallVector<Value>> SubgraphConverter::ProvideOperand(
    const HloInstruction* instr, int index, ValueRange operand_indices) {
  const HloInstruction* operand = instr->operand(index);
  if (subgraph_.instructions.contains(operand)) {
    return EmitInstruction(operand, operand_indices);
  }
  return ProvideParameter(computation_, instr, index, operand_indices,
                          call_target_provider_, this_fn_, builder_,
                          &subgraph_);
}

absl::StatusOr<SmallVector<Value>> SubgraphConverter::EmitInstruction(
    const HloInstruction* instr, ValueRange indices) {
  std::vector<void*> indices_ptrs;
  indices_ptrs.reserve(indices.size());
  for (Value index : indices) {
    indices_ptrs.push_back(index.getAsOpaquePointer());
  }
  SmallVector<Value>& entry =
      cached_instructions_[std::make_pair(instr, indices_ptrs)];
  // Only use the entry if its parent block is still in scope. Note that this
  // should always be the case normally - if not, we risk exponential code
  // size.
  // TODO(jreiffers): Remove this check / turn it into a failure.
  if (!entry.empty()) {
    Block* entry_block = entry.front().getParentBlock();
    Block* insertion_block = builder_.getInsertionBlock();
    while (insertion_block != nullptr) {
      if (insertion_block == entry_block) return entry;
      if (insertion_block->getParentOp()) {
        insertion_block = insertion_block->getParentOp()->getBlock();
      } else {
        insertion_block = nullptr;
        VLOG(2) << "Failed dominance check while looking up cache for "
                << instr->ToShortString()
                << ". This is a bug in the computation partitioner.";
      }
    }
  }

  if (HloInstruction::IsOpElementwise(instr->opcode())) {
    return EmitElementwiseInstruction(instr, indices);
  }

  TF_ASSIGN_OR_RETURN(entry,
                      HloToMlir(instr, this_fn_, indices, provide_operand_fn_,
                                call_target_provider_, builder_));
  CHECK(!absl::c_linear_search(entry, nullptr))
      << "Failed to lower " << instr->name();
  return entry;
}

absl::StatusOr<SmallVector<Value>>
SubgraphConverter::EmitElementwiseInstruction(const HloInstruction* root,
                                              ValueRange indices) {
  // `root` is elementwise, so we can emit its operands first (recursively).
  // This reduces the size of the call stack.
  std::vector<void*> indices_ptrs;
  indices_ptrs.reserve(indices.size());
  for (auto index : indices) {
    indices_ptrs.push_back(index.getAsOpaquePointer());
  }

  std::queue<const HloInstruction*> worklist;
  absl::flat_hash_set<const HloInstruction*> visited;
  worklist.push(root);
  SmallVector<const HloInstruction*> pre_order;
  while (!worklist.empty()) {
    const HloInstruction* instr = worklist.front();
    worklist.pop();
    pre_order.push_back(instr);
    if (HloInstruction::IsOpElementwise(instr->opcode())) {
      // Start with the last operand so that we will instantiate the operands
      // in order below. Not needed for correctness, but makes the generated IR
      // more readable.
      for (int i = instr->operand_count() - 1; i >= 0; --i) {
        auto* operand = instr->operand(i);
        if (subgraph_.instructions.contains(operand) &&
            !cached_instructions_.contains({operand, indices_ptrs}) &&
            visited.insert(operand).second) {
          worklist.push(operand);
        }
      }
    }
  }

  for (auto* instr : llvm::reverse(pre_order)) {
    auto& entry = cached_instructions_[{instr, indices_ptrs}];
    TF_ASSIGN_OR_RETURN(entry,
                        HloToMlir(instr, this_fn_, indices, provide_operand_fn_,
                                  call_target_provider_, builder_));
  }
  return cached_instructions_[{root, indices_ptrs}];
}

absl::StatusOr<SmallVector<Value>> SubgraphToMlir(
    const PartitionedComputation& computation,
    const PartitionedComputation::Subgraph& subgraph, func::FuncOp this_fn,
    const CallTargetProvider& call_target_provider, ValueRange parameters,
    ValueRange indices, ImplicitLocOpBuilder& builder) {
  return SubgraphConverter(computation, subgraph, this_fn, call_target_provider,
                           parameters, indices, builder)
      .Convert();
}

}  // namespace

void GetLoopBoundsFromIndexingMap(ImplicitLocOpBuilder& b,
                                  const IndexingMap& indexing_map,
                                  SmallVectorImpl<Value>& lbs,
                                  SmallVectorImpl<Value>& ubs,
                                  SmallVectorImpl<Value>& steps) {
  Value c1 = b.create<ConstantIndexOp>(1);

  for (const Interval& bound : indexing_map.GetSymbolBounds()) {
    lbs.push_back(b.create<ConstantIndexOp>(bound.lower));
    ubs.push_back(b.create<ConstantIndexOp>(bound.upper + 1));
    steps.push_back(c1);
  }
}

absl::Status SubgraphToMlirFunction(
    const PartitionedComputation& computation,
    const PartitionedComputation::Subgraph& subgraph, func::FuncOp& func,
    const CallTargetProvider& call_target_provider) {
  TF_RET_CHECK(func != nullptr);
  ImplicitLocOpBuilder builder(func.getLoc(), func->getContext());
  builder.setInsertionPointToStart(func.addEntryBlock());
  auto parameters = func.getArguments().take_front(
      computation.computation().num_parameters());
  auto indices_and_injected_values = func.getArguments().drop_front(
      computation.computation().num_parameters());
  auto indices =
      indices_and_injected_values.drop_back(subgraph.num_injected_values);
  TF_ASSIGN_OR_RETURN(
      auto results,
      SubgraphToMlir(computation, subgraph, func, call_target_provider,
                     parameters, indices, builder));
  CHECK_EQ(results.size(), func.getResultTypes().size());

  for (auto [i, result] : llvm::enumerate(results)) {
    mlir::Type expected_type = func.getResultTypes()[i];
    if (result.getType().isInteger(1) && expected_type.isInteger(8)) {
      results[i] = builder.create<arith::ExtUIOp>(builder.getI8Type(), result);
    }
  }

  builder.create<func::ReturnOp>(results);
  return absl::OkStatus();
}

ValueRange EmitZkxLoopOp(
    ImplicitLocOpBuilder& b, ValueRange dim_values, ValueRange iter_args_inits,
    const IndexingMap& indexing_map,
    mlir::function_ref<SmallVector<Value>(
        ImplicitLocOpBuilder& nested_b, ValueRange /*ivs*/,
        ValueRange /*map_results*/, ValueRange /*iter_args*/)>
        create_body,
    bool vectorize) {
  SmallVector<Value, 4> vector_inits;
  if (vectorize) {
    CHECK_EQ(indexing_map.GetSymbolBounds().back().lower, 0);
    int vector_size = indexing_map.GetSymbolBounds().back().upper + 1;
    vector_inits = iter_args_inits;
    for (auto& init : vector_inits) {
      if (!mlir::isa<mlir::ShapedType>(init.getType())) {
        auto vector_ty = mlir::VectorType::get({vector_size}, init.getType());
        init = b.create<mlir::vector::SplatOp>(vector_ty, init);
      }
    }
    iter_args_inits = vector_inits;
  }
  auto bb = [&](OpBuilder& nested_builder, Location loc, ValueRange ivs,
                ValueRange map_results, ValueRange iter_args) {
    ImplicitLocOpBuilder nested_b(loc, nested_builder);
    SmallVector<Value, 4> results;
    if (vectorize) {
      SmallVector<Value, 4> vector_args;
      vector_args = iter_args;
      // Extract the vector elements.
      for (auto& init : vector_args) {
        if (mlir::isa<mlir::VectorType>(init.getType())) {
          init = nested_b.create<mlir::vector::ExtractOp>(init, ivs.back());
        }
      }
      results = create_body(nested_b, ivs, map_results, vector_args);
      // Insert the results.
      for (auto [index, init] : llvm::enumerate(iter_args)) {
        if (mlir::isa<mlir::VectorType>(init.getType())) {
          results[index] = nested_builder.create<mlir::vector::InsertOp>(
              loc, results[index], iter_args[index], ivs.back());
        }
      }
    } else {
      results = create_body(nested_b, ivs, map_results, iter_args);
    }
    nested_b.create<zkx::YieldOp>(results);
  };
  return b.create<LoopOp>(indexing_map, dim_values, iter_args_inits, bb)
      .getResults();
}

Value ClampIndex(Value index, bool is_unsigned, int64_t high,
                 ImplicitLocOpBuilder& b) {
  auto zero = b.create<ConstantOp>(b.getIndexAttr(0));
  if (high <= 0) {
    return zero;
  }

  if (is_unsigned) {
    if (index.getType() != b.getIndexType()) {
      index = b.create<arith::IndexCastUIOp>(b.getIndexType(), index);
    }
    index = b.create<arith::MinUIOp>(
        index, b.create<ConstantOp>(b.getIndexAttr(high)));
  } else {
    if (index.getType() != b.getIndexType()) {
      index = b.create<arith::IndexCastOp>(b.getIndexType(), index);
    }
    index = b.create<arith::MinSIOp>(
        index, b.create<ConstantOp>(b.getIndexAttr(high)));
    index = b.create<arith::MaxSIOp>(index, zero);
  }
  return index;
}

}  // namespace zkx::emitters
