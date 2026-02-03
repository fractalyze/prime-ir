/* Copyright 2018 The OpenXLA Authors.
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

#include "zkx/hlo/builder/zkx_builder.h"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iterator>
#include <numeric>
#include <queue>
#include <set>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/types/span.h"

#include "xla/tsl/platform/statusor.h"
#include "zkx/hlo/ir/hlo_sharding.h"
#include "zkx/permutation_util.h"
#include "zkx/service/hlo.pb.h"
#include "zkx/service/shape_inference.h"
#include "zkx/status_macros.h"
#include "zkx/util.h"

namespace zkx {
namespace {

const char kNameSeparator = '.';

// Retrieves the base name of an instruction or computation fully qualified
// name, using separator as boundary between the initial base name part, and
// the numeric identification.
std::string GetBaseName(const std::string& name, char separator) {
  auto pos = name.rfind(separator);
  CHECK_NE(pos, std::string::npos) << name;
  return name.substr(0, pos);
}

// Generates a fully qualified computation/instruction name.
std::string GetFullName(const std::string& base_name, char separator,
                        int64_t id) {
  const char separator_str[] = {separator, '\0'};
  return absl::StrCat(base_name, separator_str, id);
}

// Common function to standardize setting name and IDs on computation and
// instruction proto entities.
template <typename T>
void SetProtoIdAndName(T* entry, const std::string& base_name, char separator,
                       int64_t id) {
  entry->set_id(id);
  entry->set_name(GetFullName(base_name, separator, id));
}

bool InstrIsSetBound(const HloInstructionProto* instr_proto) {
  // TODO(chokobole): Uncomment this. Dependency: custom_call_target
  // HloOpcode opcode = StringToHloOpcode(instr_proto->opcode()).value();
  // if (opcode == HloOpcode::kCustomCall &&
  //     instr_proto->custom_call_target() == "SetBound") {
  //   return true;
  // }
  return false;
}

absl::Status NormalizeAndAssignSharing(HloInstructionProto* instr,
                                       const OpSharding& op_sharding) {
  // Normalize tuple sharding and fail the call if the sharding is invalid.
  Shape shape(instr->shape());
  TF_ASSIGN_OR_RETURN(HloSharding sharding,
                      HloSharding::FromProto(op_sharding));
  sharding = sharding.NormalizeTupleSharding(shape);
  TF_RETURN_IF_ERROR(sharding.Validate(shape));
  *instr->mutable_sharding() = sharding.ToProto();
  return absl::OkStatus();
}

}  // namespace

namespace internal {

ZkxOp ZkxBuilderFriend::BuildBitcast(ZkxBuilder* builder, ZkxOp operand,
                                     const Shape& shape) {
  return builder->ReportErrorOrReturn([&]() -> absl::StatusOr<ZkxOp> {
    HloInstructionProto instr;
    *instr.mutable_shape() = shape.ToProto();
    return builder->AddInstruction(std::move(instr), HloOpcode::kBitcast,
                                   {operand});
  });
}

HloInstructionProto* ZkxBuilderFriend::GetInstruction(ZkxOp op) {
  return &op.builder()
              ->instructions_[op.builder()->handle_to_index_[op.handle_]];
}

HloInstructionProto* ZkxBuilderFriend::GetInstructionByHandle(
    ZkxBuilder* builder, int64_t handle) {
  return &builder->instructions_[builder->handle_to_index_[handle]];
}

}  // namespace internal

ZkxOp operator-(ZkxOp x) { return Neg(x); }
ZkxOp operator+(ZkxOp x, ZkxOp y) { return Add(x, y); }
ZkxOp operator-(ZkxOp x, ZkxOp y) { return Sub(x, y); }
ZkxOp operator*(ZkxOp x, ZkxOp y) { return Mul(x, y); }
ZkxOp operator/(ZkxOp x, ZkxOp y) { return Div(x, y); }
ZkxOp operator%(ZkxOp x, ZkxOp y) { return Rem(x, y); }

ZkxOp operator~(ZkxOp x) { return Not(x); }
ZkxOp operator&(ZkxOp x, ZkxOp y) { return And(x, y); }
ZkxOp operator|(ZkxOp x, ZkxOp y) { return Or(x, y); }
ZkxOp operator^(ZkxOp x, ZkxOp y) { return Xor(x, y); }
ZkxOp operator<<(ZkxOp x, ZkxOp y) { return ShiftLeft(x, y); }

ZkxOp operator>>(ZkxOp x, ZkxOp y) {
  ZkxBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> absl::StatusOr<ZkxOp> {
    TF_ASSIGN_OR_RETURN(const Shape* shape, builder->GetShapePtr(x));
    if (!ShapeUtil::ElementIsIntegral(*shape)) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Argument to >> operator does not have an integral type (%s).",
          ShapeUtil::HumanString(*shape)));
    }
    if (ShapeUtil::ElementIsSigned(*shape)) {
      return ShiftRightArithmetic(x, y);
    } else {
      return ShiftRightLogical(x, y);
    }
  });
}

absl::StatusOr<const Shape*> ZkxBuilder::GetShapePtr(ZkxOp op) const {
  TF_RETURN_IF_ERROR(first_error_);
  TF_RETURN_IF_ERROR(CheckOpBuilder(op));
  auto it = handle_to_index_.find(op.handle());
  if (it == handle_to_index_.end()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("No ZkxOp with handle %d", op.handle()));
  }
  return instruction_shapes_.at(it->second).get();
}

absl::StatusOr<Shape> ZkxBuilder::GetShape(ZkxOp op) const {
  TF_ASSIGN_OR_RETURN(const Shape* shape, GetShapePtr(op));
  return *shape;
}

absl::StatusOr<std::vector<Shape>> ZkxBuilder::GetOperandShapes(
    absl::Span<const ZkxOp> operands) const {
  std::vector<Shape> operand_shapes;
  operand_shapes.reserve(operands.size());
  for (ZkxOp operand : operands) {
    TF_ASSIGN_OR_RETURN(const Shape* shape, GetShapePtr(operand));
    operand_shapes.push_back(*shape);
  }
  return operand_shapes;
}

absl::StatusOr<std::optional<OpSharding>> ZkxBuilder::GetOpSharding(
    ZkxOp op) const {
  TF_ASSIGN_OR_RETURN(auto instr_proto, LookUpInstruction(op));
  if (instr_proto->has_sharding()) {
    return instr_proto->sharding();
  }
  return std::nullopt;
}

std::string ZkxBuilder::OpToString(ZkxOp op) const {
  std::string s;
  ToStringHelper(&s, /*ident=*/0, op.handle());
  return s;
}

static std::string ShapeToString(const ShapeProto& shape) {
  if (shape.tuple_shapes_size() > 1) {
    return absl::StrCat(
        "(",
        absl::StrJoin(shape.tuple_shapes(), ", ",
                      [&](std::string* s, const ShapeProto& subshape) {
                        absl::StrAppend(s, ShapeToString(subshape));
                      }),
        ")");
  }
  return absl::StrCat("[", absl::StrJoin(shape.dimensions(), ", "), "]");
}

void ZkxBuilder::ToStringHelper(std::string* out, int ident,
                                int64_t op_handle) const {
  const HloInstructionProto& instr =
      *(LookUpInstructionByHandle(op_handle).value());
  absl::StrAppend(out, std::string(ident, ' '), instr.opcode(),
                  ", shape=", ShapeToString(instr.shape()));
  if (instr.has_metadata()) {
    absl::StrAppend(out, ", metadata={", instr.metadata().source_file(), ":",
                    instr.metadata().source_line(), "}");
  }
  if (instr.operand_ids_size()) {
    absl::StrAppend(out, "\n");
  }
  absl::StrAppend(out, absl::StrJoin(instr.operand_ids(), "\n",
                                     [&](std::string* s, int64_t subop) {
                                       ToStringHelper(s, ident + 2, subop);
                                     }));
}

ZkxBuilder::ZkxBuilder(const std::string& computation_name)
    : name_(computation_name) {}

ZkxBuilder::~ZkxBuilder() = default;

ZkxOp ZkxBuilder::ReportError(const absl::Status& error) {
  CHECK(!error.ok());
  if (die_immediately_on_error_) {
    LOG(FATAL) << "error building computation: " << error;
  }

  if (first_error_.ok()) {
    first_error_ = error;
    // TODO(chokobole): Uncomment this. Dependency: SavedStackTrace,
    // first_error_backtrace_.CreateCurrent(/*skip_count=*/1);
  }
  return ZkxOp(this);
}

ZkxOp ZkxBuilder::ReportErrorOrReturn(const absl::StatusOr<ZkxOp>& op) {
  if (!first_error_.ok()) {
    return ZkxOp(this);
  }
  if (!op.ok()) {
    return ReportError(op.status());
  }
  return op.value();
}

ZkxOp ZkxBuilder::ReportErrorOrReturn(
    absl::FunctionRef<absl::StatusOr<ZkxOp>()> op_creator) {
  return ReportErrorOrReturn(op_creator());
}

absl::StatusOr<ProgramShape> ZkxBuilder::GetProgramShape(
    int64_t root_id) const {
  TF_RETURN_IF_ERROR(first_error_);
  TF_ASSIGN_OR_RETURN(const HloInstructionProto* root_proto,
                      LookUpInstructionByHandle(root_id));

  ProgramShape program_shape;

  *program_shape.mutable_result() = Shape(root_proto->shape());

  // Check that the parameter numbers are continuous from 0, and add parameter
  // shapes and names to the program shape.
  const int64_t param_count = parameter_numbers_.size();
  for (int64_t i = 0; i < param_count; i++) {
    program_shape.add_parameters();
    program_shape.add_parameter_names();
  }
  for (const HloInstructionProto& instr : instructions_) {
    // Parameter number uniqueness is guaranteed in ZkxBuilder::Parameter(). So
    // to verify continuity, we just need to verify that every parameter is in
    // the right range.
    if (instr.opcode() == HloOpcodeString(HloOpcode::kParameter)) {
      const int64_t index = instr.parameter_number();
      TF_RET_CHECK(index >= 0 && index < param_count)
          << "invalid parameter number: " << index;
      *program_shape.mutable_parameters(index) = Shape(instr.shape());
      *program_shape.mutable_parameter_names(index) = instr.name();
    }
  }
  return program_shape;
}

absl::StatusOr<ProgramShape> ZkxBuilder::GetProgramShape() const {
  TF_RET_CHECK(!instructions_.empty());
  return GetProgramShape(instructions_.back().id());
}

absl::StatusOr<ProgramShape> ZkxBuilder::GetProgramShape(ZkxOp root) const {
  if (root.builder_ != this) {
    return absl::InvalidArgumentError(
        "Given root operation is not in this computation.");
  }
  return GetProgramShape(root.handle());
}

void ZkxBuilder::IsConstantVisitor(const int64_t op_handle, int depth,
                                   absl::flat_hash_set<int64_t>* visited,
                                   bool* is_constant) const {
  if (visited->contains(op_handle) || !*is_constant) {
    return;
  }

  const HloInstructionProto& instr =
      *(LookUpInstructionByHandle(op_handle).value());
  HloInstructionProto to_print(instr);
  to_print.clear_shape();
  const HloOpcode opcode = StringToHloOpcode(instr.opcode()).value();
  const std::string indent =
      absl::StrJoin(std::vector<std::string_view>(depth, "  "), "");
  if (VLOG_IS_ON(2)) {
    VLOG(2) << indent << "Visiting:";
    for (const auto& l : absl::StrSplit(to_print.DebugString(), '\n')) {
      VLOG(2) << indent << l;
    }
  }
  switch (opcode) {
    default:
      for (const int64_t operand_id : instr.operand_ids()) {
        IsConstantVisitor(operand_id, depth + 1, visited, is_constant);
      }
      // TODO(b/32495713): We aren't checking the called computations.
      break;

    case HloOpcode::kGetDimensionSize:
      // GetDimensionSize is always considered constant in ZKX -- If a dynamic
      // dimension is presented, -1 is returned.
      break;
    // Non functional ops.
    // TODO(chokobole): Uncomment this. Dependency: HloOpcode::kRng
    // case HloOpcode::kRng:
    case HloOpcode::kAllReduce:
    case HloOpcode::kReduceScatter:
      // TODO(b/33009255): Implement constant folding for cross replica sum.
    case HloOpcode::kInfeed:
    case HloOpcode::kOutfeed:
    case HloOpcode::kCall:
      // TODO(b/32495713): We aren't checking the to_apply computation itself,
      // so we conservatively say that computations containing the Call op
      // cannot be constant.  We cannot set is_functional=false in other similar
      // cases since we're already relying on IsConstant to return true.
    case HloOpcode::kCustomCall:
      // TODO(chokobole): Uncomment this. Dependency: custom_call_target
      // if (instr.custom_call_target() == "SetBound") {
      //   // Set bound is considered constant -- the bound is used as the
      //   value. break;
      // }
      [[fallthrough]];
    case HloOpcode::kWhile:
      // TODO(b/32495713): We aren't checking the condition and body
      // computations themselves.
    case HloOpcode::kScatter:
      // TODO(b/32495713): We aren't checking the embedded computation in
      // Scatter.
    case HloOpcode::kSend:
    case HloOpcode::kRecv:
    case HloOpcode::kParameter:
      *is_constant = false;
      break;
    case HloOpcode::kGetTupleElement: {
      const HloInstructionProto& operand_instr =
          *(LookUpInstructionByHandle(instr.operand_ids(0)).value());
      if (HloOpcodeString(HloOpcode::kTuple) == operand_instr.opcode()) {
        IsConstantVisitor(operand_instr.operand_ids(instr.tuple_index()),
                          depth + 1, visited, is_constant);
      } else {
        for (const int64_t operand_id : instr.operand_ids()) {
          IsConstantVisitor(operand_id, depth + 1, visited, is_constant);
        }
      }
    }
  }
  if (VLOG_IS_ON(1) && !*is_constant) {
    VLOG(1) << indent << "Non-constant: ";
    for (const auto& l : absl::StrSplit(to_print.DebugString(), '\n')) {
      VLOG(1) << indent << l;
    }
  }
  visited->insert(op_handle);
}

absl::Status ZkxBuilder::SetInstructionFrontendAttribute(const ZkxOp op,
                                                         std::string attribute,
                                                         std::string value) {
  TF_ASSIGN_OR_RETURN(auto instr_proto, LookUpMutableInstruction(op));
  auto* frontend_attributes = instr_proto->mutable_frontend_attributes();
  (*frontend_attributes->mutable_map())[attribute] = std::move(value);
  return absl::OkStatus();
}

absl::Status ZkxBuilder::SetInstructionSharding(
    ZkxOp op, const std::optional<OpSharding>& sharding) {
  TF_ASSIGN_OR_RETURN(auto instr_proto, LookUpMutableInstruction(op));
  if (!sharding.has_value()) {
    instr_proto->clear_sharding();
    return absl::OkStatus();
  }
  return NormalizeAndAssignSharing(instr_proto, sharding.value());
}

ZkxComputation ZkxBuilder::BuildAndNoteError() {
  DCHECK(parent_builder_ != nullptr);
  auto build_status = Build();
  if (!build_status.ok()) {
    parent_builder_->ReportError(
        AddStatus(build_status.status(), absl::StrCat("error from: ", name_)));
    return {};
  }
  return std::move(build_status).value();
}

absl::Status ZkxBuilder::GetCurrentStatus() const {
  if (!first_error_.ok()) {
    std::string backtrace;
    // TODO(chokobole): Uncomment this. Dependency: SavedStackTrace
    // first_error_backtrace_.Dump(tsl::DebugWriteToString, &backtrace);
    return AppendStatus(first_error_, backtrace);
  }
  return absl::OkStatus();
}

absl::StatusOr<ZkxComputation> ZkxBuilder::Build(
    bool remove_dynamic_dimensions) {
  TF_RETURN_IF_ERROR(GetCurrentStatus());
  return Build(instructions_.back().id(), remove_dynamic_dimensions);
}

absl::StatusOr<ZkxComputation> ZkxBuilder::Build(
    ZkxOp root, bool remove_dynamic_dimensions) {
  if (root.builder_ != this) {
    return absl::InvalidArgumentError(
        "Given root operation is not in this computation.");
  }
  return Build(root.handle(), remove_dynamic_dimensions);
}

absl::StatusOr<ZkxComputation> ZkxBuilder::Build(
    int64_t root_id, bool remove_dynamic_dimensions) {
  TF_RETURN_IF_ERROR(GetCurrentStatus());

  // TODO(b/121223198): ZKX backend cannot handle dynamic dimensions yet, remove
  // all dynamic dimensions before building zkx program until we have support in
  // the backend.
  if (remove_dynamic_dimensions) {
    std::function<void(Shape*)> remove_dynamic_dimension = [&](Shape* shape) {
      if (shape->tuple_shapes_size() != 0) {
        for (int i = 0; i < shape->tuple_shapes_size(); ++i) {
          remove_dynamic_dimension(shape->mutable_tuple_shapes(i));
        }
      }
      for (int64_t i = 0; i < shape->dimensions_size(); ++i) {
        shape->set_dynamic_dimension(i, false);
      }
    };
    for (size_t index = 0; index < instructions_.size(); ++index) {
      remove_dynamic_dimension(instruction_shapes_[index].get());
      *instructions_[index].mutable_shape() =
          instruction_shapes_[index]->ToProto();
    }
  }

  HloComputationProto entry;
  SetProtoIdAndName(&entry, name_, kNameSeparator, GetNextId());
  TF_ASSIGN_OR_RETURN(ProgramShape program_shape, GetProgramShape(root_id));
  *entry.mutable_program_shape() = program_shape.ToProto();
  entry.set_root_id(root_id);

  for (auto& instruction : instructions_) {
    // Ensures that the instruction names are unique among the whole graph.
    instruction.set_name(
        GetFullName(instruction.name(), kNameSeparator, instruction.id()));
    entry.add_instructions()->Swap(&instruction);
  }

  ZkxComputation computation(entry.id());
  HloModuleProto* module = computation.mutable_proto();
  module->set_name(entry.name());
  module->set_id(entry.id());
  module->set_entry_computation_name(entry.name());
  module->set_entry_computation_id(entry.id());
  *module->mutable_host_program_shape() = entry.program_shape();
  for (auto& e : embedded_) {
    module->add_computations()->Swap(&e.second);
  }
  module->add_computations()->Swap(&entry);
  if (!input_output_aliases_.empty() || !buffer_donors_.empty()) {
    TF_RETURN_IF_ERROR(PopulateInputOutputAliasAndBufferDonor(
        module, program_shape, input_output_aliases_, buffer_donors_));
  }

  // Clear data held by this builder.
  this->instructions_.clear();
  this->instruction_shapes_.clear();
  this->handle_to_index_.clear();
  this->embedded_.clear();
  this->parameter_numbers_.clear();

  return std::move(computation);
}

// static
absl::Status ZkxBuilder::PopulateInputOutputAliasAndBufferDonor(
    HloModuleProto* module, const ProgramShape& program_shape,
    const std::vector<InputOutputAlias>& input_output_aliases,
    const absl::flat_hash_set<HloBufferDonorConfig::BufferDonor>&
        buffer_donors) {
  // Step 1: populate input output alias information.
  HloInputOutputAliasConfig io_alias_config(program_shape.result());
  for (auto& alias : input_output_aliases) {
    // The HloInputOutputAliasConfig does not do parameter validation as it only
    // carries the result shape. Maybe it should be constructed with a
    // ProgramShape to allow full validation. We will still get an error when
    // trying to compile the HLO module, but would be better to have validation
    // at this stage.
    if (alias.param_number >= program_shape.parameters_size()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Invalid parameter number %ld (total %ld)",
                          alias.param_number, program_shape.parameters_size()));
    }
    const Shape& parameter_shape = program_shape.parameters(alias.param_number);
    if (!ShapeUtil::IndexIsValid(parameter_shape, alias.param_index)) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Invalid parameter %ld index: %s", alias.param_number,
                          alias.param_index.ToString().c_str()));
    }
    TF_RETURN_IF_ERROR(io_alias_config.SetUpAlias(
        alias.output_index, alias.param_number, alias.param_index, alias.kind));
  }
  *module->mutable_input_output_alias() = io_alias_config.ToProto();

  // Step 2: populate buffer donor information.
  HloBufferDonorConfig buffer_donor_config;
  for (auto& donor : buffer_donors) {
    if (donor.param_number >= program_shape.parameters_size()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Invalid parameter number %ld (total %ld)",
                          donor.param_number, program_shape.parameters_size()));
    }
    const Shape& parameter_shape = program_shape.parameters(donor.param_number);
    if (!ShapeUtil::IndexIsValid(parameter_shape, donor.param_index)) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Invalid parameter %ld index: %s", donor.param_number,
                          donor.param_index.ToString().c_str()));
    }
    if (io_alias_config.ParameterHasAlias(donor.param_number,
                                          donor.param_index)) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Parameter %ld index %s is already aliased with one output, thus it "
          "cannot be added as a buffer donor for any output.",
          donor.param_number, donor.param_index.ToString().c_str()));
    }
    TF_RETURN_IF_ERROR(buffer_donor_config.AddBufferDonor(donor.param_number,
                                                          donor.param_index));
  }
  *module->mutable_buffer_donor() = buffer_donor_config.ToProto();

  return absl::OkStatus();
}

absl::StatusOr<ZkxOp> ZkxBuilder::InDimBroadcast(
    const Shape& shape, ZkxOp operand,
    absl::Span<const int64_t> broadcast_dimensions) {
  TF_RETURN_IF_ERROR(first_error_);

  HloInstructionProto instr;
  *instr.mutable_shape() = shape.ToProto();
  for (int64_t dim : broadcast_dimensions) {
    instr.add_dimensions(dim);
  }

  TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));
  TF_RET_CHECK(!shape.is_unbounded_dynamic())
      << "broadcast op result shapes must be static";
  for (int64_t i = 0; i < shape.rank(); i++) {
    if (auto it = absl::c_find(broadcast_dimensions, i);
        it != broadcast_dimensions.end()) {
      // Broadcast dimensions are permitted to be dynamic iff the operand
      // dimension is dynamic.
      TF_RET_CHECK(operand_shape->is_bounded_dynamic_dimension(
                       it - broadcast_dimensions.begin()) ==
                   shape.is_bounded_dynamic_dimension(i))
          << " i: " << i << ", shape: " << ShapeUtil::HumanString(shape)
          << ", operand_shape: " << ShapeUtil::HumanString(*operand_shape);
    } else {
      // Non-broadcast dimensions must be static.
      TF_RET_CHECK(shape.is_static_dimension(i));
    }
  }
  return AddInstruction(std::move(instr), HloOpcode::kBroadcast, {operand});
}

absl::StatusOr<ZkxOp> ZkxBuilder::AddBroadcastSequence(
    const Shape& output_shape, ZkxOp operand) {
  TF_RETURN_IF_ERROR(first_error_);

  TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));

  CHECK(ShapeUtil::IsScalar(*operand_shape) ||
        operand_shape->rank() == output_shape.rank());
  Shape broadcast_shape =
      ShapeUtil::ChangeElementType(output_shape, operand_shape->element_type());

  // Do explicit broadcast for scalar.
  if (ShapeUtil::IsScalar(*operand_shape)) {
    return InDimBroadcast(ShapeUtil::MakeStaticShape(broadcast_shape), operand,
                          {});
  }

  // Do explicit broadcast for degenerate broadcast.
  std::vector<int64_t> broadcast_dimensions;
  std::vector<int64_t> reshaped_dimensions;
  std::vector<bool> reshaped_dynamic_dimensions;
  for (int i = 0; i < operand_shape->rank(); i++) {
    if (operand_shape->dimensions(i) == output_shape.dimensions(i)) {
      broadcast_dimensions.push_back(i);
      reshaped_dimensions.push_back(operand_shape->dimensions(i));
      reshaped_dynamic_dimensions.push_back(
          operand_shape->is_dynamic_dimension(i));
    } else {
      TF_RET_CHECK(operand_shape->dimensions(i) == 1 &&
                   operand_shape->is_static_dimension(i))
          << "An explicit broadcast sequence requires the broadcasted "
             "dimensions to be trivial; operand shape: "
          << *operand_shape << "; output_shape: " << output_shape;
    }
    broadcast_shape.set_dynamic_dimension(
        i, operand_shape->is_dynamic_dimension(i));
  }

  Shape reshaped_shape =
      ShapeUtil::MakeShape(operand_shape->element_type(), reshaped_dimensions,
                           reshaped_dynamic_dimensions);

  // Eliminate the size one dimensions.
  // The added reshape reduces the rank of the tensor. Hence we cannot directly
  // apply the broadcast's sharding on reshape.
  ZkxOp reshaped_operand;
  {
    ZkxScopedShardingAssignment scoped_sharding(this, std::nullopt);
    TF_ASSIGN_OR_RETURN(
        reshaped_operand,
        ReshapeInternal(reshaped_shape, operand, /*inferred_dimension=*/-1));
  }
  // Broadcast 'reshape' up to the larger size.
  return InDimBroadcast(broadcast_shape, reshaped_operand,
                        broadcast_dimensions);
}

ZkxOp ZkxBuilder::UnaryOp(HloOpcode unop, ZkxOp operand) {
  return ReportErrorOrReturn([&]() -> absl::StatusOr<ZkxOp> {
    TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));
    TF_ASSIGN_OR_RETURN(
        Shape shape, ShapeInference::InferUnaryOpShape(unop, *operand_shape));
    return AddOpWithShape(unop, shape, {operand});
  });
}

namespace {

// Broadcasts an origin ZKX op to the rank of target_shape.
// Does not broadcast rank dimensions to match, only expands rank.
// Is identity function if origin rank matches target rank.
absl::StatusOr<ZkxOp> BroadcastToTargetRank(
    ZkxOp origin, const Shape& origin_shape, const Shape& target_shape,
    absl::Span<const int64_t> broadcast_dimensions) {
  if (ShapeUtil::IsScalar(origin_shape)) {
    return origin;
  }

  const int64_t origin_rank = origin_shape.rank();
  const int64_t target_rank = target_shape.rank();

  // Identity op if ranks match, should never be larger than target.
  if (origin_rank >= target_rank) {
    return origin;
  }

  // Update target_size with origin sizes using broadcast_dimensions
  absl::Span<const int64_t> target_dimensions = target_shape.dimensions();
  std::vector<int64_t> target_size{target_dimensions.begin(),
                                   target_dimensions.end()};
  for (int64_t origin_dim = 0; origin_dim < origin_rank; origin_dim++) {
    int64_t target_dim = broadcast_dimensions[origin_dim];
    target_size[target_dim] = origin_shape.dimensions(origin_dim);
  }
  return BroadcastInDim(origin, target_size, broadcast_dimensions);
}

}  // namespace

ZkxOp ZkxBuilder::BinaryOp(HloOpcode binop, ZkxOp lhs, ZkxOp rhs,
                           absl::Span<const int64_t> broadcast_dimensions,
                           std::optional<ComparisonDirection> direction) {
  return ReportErrorOrReturn([&]() -> absl::StatusOr<ZkxOp> {
    TF_ASSIGN_OR_RETURN(const Shape* lhs_shape, GetShapePtr(lhs));
    TF_ASSIGN_OR_RETURN(const Shape* rhs_shape, GetShapePtr(rhs));
    TF_ASSIGN_OR_RETURN(
        Shape shape, ShapeInference::InferBinaryOpShape(
                         binop, *lhs_shape, *rhs_shape, broadcast_dimensions));

    ZkxOp updated_lhs = lhs;
    ZkxOp updated_rhs = rhs;
    if (!lhs_shape->is_unbounded_dynamic() &&
        !rhs_shape->is_unbounded_dynamic()) {
      if (lhs_shape->rank() < shape.rank()) {
        TF_ASSIGN_OR_RETURN(updated_lhs,
                            BroadcastToTargetRank(lhs, *lhs_shape, shape,
                                                  broadcast_dimensions));
      }
      if (rhs_shape->rank() < shape.rank()) {
        TF_ASSIGN_OR_RETURN(updated_rhs,
                            BroadcastToTargetRank(rhs, *rhs_shape, shape,
                                                  broadcast_dimensions));
      }
      TF_ASSIGN_OR_RETURN(const Shape* updated_lhs_shape,
                          GetShapePtr(updated_lhs));
      TF_ASSIGN_OR_RETURN(const Shape* updated_rhs_shape,
                          GetShapePtr(updated_rhs));
      if (!ShapeUtil::SameDimensions(shape, *updated_lhs_shape)) {
        TF_ASSIGN_OR_RETURN(updated_lhs,
                            AddBroadcastSequence(shape, updated_lhs));
      }
      if (!ShapeUtil::SameDimensions(shape, *updated_rhs_shape)) {
        TF_ASSIGN_OR_RETURN(updated_rhs,
                            AddBroadcastSequence(shape, updated_rhs));
      }
    } else {
      return absl::UnimplementedError(
          "A case that lhs_shape or rhs_shape is unbounded dynamic is not "
          "supported.");
    }

    if (binop == HloOpcode::kCompare) {
      if (!direction.has_value()) {
        return absl::InvalidArgumentError(
            "kCompare expects a ComparisonDirection, but none provided.");
      }
      return Compare(shape, updated_lhs, updated_rhs, *direction);
    }

    if (direction.has_value()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "A comparison direction is provided for a non-compare opcode: %s.",
          HloOpcodeString(binop)));
    }
    return BinaryOpNoBroadcast(binop, shape, updated_lhs, updated_rhs);
  });
}

ZkxOp ZkxBuilder::BinaryOpNoBroadcast(HloOpcode binop, const Shape& shape,
                                      ZkxOp lhs, ZkxOp rhs) {
  return ReportErrorOrReturn([&]() -> absl::StatusOr<ZkxOp> {
    HloInstructionProto instr;
    *instr.mutable_shape() = shape.ToProto();
    return AddInstruction(std::move(instr), binop, {lhs, rhs});
  });
}

absl::StatusOr<ZkxOp> ZkxBuilder::Compare(const Shape& shape, ZkxOp lhs,
                                          ZkxOp rhs,
                                          ComparisonDirection direction) {
  HloInstructionProto instr;
  instr.set_comparison_direction(ComparisonDirectionToString(direction));
  *instr.mutable_shape() = shape.ToProto();
  return AddInstruction(std::move(instr), HloOpcode::kCompare, {lhs, rhs});
}

absl::StatusOr<ZkxOp> ZkxBuilder::BroadcastScalarToOutputShape(ZkxOp scalar,
                                                               ZkxOp output) {
  TF_ASSIGN_OR_RETURN(const Shape* output_shape, GetShapePtr(output));

  ZkxOp updated_output = scalar;
  if (output_shape->is_unbounded_dynamic()) {
    return absl::UnimplementedError(
        "A case that output_shape is unbounded dynamic is not supported.");
  }

  TF_ASSIGN_OR_RETURN(updated_output,
                      AddBroadcastSequence(*output_shape, updated_output));
  return updated_output;
}

ZkxOp ZkxBuilder::TernaryOp(HloOpcode triop, ZkxOp lhs, ZkxOp rhs, ZkxOp ehs) {
  return ReportErrorOrReturn([&]() -> absl::StatusOr<ZkxOp> {
    ZkxOp updated_lhs = lhs;
    ZkxOp updated_rhs = rhs;
    ZkxOp updated_ehs = ehs;

    // The client API supports implicit broadcast for kSelect and kClamp, but
    // ZKX does not support implicit broadcast. Make implicit broadcast explicit
    // and update the operands.
    if (triop == HloOpcode::kSelect || triop == HloOpcode::kClamp) {
      TF_ASSIGN_OR_RETURN(const Shape* lhs_shape, GetShapePtr(lhs));
      TF_ASSIGN_OR_RETURN(const Shape* rhs_shape, GetShapePtr(rhs));
      TF_ASSIGN_OR_RETURN(const Shape* ehs_shape, GetShapePtr(ehs));
      TF_ASSIGN_OR_RETURN(
          std::optional<Shape> output_shape,
          ShapeInference::InferScalarBroadcastShape(
              absl::Span<const Shape>({*lhs_shape, *rhs_shape, *ehs_shape})));

      // Scalar broadcast if mix of scalars and non-scalars
      if (output_shape.has_value()) {
        if (ShapeUtil::IsScalar(*lhs_shape)) {
          TF_ASSIGN_OR_RETURN(
              updated_lhs,
              BroadcastScalarToOutputShape(
                  /*scalar=*/lhs,
                  /*output=*/
                  ShapeUtil::Equal(*output_shape, *rhs_shape) ? rhs : ehs));
        }
        if (ShapeUtil::IsScalar(*rhs_shape)) {
          TF_ASSIGN_OR_RETURN(
              updated_rhs,
              BroadcastScalarToOutputShape(
                  /*scalar=*/rhs,
                  /*output=*/
                  ShapeUtil::Equal(*output_shape, *lhs_shape) ? lhs : ehs));
        }
        if (ShapeUtil::IsScalar(*ehs_shape)) {
          TF_ASSIGN_OR_RETURN(
              updated_ehs,
              BroadcastScalarToOutputShape(
                  /*scalar=*/ehs,
                  /*output=*/
                  ShapeUtil::Equal(*output_shape, *lhs_shape) ? lhs : rhs));
        }
      }
    }

    TF_ASSIGN_OR_RETURN(const Shape* lhs_shape, GetShapePtr(updated_lhs));
    TF_ASSIGN_OR_RETURN(const Shape* rhs_shape, GetShapePtr(updated_rhs));
    TF_ASSIGN_OR_RETURN(const Shape* ehs_shape, GetShapePtr(updated_ehs));
    TF_ASSIGN_OR_RETURN(const Shape inferred_shape,
                        ShapeInference::InferTernaryOpShape(
                            triop, *lhs_shape, *rhs_shape, *ehs_shape));

    return AddOpWithShape(triop, inferred_shape,
                          {updated_lhs, updated_rhs, updated_ehs});
  });
}

ZkxOp ZkxBuilder::ConstantLiteral(const LiteralSlice& literal) {
  return ReportErrorOrReturn([&]() -> absl::StatusOr<ZkxOp> {
    if (literal.shape().IsArray() && literal.element_count() > 1 &&
        literal.IsAllFirst()) {
      Literal scalar = LiteralUtil::GetFirstScalarLiteral(literal);
      HloInstructionProto instr;
      *instr.mutable_shape() = scalar.shape().ToProto();
      *instr.mutable_literal() = scalar.ToProto();
      ZkxOp scalar_op;
      {
        // If the builder has a sharding, it should only be added to the
        // broadcast (and not the scalar constant).
        ZkxScopedShardingAssignment scoped_sharding(this, std::nullopt);
        TF_ASSIGN_OR_RETURN(
            scalar_op, AddInstruction(std::move(instr), HloOpcode::kConstant));
      }
      return Broadcast(scalar_op, literal.shape().dimensions());
    } else {
      HloInstructionProto instr;
      *instr.mutable_shape() = literal.shape().ToProto();
      *instr.mutable_literal() = literal.ToProto();
      return AddInstruction(std::move(instr), HloOpcode::kConstant);
    }
  });
}

ZkxOp ZkxBuilder::Iota(const Shape& shape, int64_t iota_dimension) {
  return ReportErrorOrReturn([&]() -> absl::StatusOr<ZkxOp> {
    if (!shape.is_static()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "The output of iota must not have dynamic dimensions: %s",
          ShapeUtil::HumanString(shape)));
    }
    HloInstructionProto instr;
    *instr.mutable_shape() = shape.ToProto();
    instr.add_dimensions(iota_dimension);
    return AddInstruction(std::move(instr), HloOpcode::kIota);
  });
}

ZkxOp ZkxBuilder::Iota(PrimitiveType type, int64_t size) {
  return Iota(ShapeUtil::MakeShape(type, {size}), /*iota_dimension=*/0);
}

ZkxOp ZkxBuilder::Call(const ZkxComputation& computation,
                       absl::Span<const ZkxOp> operands) {
  return ReportErrorOrReturn([&]() -> absl::StatusOr<ZkxOp> {
    HloInstructionProto instr;
    std::vector<const Shape*> operand_shape_ptrs;
    TF_ASSIGN_OR_RETURN(const auto& operand_shapes, GetOperandShapes(operands));
    absl::c_transform(operand_shapes, std::back_inserter(operand_shape_ptrs),
                      [](const Shape& shape) { return &shape; });
    TF_ASSIGN_OR_RETURN(const ProgramShape& called_program_shape,
                        computation.GetProgramShape());
    TF_ASSIGN_OR_RETURN(Shape shape, ShapeInference::InferCallShape(
                                         operand_shape_ptrs,
                                         /*to_apply=*/called_program_shape));
    *instr.mutable_shape() = shape.ToProto();

    AddCalledComputation(computation, &instr);

    return AddInstruction(std::move(instr), HloOpcode::kCall, operands);
  });
}

ZkxOp ZkxBuilder::CompositeCall(const ZkxComputation& computation,
                                absl::Span<const ZkxOp> operands,
                                const std::string& name,
                                std::optional<std::string_view> attributes,
                                std::optional<int64_t> version) {
  return ReportErrorOrReturn([&]() -> absl::StatusOr<ZkxOp> {
    HloInstructionProto instr;
    std::vector<const Shape*> operand_shape_ptrs;
    TF_ASSIGN_OR_RETURN(const auto& operand_shapes, GetOperandShapes(operands));
    absl::c_transform(operand_shapes, std::back_inserter(operand_shape_ptrs),
                      [](const Shape& shape) { return &shape; });
    TF_ASSIGN_OR_RETURN(const ProgramShape& called_program_shape,
                        computation.GetProgramShape());
    TF_ASSIGN_OR_RETURN(Shape shape, ShapeInference::InferCallShape(
                                         operand_shape_ptrs,
                                         /*to_apply=*/called_program_shape));
    *instr.mutable_shape() = shape.ToProto();

    AddCalledComputation(computation, &instr);
    instr.set_is_composite(true);

    TF_ASSIGN_OR_RETURN(
        ZkxOp instruction,
        AddInstruction(std::move(instr), HloOpcode::kCall, operands));
    TF_RETURN_IF_ERROR(
        SetInstructionFrontendAttribute(instruction, "composite.name", name));
    TF_RETURN_IF_ERROR(SetInstructionFrontendAttribute(
        instruction, "composite.attributes",
        attributes.has_value() ? std::string(*attributes) : "{}"));
    TF_RETURN_IF_ERROR(SetInstructionFrontendAttribute(
        instruction, "composite.version",
        version.has_value() ? std::to_string(*version) : "0"));
    return instruction;
  });
}

ZkxOp ZkxBuilder::Parameter(
    int64_t parameter_number, const Shape& shape, const std::string& name,
    const std::vector<bool>& replicated_at_leaf_buffers) {
  return ReportErrorOrReturn([&]() -> absl::StatusOr<ZkxOp> {
    HloInstructionProto instr;
    if (!parameter_numbers_.insert(parameter_number).second) {
      return absl::InvalidArgumentError(
          absl::StrFormat("parameter %d already registered", parameter_number));
    }
    instr.set_parameter_number(parameter_number);
    instr.set_name(name);
    *instr.mutable_shape() = shape.ToProto();
    if (!replicated_at_leaf_buffers.empty()) {
      auto replication = instr.mutable_parameter_replication();
      for (bool replicated : replicated_at_leaf_buffers) {
        replication->add_replicated_at_leaf_buffers(replicated);
      }
    }
    return AddInstruction(std::move(instr), HloOpcode::kParameter);
  });
}

ZkxOp ZkxBuilder::Broadcast(ZkxOp operand,
                            absl::Span<const int64_t> broadcast_sizes) {
  return ReportErrorOrReturn([&]() -> absl::StatusOr<ZkxOp> {
    TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));
    TF_ASSIGN_OR_RETURN(
        const Shape& shape,
        ShapeInference::InferBroadcastShape(*operand_shape, broadcast_sizes));

    // The client-level broadcast op just appends dimensions on the left (adds
    // lowest numbered dimensions). The HLO broadcast instruction is more
    // flexible and can add new dimensions anywhere. The instruction's
    // dimensions field maps operand dimensions to dimensions in the broadcast
    // output, so to append dimensions on the left the instruction's dimensions
    // should just be the n highest dimension numbers of the output shape where
    // n is the number of input dimensions.
    const int64_t operand_rank = operand_shape->rank();
    std::vector<int64_t> dimensions(operand_rank);
    for (int i = 0; i < operand_rank; ++i) {
      dimensions[i] = i + shape.rank() - operand_rank;
    }
    return InDimBroadcast(shape, operand, dimensions);
  });
}

ZkxOp ZkxBuilder::BroadcastInDim(
    ZkxOp operand, absl::Span<const int64_t> out_dim_size,
    absl::Span<const int64_t> broadcast_dimensions) {
  return ReportErrorOrReturn([&]() -> absl::StatusOr<ZkxOp> {
    TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));
    // Output shape, in the case of degenerate broadcast, the out_dim_size is
    // not necessarily the same as the dimension sizes of the output shape.
    TF_ASSIGN_OR_RETURN(auto output_shape,
                        ShapeUtil::MakeValidatedShape(
                            operand_shape->element_type(), out_dim_size));
    TF_RET_CHECK(!output_shape.is_unbounded_dynamic())
        << "BroadcastInDim output must shape be static or bounded dynamic "
        << ShapeUtil::HumanString(output_shape);
    int64_t broadcast_rank = broadcast_dimensions.size();
    if (operand_shape->rank() != broadcast_rank) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Size of broadcast_dimensions has to match operand's rank; operand "
          "rank: %lld, size of broadcast_dimensions %u.",
          operand_shape->rank(), broadcast_dimensions.size()));
    }
    for (int i = 0; i < broadcast_rank; i++) {
      const int64_t num_dims = out_dim_size.size();
      if (broadcast_dimensions[i] < 0 || broadcast_dimensions[i] > num_dims) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Broadcast dimension %lld is out of bound",
                            broadcast_dimensions[i]));
      }
      output_shape.set_dynamic_dimension(
          broadcast_dimensions[i],
          operand_shape->is_bounded_dynamic_dimension(i));
    }

    TF_RETURN_IF_ERROR(ShapeInference::InferBroadcastShape(
                           *operand_shape, output_shape, broadcast_dimensions)
                           .status());
    std::vector<int64_t> in_dim_size(out_dim_size.begin(), out_dim_size.end());
    std::vector<bool> in_dim_dynamic(out_dim_size.size(), false);
    for (int i = 0; i < broadcast_rank; i++) {
      in_dim_size[broadcast_dimensions[i]] =
          (operand_shape->is_unbounded_dynamic_dimension(i))
              ? out_dim_size[broadcast_dimensions[i]]
              : operand_shape->dimensions(i);
      in_dim_dynamic[broadcast_dimensions[i]] =
          operand_shape->is_bounded_dynamic_dimension(i);
    }
    const auto& in_dim_shape = ShapeUtil::MakeShape(
        operand_shape->element_type(), in_dim_size, in_dim_dynamic);
    TF_ASSIGN_OR_RETURN(
        ZkxOp in_dim_broadcast,
        InDimBroadcast(in_dim_shape, operand, broadcast_dimensions));

    // If broadcast is not degenerate, return broadcasted result.
    if (ShapeUtil::Equal(in_dim_shape, output_shape)) {
      return in_dim_broadcast;
    }

    // Otherwise handle degenerate broadcast case.
    return AddBroadcastSequence(output_shape, in_dim_broadcast);
  });
}

absl::StatusOr<ZkxOp> ZkxBuilder::ReshapeInternal(const Shape& shape,
                                                  ZkxOp operand,
                                                  int64_t inferred_dimension) {
  TF_RETURN_IF_ERROR(first_error_);
  if (shape.is_unbounded_dynamic()) {
    return absl::InvalidArgumentError(
        "Reshaping with unbounded result shape is not supported.");
  }

  HloInstructionProto instr;
  *instr.mutable_shape() = shape.ToProto();
  if (inferred_dimension != -1) {
    instr.add_dimensions(inferred_dimension);
  }
  return AddInstruction(std::move(instr), HloOpcode::kReshape, {operand});
}

ZkxOp ZkxBuilder::Slice(ZkxOp operand, absl::Span<const int64_t> start_indices,
                        absl::Span<const int64_t> limit_indices,
                        absl::Span<const int64_t> strides) {
  return ReportErrorOrReturn([&]() -> absl::StatusOr<ZkxOp> {
    TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));
    TF_ASSIGN_OR_RETURN(Shape shape, ShapeInference::InferSliceShape(
                                         *operand_shape, start_indices,
                                         limit_indices, strides));
    return SliceInternal(shape, operand, start_indices, limit_indices, strides);
  });
}

absl::StatusOr<ZkxOp> ZkxBuilder::SliceInternal(
    const Shape& shape, ZkxOp operand, absl::Span<const int64_t> start_indices,
    absl::Span<const int64_t> limit_indices,
    absl::Span<const int64_t> strides) {
  HloInstructionProto instr;
  *instr.mutable_shape() = shape.ToProto();
  for (int i = 0, end = start_indices.size(); i < end; i++) {
    auto* slice_config = instr.add_slice_dimensions();
    slice_config->set_start(start_indices[i]);
    slice_config->set_limit(limit_indices[i]);
    slice_config->set_stride(strides[i]);
  }
  return AddInstruction(std::move(instr), HloOpcode::kSlice, {operand});
}

ZkxOp ZkxBuilder::SliceInDim(ZkxOp operand, int64_t start_index,
                             int64_t limit_index, int64_t stride,
                             int64_t dimno) {
  return ReportErrorOrReturn([&]() -> absl::StatusOr<ZkxOp> {
    TF_ASSIGN_OR_RETURN(const Shape* shape, GetShapePtr(operand));
    std::vector<int64_t> starts(shape->rank(), 0);
    std::vector<int64_t> limits(shape->dimensions().begin(),
                                shape->dimensions().end());
    std::vector<int64_t> strides(shape->rank(), 1);
    starts[dimno] = start_index;
    limits[dimno] = limit_index;
    strides[dimno] = stride;
    return Slice(operand, starts, limits, strides);
  });
}

ZkxOp ZkxBuilder::DynamicSlice(ZkxOp operand,
                               absl::Span<const ZkxOp> start_indices,
                               absl::Span<const int64_t> slice_sizes) {
  return ReportErrorOrReturn([&]() -> absl::StatusOr<ZkxOp> {
    TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));
    std::vector<const Shape*> start_indices_shape_ptrs;
    TF_ASSIGN_OR_RETURN(const auto& start_indices_shapes,
                        GetOperandShapes(start_indices));
    absl::c_transform(start_indices_shapes,
                      std::back_inserter(start_indices_shape_ptrs),
                      [](const Shape& shape) { return &shape; });
    TF_ASSIGN_OR_RETURN(Shape shape,
                        ShapeInference::InferDynamicSliceShape(
                            *operand_shape, start_indices_shapes, slice_sizes));
    return DynamicSliceInternal(shape, operand, start_indices, slice_sizes);
  });
}

absl::StatusOr<ZkxOp> ZkxBuilder::DynamicSliceInternal(
    const Shape& shape, ZkxOp operand, absl::Span<const ZkxOp> start_indices,
    absl::Span<const int64_t> slice_sizes) {
  HloInstructionProto instr;
  *instr.mutable_shape() = shape.ToProto();

  for (int64_t size : slice_sizes) {
    instr.add_dynamic_slice_sizes(size);
  }

  std::vector<ZkxOp> operands = {operand};
  operands.insert(operands.end(), start_indices.begin(), start_indices.end());
  return AddInstruction(std::move(instr), HloOpcode::kDynamicSlice, operands);
}

ZkxOp ZkxBuilder::DynamicUpdateSlice(ZkxOp operand, ZkxOp update,
                                     absl::Span<const ZkxOp> start_indices) {
  return ReportErrorOrReturn([&]() -> absl::StatusOr<ZkxOp> {
    TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));
    TF_ASSIGN_OR_RETURN(const Shape* update_shape, GetShapePtr(update));
    std::vector<const Shape*> start_indices_shape_ptrs;
    TF_ASSIGN_OR_RETURN(const auto& start_indices_shapes,
                        GetOperandShapes(start_indices));
    absl::c_transform(start_indices_shapes,
                      std::back_inserter(start_indices_shape_ptrs),
                      [](const Shape& shape) { return &shape; });
    TF_ASSIGN_OR_RETURN(
        Shape shape, ShapeInference::InferDynamicUpdateSliceShape(
                         *operand_shape, *update_shape, start_indices_shapes));
    return DynamicUpdateSliceInternal(shape, operand, update, start_indices);
  });
}

absl::StatusOr<ZkxOp> ZkxBuilder::DynamicUpdateSliceInternal(
    const Shape& shape, ZkxOp operand, ZkxOp update,
    absl::Span<const ZkxOp> start_indices) {
  HloInstructionProto instr;
  *instr.mutable_shape() = shape.ToProto();

  std::vector<ZkxOp> operands = {operand, update};
  operands.insert(operands.end(), start_indices.begin(), start_indices.end());
  return AddInstruction(std::move(instr), HloOpcode::kDynamicUpdateSlice,
                        operands);
}

ZkxOp ZkxBuilder::ConcatInDim(absl::Span<const ZkxOp> operands,
                              int64_t dimension) {
  return ReportErrorOrReturn([&]() -> absl::StatusOr<ZkxOp> {
    std::vector<const Shape*> operand_shape_ptrs;
    TF_ASSIGN_OR_RETURN(const auto& operand_shapes, GetOperandShapes(operands));
    absl::c_transform(operand_shapes, std::back_inserter(operand_shape_ptrs),
                      [](const Shape& shape) { return &shape; });
    TF_ASSIGN_OR_RETURN(Shape shape, ShapeInference::InferConcatOpShape(
                                         operand_shape_ptrs, dimension));
    return ConcatInDimInternal(shape, operands, dimension);
  });
}

absl::StatusOr<ZkxOp> ZkxBuilder::ConcatInDimInternal(
    const Shape& shape, absl::Span<const ZkxOp> operands, int64_t dimension) {
  HloInstructionProto instr;
  *instr.mutable_shape() = shape.ToProto();

  instr.add_dimensions(dimension);

  return AddInstruction(std::move(instr), HloOpcode::kConcatenate, operands);
}

ZkxOp ZkxBuilder::Pad(ZkxOp operand, ZkxOp padding_value,
                      const PaddingConfig& padding_config) {
  return ReportErrorOrReturn([&]() -> absl::StatusOr<ZkxOp> {
    TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));
    TF_ASSIGN_OR_RETURN(const Shape* padding_value_shape,
                        GetShapePtr(padding_value));
    TF_ASSIGN_OR_RETURN(
        Shape shape, ShapeInference::InferPadShape(
                         *operand_shape, *padding_value_shape, padding_config));
    return PadInternal(shape, operand, padding_value, padding_config);
  });
}

ZkxOp ZkxBuilder::PadInDim(ZkxOp operand, ZkxOp padding_value, int64_t dimno,
                           int64_t pad_lo, int64_t pad_hi) {
  return ReportErrorOrReturn([&]() -> absl::StatusOr<ZkxOp> {
    TF_ASSIGN_OR_RETURN(const Shape* shape, GetShapePtr(operand));
    PaddingConfig padding_config = MakeNoPaddingConfig(shape->rank());
    auto* dims = padding_config.mutable_dimensions(dimno);
    dims->set_edge_padding_low(pad_lo);
    dims->set_edge_padding_high(pad_hi);
    return Pad(operand, padding_value, padding_config);
  });
}

absl::StatusOr<ZkxOp> ZkxBuilder::PadInternal(
    const Shape& shape, ZkxOp operand, ZkxOp padding_value,
    const PaddingConfig& padding_config) {
  HloInstructionProto instr;
  *instr.mutable_shape() = shape.ToProto();
  *instr.mutable_padding_config() = padding_config;
  return AddInstruction(std::move(instr), HloOpcode::kPad,
                        {operand, padding_value});
}

ZkxOp ZkxBuilder::Reshape(ZkxOp operand, absl::Span<const int64_t> dimensions,
                          absl::Span<const int64_t> new_sizes,
                          int64_t inferred_dimension) {
  return ReportErrorOrReturn([&]() -> absl::StatusOr<ZkxOp> {
    TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));
    TF_ASSIGN_OR_RETURN(const Shape shape, ShapeInference::InferReshapeShape(
                                               *operand_shape, dimensions,
                                               new_sizes, inferred_dimension));
    ZkxOp transposed = IsIdentityPermutation(dimensions)
                           ? operand
                           : Transpose(operand, dimensions);
    return ReshapeInternal(shape, transposed, inferred_dimension);
  });
}

ZkxOp ZkxBuilder::Reshape(ZkxOp operand, absl::Span<const int64_t> new_sizes,
                          int64_t inferred_dimension) {
  return ReportErrorOrReturn([&]() -> absl::StatusOr<ZkxOp> {
    TF_ASSIGN_OR_RETURN(const Shape* shape, GetShapePtr(operand));
    std::vector<int64_t> dimensions(shape->dimensions_size());
    std::iota(dimensions.begin(), dimensions.end(), 0);
    return Reshape(operand, dimensions, new_sizes, inferred_dimension);
  });
}

ZkxOp ZkxBuilder::Reshape(const Shape& shape, ZkxOp operand,
                          int64_t inferred_dimension) {
  return ReportErrorOrReturn([&]() -> absl::StatusOr<ZkxOp> {
    return ReshapeInternal(shape, operand, inferred_dimension);
  });
}

ZkxOp ZkxBuilder::DynamicReshape(ZkxOp operand,
                                 absl::Span<const ZkxOp> dim_sizes,
                                 absl::Span<const int64_t> new_size_bounds,
                                 const std::vector<bool>& dims_are_dynamic) {
  return ReportErrorOrReturn([&]() -> absl::StatusOr<ZkxOp> {
    TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));
    std::vector<const Shape*> dim_size_shape_ptrs;
    TF_ASSIGN_OR_RETURN(const auto& dim_size_shapes,
                        GetOperandShapes(dim_sizes));

    absl::c_transform(dim_size_shapes, std::back_inserter(dim_size_shape_ptrs),
                      [](const Shape& shape) { return &shape; });
    TF_ASSIGN_OR_RETURN(const Shape shape,
                        ShapeInference::InferDynamicReshapeShape(
                            *operand_shape, dim_size_shape_ptrs,
                            new_size_bounds, dims_are_dynamic));
    TF_RETURN_IF_ERROR(first_error_);
    std::vector<ZkxOp> operands;
    operands.reserve(1 + dim_sizes.size());
    operands.push_back(operand);
    for (const ZkxOp& dim_size : dim_sizes) {
      operands.push_back(dim_size);
    }
    HloInstructionProto instr;
    *instr.mutable_shape() = shape.ToProto();
    return AddInstruction(std::move(instr), HloOpcode::kDynamicReshape,
                          operands);
  });
}

ZkxOp ZkxBuilder::Collapse(ZkxOp operand,
                           absl::Span<const int64_t> dimensions) {
  return ReportErrorOrReturn([&]() -> absl::StatusOr<ZkxOp> {
    if (dimensions.size() <= 1) {
      // Not collapsing anything, trivially we can return the operand versus
      // enqueueing a trivial reshape.
      return operand;
    }

    // Out-of-order collapse is not supported.
    // Checks that the collapsed dimensions are in order and consecutive.
    for (absl::Span<const int64_t>::size_type i = 1; i < dimensions.size();
         ++i) {
      if (dimensions[i] - 1 != dimensions[i - 1]) {
        return absl::InvalidArgumentError(
            "Collapsed dimensions are not in consecutive order.");
      }
    }

    // Create a new sizes vector from the old shape, replacing the collapsed
    // dimensions by the product of their sizes.
    TF_ASSIGN_OR_RETURN(const Shape* original_shape, GetShapePtr(operand));

    VLOG(3) << "original shape: " << ShapeUtil::HumanString(*original_shape);
    VLOG(3) << "dims to collapse: " << absl::StrJoin(dimensions, ",");

    std::vector<int64_t> new_sizes;
    for (int i = 0; i < original_shape->rank(); ++i) {
      if (i <= dimensions.front() || i > dimensions.back()) {
        new_sizes.push_back(original_shape->dimensions(i));
      } else {
        new_sizes.back() *= original_shape->dimensions(i);
      }
    }

    VLOG(3) << "new sizes: [" << absl::StrJoin(new_sizes, ",") << "]";

    return Reshape(operand, new_sizes);
  });
}

namespace {

// Dummy pass-through computation returning it's parameter of shape `shape`.
absl::StatusOr<ZkxComputation> PassthroughComputation(const Shape& shape) {
  ZkxBuilder builder("dummy");
  ZkxOp out = Parameter(&builder, 0, shape, "p");
  return builder.Build(out);
}

}  // namespace

ZkxOp ZkxBuilder::Select(ZkxOp pred, ZkxOp on_true, ZkxOp on_false) {
  return ReportErrorOrReturn([&]() -> absl::StatusOr<ZkxOp> {
    TF_ASSIGN_OR_RETURN(const Shape* true_shape, GetShapePtr(on_true));
    TF_ASSIGN_OR_RETURN(const Shape* false_shape, GetShapePtr(on_false));
    TF_RET_CHECK(true_shape->IsTuple() == false_shape->IsTuple());
    if (true_shape->IsTuple()) {
      TF_ASSIGN_OR_RETURN(ZkxComputation passthrough_true,
                          PassthroughComputation(*true_shape));
      TF_ASSIGN_OR_RETURN(ZkxComputation passthrough_false,
                          PassthroughComputation(*false_shape));
      return Conditional(pred, on_true, passthrough_true, on_false,
                         passthrough_false);
    }
    return TernaryOp(HloOpcode::kSelect, pred, on_true, on_false);
  });
}

ZkxOp ZkxBuilder::Tuple(absl::Span<const ZkxOp> elements) {
  return ReportErrorOrReturn([&]() -> absl::StatusOr<ZkxOp> {
    std::vector<const Shape*> operand_shape_ptrs;
    TF_ASSIGN_OR_RETURN(const auto& operand_shapes, GetOperandShapes(elements));
    absl::c_transform(operand_shapes, std::back_inserter(operand_shape_ptrs),
                      [](const Shape& shape) { return &shape; });
    TF_ASSIGN_OR_RETURN(const Shape shape,
                        ShapeInference::InferVariadicOpShape(
                            HloOpcode::kTuple, operand_shape_ptrs));
    return TupleInternal(shape, elements);
  });
}

absl::StatusOr<ZkxOp> ZkxBuilder::TupleInternal(
    const Shape& shape, absl::Span<const ZkxOp> elements) {
  HloInstructionProto instr;
  *instr.mutable_shape() = shape.ToProto();
  return AddInstruction(std::move(instr), HloOpcode::kTuple, elements);
}

ZkxOp ZkxBuilder::GetTupleElement(ZkxOp tuple_data, int64_t index) {
  return ReportErrorOrReturn([&]() -> absl::StatusOr<ZkxOp> {
    TF_ASSIGN_OR_RETURN(const Shape* tuple_shape, GetShapePtr(tuple_data));
    if (!tuple_shape->IsTuple()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Operand to GetTupleElement() is not a tuple; got %s",
                          ShapeUtil::HumanString(*tuple_shape)));
    }
    if (index < 0 || index >= ShapeUtil::TupleElementCount(*tuple_shape)) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "GetTupleElement() index (%d) out of range for tuple shape %s", index,
          ShapeUtil::HumanString(*tuple_shape)));
    }
    return GetTupleElementInternal(
        ShapeUtil::GetTupleElementShape(*tuple_shape, index), tuple_data,
        index);
  });
}

absl::StatusOr<ZkxOp> ZkxBuilder::GetTupleElementInternal(const Shape& shape,
                                                          ZkxOp tuple_data,
                                                          int64_t index) {
  HloInstructionProto instr;
  *instr.mutable_shape() = shape.ToProto();
  instr.set_tuple_index(index);
  return AddInstruction(std::move(instr), HloOpcode::kGetTupleElement,
                        {tuple_data});
}

ZkxOp ZkxBuilder::CreateToken() {
  return ReportErrorOrReturn([&]() -> absl::StatusOr<ZkxOp> {
    HloInstructionProto instr;
    *instr.mutable_shape() = ShapeUtil::MakeTokenShape().ToProto();
    return AddInstruction(std::move(instr), HloOpcode::kAfterAll);
  });
}

ZkxOp ZkxBuilder::Transpose(ZkxOp operand,
                            absl::Span<const int64_t> permutation) {
  return ReportErrorOrReturn([&]() -> absl::StatusOr<ZkxOp> {
    TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));
    TF_ASSIGN_OR_RETURN(Shape shape, ShapeInference::InferTransposeShape(
                                         *operand_shape, permutation));
    return TransposeInternal(shape, operand, permutation);
  });
}

absl::StatusOr<ZkxOp> ZkxBuilder::TransposeInternal(
    const Shape& shape, ZkxOp operand, absl::Span<const int64_t> permutation) {
  HloInstructionProto instr;
  *instr.mutable_shape() = shape.ToProto();
  for (int64_t dim : permutation) {
    instr.add_dimensions(dim);
  }
  return AddInstruction(std::move(instr), HloOpcode::kTranspose, {operand});
}

ZkxOp ZkxBuilder::Rev(ZkxOp operand, absl::Span<const int64_t> dimensions) {
  return ReportErrorOrReturn([&]() -> absl::StatusOr<ZkxOp> {
    TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));
    TF_ASSIGN_OR_RETURN(Shape shape, ShapeInference::InferReverseShape(
                                         *operand_shape, dimensions));
    return RevInternal(shape, operand, dimensions);
  });
}

absl::StatusOr<ZkxOp> ZkxBuilder::RevInternal(
    const Shape& shape, ZkxOp operand, absl::Span<const int64_t> dimensions) {
  HloInstructionProto instr;
  *instr.mutable_shape() = shape.ToProto();
  for (int64_t dim : dimensions) {
    instr.add_dimensions(dim);
  }
  return AddInstruction(std::move(instr), HloOpcode::kReverse, {operand});
}

ZkxOp ZkxBuilder::Sort(absl::Span<const ZkxOp> operands,
                       const ZkxComputation& comparator, int64_t dimension,
                       bool is_stable) {
  return ReportErrorOrReturn([&]() -> absl::StatusOr<ZkxOp> {
    std::vector<const Shape*> operand_shape_ptrs;
    TF_ASSIGN_OR_RETURN(std::vector<Shape> operand_shapes,
                        GetOperandShapes(operands));
    absl::c_transform(operand_shapes, std::back_inserter(operand_shape_ptrs),
                      [](const Shape& shape) { return &shape; });
    TF_ASSIGN_OR_RETURN(Shape shape, ShapeInference::InferVariadicOpShape(
                                         HloOpcode::kSort, operand_shape_ptrs));
    return SortInternal(shape, operands, comparator, dimension, is_stable);
  });
}

absl::StatusOr<ZkxOp> ZkxBuilder::SortInternal(const Shape& shape,
                                               absl::Span<const ZkxOp> operands,
                                               const ZkxComputation& comparator,
                                               int64_t dimension,
                                               bool is_stable) {
  HloInstructionProto instr;
  *instr.mutable_shape() = shape.ToProto();
  instr.set_is_stable(is_stable);
  if (dimension == -1) {
    TF_ASSIGN_OR_RETURN(const Shape* keys_shape, GetShapePtr(operands[0]));
    dimension = keys_shape->rank() - 1;
  }
  instr.add_dimensions(dimension);
  AddCalledComputation(comparator, &instr);
  return AddInstruction(std::move(instr), HloOpcode::kSort, operands);
}

ZkxOp ZkxBuilder::ConvertElementType(ZkxOp operand,
                                     PrimitiveType new_element_type) {
  return ReportErrorOrReturn([&]() -> absl::StatusOr<ZkxOp> {
    TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));
    TF_ASSIGN_OR_RETURN(Shape shape, ShapeInference::InferConvertShape(
                                         *operand_shape, new_element_type));
    return AddOpWithShape(HloOpcode::kConvert, shape, {operand});
  });
}

ZkxOp ZkxBuilder::BitcastConvertType(ZkxOp operand,
                                     PrimitiveType new_element_type) {
  return ReportErrorOrReturn([&]() -> absl::StatusOr<ZkxOp> {
    TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));
    TF_ASSIGN_OR_RETURN(Shape shape, ShapeInference::InferBitcastConvertShape(
                                         *operand_shape, new_element_type));
    return BitcastConvertTypeInternal(shape, operand);
  });
}

absl::StatusOr<ZkxOp> ZkxBuilder::BitcastConvertTypeInternal(const Shape& shape,
                                                             ZkxOp operand) {
  HloInstructionProto instr;
  *instr.mutable_shape() = shape.ToProto();
  return AddInstruction(std::move(instr), HloOpcode::kBitcastConvert,
                        {operand});
}

ZkxOp ZkxBuilder::Clamp(ZkxOp min, ZkxOp operand, ZkxOp max) {
  return TernaryOp(HloOpcode::kClamp, min, operand, max);
}

ZkxOp ZkxBuilder::Map(absl::Span<const ZkxOp> operands,
                      const ZkxComputation& computation,
                      absl::Span<const int64_t> dimensions,
                      absl::Span<const ZkxOp> static_operands) {
  return ReportErrorOrReturn([&]() -> absl::StatusOr<ZkxOp> {
    if (!static_operands.empty()) {
      return absl::UnimplementedError(
          "static_operands is not supported in Map");
    }

    HloInstructionProto instr;
    std::vector<const Shape*> operand_shape_ptrs;
    TF_ASSIGN_OR_RETURN(const auto& operand_shapes, GetOperandShapes(operands));
    absl::c_transform(operand_shapes, std::back_inserter(operand_shape_ptrs),
                      [](const Shape& shape) { return &shape; });
    TF_ASSIGN_OR_RETURN(const ProgramShape& called_program_shape,
                        computation.GetProgramShape());
    TF_ASSIGN_OR_RETURN(
        Shape shape, ShapeInference::InferMapShape(
                         operand_shape_ptrs, called_program_shape, dimensions));
    *instr.mutable_shape() = shape.ToProto();

    Shape output_shape(instr.shape());
    const int64_t output_rank = output_shape.rank();
    AddCalledComputation(computation, &instr);
    std::vector<ZkxOp> new_operands(operands.begin(), operands.end());
    for (ZkxOp& new_operand : new_operands) {
      TF_ASSIGN_OR_RETURN(const Shape* shape, GetShapePtr(new_operand));
      const int64_t rank = shape->rank();
      if (rank != output_rank) {
        TF_ASSIGN_OR_RETURN(new_operand,
                            InDimBroadcast(output_shape, new_operand, {}));
        TF_ASSIGN_OR_RETURN(shape, GetShapePtr(new_operand));
      }
      if (!ShapeUtil::SameDimensions(output_shape, *shape)) {
        TF_ASSIGN_OR_RETURN(new_operand,
                            AddBroadcastSequence(output_shape, new_operand));
      }
    }

    return AddInstruction(std::move(instr), HloOpcode::kMap, new_operands);
  });
}

ZkxOp ZkxBuilder::While(const ZkxComputation& condition,
                        const ZkxComputation& body, ZkxOp init) {
  return ReportErrorOrReturn([&]() -> absl::StatusOr<ZkxOp> {
    // Infer shape.
    TF_ASSIGN_OR_RETURN(const auto& body_program_shape, body.GetProgramShape());
    TF_ASSIGN_OR_RETURN(const auto& condition_program_shape,
                        condition.GetProgramShape());
    TF_ASSIGN_OR_RETURN(const Shape* init_shape, GetShapePtr(init));
    TF_ASSIGN_OR_RETURN(Shape shape, ShapeInference::InferWhileShape(
                                         condition_program_shape,
                                         body_program_shape, *init_shape));
    return WhileInternal(shape, condition, body, init);
  });
}

absl::StatusOr<ZkxOp> ZkxBuilder::WhileInternal(const Shape& shape,
                                                const ZkxComputation& condition,
                                                const ZkxComputation& body,
                                                ZkxOp init) {
  HloInstructionProto instr;
  *instr.mutable_shape() = shape.ToProto();
  // Body comes before condition computation in the vector.
  AddCalledComputation(body, &instr);
  AddCalledComputation(condition, &instr);
  return AddInstruction(std::move(instr), HloOpcode::kWhile, {init});
}

ZkxOp ZkxBuilder::Scatter(ZkxOp input, ZkxOp scatter_indices, ZkxOp updates,
                          const ZkxComputation& update_computation,
                          const ScatterDimensionNumbers& dimension_numbers,
                          bool indices_are_sorted, bool unique_indices) {
  return Scatter(absl::MakeConstSpan(&input, 1), scatter_indices,
                 absl::MakeConstSpan(&updates, 1), update_computation,
                 dimension_numbers, indices_are_sorted, unique_indices);
}

ZkxOp ZkxBuilder::Scatter(absl::Span<const ZkxOp> inputs, ZkxOp scatter_indices,
                          absl::Span<const ZkxOp> updates,
                          const ZkxComputation& update_computation,
                          const ScatterDimensionNumbers& dimension_numbers,
                          bool indices_are_sorted, bool unique_indices) {
  return ReportErrorOrReturn([&]() -> absl::StatusOr<ZkxOp> {
    if (inputs.empty()) {
      return absl::InvalidArgumentError("Scatter inputs cannot be empty.");
    }
    if (inputs.size() != updates.size()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Scatter should have same number of inputs and updates: %d vs %d.",
          inputs.size(), updates.size()));
    }
    absl::InlinedVector<const Shape*, 3> operand_shapes;
    operand_shapes.reserve(inputs.size() + 1 + updates.size());
    for (const ZkxOp& input : inputs) {
      TF_ASSIGN_OR_RETURN(const Shape* input_shape, GetShapePtr(input));
      operand_shapes.push_back(input_shape);
    }
    TF_ASSIGN_OR_RETURN(const Shape* scatter_indices_shape,
                        GetShapePtr(scatter_indices));
    operand_shapes.push_back(scatter_indices_shape);
    for (const ZkxOp& update : updates) {
      TF_ASSIGN_OR_RETURN(const Shape* update_shape, GetShapePtr(update));
      operand_shapes.push_back(update_shape);
    }
    TF_ASSIGN_OR_RETURN(const ProgramShape& to_apply_shape,
                        update_computation.GetProgramShape());
    TF_ASSIGN_OR_RETURN(Shape shape,
                        ShapeInference::InferScatterShape(
                            operand_shapes, to_apply_shape, dimension_numbers));
    return ScatterInternal(shape, inputs, scatter_indices, updates,
                           update_computation, dimension_numbers,
                           indices_are_sorted, unique_indices);
  });
}

absl::StatusOr<ZkxOp> ZkxBuilder::ScatterInternal(
    const Shape& shape, absl::Span<const ZkxOp> inputs, ZkxOp scatter_indices,
    absl::Span<const ZkxOp> updates, const ZkxComputation& update_computation,
    const ScatterDimensionNumbers& dimension_numbers, bool indices_are_sorted,
    bool unique_indices) {
  return ReportErrorOrReturn([&]() -> absl::StatusOr<ZkxOp> {
    HloInstructionProto instr;
    instr.set_indices_are_sorted(indices_are_sorted);
    instr.set_unique_indices(unique_indices);
    *instr.mutable_shape() = shape.ToProto();
    *instr.mutable_scatter_dimension_numbers() = dimension_numbers;

    AddCalledComputation(update_computation, &instr);
    absl::InlinedVector<ZkxOp, 3> operands;
    operands.reserve(inputs.size() + 1 + updates.size());
    absl::c_copy(inputs, std::back_inserter(operands));
    operands.push_back(scatter_indices);
    absl::c_copy(updates, std::back_inserter(operands));
    return AddInstruction(std::move(instr), HloOpcode::kScatter, operands);
  });
}

ZkxOp ZkxBuilder::Conditional(ZkxOp predicate, ZkxOp true_operand,
                              const ZkxComputation& true_computation,
                              ZkxOp false_operand,
                              const ZkxComputation& false_computation) {
  return ReportErrorOrReturn([&]() -> absl::StatusOr<ZkxOp> {
    TF_ASSIGN_OR_RETURN(const Shape* shape, GetShapePtr(predicate));

    if (!ShapeUtil::IsScalar(*shape) || shape->element_type() != PRED) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Argument to predicated-Conditional is not a scalar of PRED type "
          "(%s).",
          ShapeUtil::HumanString(*shape)));
    }
    // The index of true_computation must be 0 and that of false computation
    // must be 1.
    return ConditionalImpl(predicate, {&true_computation, &false_computation},
                           {true_operand, false_operand});
  });
}

ZkxOp ZkxBuilder::Conditional(
    ZkxOp branch_index,
    absl::Span<const ZkxComputation* const> branch_computations,
    absl::Span<const ZkxOp> branch_operands) {
  return ReportErrorOrReturn([&]() -> absl::StatusOr<ZkxOp> {
    TF_ASSIGN_OR_RETURN(const Shape* shape, GetShapePtr(branch_index));

    if (!ShapeUtil::IsScalar(*shape) || shape->element_type() != S32) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Argument to indexed-Conditional is not a scalar of S32 type (%s).",
          ShapeUtil::HumanString(*shape)));
    }
    return ConditionalImpl(branch_index, branch_computations, branch_operands);
  });
}

ZkxOp ZkxBuilder::ConditionalImpl(
    ZkxOp branch_index,
    absl::Span<const ZkxComputation* const> branch_computations,
    absl::Span<const ZkxOp> branch_operands) {
  return ReportErrorOrReturn([&]() -> absl::StatusOr<ZkxOp> {
    HloInstructionProto instr;

    TF_ASSIGN_OR_RETURN(const Shape* branch_index_shape,
                        GetShapePtr(branch_index));
    std::vector<Shape> branch_operand_shapes(branch_operands.size());
    std::vector<ProgramShape> branch_computation_shapes(
        branch_computations.size());
    for (int j = 0, end = branch_operands.size(); j < end; ++j) {
      TF_ASSIGN_OR_RETURN(branch_operand_shapes[j],
                          GetShape(branch_operands[j]));
      TF_ASSIGN_OR_RETURN(branch_computation_shapes[j],
                          branch_computations[j]->GetProgramShape());
    }
    TF_ASSIGN_OR_RETURN(const Shape shape,
                        ShapeInference::InferConditionalShape(
                            *branch_index_shape, branch_computation_shapes,
                            branch_operand_shapes));
    *instr.mutable_shape() = shape.ToProto();

    for (const ZkxComputation* branch_computation : branch_computations) {
      AddCalledComputation(*branch_computation, &instr);
    }

    std::vector<ZkxOp> operands(1, branch_index);
    for (const ZkxOp branch_operand : branch_operands) {
      operands.push_back(branch_operand);
    }
    return AddInstruction(std::move(instr), HloOpcode::kConditional,
                          absl::MakeSpan(operands));
  });
}

absl::Status ZkxBuilder::CheckOpBuilder(ZkxOp op) const {
  if (this != op.builder()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "ZkxOp with handle %d is built by builder '%s', but is trying to use "
        "it in builder '%s'",
        op.handle(), op.builder()->name(), name()));
  }
  return absl::OkStatus();
}

ZkxOp ZkxBuilder::Reduce(ZkxOp operand, ZkxOp init_value,
                         const ZkxComputation& computation,
                         absl::Span<const int64_t> dimensions_to_reduce) {
  return Reduce(absl::Span<const ZkxOp>({operand}),
                absl::Span<const ZkxOp>({init_value}), computation,
                dimensions_to_reduce);
}

ZkxOp ZkxBuilder::Reduce(absl::Span<const ZkxOp> operands,
                         absl::Span<const ZkxOp> init_values,
                         const ZkxComputation& computation,
                         absl::Span<const int64_t> dimensions_to_reduce) {
  return ReportErrorOrReturn([&]() -> absl::StatusOr<ZkxOp> {
    TF_ASSIGN_OR_RETURN(const ProgramShape& called_program_shape,
                        computation.GetProgramShape());

    std::vector<ZkxOp> all_operands;
    all_operands.insert(all_operands.end(), operands.begin(), operands.end());
    all_operands.insert(all_operands.end(), init_values.begin(),
                        init_values.end());

    std::vector<const Shape*> operand_shape_ptrs;
    TF_ASSIGN_OR_RETURN(const auto& operand_shapes,
                        GetOperandShapes(all_operands));
    absl::c_transform(operand_shapes, std::back_inserter(operand_shape_ptrs),
                      [](const Shape& shape) { return &shape; });

    TF_ASSIGN_OR_RETURN(
        Shape shape,
        ShapeInference::InferReduceShape(
            operand_shape_ptrs, dimensions_to_reduce, called_program_shape));
    return ReduceInternal(shape, all_operands, computation,
                          dimensions_to_reduce);
  });
}

absl::StatusOr<ZkxOp> ZkxBuilder::ReduceInternal(
    const Shape& shape, absl::Span<const ZkxOp> all_operands,
    const ZkxComputation& computation,
    absl::Span<const int64_t> dimensions_to_reduce) {
  return ReportErrorOrReturn([&]() -> absl::StatusOr<ZkxOp> {
    HloInstructionProto instr;
    *instr.mutable_shape() = shape.ToProto();

    for (int64_t dim : dimensions_to_reduce) {
      instr.add_dimensions(dim);
    }

    AddCalledComputation(computation, &instr);
    return AddInstruction(std::move(instr), HloOpcode::kReduce, all_operands);
  });
}

ZkxOp ZkxBuilder::ReduceWindow(ZkxOp operand, ZkxOp init_value,
                               const ZkxComputation& computation,
                               const Window& window) {
  return ReduceWindow(absl::Span<const ZkxOp>({operand}),
                      absl::Span<const ZkxOp>({init_value}), computation,
                      window);
}

ZkxOp ZkxBuilder::ReduceWindow(absl::Span<const ZkxOp> operands,
                               absl::Span<const ZkxOp> init_values,
                               const ZkxComputation& computation,
                               const Window& window) {
  return ReportErrorOrReturn([&]() -> absl::StatusOr<ZkxOp> {
    TF_ASSIGN_OR_RETURN(const ProgramShape& called_program_shape,
                        computation.GetProgramShape());

    std::vector<ZkxOp> all_operands;
    all_operands.insert(all_operands.end(), operands.begin(), operands.end());
    all_operands.insert(all_operands.end(), init_values.begin(),
                        init_values.end());

    std::vector<const Shape*> operand_shape_ptrs;
    std::vector<const Shape*> init_value_shape_ptrs;
    TF_ASSIGN_OR_RETURN(const auto& all_shapes, GetOperandShapes(all_operands));
    for (size_t i = 0; i < operands.size(); ++i) {
      operand_shape_ptrs.push_back(&all_shapes[i]);
    }
    for (size_t i = 0; i < init_values.size(); ++i) {
      init_value_shape_ptrs.push_back(&all_shapes[operands.size() + i]);
    }

    TF_ASSIGN_OR_RETURN(Shape shape,
                        ShapeInference::InferReduceWindowShape(
                            operand_shape_ptrs, init_value_shape_ptrs, window,
                            called_program_shape));
    return ReduceWindowInternal(shape, all_operands, computation, window);
  });
}

absl::StatusOr<ZkxOp> ZkxBuilder::ReduceWindowInternal(
    const Shape& shape, absl::Span<const ZkxOp> all_operands,
    const ZkxComputation& computation, const Window& window) {
  return ReportErrorOrReturn([&]() -> absl::StatusOr<ZkxOp> {
    HloInstructionProto instr;
    *instr.mutable_shape() = shape.ToProto();
    *instr.mutable_window() = window;

    AddCalledComputation(computation, &instr);
    return AddInstruction(std::move(instr), HloOpcode::kReduceWindow,
                          all_operands);
  });
}

ZkxOp ZkxBuilder::GetDimensionSize(ZkxOp operand, int64_t dimension) {
  return ReportErrorOrReturn([&]() -> absl::StatusOr<ZkxOp> {
    HloInstructionProto instr;
    TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));
    TF_ASSIGN_OR_RETURN(Shape shape, ShapeInference::InferGetDimensionSizeShape(
                                         *operand_shape, dimension));
    // Calling GetDimensionSize on a static dimension returns a constant
    // instruction.
    if (operand_shape->is_static_dimension(dimension)) {
      return ConstantR0<int32_t>(this, operand_shape->dimensions(dimension));
    }
    *instr.mutable_shape() = shape.ToProto();
    instr.add_dimensions(dimension);
    return AddInstruction(std::move(instr), HloOpcode::kGetDimensionSize,
                          {operand});
  });
}

ZkxOp ZkxBuilder::RemoveDynamicDimension(ZkxOp operand, int64_t dimension) {
  return ReportErrorOrReturn([&]() -> absl::StatusOr<ZkxOp> {
    TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));

    Shape shape = *operand_shape;
    shape.set_dynamic_dimension(dimension, false);
    // Setting an op's dynamic dimension to its static size removes the dynamic
    // dimension.
    ZkxOp static_size =
        ConstantR0<int32_t>(this, operand_shape->dimensions(dimension));
    return SetDimensionSizeInternal(shape, operand, static_size, dimension);
  });
}

ZkxOp ZkxBuilder::SetDimensionSize(ZkxOp operand, ZkxOp val,
                                   int64_t dimension) {
  return ReportErrorOrReturn([&]() -> absl::StatusOr<ZkxOp> {
    TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));
    TF_ASSIGN_OR_RETURN(const Shape* val_shape, GetShapePtr(val));

    TF_ASSIGN_OR_RETURN(Shape shape,
                        ShapeInference::InferSetDimensionSizeShape(
                            *operand_shape, *val_shape, dimension));
    return SetDimensionSizeInternal(shape, operand, val, dimension);
  });
}

absl::StatusOr<ZkxOp> ZkxBuilder::SetDimensionSizeInternal(const Shape& shape,
                                                           ZkxOp operand,
                                                           ZkxOp val,
                                                           int64_t dimension) {
  // Note that both SetDimensionSize and RemoveDynamicDimension use
  // HloOpcode::kSetDimensionSize internally. However, The SetDimensionSize
  // builder always produces an output with a dynamic bound on the given
  // dimension, while RemoveDynamicDimension removes the dynamic dimension from
  // the shape. The only case where HloOpcode::kSetDimensionSize should have a
  // non-dynamic bound on the given dimension is where the operand is constant
  // and exactly equal to the size of the dimension.
  // TODO(b/298671312): Clarify the semantics of SetDimensionSize and consider
  // adding a separate RemoveDynamicDimension opcode.
  HloInstructionProto instr;
  *instr.mutable_shape() = shape.ToProto();
  instr.add_dimensions(dimension);
  return AddInstruction(std::move(instr), HloOpcode::kSetDimensionSize,
                        {operand, val});
}

absl::StatusOr<bool> ZkxBuilder::IsConstant(ZkxOp operand) const {
  TF_RETURN_IF_ERROR(first_error_);

  // Verify that the handle is valid.
  TF_RETURN_IF_ERROR(LookUpInstruction(operand).status());

  bool is_constant = true;
  absl::flat_hash_set<int64_t> visited;
  IsConstantVisitor(operand.handle(), /*depth=*/0, &visited, &is_constant);
  return is_constant;
}

absl::StatusOr<ZkxComputation> ZkxBuilder::BuildConstantSubGraph(
    ZkxOp root_op, bool dynamic_dimension_is_minus_one) {
  TF_ASSIGN_OR_RETURN(bool is_constant, IsConstant(root_op));
  if (!is_constant) {
    auto op_status = LookUpInstruction(root_op);
    std::string op_string =
        op_status.ok() ? op_status.value()->name() : "<unknown operation>";
    return absl::InvalidArgumentError(absl::StrFormat(
        "Operand to BuildConstantSubGraph depends on a parameter.\n\n"
        "  op requested for constant subgraph: %s\n\n"
        "This is an internal error that typically happens when the ZKX user "
        "(e.g. TensorFlow) is attempting to determine a value that must be a "
        "compile-time constant (e.g. an array dimension) but it is not capable "
        "of being evaluated at ZKX compile time.\n\n"
        "Please file a usability bug with the framework being used (e.g. "
        "TensorFlow).",
        op_string));
  }

  TF_ASSIGN_OR_RETURN(const HloInstructionProto* root,
                      LookUpInstruction(root_op));
  if (VLOG_IS_ON(4)) {
    VLOG(4) << "Build constant subgraph for:\n" << OpToString(root_op);
  }

  HloComputationProto entry;
  SetProtoIdAndName(&entry, absl::StrCat(name_, "_compute_constant"),
                    kNameSeparator, GetNextId());
  ProgramShapeProto* program_shape = entry.mutable_program_shape();
  *program_shape->mutable_result() = root->shape();

  // We use std::set to keep the instruction ids in ascending order (which is
  // also a valid dependency order). The related ops will be added to the
  // subgraph in the same order.
  std::set<int64_t> related_ops;
  absl::flat_hash_map<int64_t, int64_t> substitutions;
  absl::flat_hash_set<int64_t> related_calls;  // Related computations.
  std::queue<int64_t> worklist;
  worklist.push(root->id());
  related_ops.insert(root->id());

  while (!worklist.empty()) {
    int64_t handle = worklist.front();
    worklist.pop();
    TF_ASSIGN_OR_RETURN(const HloInstructionProto* instr_proto,
                        LookUpInstructionByHandle(handle));

    auto default_behavior = [&related_ops, &worklist, &related_calls,
                             instr_proto]() {
      for (int64_t id : instr_proto->operand_ids()) {
        if (related_ops.insert(id).second) {
          worklist.push(id);
        }
      }
      for (int64_t called_id : instr_proto->called_computation_ids()) {
        related_calls.insert(called_id);
      }
    };

    if (instr_proto->opcode() ==
            HloOpcodeString(HloOpcode::kGetDimensionSize) ||
        InstrIsSetBound(instr_proto)) {
      int32_t constant_value = -1;
      HloInstructionProto const_instr;

      if (instr_proto->opcode() ==
          HloOpcodeString(HloOpcode::kGetDimensionSize)) {
        // At this point, BuildConstantSubGraph should never encounter a
        // GetDimensionSize with a dynamic dimension. IsConstant check would
        // have failed at the beginning of this function.
        //
        // Replace GetDimensionSize with a Constant representing the static
        // bound of the shape.
        int64_t dimension = instr_proto->dimensions(0);
        int64_t operand_handle = instr_proto->operand_ids(0);
        TF_ASSIGN_OR_RETURN(const HloInstructionProto* operand_proto,
                            LookUpInstructionByHandle(operand_handle));

        if (!(operand_proto->shape().is_dynamic_dimension(dimension) &&
              dynamic_dimension_is_minus_one)) {
          constant_value = static_cast<int32_t>(
              operand_proto->shape().dimensions(dimension));
        }
        Literal literal = LiteralUtil::CreateR0(constant_value);
        *const_instr.mutable_literal() = literal.ToProto();
        *const_instr.mutable_shape() = literal.shape().ToProto();
      } else {
        if (instr_proto->literal().shape().element_type() == TUPLE) {
          *const_instr.mutable_literal() =
              // First literal of SetBound contains bounds, second literal
              // contains dynamism indicators.
              instr_proto->literal().tuple_literals(0);
        } else {
          *const_instr.mutable_literal() = instr_proto->literal();
        }

        *const_instr.mutable_shape() = instr_proto->shape();
      }
      *const_instr.mutable_opcode() =
          std::string(HloOpcodeString(HloOpcode::kConstant));
      const_instr.set_id(handle);
      *const_instr.mutable_name() =
          GetFullName(const_instr.opcode(), kNameSeparator, const_instr.id());
      *entry.add_instructions() =
          const_instr;  // Add to the result constant graph.

    } else if (instr_proto->opcode() ==
               HloOpcodeString(HloOpcode::kGetTupleElement)) {
      // Look through GTE(Tuple(..), i).
      TF_ASSIGN_OR_RETURN(
          const HloInstructionProto* maybe_tuple_instr,
          LookUpInstructionByHandle(instr_proto->operand_ids(0)));

      if (maybe_tuple_instr->opcode() == HloOpcodeString(HloOpcode::kTuple)) {
        int64_t id = maybe_tuple_instr->operand_ids(instr_proto->tuple_index());
        // Enqueue any dependencies of `id`.
        if (related_ops.insert(id).second) {
          worklist.push(id);
        }
        substitutions[handle] = id;

      } else {
        default_behavior();
      }

    } else {
      default_behavior();
    }
  }

  // Resolve any substitutions for the root id.
  int64_t root_id = root->id();
  auto it = substitutions.find(root_id);
  while (it != substitutions.end()) {
    root_id = it->second;
    it = substitutions.find(root_id);
  }
  entry.set_root_id(root_id);

  // Add related ops to the computation.
  for (int64_t id : related_ops) {
    if (substitutions.find(id) != substitutions.end()) {
      // Skip adding this instruction; we will replace references to it with the
      // substitution instruction's id.
      continue;
    }
    TF_ASSIGN_OR_RETURN(const HloInstructionProto* instr_src,
                        LookUpInstructionByHandle(id));

    if (instr_src->opcode() == HloOpcodeString(HloOpcode::kGetDimensionSize) ||
        InstrIsSetBound(instr_src)) {
      continue;
    }
    HloInstructionProto* instr = entry.add_instructions();
    *instr = *instr_src;
    // Replace operands in case we have substitutions mapped.
    instr->clear_operand_ids();
    for (int64_t operand_id : instr_src->operand_ids()) {
      auto it = substitutions.find(operand_id);
      while (it != substitutions.end()) {
        operand_id = it->second;
        it = substitutions.find(operand_id);
      }
      instr->add_operand_ids(operand_id);
    }
    // Ensures that the instruction names are unique among the graph.
    const std::string& new_name =
        absl::StrCat(instr->name(), ".", entry.id(), ".", instr->id());
    instr->set_name(new_name);
  }

  ZkxComputation computation(entry.id());
  HloModuleProto* module = computation.mutable_proto();
  module->set_name(entry.name());
  module->set_id(entry.id());
  module->set_entry_computation_name(entry.name());
  module->set_entry_computation_id(entry.id());
  *module->mutable_host_program_shape() = *program_shape;
  for (auto& e : embedded_) {
    if (related_calls.find(e.second.id()) != related_calls.end()) {
      *module->add_computations() = e.second;
    }
  }
  *module->add_computations() = std::move(entry);
  if (VLOG_IS_ON(4)) {
    VLOG(4) << "Constant computation:\n" << module->DebugString();
  }
  return std::move(computation);
}

std::unique_ptr<ZkxBuilder> ZkxBuilder::CreateSubBuilder(
    const std::string& computation_name) {
  auto sub_builder = std::make_unique<ZkxBuilder>(computation_name);
  sub_builder->parent_builder_ = this;
  sub_builder->die_immediately_on_error_ = this->die_immediately_on_error_;
  return sub_builder;
}

absl::StatusOr<ZkxOp> ZkxBuilder::AddInstruction(
    HloInstructionProto&& instr, HloOpcode opcode,
    absl::Span<const ZkxOp> operands) {
  TF_RETURN_IF_ERROR(first_error_);

  const int64_t handle = GetNextId();
  instr.set_id(handle);
  *instr.mutable_opcode() = std::string(HloOpcodeString(opcode));
  if (instr.name().empty()) {
    instr.set_name(instr.opcode());
  }
  for (const auto& operand : operands) {
    if (operand.builder_ == nullptr) {
      return absl::InvalidArgumentError(
          absl::StrFormat("invalid ZkxOp with handle %d", operand.handle()));
    }
    if (operand.builder_ != this) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Do not add ZkxOp from builder %s to builder %s",
                          operand.builder_->name(), this->name()));
    }
    instr.add_operand_ids(operand.handle());
  }

  if (one_shot_metadata_.has_value()) {
    *instr.mutable_metadata() = one_shot_metadata_.value();
    one_shot_metadata_.reset();
  } else {
    *instr.mutable_metadata() = metadata_;
  }
  if (sharding_) {
    TF_RETURN_IF_ERROR(NormalizeAndAssignSharing(&instr, *sharding_));
  }
  *instr.mutable_frontend_attributes() = frontend_attributes_;

  handle_to_index_[handle] = instructions_.size();
  instructions_.push_back(std::move(instr));
  instruction_shapes_.push_back(
      std::make_unique<Shape>(instructions_.back().shape()));

  ZkxOp op(handle, this);
  return op;
}

absl::StatusOr<ZkxOp> ZkxBuilder::AddOpWithShape(
    HloOpcode opcode, const Shape& shape, absl::Span<const ZkxOp> operands) {
  HloInstructionProto instr;
  *instr.mutable_shape() = shape.ToProto();
  return AddInstruction(std::move(instr), opcode, operands);
}

void ZkxBuilder::AddCalledComputation(const ZkxComputation& computation,
                                      HloInstructionProto* instr) {
  absl::flat_hash_map<int64_t, int64_t> remapped_ids;
  std::vector<HloComputationProto> imported_computations;
  imported_computations.reserve(computation.proto().computations_size());
  // Before we import the computations by remapping IDs, and capturing the
  // old->new mappings in remapped_ids.
  for (const HloComputationProto& e : computation.proto().computations()) {
    HloComputationProto new_computation(e);
    int64_t computation_id = GetNextId();
    remapped_ids[new_computation.id()] = computation_id;
    SetProtoIdAndName(&new_computation,
                      GetBaseName(new_computation.name(), kNameSeparator),
                      kNameSeparator, computation_id);
    for (auto& instruction : *new_computation.mutable_instructions()) {
      int64_t instruction_id = GetNextId();
      remapped_ids[instruction.id()] = instruction_id;
      SetProtoIdAndName(&instruction,
                        GetBaseName(instruction.name(), kNameSeparator),
                        kNameSeparator, instruction_id);
    }
    new_computation.set_root_id(remapped_ids.at(new_computation.root_id()));

    imported_computations.push_back(std::move(new_computation));
  }
  // Once we have imported all the computations, and captured all the ID
  // mappings, we go back and fixup the IDs in the imported computations.
  instr->add_called_computation_ids(
      remapped_ids.at(computation.proto().entry_computation_id()));
  for (auto& imported_computation : imported_computations) {
    for (auto& instruction : *imported_computation.mutable_instructions()) {
      for (auto& operand_id : *instruction.mutable_operand_ids()) {
        operand_id = remapped_ids.at(operand_id);
      }
      for (auto& control_predecessor_id :
           *instruction.mutable_control_predecessor_ids()) {
        control_predecessor_id = remapped_ids.at(control_predecessor_id);
      }
      for (auto& called_computation_id :
           *instruction.mutable_called_computation_ids()) {
        called_computation_id = remapped_ids.at(called_computation_id);
      }
    }

    int64_t computation_id = imported_computation.id();
    for (int64_t i = 0; i < imported_computation.instructions_size(); ++i) {
      ImportedInstruction imported_instruction;
      imported_instruction.computation_id = computation_id;
      imported_instruction.instruction_index = i;
      handle_to_imported_index_.insert(
          {imported_computation.instructions(i).id(), imported_instruction});
    }
    embedded_.insert({computation_id, std::move(imported_computation)});
  }
}

absl::StatusOr<const HloInstructionProto*> ZkxBuilder::LookUpInstruction(
    const ZkxOp op) const {
  TF_RETURN_IF_ERROR(first_error_);
  return LookUpInstructionInternal<const HloInstructionProto*>(op);
}

absl::StatusOr<const HloInstructionProto*>
ZkxBuilder::LookUpInstructionByHandle(int64_t handle) const {
  return LookUpInstructionByHandleInternal<const HloInstructionProto*>(handle);
}

absl::StatusOr<HloInstructionProto*> ZkxBuilder::LookUpMutableInstruction(
    const ZkxOp op) {
  TF_RETURN_IF_ERROR(first_error_);
  return LookUpInstructionInternal<HloInstructionProto*>(op);
}

absl::StatusOr<HloInstructionProto*>
ZkxBuilder::LookUpMutableInstructionByHandle(int64_t handle) {
  return LookUpInstructionByHandleInternal<HloInstructionProto*>(handle);
}

// Enqueues a "retrieve parameter value" instruction for a parameter that was
// passed to the computation.
ZkxOp Parameter(ZkxBuilder* builder, int64_t parameter_number,
                const Shape& shape, const std::string& name) {
  std::vector<bool> empty_bools;
  return Parameter(builder, parameter_number, shape, name, empty_bools);
}

ZkxOp Parameter(ZkxBuilder* builder, int64_t parameter_number,
                const Shape& shape, const std::string& name,
                const std::vector<bool>& replicated_at_leaf_buffers) {
  return builder->Parameter(parameter_number, shape, name,
                            replicated_at_leaf_buffers);
}

// Enqueues a constant with the value of the given literal onto the
// computation.
ZkxOp ConstantLiteral(ZkxBuilder* builder, const LiteralSlice& literal) {
  return builder->ConstantLiteral(literal);
}

ZkxOp Broadcast(const ZkxOp operand,
                absl::Span<const int64_t> broadcast_sizes) {
  return operand.builder()->Broadcast(operand, broadcast_sizes);
}

ZkxOp BroadcastInDim(const ZkxOp operand,
                     absl::Span<const int64_t> out_dim_size,
                     absl::Span<const int64_t> broadcast_dimensions) {
  return operand.builder()->BroadcastInDim(operand, out_dim_size,
                                           broadcast_dimensions);
}

ZkxOp Copy(const ZkxOp operand) {
  return operand.builder()->UnaryOp(HloOpcode::kCopy, operand);
}

ZkxOp Pad(const ZkxOp operand, const ZkxOp padding_value,
          const PaddingConfig& padding_config) {
  return operand.builder()->Pad(operand, padding_value, padding_config);
}

ZkxOp PadInDim(ZkxOp operand, ZkxOp padding_value, int64_t dimno,
               int64_t pad_lo, int64_t pad_hi) {
  return operand.builder()->PadInDim(operand, padding_value, dimno, pad_lo,
                                     pad_hi);
}

ZkxOp Reshape(const ZkxOp operand, absl::Span<const int64_t> dimensions,
              absl::Span<const int64_t> new_sizes) {
  return operand.builder()->Reshape(operand, dimensions, new_sizes);
}

ZkxOp Reshape(const ZkxOp operand, absl::Span<const int64_t> new_sizes) {
  return operand.builder()->Reshape(operand, new_sizes);
}

ZkxOp Reshape(const Shape& shape, ZkxOp operand) {
  return operand.builder()->Reshape(shape, operand);
}

ZkxOp DynamicReshape(ZkxOp operand, absl::Span<const ZkxOp> dim_sizes,
                     absl::Span<const int64_t> new_size_bounds,
                     const std::vector<bool>& dims_are_dynamic) {
  return operand.builder()->DynamicReshape(operand, dim_sizes, new_size_bounds,
                                           dims_are_dynamic);
}

ZkxOp ReshapeWithInferredDimension(ZkxOp operand,
                                   absl::Span<const int64_t> new_sizes,
                                   int64_t inferred_dimension) {
  return operand.builder()->Reshape(operand, new_sizes, inferred_dimension);
}

ZkxOp Collapse(const ZkxOp operand, absl::Span<const int64_t> dimensions) {
  return operand.builder()->Collapse(operand, dimensions);
}

ZkxOp Slice(const ZkxOp operand, absl::Span<const int64_t> start_indices,
            absl::Span<const int64_t> limit_indices,
            absl::Span<const int64_t> strides) {
  return operand.builder()->Slice(operand, start_indices, limit_indices,
                                  strides);
}

ZkxOp SliceInDim(const ZkxOp operand, int64_t start_index, int64_t limit_index,
                 int64_t stride, int64_t dimno) {
  return operand.builder()->SliceInDim(operand, start_index, limit_index,
                                       stride, dimno);
}

ZkxOp DynamicSlice(const ZkxOp operand, absl::Span<const ZkxOp> start_indices,
                   absl::Span<const int64_t> slice_sizes) {
  return operand.builder()->DynamicSlice(operand, start_indices, slice_sizes);
}

ZkxOp DynamicUpdateSlice(const ZkxOp operand, const ZkxOp update,
                         absl::Span<const ZkxOp> start_indices) {
  return operand.builder()->DynamicUpdateSlice(operand, update, start_indices);
}

ZkxOp ConcatInDim(ZkxBuilder* builder, absl::Span<const ZkxOp> operands,
                  int64_t dimension) {
  return builder->ConcatInDim(operands, dimension);
}

ZkxOp Select(const ZkxOp pred, const ZkxOp on_true, const ZkxOp on_false) {
  return pred.builder()->Select(pred, on_true, on_false);
}

ZkxOp Tuple(ZkxBuilder* builder, absl::Span<const ZkxOp> elements) {
  return builder->Tuple(elements);
}

ZkxOp GetTupleElement(const ZkxOp tuple_data, int64_t index) {
  return tuple_data.builder()->GetTupleElement(tuple_data, index);
}

ZkxOp Eq(const ZkxOp lhs, const ZkxOp rhs,
         absl::Span<const int64_t> broadcast_dimensions) {
  return Compare(lhs, rhs, broadcast_dimensions, ComparisonDirection::kEq);
}

ZkxOp Ne(const ZkxOp lhs, const ZkxOp rhs,
         absl::Span<const int64_t> broadcast_dimensions) {
  return Compare(lhs, rhs, broadcast_dimensions, ComparisonDirection::kNe);
}

ZkxOp Ge(const ZkxOp lhs, const ZkxOp rhs,
         absl::Span<const int64_t> broadcast_dimensions) {
  return Compare(lhs, rhs, broadcast_dimensions, ComparisonDirection::kGe);
}

ZkxOp Gt(const ZkxOp lhs, const ZkxOp rhs,
         absl::Span<const int64_t> broadcast_dimensions) {
  return Compare(lhs, rhs, broadcast_dimensions, ComparisonDirection::kGt);
}

ZkxOp Le(const ZkxOp lhs, const ZkxOp rhs,
         absl::Span<const int64_t> broadcast_dimensions) {
  return Compare(lhs, rhs, broadcast_dimensions, ComparisonDirection::kLe);
}

ZkxOp Lt(const ZkxOp lhs, const ZkxOp rhs,
         absl::Span<const int64_t> broadcast_dimensions) {
  return Compare(lhs, rhs, broadcast_dimensions, ComparisonDirection::kLt);
}

ZkxOp Compare(const ZkxOp lhs, const ZkxOp rhs,
              absl::Span<const int64_t> broadcast_dimensions,
              ComparisonDirection direction) {
  return lhs.builder()->BinaryOp(HloOpcode::kCompare, lhs, rhs,
                                 broadcast_dimensions, direction);
}

ZkxOp Compare(const ZkxOp lhs, const ZkxOp rhs, ComparisonDirection direction) {
  return Compare(lhs, rhs, {}, direction);
}

ZkxOp Call(ZkxBuilder* builder, const ZkxComputation& computation,
           absl::Span<const ZkxOp> operands) {
  return builder->Call(computation, operands);
}

ZkxOp CompositeCall(ZkxBuilder* builder, const ZkxComputation& computation,
                    absl::Span<const ZkxOp> operands, const std::string& name,
                    std::optional<std::string_view> attributes,
                    std::optional<int64_t> version) {
  return builder->CompositeCall(computation, operands, name, attributes,
                                version);
}

ZkxOp Add(const ZkxOp lhs, const ZkxOp rhs,
          absl::Span<const int64_t> broadcast_dimensions) {
  return lhs.builder()->BinaryOp(HloOpcode::kAdd, lhs, rhs,
                                 broadcast_dimensions);
}

ZkxOp Sub(const ZkxOp lhs, const ZkxOp rhs,
          absl::Span<const int64_t> broadcast_dimensions) {
  return lhs.builder()->BinaryOp(HloOpcode::kSubtract, lhs, rhs,
                                 broadcast_dimensions);
}

ZkxOp Mul(const ZkxOp lhs, const ZkxOp rhs,
          absl::Span<const int64_t> broadcast_dimensions) {
  return lhs.builder()->BinaryOp(HloOpcode::kMultiply, lhs, rhs,
                                 broadcast_dimensions);
}

ZkxOp Div(const ZkxOp lhs, const ZkxOp rhs,
          absl::Span<const int64_t> broadcast_dimensions) {
  return lhs.builder()->BinaryOp(HloOpcode::kDivide, lhs, rhs,
                                 broadcast_dimensions);
}

ZkxOp Rem(const ZkxOp lhs, const ZkxOp rhs,
          absl::Span<const int64_t> broadcast_dimensions) {
  return lhs.builder()->BinaryOp(HloOpcode::kRemainder, lhs, rhs,
                                 broadcast_dimensions);
}

ZkxOp Max(const ZkxOp lhs, const ZkxOp rhs,
          absl::Span<const int64_t> broadcast_dimensions) {
  return lhs.builder()->BinaryOp(HloOpcode::kMaximum, lhs, rhs,
                                 broadcast_dimensions);
}

ZkxOp Min(const ZkxOp lhs, const ZkxOp rhs,
          absl::Span<const int64_t> broadcast_dimensions) {
  return lhs.builder()->BinaryOp(HloOpcode::kMinimum, lhs, rhs,
                                 broadcast_dimensions);
}

ZkxOp And(const ZkxOp lhs, const ZkxOp rhs,
          absl::Span<const int64_t> broadcast_dimensions) {
  return lhs.builder()->BinaryOp(HloOpcode::kAnd, lhs, rhs,
                                 broadcast_dimensions);
}

ZkxOp Or(const ZkxOp lhs, const ZkxOp rhs,
         absl::Span<const int64_t> broadcast_dimensions) {
  return lhs.builder()->BinaryOp(HloOpcode::kOr, lhs, rhs,
                                 broadcast_dimensions);
}

ZkxOp Xor(const ZkxOp lhs, const ZkxOp rhs,
          absl::Span<const int64_t> broadcast_dimensions) {
  return lhs.builder()->BinaryOp(HloOpcode::kXor, lhs, rhs,
                                 broadcast_dimensions);
}

ZkxOp Not(const ZkxOp operand) {
  return operand.builder()->UnaryOp(HloOpcode::kNot, operand);
}

ZkxOp PopulationCount(const ZkxOp operand) {
  return operand.builder()->UnaryOp(HloOpcode::kPopulationCount, operand);
}

ZkxOp ShiftLeft(const ZkxOp lhs, const ZkxOp rhs,
                absl::Span<const int64_t> broadcast_dimensions) {
  return lhs.builder()->BinaryOp(HloOpcode::kShiftLeft, lhs, rhs,
                                 broadcast_dimensions);
}

ZkxOp ShiftRightArithmetic(const ZkxOp lhs, const ZkxOp rhs,
                           absl::Span<const int64_t> broadcast_dimensions) {
  return lhs.builder()->BinaryOp(HloOpcode::kShiftRightArithmetic, lhs, rhs,
                                 broadcast_dimensions);
}

ZkxOp ShiftRightLogical(const ZkxOp lhs, const ZkxOp rhs,
                        absl::Span<const int64_t> broadcast_dimensions) {
  return lhs.builder()->BinaryOp(HloOpcode::kShiftRightLogical, lhs, rhs,
                                 broadcast_dimensions);
}

ZkxOp Reduce(const ZkxOp operand, const ZkxOp init_value,
             const ZkxComputation& computation,
             absl::Span<const int64_t> dimensions_to_reduce) {
  return operand.builder()->Reduce(operand, init_value, computation,
                                   dimensions_to_reduce);
}

// Reduces several arrays simultaneously among the provided dimensions, given
// "computation" as a reduction operator.
ZkxOp Reduce(ZkxBuilder* builder, absl::Span<const ZkxOp> operands,
             absl::Span<const ZkxOp> init_values,
             const ZkxComputation& computation,
             absl::Span<const int64_t> dimensions_to_reduce) {
  return builder->Reduce(operands, init_values, computation,
                         dimensions_to_reduce);
}

ZkxOp ReduceWindow(const ZkxOp operand, const ZkxOp init_value,
                   const ZkxComputation& computation, const Window& window) {
  return operand.builder()->ReduceWindow(operand, init_value, computation,
                                         window);
}

ZkxOp ReduceWindow(ZkxBuilder* builder, absl::Span<const ZkxOp> operands,
                   absl::Span<const ZkxOp> init_values,
                   const ZkxComputation& computation, const Window& window) {
  return builder->ReduceWindow(operands, init_values, computation, window);
}

ZkxOp Abs(const ZkxOp operand) {
  return operand.builder()->UnaryOp(HloOpcode::kAbs, operand);
}

ZkxOp Sign(const ZkxOp operand) {
  return operand.builder()->UnaryOp(HloOpcode::kSign, operand);
}

ZkxOp Clz(const ZkxOp operand) {
  return operand.builder()->UnaryOp(HloOpcode::kClz, operand);
}

ZkxOp Pow(const ZkxOp lhs, const ZkxOp rhs,
          absl::Span<const int64_t> broadcast_dimensions) {
  return lhs.builder()->BinaryOp(HloOpcode::kPower, lhs, rhs,
                                 broadcast_dimensions);
}

ZkxOp ConvertElementType(const ZkxOp operand, PrimitiveType new_element_type) {
  return operand.builder()->ConvertElementType(operand, new_element_type);
}

ZkxOp BitcastConvertType(const ZkxOp operand, PrimitiveType new_element_type) {
  return operand.builder()->BitcastConvertType(operand, new_element_type);
}

ZkxOp Neg(const ZkxOp operand) {
  return operand.builder()->UnaryOp(HloOpcode::kNegate, operand);
}

ZkxOp Transpose(const ZkxOp operand, absl::Span<const int64_t> permutation) {
  return operand.builder()->Transpose(operand, permutation);
}

ZkxOp Rev(const ZkxOp operand, absl::Span<const int64_t> dimensions) {
  return operand.builder()->Rev(operand, dimensions);
}

ZkxOp Sort(absl::Span<const ZkxOp> operands, const ZkxComputation& comparator,
           int64_t dimension, bool is_stable) {
  return operands[0].builder()->Sort(operands, comparator, dimension,
                                     is_stable);
}

ZkxOp Clamp(const ZkxOp min, const ZkxOp operand, const ZkxOp max) {
  return min.builder()->Clamp(min, operand, max);
}

ZkxOp Map(ZkxBuilder* builder, absl::Span<const ZkxOp> operands,
          const ZkxComputation& computation,
          absl::Span<const int64_t> dimensions,
          absl::Span<const ZkxOp> static_operands) {
  return builder->Map(operands, computation, dimensions, static_operands);
}

ZkxOp While(const ZkxComputation& condition, const ZkxComputation& body,
            const ZkxOp init) {
  return init.builder()->While(condition, body, init);
}

ZkxOp Conditional(const ZkxOp predicate, const ZkxOp true_operand,
                  const ZkxComputation& true_computation,
                  const ZkxOp false_operand,
                  const ZkxComputation& false_computation) {
  return predicate.builder()->Conditional(predicate, true_operand,
                                          true_computation, false_operand,
                                          false_computation);
}

ZkxOp Conditional(const ZkxOp branch_index,
                  absl::Span<const ZkxComputation* const> branch_computations,
                  absl::Span<const ZkxOp> branch_operands) {
  return branch_index.builder()->Conditional(branch_index, branch_computations,
                                             branch_operands);
}

ZkxOp Scatter(const ZkxOp input, const ZkxOp scatter_indices,
              const ZkxOp updates, const ZkxComputation& update_computation,
              const ScatterDimensionNumbers& dimension_numbers,
              bool indices_are_sorted, bool unique_indices) {
  return input.builder()->Scatter(input, scatter_indices, updates,
                                  update_computation, dimension_numbers,
                                  indices_are_sorted, unique_indices);
}

ZkxOp Scatter(absl::Span<const ZkxOp> inputs, ZkxOp scatter_indices,
              absl::Span<const ZkxOp> updates,
              const ZkxComputation& update_computation,
              const ScatterDimensionNumbers& dimension_numbers,
              bool indices_are_sorted, bool unique_indices) {
  return scatter_indices.builder()->Scatter(
      inputs, scatter_indices, updates, update_computation, dimension_numbers,
      indices_are_sorted, unique_indices);
}

ZkxOp CreateToken(ZkxBuilder* builder) { return builder->CreateToken(); }

ZkxOp Iota(ZkxBuilder* builder, PrimitiveType type, int64_t size) {
  return builder->Iota(type, size);
}

ZkxOp Iota(ZkxBuilder* builder, const Shape& shape, int64_t iota_dimension) {
  return builder->Iota(shape, iota_dimension);
}

ZkxOp GetDimensionSize(const ZkxOp operand, int64_t dimension) {
  return operand.builder()->GetDimensionSize(operand, dimension);
}

ZkxOp SetDimensionSize(const ZkxOp operand, const ZkxOp val,
                       int64_t dimension) {
  return operand.builder()->SetDimensionSize(operand, val, dimension);
}

ZkxOp RemoveDynamicDimension(const ZkxOp operand, int64_t dimension) {
  return operand.builder()->RemoveDynamicDimension(operand, dimension);
}

}  // namespace zkx
