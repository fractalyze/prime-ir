/* Copyright 2017 The OpenXLA Authors.
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

#include "zkx/tests/test_utils.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "zkx/hlo/analysis/hlo_dataflow_analysis.h"
#include "zkx/hlo/ir/hlo_casting_utils.h"
#include "zkx/literal_util.h"

namespace zkx {

namespace {

enum class ConstantType { kUnknown, kZero, kOne };

// Return the constant type required by this computation, if known.
ConstantType GetInitValue(const HloComputation& computation) {
  // TODO(b/77635120): Add init values, for min, max, and their arg variants.
  const HloInstruction* const root = computation.root_instruction();
  if (computation.num_parameters() != 2 || root->operand_count() != 2 ||
      root->operand(0)->opcode() != HloOpcode::kParameter ||
      root->operand(1)->opcode() != HloOpcode::kParameter ||
      root->operand(0) == root->operand(1)) {
    return ConstantType::kUnknown;
  }

  switch (root->opcode()) {
    case HloOpcode::kAdd:
      return ConstantType::kZero;
    case HloOpcode::kMultiply:
      return ConstantType::kOne;
    default:
      return ConstantType::kUnknown;
  }
}

// Reduce op may need a non-random initialization value.
bool NeedsInitValue(const HloUse& use) {
  const HloInstruction* const instruction = use.instruction;
  const HloOpcode opcode = instruction->opcode();
  const int64_t op_num = use.operand_number;
  return ((opcode == HloOpcode::kReduce &&
           op_num >= instruction->operand_count() / 2));
}

// Returns true if `dest' is reachable from `src' through data-formatting and
// custom call instructions within the same computation.
bool ReachableViaDataFormatting(const HloInstruction* src,
                                const HloInstruction* dest,
                                bool treat_gte_as_data_formatting) {
  if (src == dest) {
    return true;
  }
  switch (dest->opcode()) {
    case HloOpcode::kReshape:
    case HloOpcode::kTranspose:
    case HloOpcode::kCopy:
    case HloOpcode::kSlice:
      break;
    case HloOpcode::kCustomCall:
      // clang-format off
      // TODO(chokobole): Uncomment this. Dependency: HloInstruction::custom_call_target
      // clang-format on
      // if (dest->custom_call_target() == "AssumeGatherIndicesInBound") {
      //   break;
      // }
      return false;
    // TODO(b/249417724): a workaround for tuple param.
    case HloOpcode::kGetTupleElement:
      if (treat_gte_as_data_formatting) {
        break;
      } else {
        return false;
      }
    default:
      return false;
  }
  for (const auto* operand : dest->operands()) {
    if (ReachableViaDataFormatting(src, operand,
                                   treat_gte_as_data_formatting)) {
      return true;
    }
  }
  return false;
}

// Use dataflow analysis on each parameter to see if there are uses that would
// be problematic when generating input data.  Returns the list of
// instructions that correspond to their uses.
//
// Should be paired with the CreateLiteralForConstrainedUses() function below.
std::vector<HloInstruction*> FindConstrainedUses(
    const HloDataflowAnalysis& dataflow, const HloInstruction& param,
    bool treat_gte_as_data_formatting) {
  std::vector<HloInstruction*> constrained_uses;
  for (const auto& pair : dataflow.GetInstructionValueSet(&param)) {
    const HloValue& value = dataflow.GetUniqueValueAt(&param, pair.first);
    for (const HloUse& use : value.GetUses()) {
      HloInstruction* instruction = use.instruction;
      const HloOpcode opcode = instruction->opcode();
      const int64_t op_num = use.operand_number;
      if ((opcode == HloOpcode::kDynamicSlice && op_num >= 1) ||
          (opcode == HloOpcode::kDynamicUpdateSlice && op_num >= 2)) {
        constrained_uses.push_back(instruction);
      } else if ((opcode == HloOpcode::kGather ||
                  opcode == HloOpcode::kScatter) &&
                 op_num == 1) {
        constrained_uses.push_back(instruction);
      } else if (opcode == HloOpcode::kFusion) {
        const HloInstruction* const to_analyze =
            instruction->fused_parameter(op_num);
        auto fused_uses = FindConstrainedUses(dataflow, *to_analyze,
                                              treat_gte_as_data_formatting);
        constrained_uses.insert(constrained_uses.end(), fused_uses.begin(),
                                fused_uses.end());
      } else if (NeedsInitValue(use)) {
        constrained_uses.push_back(instruction);
      } else if (opcode == HloOpcode::kConvert) {
        auto converted_uses = FindConstrainedUses(dataflow, *instruction,
                                                  treat_gte_as_data_formatting);
        constrained_uses.insert(constrained_uses.end(), converted_uses.begin(),
                                converted_uses.end());
      } else if (opcode == HloOpcode::kSort &&
                 instruction->operand_count() >= 2 && op_num == 0) {
        // Operand 0 of sort is the array of keys used for key/value
        // (two-operand) kSort instructions. Since sort stability is not
        // guaranteed, constrain keys of key-value sort not to have
        // duplicates, since otherwise the value order may legitimately
        // differ.
        constrained_uses.push_back(instruction);
      }
    }
  }

  for (auto* instruction : param.parent()->instructions()) {
    const HloOpcode opcode = instruction->opcode();
    if (opcode == HloOpcode::kGather || opcode == HloOpcode::kScatter) {
      if (instruction->operand(1) == &param) {
        // Above already covers this case.
        continue;
      }
      if (ReachableViaDataFormatting(&param, instruction->operand(1),
                                     treat_gte_as_data_formatting)) {
        constrained_uses.push_back(instruction);
      }
    }
  }
  return constrained_uses;
}

// Given a parameter, generate a random Literal to use as input if there exist
// no constrained uses in the dataflow graph.  If such constraints exist,
// generate a constrained literal (either bounded in the case of indices, or
// zero in the case of init_values for reductions).
absl::StatusOr<Literal> CreateLiteralForConstrainedUses(
    const absl::Span<HloInstruction* const> constrained_uses,
    const HloInstruction& param, const Shape& param_shape,
    std::minstd_rand0* engine, std::optional<int64_t> max_bits_of_precision) {
  int64_t index_bound = INT64_MAX;
  bool no_duplicates = false;
  bool needs_constant = false;
  bool needs_sorted_indices = false;
  ConstantType constant_type = ConstantType::kUnknown;
  for (HloInstruction* use : constrained_uses) {
    switch (use->opcode()) {
      case HloOpcode::kDynamicSlice:
      case HloOpcode::kDynamicUpdateSlice: {
        // TODO(chokobole): Uncomment this. HloDynamicIndexInstruction
        // const Shape& indexed_shape = use->operand(0)->shape();
        // const Shape& slice_shape = use->opcode() == HloOpcode::kDynamicSlice
        //                                ? use->shape()
        //                                : use->operand(1)->shape();
        // const int64_t first_index =
        //     Cast<HloDynamicIndexInstruction>(use)->first_index_operand_number();
        // for (int64_t operand = first_index; operand < use->operand_count();
        //      ++operand) {
        //   if (use->operand(operand) == &param) {
        //     index_bound = std::min(
        //         index_bound,
        //         ShapeUtil::GetDimension(indexed_shape, operand - first_index)
        //         -
        //             ShapeUtil::GetDimension(slice_shape,
        //                                     operand - first_index));
        //   }
        // }
        break;
      }
      case HloOpcode::kScatter: {
        const Shape& operand_shape = use->operand(0)->shape();
        auto index_map =
            use->scatter_dimension_numbers().scatter_dims_to_operand_dims();
        for (const auto dim_in_operand : index_map) {
          index_bound = std::min(index_bound,
                                 operand_shape.dimensions(dim_in_operand) - 1);
        }
        needs_sorted_indices |=
            Cast<const HloScatterInstruction>(use)->indices_are_sorted();
        break;
      }
      case HloOpcode::kGather: {
        // TODO(chokobole): Uncomment this. Dependency:
        // HloGatherInstruction, gather_dimension_numbers.
        // const Shape& operand_shape = use->operand(0)->shape();
        // auto index_map =
        //     use->gather_dimension_numbers().start_index_map();
        // for (const auto dim_in_operand : index_map) {
        //   index_bound = std::min(index_bound,
        //                          operand_shape.dimensions(dim_in_operand) -
        //                          1);
        // }
        // needs_sorted_indices |=
        //     Cast<const HloGatherInstruction>(use)->indices_are_sorted();
        break;
      }
      case HloOpcode::kReduce:
        needs_constant = true;
        constant_type = GetInitValue(*use->to_apply());
        break;

      case HloOpcode::kSort:
        no_duplicates = true;
        break;

      default:
        return absl::UnimplementedError(absl::StrFormat(
            "Constrained operand generation not implemented for %s.",
            use->ToString()));
    }
  }
  int constraint_count = 0;
  constraint_count += no_duplicates ? 1 : 0;
  constraint_count += (index_bound != INT64_MAX) ? 1 : 0;
  constraint_count += needs_constant ? 1 : 0;
  if (constraint_count > 1) {
    return absl::UnimplementedError(
        "Conflicting operand generation constraints.");
  }
  if (index_bound != INT64_MAX) {
    return MakeFakeLiteral(
        param_shape, engine, std::pair<int64_t, int64_t>(0, index_bound),
        needs_sorted_indices, no_duplicates, max_bits_of_precision);
  } else if (needs_constant) {
    switch (constant_type) {
      case ConstantType::kZero:
        return LiteralUtil::Zero(param_shape.element_type());
      case ConstantType::kOne:
        return LiteralUtil::One(param_shape.element_type());
      case ConstantType::kUnknown:
        // We want the identity element for the computation, but we don't
        // really know what it is - so any value we generate will be just as
        // wrong.
        return MakeFakeLiteral(param_shape, engine, /*limit=*/std::nullopt,
                               /*is_sorted=*/needs_sorted_indices,
                               /*no_duplicates=*/false, max_bits_of_precision);
    }
  } else {
    return MakeFakeLiteral(param_shape, engine, /*limit=*/std::nullopt,
                           /*is_sorted=*/needs_sorted_indices, no_duplicates,
                           max_bits_of_precision);
  }
}

// Given a module entry parameter, use the dataflow analysis to see if a
// special case literal must be created, or if we can generate fake data.
absl::StatusOr<Literal> MakeConstrainedArgument(
    const HloDataflowAnalysis& dataflow, const HloInstruction& param,
    const Shape& param_shape, std::minstd_rand0* engine,
    bool treat_gte_as_data_formatting,
    std::optional<int64_t> max_bits_of_precision) {
  const auto constrained_uses =
      FindConstrainedUses(dataflow, param, treat_gte_as_data_formatting);
  return CreateLiteralForConstrainedUses(constrained_uses, param, param_shape,
                                         engine, max_bits_of_precision);
}

}  // namespace

absl::StatusOr<std::vector<Literal>> MakeFakeArguments(
    const HloModule* module, bool pseudo_random,
    bool treat_gte_as_data_formatting,
    std::optional<int64_t> max_bits_of_precision, std::minstd_rand0* engine) {
  if (!pseudo_random) {
    return MakeFakeArguments(module, nullptr, treat_gte_as_data_formatting,
                             max_bits_of_precision);
  }
  if (engine == nullptr) {
    auto new_engine =
        pseudo_random ? std::make_unique<std::minstd_rand0>() : nullptr;
    return MakeFakeArguments(module, new_engine.get(),
                             treat_gte_as_data_formatting,
                             max_bits_of_precision);
  }
  return MakeFakeArguments(module, engine, treat_gte_as_data_formatting,
                           max_bits_of_precision);
}

absl::StatusOr<std::vector<Literal>> MakeFakeArguments(
    const HloModule* module, std::minstd_rand0* engine,
    bool treat_gte_as_data_formatting,
    std::optional<int64_t> max_bits_of_precision) {
  TF_ASSIGN_OR_RETURN(auto dataflow, HloDataflowAnalysis::Run(*module));
  const auto params = module->entry_computation()->parameter_instructions();
  std::vector<Literal> arguments(params.size());
  for (int i = 0; i < params.size(); ++i) {
    const HloModuleConfig& module_config = module->config();
    const Shape& param_shape = (module_config.has_entry_computation_layout() &&
                                module_config.entry_computation_layout()
                                    .parameter_layout(i)
                                    .shape()
                                    .is_static())
                                   ? module_config.entry_computation_layout()
                                         .parameter_layout(i)
                                         .shape()
                                   : params[i]->shape();

    TF_ASSIGN_OR_RETURN(
        arguments[i], MakeConstrainedArgument(
                          *dataflow, *params[i], param_shape, engine,
                          treat_gte_as_data_formatting, max_bits_of_precision));
  }
  return std::move(arguments);
}

std::unique_ptr<HloDotInstruction> CreateCanonicalDot(const Shape& shape,
                                                      HloInstruction* lhs,
                                                      HloInstruction* rhs) {
  CHECK_LE(lhs->shape().rank(), 2);
  CHECK_LE(rhs->shape().rank(), 2);
  DotDimensionNumbers dot_dimension_numbers;
  dot_dimension_numbers.add_lhs_contracting_dimensions(
      lhs->shape().rank() > 1 ? 1 : 0);
  dot_dimension_numbers.add_rhs_contracting_dimensions(0);
  return std::make_unique<HloDotInstruction>(shape, lhs, rhs,
                                             dot_dimension_numbers);
}

}  // namespace zkx
