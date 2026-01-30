/* Copyright 2017 The OpenXLA Authors.
Copyright 2026 The ZKX Authors.

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
#include "zkx/hlo/evaluator/hlo_evaluator.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <atomic>
#include <string>
#include <utility>
#include <variant>

#include "absl/base/internal/endian.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/log/log.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"

#include "xla/tsl/platform/cpu_info.h"
#include "xla/tsl/platform/statusor.h"
#include "zk_dtypes/include/comparable_traits.h"
#include "zkx/base/logging.h"
#include "zkx/hlo/evaluator/hlo_evaluator_typed_visitor.h"
#include "zkx/hlo/ir/hlo_clone_context.h"
#include "zkx/hlo/utils/hlo_query.h"
#include "zkx/layout.h"
#include "zkx/layout_util.h"
#include "zkx/literal_util.h"
#include "zkx/primitive_util.h"
#include "zkx/service/compilation_environments.h"
#include "zkx/service/hlo_module_config.h"
#include "zkx/service/logical_buffer.h"
#include "zkx/service/pattern_matcher.h"
#include "zkx/service/shape_inference.h"

namespace zkx {
namespace {

using primitive_util::NativeTypeOf;

template <typename OperandT>
absl::StatusOr<Literal> Compare(const Shape& shape, Comparison comparison,
                                LiteralSlice lhs_literal,
                                LiteralSlice rhs_literal) {
  auto populate = [&](auto compare_op) -> absl::StatusOr<Literal> {
    Literal result(shape);
    TF_RETURN_IF_ERROR(result.PopulateParallel<bool>(
        [&](absl::Span<const int64_t> multi_index, int /*thread_id*/) {
          auto lhs = lhs_literal.Get<OperandT>(multi_index);
          auto rhs = rhs_literal.Get<OperandT>(multi_index);
          return compare_op(lhs, rhs);
        }));
    return std::move(result);
  };
  switch (comparison.GetDirection()) {
    case ComparisonDirection::kEq:
      return populate([](auto lhs, auto rhs) { return lhs == rhs; });
    case ComparisonDirection::kNe:
      return populate([](auto lhs, auto rhs) { return lhs != rhs; });
    case ComparisonDirection::kGe:
      if constexpr (zk_dtypes::IsComparable<OperandT>) {
        return populate([](auto lhs, auto rhs) { return lhs >= rhs; });
      }
      break;
    case ComparisonDirection::kGt:
      if constexpr (zk_dtypes::IsComparable<OperandT>) {
        return populate([](auto lhs, auto rhs) { return lhs > rhs; });
        break;
      }
    case ComparisonDirection::kLe:
      if constexpr (zk_dtypes::IsComparable<OperandT>) {
        return populate([](auto lhs, auto rhs) { return lhs <= rhs; });
        break;
      }
    case ComparisonDirection::kLt:
      if constexpr (zk_dtypes::IsComparable<OperandT>) {
        return populate([](auto lhs, auto rhs) { return lhs < rhs; });
        break;
      }
  }

  LOG(FATAL) << "unhandled direction for conversion to Comparison: "
             << comparison.ToString();
}

std::optional<bool> GetInstructionStaticValueAsBool(
    const HloInstruction* instruction) {
  HloEvaluator evaluator;
  absl::StatusOr<Literal> static_value =
      evaluator.Evaluate(instruction, /*precomputed_analyses=*/{},
                         /*recursively_evaluate_nonconstant_operands=*/true);
  if (static_value.ok()) {
    return static_value->GetFirstElement<bool>();
  }
  return std::nullopt;
}

template <PrimitiveType kType>
struct PopulateParallelImpl {
  using NativeT = NativeTypeOf<kType>;
  static absl::Status Run(
      Literal& literal,
      absl::FunctionRef<Literal(absl::Span<const int64_t>, int)>
          literal_generator) {
    return literal.PopulateParallel<NativeT>(
        [&literal_generator](absl::Span<const int64_t> output_index,
                             int thread_id) {
          return literal_generator(output_index, thread_id)
              .template Get<NativeT>({});
        });
  }
};

template <PrimitiveType kType>
struct PopulateImpl {
  using NativeT = NativeTypeOf<kType>;
  static absl::Status Run(
      Literal& literal,
      absl::FunctionRef<Literal(absl::Span<const int64_t>)> literal_generator) {
    return literal.Populate<NativeT>(
        [&literal_generator](absl::Span<const int64_t> output_index) {
          return literal_generator(output_index).template Get<NativeT>({});
        });
  }
};

// Helper function for when it has a Literal generator (typically from an
// embedded evaluator to evaluate subcomputations) but needs to extract the
// scalar value of a specific type from it to populate a Literal. Putting such
// evaluation implementations in typed visitors gives no performance benefits
// but leads to unnecessarily large code size therefore providing a delegation
// to small templated helpers just for the parts that require manipulating the
// native types to avoid templating the whole implementations.
template <template <PrimitiveType> typename Trait, typename F>
absl::Status Apply(Literal& literal, F&& literal_generator) {
  return primitive_util::PrimitiveTypeSwitch<absl::Status>(
      [&, literal_generator = std::forward<F>(literal_generator)](
          auto primitive_type_constant) -> absl::Status {
        if constexpr (primitive_util::IsArrayType(primitive_type_constant)) {
          return Trait<primitive_type_constant>::Run(
              literal, std::move(literal_generator));
        }
        LOG(FATAL) << "Unhandled primitive type "
                   << literal.shape().element_type();
      },
      literal.shape().element_type());
}

absl::Status MakeEvalErrorDueToParamOrInfeed(
    const HloInstruction& eval_instruction) {
  absl::Status error = absl::FailedPreconditionError(absl::StrCat(
      "Failed to evaluate instruction (", eval_instruction.name(),
      ") since it depends on infeed or parameters to its parent computation (",
      eval_instruction.parent()->name(), ")."));
  std::string error_payload;
  error_payload.resize(sizeof(internal::EvalErrorDetail));
  absl::little_endian::Store32(
      const_cast<char*>(error_payload.data()),
      static_cast<uint32_t>(
          internal::EvalErrorDetail::kDynamicValueDependence));
  error.SetPayload(internal::kEvalErrorDetailUrl, absl::Cord(error_payload));
  return error;
}

// Represents a value that might or might not be determined statically.
struct DynamicOrStaticInteger {
  std::optional<int64_t> static_value;
  bool is_dynamic() const { return !static_value.has_value(); }

  std::string ToString() const {
    return is_dynamic() ? std::string("DYNAMIC") : absl::StrCat(*static_value);
  }
};

std::optional<DynamicOrStaticInteger> GetInstructionValueAsInteger(
    const HloInstruction* instruction,
    HloEvaluator::PrecomputedAnalyses precomputed_analyses) {
  HloEvaluator evaluator;
  absl::StatusOr<Literal> static_value =
      evaluator.Evaluate(instruction, precomputed_analyses,
                         /*recursively_evaluate_nonconstant_operands=*/true);
  if (static_value.ok()) {
    if (instruction->shape().element_type() == PrimitiveType::PRED) {
      return DynamicOrStaticInteger{
          static_cast<int64_t>(static_value->GetFirstElement<bool>())};
    } else {
      return DynamicOrStaticInteger{static_value->GetFirstInteger()};
    }
  }

  std::optional<internal::EvalErrorDetail> eval_error_detail =
      internal::ParseEvalErrorDetail(static_value.status());
  if (eval_error_detail.has_value() &&
      *eval_error_detail ==
          internal::EvalErrorDetail::kDynamicValueDependence) {
    return DynamicOrStaticInteger{std::nullopt};
  }
  return std::nullopt;
}

// Represents an index into the while argument tuple and / or a value.
// At least one of param_index and value has a value; both of them could have
// a value.
struct ParamIndexAndValue {
  std::optional<int64_t> param_index;
  std::optional<DynamicOrStaticInteger> value;

  bool IsValid() const { return param_index.has_value() || value.has_value(); }
  std::string ToString() const {
    return absl::StrCat(
        "param_index:",
        !param_index.has_value() ? std::string("UNKNOWN")
                                 : absl::StrCat(*param_index),
        ",", "value:",
        !value.has_value() ? std::string("UNKNOWN") : value->ToString());
  }
};

std::optional<ParamIndexAndValue> TryParsingInstructionAsParameterAndInteger(
    const HloInstruction* instruction,
    HloEvaluator::PrecomputedAnalyses precomputed_analyses) {
  // Skip copies.
  if (instruction->opcode() == HloOpcode::kCopy) {
    return TryParsingInstructionAsParameterAndInteger(instruction->operand(0),
                                                      precomputed_analyses);
  }
  if (instruction->opcode() == HloOpcode::kCopyDone) {
    return TryParsingInstructionAsParameterAndInteger(
        instruction->operand(0)->operand(1), precomputed_analyses);
  }
  ParamIndexAndValue result;
  if (Match(instruction, match::GetTupleElement().WithOperand(
                             0, match::Parameter().WithParameterNum(0)))) {
    result.param_index = instruction->tuple_index();
  }
  std::optional<DynamicOrStaticInteger> integer_value =
      GetInstructionValueAsInteger(instruction, precomputed_analyses);
  result.value = std::move(integer_value);
  if (!result.IsValid()) {
    return std::nullopt;
  }
  return std::optional<ParamIndexAndValue>(std::move(result));
}

// Represents the while loop condition comparison.
// We assume comparison is of the form: lhs comp rhs.
struct WhileCondComparison {
  ComparisonDirection comparison_direction;
  ParamIndexAndValue lhs;
  ParamIndexAndValue rhs;

  std::string ToString() const {
    return absl::StrCat("WhileCondComparison{", "LHS:{", lhs.ToString(),
                        "},RHS:{", rhs.ToString(), "}}");
  }
};

// Represents the parsed while loop condition. The loop induction variable may
// either be used in a comparison or returned directly, i.e., NoOp. In the case
// of NoOp, it contains the parameter index and initial value of the loop
// induction variable.
using WhileCondComparisonOrNoOp =
    std::variant<WhileCondComparison, ParamIndexAndValue>;

std::optional<ParamIndexAndValue> ParseComparisonOperand(
    const HloInstruction* operand,
    HloEvaluator::PrecomputedAnalyses precomputed_analyses) {
  if (operand->opcode() == HloOpcode::kCopy ||
      operand->opcode() == HloOpcode::kCopyStart ||
      operand->opcode() == HloOpcode::kCopyDone) {
    return ParseComparisonOperand(operand->operand(0), precomputed_analyses);
  }
  std::optional<int64_t> param_index;
  if (Match(operand, match::GetTupleElement().WithOperand(
                         0, match::Parameter().WithParameterNum(0)))) {
    param_index = operand->tuple_index();
  }
  std::optional<DynamicOrStaticInteger> operand_value =
      GetInstructionValueAsInteger(operand, precomputed_analyses);
  if (!param_index.has_value() && !operand_value.has_value()) {
    return std::nullopt;
  }
  return ParamIndexAndValue{param_index, operand_value};
}

std::optional<WhileCondComparisonOrNoOp> PatternMatchLoopCondComparison(
    const HloInstruction* comparison,
    HloEvaluator::PrecomputedAnalyses precomputed_analyses) {
  CHECK_EQ(comparison->opcode(), HloOpcode::kCompare);
  std::optional<ParamIndexAndValue> lhs =
      ParseComparisonOperand(comparison->operand(0), precomputed_analyses);
  std::optional<ParamIndexAndValue> rhs =
      ParseComparisonOperand(comparison->operand(1), precomputed_analyses);
  if (!lhs.has_value() || !rhs.has_value()) {
    return std::nullopt;
  }
  return WhileCondComparison{comparison->comparison_direction(),
                             *std::move(lhs), *std::move(rhs)};
}
// Finds the while loop condition comparison by matching the loop condition root
// with known patterns.
std::optional<WhileCondComparisonOrNoOp> PatternMatchLoopCondRoot(
    const HloInstruction* loop_cond_root,
    HloEvaluator::PrecomputedAnalyses precomputed_analyses) {
  if (loop_cond_root->opcode() == HloOpcode::kCopy) {
    return PatternMatchLoopCondRoot(loop_cond_root->operand(0),
                                    precomputed_analyses);
  }
  if (loop_cond_root->opcode() == HloOpcode::kCopyDone) {
    return PatternMatchLoopCondRoot(loop_cond_root->operand(0)->operand(1),
                                    precomputed_analyses);
  }
  if (loop_cond_root->opcode() == HloOpcode::kCompare) {
    // Base pattern #1: gte-0 comp gte-1
    // Base pattern #2: constant comp gte
    // Base pattern #3: gte comp constant
    return PatternMatchLoopCondComparison(loop_cond_root, precomputed_analyses);
  }
  // Base pattern #4: gte is a boolean scalar and it was return immediately.
  if (Match(loop_cond_root, match::GetTupleElement().WithOperand(
                                0, match::Parameter().WithParameterNum(0)))) {
    if (loop_cond_root->shape().element_type() != PrimitiveType::PRED &&
        loop_cond_root->shape().rank() != 0) {
      return std::nullopt;
    }
    return ParamIndexAndValue{{/*param_index=*/loop_cond_root->tuple_index()}};
  }

  // Recursive pattern #1:
  // loop_cond_root is a GetTupleElement whose operand is a call with a single
  // parameter which takes the computation's single parameter.
  // In this case, if the called computation's root is a tuple, we can recurse
  // on that tuple's element as the new loop_cond_root.
  if (Match(loop_cond_root,
            match::GetTupleElement().WithOperand(
                0, match::Call().WithNumOperands(1).WithOperand(
                       0, match::Parameter().WithParameterNum(0))))) {
    const HloInstruction* call_instruction = loop_cond_root->operand(0);
    const HloComputation* to_apply = call_instruction->to_apply();
    const HloInstruction* to_apply_root = to_apply->root_instruction();
    if (Match(to_apply_root, match::Tuple())) {
      return PatternMatchLoopCondRoot(
          to_apply_root->operand(loop_cond_root->tuple_index()),
          precomputed_analyses);
    }
  }
  // Recursive pattern #2:
  // loop_cond_root is a GetTupleElement whose operand is a tuple.
  // We can recurse on the tuple's element as the new loop_cond_root.
  if (Match(loop_cond_root,
            match::GetTupleElement().WithOperand(0, match::Tuple()))) {
    const HloInstruction* new_cond_root =
        loop_cond_root->operand(0)->operand(loop_cond_root->tuple_index());
    return PatternMatchLoopCondRoot(new_cond_root, precomputed_analyses);
  }
  return std::nullopt;
}

std::optional<DynamicOrStaticInteger> PatternMatchInductionVarUpdate(
    const HloInstruction* induction_var_update, int64_t tuple_index,
    HloEvaluator::PrecomputedAnalyses precomputed_analyses) {
  if (induction_var_update->opcode() == HloOpcode::kCopy) {
    return PatternMatchInductionVarUpdate(induction_var_update->operand(0),
                                          tuple_index, precomputed_analyses);
  }
  if (induction_var_update->opcode() == HloOpcode::kCopyDone) {
    return PatternMatchInductionVarUpdate(
        induction_var_update->operand(0)->operand(1), tuple_index,
        precomputed_analyses);
  }
  std::optional<ParamIndexAndValue> update_param_index_and_value =
      TryParsingInstructionAsParameterAndInteger(induction_var_update,
                                                 precomputed_analyses);

  if (update_param_index_and_value.has_value()) {
    if (update_param_index_and_value->param_index.has_value()) {
      if (*update_param_index_and_value->param_index == tuple_index) {
        // Pattern: the induc_var is directly returned from the loop body with
        // no changes.
        VLOG(3) << "PatternMatchInductionVarUpdate, pattern: [induc_var].";
        return DynamicOrStaticInteger{/*static_value=*/0};
      } else {
        VLOG(3)
            << "PatternMatchInductionVarUpdate, induction variable is set to "
               "another parameter value. Parsed update: "
            << update_param_index_and_value->ToString();
        return std::nullopt;
      }
    }
    if (update_param_index_and_value->value.has_value() &&
        !update_param_index_and_value->value->is_dynamic()) {
      VLOG(3) << "PatternMatchInductionVarUpdate, induction variable is set to "
                 "a constant. Parsed update: "
              << update_param_index_and_value->ToString();
      return std::nullopt;
    }
  }

  if (induction_var_update->opcode() != HloOpcode::kAdd &&
      induction_var_update->opcode() != HloOpcode::kSubtract) {
    return std::nullopt;
  }
  bool negate_update = induction_var_update->opcode() == HloOpcode::kSubtract;
  const HloInstruction* update_lhs = induction_var_update->operand(0);
  VLOG(3) << "PatternMatchInductionVarUpdate, LHS: " << update_lhs->ToString();
  std::optional<ParamIndexAndValue> update_lhs_param_index_and_value =
      TryParsingInstructionAsParameterAndInteger(update_lhs,
                                                 precomputed_analyses);

  const HloInstruction* update_rhs = induction_var_update->operand(1);
  VLOG(3) << "PatternMatchInductionVarUpdate, RHS: " << update_rhs->ToString();
  std::optional<ParamIndexAndValue> update_rhs_param_index_and_value =
      TryParsingInstructionAsParameterAndInteger(update_rhs,
                                                 precomputed_analyses);

  if (!update_lhs_param_index_and_value.has_value() ||
      !update_lhs_param_index_and_value->value.has_value() ||
      !update_rhs_param_index_and_value.has_value() ||
      !update_rhs_param_index_and_value->value.has_value()) {
    VLOG(3) << "PatternMatchInductionVarUpdate, failed to parse operands. "
               "Induction var update instruction: "
            << induction_var_update->ToString();
    return std::nullopt;
  }

  VLOG(3) << "update_lhs: " << update_lhs->ToString();
  VLOG(3) << "update_rhs: " << update_rhs->ToString();

  if (update_lhs_param_index_and_value->param_index.has_value() &&
      *update_lhs_param_index_and_value->param_index == tuple_index &&
      update_lhs_param_index_and_value->value->is_dynamic()) {
    if (update_rhs_param_index_and_value->value->is_dynamic()) {
      return update_rhs_param_index_and_value->value;
    }
    int64_t update_value =
        *update_rhs_param_index_and_value->value->static_value;
    return negate_update
               ? DynamicOrStaticInteger{/*static_value=*/-update_value}
               : DynamicOrStaticInteger{/*static_value=*/update_value};
  }

  if (update_rhs_param_index_and_value->param_index.has_value() &&
      *update_rhs_param_index_and_value->param_index == tuple_index &&
      update_rhs_param_index_and_value->value->is_dynamic() && !negate_update) {
    return update_lhs_param_index_and_value->value;
  }
  VLOG(3) << "Failed to pattern match induction variable update.";
  return std::nullopt;
}

// Tries to parse the loop body to find how the induction variable is updated
// using pattern matching.
std::optional<DynamicOrStaticInteger>
PatternMatchInductionVarUpdateFromLoopBodyRoot(
    const HloInstruction* loop_body_root, int64_t tuple_index,
    HloEvaluator::PrecomputedAnalyses precomputed_analyses) {
  if (loop_body_root->opcode() != HloOpcode::kTuple ||
      loop_body_root->operand_count() <= tuple_index) {
    return std::nullopt;
  }
  const HloInstruction* induction_var_update =
      loop_body_root->operand(tuple_index);
  return PatternMatchInductionVarUpdate(induction_var_update, tuple_index,
                                        precomputed_analyses);
}

std::optional<bool> PatternMatchLoopCondVarOverride(
    const HloInstruction* loop_body_root, int64_t tuple_index) {
  if (!Match(loop_body_root, match::Tuple()) ||
      loop_body_root->operand_count() <= tuple_index) {
    return std::nullopt;
  }
  const HloInstruction* cond_var_override =
      loop_body_root->operand(tuple_index);
  return GetInstructionStaticValueAsBool(cond_var_override);
}

// A convenience wrapper to compute the while loop's argument's init value at
// the given tuple_index. If the init value depends on parameters to the
// while loop's parent computation or infeed, we consider the init value
// dynamic.
std::optional<DynamicOrStaticInteger> EvaluateWhileLoopParamInitValue(
    const HloInstruction* param_instruction, int64_t tuple_index) {
  if (param_instruction->opcode() != HloOpcode::kTuple) {
    return std::nullopt;
  }
  const HloInstruction* element_instruction =
      param_instruction->operand(tuple_index);
  return GetInstructionValueAsInteger(element_instruction,
                                      /*precomputed_analyses=*/{});
}

}  // namespace

namespace internal {

constexpr std::string_view kEvalErrorDetailUrl = "EvalErrorDetailUrl";

std::optional<EvalErrorDetail> ParseEvalErrorDetail(const absl::Status& error) {
  auto error_detail = error.GetPayload(kEvalErrorDetailUrl);
  if (!error_detail.has_value() || error_detail->empty()) {
    return std::nullopt;
  }
  return static_cast<EvalErrorDetail>(
      absl::little_endian::Load32(error_detail->Flatten().data()));
}

}  // namespace internal

std::optional<ParsedWhileLoop> HandleNoopLoopCondition(
    const ParamIndexAndValue& parameter_index_and_value,
    const HloInstruction* while_operand, const HloComputation* while_body) {
  CHECK(parameter_index_and_value.param_index.has_value());
  int64_t loop_cond_var_index = *parameter_index_and_value.param_index;
  std::optional<DynamicOrStaticInteger> noop_value =
      EvaluateWhileLoopParamInitValue(while_operand, loop_cond_var_index);

  if (noop_value.has_value()) {
    if (noop_value->is_dynamic()) {
      return kParsedDynamicWhileLoop;
    } else if (*noop_value->static_value == 0) {
      return ParsedWhileLoop{
          ParsedStaticWhileLoop{/*trip_count=*/0,
                                /*induction_var_index=*/loop_cond_var_index,
                                /*induction_var_init_value=*/0,
                                /*step_size=*/0,
                                /*loop_bound=*/0}};
    }
    std::optional<bool> updated_loop_cond_var = PatternMatchLoopCondVarOverride(
        while_body->root_instruction(), loop_cond_var_index);
    if (updated_loop_cond_var.has_value()) {
      if (!*updated_loop_cond_var) {
        return ParsedWhileLoop{
            ParsedStaticWhileLoop{/*trip_count=*/1,
                                  /*induction_var_index=*/loop_cond_var_index,
                                  /*induction_var_init_value=*/0,
                                  /*step_size=*/1,
                                  /*loop_bound=*/1}};
      } else {
        // This is an infinite loop and we set trip_count to -1.
        return ParsedWhileLoop{
            ParsedStaticWhileLoop{/*trip_count=*/-1,
                                  /*induction_var_index=*/loop_cond_var_index,
                                  /*induction_var_init_value=*/0,
                                  /*step_size=*/0,
                                  /*loop_bound=*/1}};
      }
    }
  }
  return std::nullopt;
}

int64_t ComputeTripCountFromComparison(int64_t init, int64_t bound,
                                       int64_t update,
                                       bool comparison_with_equal) {
  if (comparison_with_equal && init > bound) {
    return 0;
  }
  if (!comparison_with_equal && init >= bound) {
    return 0;
  }
  int64_t distance = bound - init;
  int64_t trip_count = (distance + update - 1) / update;
  CHECK_GE(trip_count, 0);
  // Additional logic to deal with equal comparison.
  if (comparison_with_equal && (bound - init) % update == 0) {
    trip_count += 1;
  }
  return trip_count;
}

std::optional<ParsedWhileLoop> HandleStaticLoopComparison(
    int64_t lhs, int64_t rhs, Comparison::Direction comparison_direction) {
  if ((comparison_direction == Comparison::Direction::kLt && lhs < rhs) ||
      (comparison_direction == Comparison::Direction::kLe && lhs <= rhs) ||
      (comparison_direction == Comparison::Direction::kGt && lhs > rhs) ||
      (comparison_direction == Comparison::Direction::kGe && lhs >= rhs) ||
      (comparison_direction == Comparison::Direction::kEq && lhs == rhs) ||
      (comparison_direction == Comparison::Direction::kNe && lhs != rhs)) {
    // This is an infinite loop and we set trip_count to -1.
    // There is no induction variable.
    return ParsedWhileLoop{ParsedStaticWhileLoop{/*trip_count=*/-1,
                                                 /*induction_var_index=*/-1,
                                                 /*induction_var_init_value=*/0,
                                                 /*step_size=*/0,
                                                 /*loop_bound=*/1}};
  }
  return ParsedWhileLoop{ParsedStaticWhileLoop{/*trip_count=*/0,
                                               /*induction_var_index=*/-1,
                                               /*induction_var_init_value=*/0,
                                               /*step_size=*/0,
                                               /*loop_bound=*/0}};
}

std::optional<ParsedWhileLoop> PatternMatchParseWhileLoop(
    const HloInstruction* while_op,
    HloEvaluator::PrecomputedAnalyses precomputed_analyses) {
  VLOG(3) << "PatternMatchParseWhileLoop, while_op: " << while_op->name();
  const HloComputation* while_cond = while_op->while_condition();
  const HloComputation* while_body = while_op->while_body();
  const HloInstruction* while_operand = while_op->operand(0);
  // Try to parse the loop condition comparison.
  std::optional<WhileCondComparisonOrNoOp> loop_comparison_or_noop =
      PatternMatchLoopCondRoot(while_cond->root_instruction(),
                               precomputed_analyses);
  if (!loop_comparison_or_noop.has_value()) {
    return std::nullopt;
  }
  if (loop_comparison_or_noop->index() == 1) {
    return HandleNoopLoopCondition(
        std::get<ParamIndexAndValue>(*loop_comparison_or_noop), while_operand,
        while_body);
  }
  CHECK_EQ(loop_comparison_or_noop->index(), 0);
  WhileCondComparison loop_comparison =
      std::get<WhileCondComparison>(*loop_comparison_or_noop);
  CHECK(loop_comparison.lhs.IsValid() && loop_comparison.rhs.IsValid());

  // We can't handle the case when the while loop argument is not a Tuple
  // instruction.
  if (while_operand->opcode() != HloOpcode::kTuple) {
    return std::nullopt;
  }

  if (!loop_comparison.lhs.value.has_value() ||
      !loop_comparison.rhs.value.has_value()) {
    return std::nullopt;
  }

  // We have either successfully parsed the init value for both LHS and RHS
  // or have returned as failure.
  CHECK(loop_comparison.lhs.value.has_value());
  CHECK(loop_comparison.rhs.value.has_value());

  VLOG(3) << loop_comparison.ToString();

  // If both operands of the loop condition comparison have dynamic value, the
  // trip count might be dynamic or static. This is a case that our existing
  // patterns could not yet handle, so we return std::nullopt.
  if (loop_comparison.lhs.value->is_dynamic() &&
      loop_comparison.rhs.value->is_dynamic()) {
    VLOG(3) << "Both operands of the loop condition comparison are dynamic.";
    return std::nullopt;
  }
  // We would have returned if both operands are dynamic. So there is at most
  // one dynamic operand, which is potentially the loop induction variable.
  CHECK(!loop_comparison.lhs.value->is_dynamic() ||
        !loop_comparison.rhs.value->is_dynamic());

  if (!loop_comparison.lhs.value->is_dynamic() &&
      !loop_comparison.rhs.value->is_dynamic()) {
    int64_t lhs_value = *loop_comparison.lhs.value->static_value;
    int64_t rhs_value = *loop_comparison.rhs.value->static_value;
    Comparison::Direction comparison_direction =
        loop_comparison.comparison_direction;
    return HandleStaticLoopComparison(lhs_value, rhs_value,
                                      comparison_direction);
  }
  std::optional<DynamicOrStaticInteger> induction_var_init;
  std::optional<DynamicOrStaticInteger> induction_var_update;
  bool lhs_is_induction_var = true;
  if (loop_comparison.lhs.value->is_dynamic()) {
    if (loop_comparison.lhs.param_index.has_value()) {
      VLOG(3) << "Comparison LHS is induction variable.";
      induction_var_init = EvaluateWhileLoopParamInitValue(
          while_operand, *loop_comparison.lhs.param_index);
      induction_var_update = PatternMatchInductionVarUpdateFromLoopBodyRoot(
          while_body->root_instruction(), *loop_comparison.lhs.param_index,
          precomputed_analyses);
      lhs_is_induction_var = true;
    }
  } else {
    CHECK(loop_comparison.rhs.value->is_dynamic());
    if (loop_comparison.rhs.param_index.has_value()) {
      VLOG(3) << "Comparison RHS is induction variable.";
      induction_var_init = EvaluateWhileLoopParamInitValue(
          while_operand, *loop_comparison.rhs.param_index);
      induction_var_update = PatternMatchInductionVarUpdateFromLoopBodyRoot(
          while_body->root_instruction(), *loop_comparison.rhs.param_index,
          precomputed_analyses);
      lhs_is_induction_var = false;
    }
  }

  if (!induction_var_init.has_value() || !induction_var_update.has_value()) {
    return std::nullopt;
  }
  VLOG(3) << "induction_var_init: " << induction_var_init->ToString();
  VLOG(3) << "induction_var_update: " << induction_var_update->ToString();
  if (induction_var_init->is_dynamic() || induction_var_update->is_dynamic()) {
    return kParsedDynamicWhileLoop;
  }

  int64_t init_value = *induction_var_init->static_value;
  int64_t update_value = *induction_var_update->static_value;
  Comparison::Direction comparison_direction =
      loop_comparison.comparison_direction;
  ParsedWhileLoop parsed_static_while_loop = ParsedWhileLoop{
      ParsedStaticWhileLoop{/*trip_count=*/0,
                            // Unassigned.
                            /*induction_var_index=*/-1,
                            /*induction_var_init_value=*/init_value,
                            /*step_size=*/update_value,
                            // Unassigned.
                            /*loop_bound=*/-1}};
  // Lhs is the induction variable.
  if (lhs_is_induction_var) {
    CHECK(loop_comparison.rhs.value.has_value() &&
          !loop_comparison.rhs.value->is_dynamic());
    int64_t bound = *loop_comparison.rhs.value->static_value;
    parsed_static_while_loop.static_while_loop->induction_var_index =
        *loop_comparison.lhs.param_index;
    parsed_static_while_loop.static_while_loop->loop_bound = bound;
    if (update_value > 0 &&
        (comparison_direction == Comparison::Direction::kLt ||
         comparison_direction == Comparison::Direction::kLe)) {
      int64_t trip_count = ComputeTripCountFromComparison(
          init_value, bound, update_value,
          comparison_direction == Comparison::Direction::kLe);
      parsed_static_while_loop.static_while_loop->trip_count = trip_count;
      return parsed_static_while_loop;
    }
    if (update_value < 0 &&
        (comparison_direction == Comparison::Direction::kGt ||
         comparison_direction == Comparison::Direction::kGe)) {
      int64_t trip_count = ComputeTripCountFromComparison(
          bound, init_value, -update_value,
          comparison_direction == Comparison::Direction::kGe);
      parsed_static_while_loop.static_while_loop->trip_count = trip_count;
      return parsed_static_while_loop;
    }
    return std::nullopt;
  }
  // Rhs is the induction variable.
  CHECK(loop_comparison.lhs.value.has_value() &&
        !loop_comparison.lhs.value->is_dynamic());
  int64_t bound = *loop_comparison.lhs.value->static_value;
  parsed_static_while_loop.static_while_loop->induction_var_index =
      *loop_comparison.rhs.param_index;
  parsed_static_while_loop.static_while_loop->loop_bound = bound;
  if (update_value > 0 &&
      (comparison_direction == Comparison::Direction::kGt ||
       comparison_direction == Comparison::Direction::kGe)) {
    int64_t trip_count = ComputeTripCountFromComparison(
        init_value, bound, update_value,
        comparison_direction == Comparison::Direction::kGe);
    parsed_static_while_loop.static_while_loop->trip_count = trip_count;
    return parsed_static_while_loop;
  }
  if (update_value < 0 &&
      (comparison_direction == Comparison::Direction::kLt ||
       comparison_direction == Comparison::Direction::kLe)) {
    int64_t trip_count = ComputeTripCountFromComparison(
        bound, init_value, -update_value,
        comparison_direction == Comparison::Direction::kLe);
    parsed_static_while_loop.static_while_loop->trip_count = trip_count;
    return parsed_static_while_loop;
  }
  return std::nullopt;
}

// Note that unsupported types by the typed visitor does not necessarily imply
// the non-typed HloEvaluator (parent evaluator) would not support them either
// in the type-agnostic handler. For e.g., HandleGetTupleElement in the parent
// type-agnostic evaluator will be able to accept Tuple primitive type, whereas
// HloEvaluatorTypedVisitor cannot.
HloEvaluator::HloEvaluator(int64_t max_loop_iterations)
    : max_loop_iterations_(max_loop_iterations) {
  for (int i = PrimitiveType_MIN; i < PrimitiveType_ARRAYSIZE; ++i) {
    if (!primitive_util::IsArrayType(PrimitiveType{i})) {
      continue;
    }
    primitive_util::PrimitiveTypeSwitch<void>(
        [&](auto primitive_type) {
          if constexpr (primitive_util::IsArrayType(primitive_type)) {
            using NativeT = primitive_util::NativeTypeOf<primitive_type>;
            if constexpr (primitive_util::IsSignedIntegralType(
                              primitive_type)) {
              typed_visitors_[primitive_type] =
                  std::make_unique<HloEvaluatorTypedVisitor<NativeT, int64_t>>(
                      this);
              // NOLINTNEXTLINE(readability/braces)
            } else if constexpr (primitive_util::IsUnsignedIntegralType(
                                     primitive_type)) {
              typed_visitors_[primitive_type] =
                  std::make_unique<HloEvaluatorTypedVisitor<NativeT, uint64_t>>(
                      this);
            } else {
              typed_visitors_[primitive_type] =
                  std::make_unique<HloEvaluatorTypedVisitor<NativeT>>(this);
            }
          }
        },
        PrimitiveType{i});
  }

  typed_visitors_[TUPLE] =
      std::make_unique<ConstFunctionVisitor>([](const HloInstruction*) {
        return absl::UnimplementedError(
            "HloEvaluatorTypedVisitor: unhandled primitive type: TUPLE.");
      });
  typed_visitors_[OPAQUE_TYPE] =
      std::make_unique<ConstFunctionVisitor>([](const HloInstruction*) {
        return absl::UnimplementedError(
            "HloEvaluatorTypedVisitor: unhandled primitive type: OPAQUE_TYPE.");
      });
  typed_visitors_[TOKEN] =
      std::make_unique<ConstFunctionVisitor>([](const HloInstruction*) {
        return absl::UnimplementedError(
            "HloEvaluatorTypedVisitor: unhandled primitive type: TOKEN.");
      });
}

absl::StatusOr<Literal> HloEvaluator::Evaluate(
    const HloComputation& computation,
    absl::Span<const Literal* const> arg_literals) {
  CHECK(computation.parent() != nullptr);
  ZKX_VLOG_LINES(
      2, "HloEvaluator::Evaluate computation:\n" + computation.ToString());
  OnEvaluateComputation(computation);

  if (arg_literals.size() != computation.num_parameters()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Expected %d argument%s, but got %d.", computation.num_parameters(),
        computation.num_parameters() == 1 ? "" : "s", arg_literals.size()));
  }
  for (int64_t i = 0; i < arg_literals.size(); ++i) {
    const auto& computation_shape =
        computation.parameter_instruction(i)->shape();
    const auto& arg_shape = arg_literals[i]->shape();
    bool ignore_layout = !computation_shape.has_layout();
    if (!Shape::Equal()
             .IgnoreLayout(ignore_layout)
             .MinorToMajorOnlyInLayout()(computation_shape, arg_shape)) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Shape mismatch at parameter %d. Computation expected %s, but arg "
          "was %s.",
          i, ShapeUtil::HumanStringWithLayout(computation_shape),
          ShapeUtil::HumanStringWithLayout(arg_shape)));
    }
  }

  evaluated_.clear();
  arg_literals_.clear();
  call_graph_cache_.reset();
  tuple_points_to_analysis_cache_.reset();
  for (const auto& literal_ptr : arg_literals) {
    arg_literals_.push_back(&*literal_ptr);
  }

  // Re-seed RNG, either from the configuration's seed or a monotonic
  // per-evaluator seed (which prevents two evaluators from returning the same
  // random sequence).
  if (computation.parent()->config().seed()) {
    seed_ = computation.parent()->config().seed();
  } else {
    // Start global_seed at a (true) random value.
    static std::atomic<uint64_t> global_seed{std::random_device()()};
    seed_ = global_seed.fetch_add(1);
  }
  engine_.seed(seed_);

  TF_RETURN_IF_ERROR(computation.Accept(this));
  const Literal& result =
      GetEvaluatedLiteralFor(computation.root_instruction());
  if (VLOG_IS_ON(100)) {
    for (const HloInstruction* instr : computation.instructions()) {
      VLOG(100) << instr->name() << " = " << GetEvaluatedLiteralFor(instr);
    }
  }
  if (!result.IsKnown()) {
    return MakeEvalErrorDueToParamOrInfeed(*computation.root_instruction());
  }
  return result.Clone();
}

absl::StatusOr<Literal> HloEvaluator::Evaluate(
    const HloInstruction* instruction, PrecomputedAnalyses precomputed_analyses,
    bool recursively_evaluate_nonconstant_operands) {
  arg_literals_.clear();
  evaluated_.clear();
  call_graph_cache_.reset();
  tuple_points_to_analysis_cache_.reset();
  auto enable_partial_evaluation_cleanup =
      absl::MakeCleanup([this] { enable_partial_evaluation_ = false; });
  enable_partial_evaluation_ = recursively_evaluate_nonconstant_operands;
  TF_RETURN_IF_ERROR(
      EvaluateInternal(instruction, precomputed_analyses, /*shape_index=*/{},
                       recursively_evaluate_nonconstant_operands));
  const Literal& result = GetEvaluatedLiteralFor(instruction);
  if (!result.IsKnown()) {
    return MakeEvalErrorDueToParamOrInfeed(*instruction);
  }
  return result.Clone();
}

bool HloEvaluator::TryEvaluate(const HloInstruction* instruction,
                               Literal* result,
                               bool recursively_evaluate_nonconstant_operands) {
  CHECK_NE(result, nullptr);
  auto result_or = Evaluate(instruction, /*precomputed_analyses=*/{},
                            recursively_evaluate_nonconstant_operands);
  if (!result_or.ok()) {
    VLOG(1) << "TryEvaluate failed:" << result_or.status();
    return false;
  }

  *result = std::move(result_or).value();
  return true;
}

absl::StatusOr<Literal> HloEvaluator::EvaluateWithSubstitutions(
    const HloInstruction* instruction,
    const absl::flat_hash_map<const HloInstruction*, const LiteralBase*>&
        substitutions,
    bool recursively_evaluate_nonconstant_operands) {
  std::vector<std::unique_ptr<HloInstruction>> owned_operands;
  for (const HloInstruction* operand : instruction->operands()) {
    auto it = substitutions.find(operand);
    if (it == substitutions.end()) {
      if (recursively_evaluate_nonconstant_operands) {
        TF_ASSIGN_OR_RETURN(Literal value,
                            EvaluateWithSubstitutions(
                                operand, substitutions,
                                recursively_evaluate_nonconstant_operands));
        owned_operands.push_back(HloInstruction::CreateConstant(value.Clone()));
      } else {
        if (!operand->IsConstant()) {
          VLOG(2) << "EvaluateWithSubstitutions called when not all operands "
                     "are constant. Consider calling it with "
                     "`recursively_evaluate_non_constant_operands` true.";
        }
        owned_operands.push_back(operand->Clone());
      }
    } else {
      owned_operands.push_back(
          HloInstruction::CreateConstant(it->second->Clone()));
    }
  }

  std::vector<HloInstruction*> operands;
  operands.reserve(owned_operands.size());
  for (auto& operand : owned_operands) {
    operands.push_back(operand.get());
  }

  std::unique_ptr<HloInstruction> cloned_instruction =
      instruction->CloneWithNewOperands(instruction->shape(), operands);
  auto result = Evaluate(cloned_instruction.get());

  return result;
}

absl::StatusOr<Literal> HloEvaluator::EvaluateElementwiseBinaryOp(
    HloOpcode opcode, const Literal& lhs, const Literal& rhs) {
  std::unique_ptr<HloInstruction> lhs_instr =
      HloInstruction::CreateConstant(lhs.Clone());
  std::unique_ptr<HloInstruction> rhs_instr =
      HloInstruction::CreateConstant(rhs.Clone());

  std::unique_ptr<HloInstruction> cloned_instruction =
      HloInstruction::CreateBinary(lhs.shape(), opcode, lhs_instr.get(),
                                   rhs_instr.get());
  auto result = Evaluate(cloned_instruction.get());

  return result;
}

absl::StatusOr<Literal> HloEvaluator::EvaluateElementwiseTernaryOp(
    HloOpcode opcode, const Literal& lhs, const Literal& rhs,
    const Literal& ehs) {
  std::unique_ptr<HloInstruction> lhs_instr =
      HloInstruction::CreateConstant(lhs.Clone());
  std::unique_ptr<HloInstruction> rhs_instr =
      HloInstruction::CreateConstant(rhs.Clone());
  std::unique_ptr<HloInstruction> ehs_instr =
      HloInstruction::CreateConstant(ehs.Clone());
  TF_ASSIGN_OR_RETURN(auto output_shape,
                      ShapeInference::InferTernaryOpShape(
                          opcode, lhs.shape(), rhs.shape(), ehs.shape()));
  std::unique_ptr<HloInstruction> cloned_instruction =
      HloInstruction::CreateTernary(output_shape, opcode, lhs_instr.get(),
                                    rhs_instr.get(), ehs_instr.get());
  return Evaluate(cloned_instruction.get());
}

absl::StatusOr<Literal> HloEvaluator::EvaluateElementwiseCompareOp(
    ComparisonDirection direction, const Literal& lhs, const Literal& rhs) {
  std::unique_ptr<HloInstruction> lhs_instr =
      HloInstruction::CreateConstant(lhs.Clone());
  std::unique_ptr<HloInstruction> rhs_instr =
      HloInstruction::CreateConstant(rhs.Clone());

  std::unique_ptr<HloInstruction> cloned_instruction =
      HloInstruction::CreateCompare(
          ShapeUtil::ChangeElementType(lhs.shape(), PRED), lhs_instr.get(),
          rhs_instr.get(), direction);
  auto result = Evaluate(cloned_instruction.get());

  return result;
}

absl::StatusOr<Literal> HloEvaluator::EvaluateElementwiseUnaryOp(
    HloOpcode opcode, const Literal& operand) {
  std::unique_ptr<HloInstruction> operand_instr =
      HloInstruction::CreateConstant(operand.Clone());

  TF_ASSIGN_OR_RETURN(Shape inferred_shape, ShapeInference::InferUnaryOpShape(
                                                opcode, operand.shape()));
  std::unique_ptr<HloInstruction> cloned_instruction =
      HloInstruction::CreateUnary(inferred_shape, opcode, operand_instr.get());
  auto result = Evaluate(cloned_instruction.get());

  return result;
}

absl::StatusOr<Literal> HloEvaluator::EvaluateDotOp(
    const DotDimensionNumbers& dim_numbers, const Literal& lhs,
    const Literal& rhs) {
  std::unique_ptr<HloInstruction> lhs_instr =
      HloInstruction::CreateConstant(lhs.Clone());
  std::unique_ptr<HloInstruction> rhs_instr =
      HloInstruction::CreateConstant(rhs.Clone());

  TF_ASSIGN_OR_RETURN(
      Shape dot_shape,
      ShapeInference::InferDotOpShape(lhs.shape(), rhs.shape(), dim_numbers,
                                      /*preferred_element_type=*/std::nullopt));

  std::unique_ptr<HloInstruction> cloned_instruction =
      HloInstruction::CreateDot(dot_shape, lhs_instr.get(), rhs_instr.get(),
                                dim_numbers);
  return Evaluate(cloned_instruction.get());
}

absl::Status HloEvaluator::EvaluateParameterFromCallerArgument(
    const HloInstruction* parameter, const ShapeIndex& shape_index,
    PrecomputedAnalyses analyses) {
  CHECK(!evaluated_.contains(parameter));
  const HloComputation* parent_computation = parameter->parent();
  std::vector<HloInstruction*> computation_callers =
      analyses.call_graph->GetComputationCallers(parent_computation);
  // If the parent computation has multiple callers, we cannot determine from
  // which caller the arguments are passed.
  if (computation_callers.size() != 1) {
    return absl::FailedPreconditionError(
        absl::StrCat("The computation ", parent_computation->name(),
                     " is called by ", computation_callers.size(),
                     " callers and thus its argument value "
                     "cannot be determined statically."));
  }
  const HloInstruction* computation_caller = computation_callers[0];
  const HloInstruction* caller_operand = computation_caller->operand(0);
  if (computation_caller->opcode() != HloOpcode::kWhile &&
      computation_caller->opcode() != HloOpcode::kCall) {
    return absl::FailedPreconditionError(absl::StrCat(
        "The computation ", parent_computation->name(), " is called by ",
        "instruction ", computation_caller->name(),
        ", which is not yet supported."));
  }
  if (computation_caller->opcode() == HloOpcode::kWhile) {
    HloComputation* while_body = computation_caller->while_body();
    TF_ASSIGN_OR_RETURN(
        const LogicalBuffer* logical_buffer,
        analyses.tuple_points_to->GetBufferDefinedAt(
            while_body->parameter_instruction(parameter->parameter_number()),
            shape_index));
    const TuplePointsToAnalysis::BufferAliasVector& buffer_aliases =
        analyses.tuple_points_to->GetBufferAliases(*logical_buffer);
    bool unchanged_in_return = false;
    for (const BufferAlias& buffer_alias : buffer_aliases) {
      if (buffer_alias.instruction() == while_body->root_instruction() &&
          buffer_alias.index() == shape_index) {
        unchanged_in_return = true;
      }
    }
    if (!unchanged_in_return) {
      return MakeEvalErrorDueToParamOrInfeed(*parameter);
    }
  }
  TF_RETURN_IF_ERROR(
      EvaluateInternal(caller_operand, analyses, shape_index, true));
  const Literal& caller_operand_literal =
      GetEvaluatedLiteralFor(caller_operand);
  evaluated_[parameter] =
      Literal::CreateFromShapeWithUnknownLeafArrays(parameter->shape());
  TF_RETURN_IF_ERROR(evaluated_[parameter].CopyFrom(
      caller_operand_literal, /*dest_shape_index=*/shape_index,
      /*src_shape_index=*/shape_index));
  return absl::OkStatus();
}

std::vector<int64_t> HloEvaluator::GetS64Indices(
    absl::Span<HloInstruction* const> start_indices) {
  auto get_first_s64 = [&](const Literal& index) -> int64_t {
    return primitive_util::PrimitiveTypeSwitch<int64_t>(
        [&](auto primitive_type_constant) -> int64_t {
          if constexpr (primitive_util::IsIntegralType(
                            primitive_type_constant)) {
            return static_cast<int64_t>(
                index.GetFirstElement<NativeTypeOf<primitive_type_constant>>());
          }
          LOG(FATAL) << "GetS64Indices: unhandled primitive type for "
                     << PrimitiveType_Name(index.shape().element_type());
        },
        index.shape().element_type());
  };
  std::vector<int64_t> start;
  start.reserve(start_indices.size());
  for (HloInstruction* index : start_indices) {
    start.push_back(get_first_s64(GetEvaluatedLiteralFor(index)));
  }
  return start;
}

DimensionVector HloEvaluator::MakeDimMultipliers(const Shape& shape) {
  DimensionVector v(shape.rank());
  int64_t scale = 1;
  for (auto dim : LayoutUtil::MinorToMajor(shape)) {
    v[dim] = scale;
    scale *= shape.dimensions(dim);
  }
  return v;
}

absl::Status HloEvaluator::EvaluateInternal(
    const HloInstruction* instruction, PrecomputedAnalyses precomputed_analyses,
    const ShapeIndex& shape_index,
    bool recursively_evaluate_nonconstant_operands) {
  // Don't need to evaluate this instruction again if it has already been
  // evaluated.
  if (IsAlreadyEvaluated(instruction, shape_index)) {
    return absl::OkStatus();
  }

  if (!recursively_evaluate_nonconstant_operands) {
    if (!hlo_query::AllOperandsAreConstants(*instruction)) {
      return absl::FailedPreconditionError(
          absl::StrCat("Not all operands are constants. Instruction: ",
                       instruction->ToString()));
    }
  } else {
    if (instruction->opcode() == HloOpcode::kGetTupleElement) {
      ShapeIndex new_shape_index = shape_index;
      new_shape_index.push_front(instruction->tuple_index());
      TF_RETURN_IF_ERROR(EvaluateInternal(
          instruction->operand(0), precomputed_analyses, new_shape_index,
          /*recursively_evaluate_nonconstant_operands=*/true));
    } else if (instruction->opcode() == HloOpcode::kTuple &&
               !shape_index.empty()) {
      ShapeIndex new_shape_index = shape_index;
      int64_t tuple_index = new_shape_index.front();
      new_shape_index.pop_front();
      TF_RETURN_IF_ERROR(
          EvaluateInternal(instruction->operand(tuple_index),
                           precomputed_analyses, new_shape_index,
                           /*recursively_evaluate_nonconstant_operands=*/true));
    } else if (instruction->opcode() == HloOpcode::kParameter) {
      CallGraph* call_graph =
          (precomputed_analyses.call_graph != nullptr)
              ? precomputed_analyses.call_graph
              : std::invoke([this, instruction]() -> CallGraph* {
                  call_graph_cache_ =
                      CallGraph::Build(instruction->GetModule());
                  return call_graph_cache_.get();
                });
      TuplePointsToAnalysis* tuple_points_to_analysis =
          (precomputed_analyses.tuple_points_to != nullptr)
              ? precomputed_analyses.tuple_points_to
              : std::invoke([this, instruction]() -> TuplePointsToAnalysis* {
                  absl::StatusOr<std::unique_ptr<TuplePointsToAnalysis>>
                      tuple_points_to_analysis =
                          TuplePointsToAnalysis::Run(instruction->GetModule());
                  if (!tuple_points_to_analysis.ok()) {
                    return nullptr;
                  }
                  tuple_points_to_analysis_cache_ =
                      *std::move(tuple_points_to_analysis);
                  return tuple_points_to_analysis_cache_.get();
                });
      if (call_graph && tuple_points_to_analysis) {
        absl::Status argument_eval_status = EvaluateParameterFromCallerArgument(
            instruction, shape_index, {tuple_points_to_analysis, call_graph});
        if (!argument_eval_status.ok()) {
          VLOG(4) << "Failed to evaluate parameter " << instruction->name()
                  << " from caller. Reason: " << argument_eval_status.message();
        } else {
          VLOG(4) << "Successfully evaluated parameter: "
                  << instruction->name();
        }
      }
    } else {
      for (HloInstruction* operand : instruction->operands()) {
        TF_RETURN_IF_ERROR(EvaluateInternal(
            operand, precomputed_analyses, /*shape_index=*/{},
            /*recursively_evaluate_nonconstant_operands=*/true));
        // Except for the above and following cases, we do not support handling
        // unknown operands for other HLOs. So mark the result as unknown.
        if ((!GetEvaluatedLiteralFor(operand).IsKnown() &&
             instruction->opcode() != HloOpcode::kCopy &&
             instruction->opcode() != HloOpcode::kCopyStart &&
             instruction->opcode() != HloOpcode::kCopyDone &&
             instruction->opcode() != HloOpcode::kAsyncStart &&
             instruction->opcode() != HloOpcode::kAsyncUpdate &&
             instruction->opcode() != HloOpcode::kAsyncDone &&
             instruction->opcode() != HloOpcode::kWhile)) {
          evaluated_[instruction] =
              Literal::CreateFromShapeWithUnknownLeafArrays(
                  instruction->shape());
          return absl::OkStatus();
        }
      }
    }
  }
  visitor_shape_index_ = shape_index;
  TF_RETURN_IF_ERROR(Preprocess(instruction));
  TF_RETURN_IF_ERROR(instruction->Visit(this));
  TF_RETURN_IF_ERROR(Postprocess(instruction));
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleBitcast(const HloInstruction* bitcast) {
  Shape result_shape = bitcast->shape();

  // Allow effective scalars without layouts as the result is unambiguous.
  if (!result_shape.has_layout() &&
      ShapeUtil::IsEffectiveScalar(result_shape)) {
    result_shape = LayoutUtil::GetWithDefaultLayout(result_shape);
  }

  // In general, we require a layout to evaluate a bitcast: this is the only
  // operation where indexing is physical rather than logical.
  if (!result_shape.has_layout()) {
    return absl::InvalidArgumentError(
        "Evaluator cannot evaluate bitcast for non-scalar operand without "
        "assigned layout.");
  }
  TF_RETURN_IF_ERROR(ShapeUtil::ValidateShape(result_shape));

  Literal result(result_shape);

  // Bitcast output is allowed to be smaller than the input if the backend-
  // specific buffer sizes for the input and output are the same. Since the HLO
  // evaluator doesn't have access to the backend-specific shape size function,
  // assume it's OK to bitcast if output <= input.
  const Literal& operand_literal = GetEvaluatedLiteralFor(bitcast->operand(0));
  TF_RET_CHECK(operand_literal.size_bytes() >= result.size_bytes());
  memcpy(result.untyped_data(), operand_literal.untyped_data(),
         result.size_bytes());
  evaluated_[bitcast] = std::move(result);
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleBitcastConvert(const HloInstruction* convert) {
  const HloInstruction* operand = convert->operand(0);
  TF_ASSIGN_OR_RETURN(
      Literal result,
      GetEvaluatedLiteralFor(operand).BitcastConvert(convert->shape()));

  evaluated_[convert] = std::move(result);
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleGetDimensionSize(
    const HloInstruction* get_dimension_size) {
  const HloInstruction* operand = get_dimension_size->operand(0);
  int64_t dim = get_dimension_size->dimension();
  if (dynamic_dimension_inference_ == nullptr) {
    return absl::InvalidArgumentError(
        "Evaluator cannot evaluate get_dimension_size without "
        "set_dynamic_dimension_inference.");
  }
  const HloInstruction* dynamic_size =
      dynamic_dimension_inference_->GetDynamicSize(operand, {}, dim);
  if (dynamic_size != nullptr) {
    evaluated_[get_dimension_size] =
        GetEvaluatedLiteralFor(dynamic_size).Clone();
    return absl::OkStatus();
  }

  const Shape& shape = get_dimension_size->operand(0)->shape();
  Literal output(ShapeUtil::MakeShape(S32, {}));
  output.PopulateWithValue(
      static_cast<int32_t>(shape.dimensions(get_dimension_size->dimension())));
  evaluated_[get_dimension_size] = std::move(output);
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleSetDimensionSize(
    const HloInstruction* set_dimension_size) {
  const Literal& operand_literal =
      GetEvaluatedLiteralFor(set_dimension_size->operand(0));
  Literal result(set_dimension_size->shape());
  memcpy(result.untyped_data(), operand_literal.untyped_data(),
         operand_literal.size_bytes());
  const Literal& size_literal =
      GetEvaluatedLiteralFor(set_dimension_size->operand(1));
  result.SetDynamicSize(set_dimension_size->dimension(),
                        size_literal.Get<int32_t>({}));
  evaluated_[set_dimension_size] = std::move(result);
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleParameter(const HloInstruction* parameter) {
  if (!IsAlreadyEvaluated(parameter, visitor_shape_index_)) {
    if (!enable_partial_evaluation_) {
      return absl::FailedPreconditionError(
          "Failed to evaluate instruction since its operands are unknown "
          "or undetermined and partial evaluation is not enabled.");
    }
    evaluated_[parameter] =
        Literal::CreateFromShapeWithUnknownLeafArrays(parameter->shape());
    return absl::OkStatus();
  }

  if (!arg_literals_.empty()) {
    // Nothing to do other than sanity checks. Parameters' values are stored in
    // arg_literals_.
    CHECK_LT(parameter->parameter_number(), arg_literals_.size());
#ifndef NDEBUG
    const Literal* input_literal = arg_literals_[parameter->parameter_number()];
    VLOG(2) << "Parameter evaluated to: " << input_literal->ToString();
    bool check_layout = parameter->shape().has_layout();
    DCHECK(Shape::Equal()
               .IgnoreLayout(!check_layout)
               .MinorToMajorOnlyInLayout()(parameter->shape(),
                                           input_literal->shape()))
        << "parameter shape is: "
        << ShapeUtil::HumanStringWithLayout(parameter->shape())
        << ", but input literal shape is: "
        << ShapeUtil::HumanStringWithLayout(input_literal->shape());
#endif
  }

  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleInfeed(const HloInstruction* infeed) {
  if (!enable_partial_evaluation_) {
    return absl::FailedPreconditionError(
        "Failed to evaluate instruction since its operands are unknown "
        "or undetermined and partial evaluation is not enabled.");
  }
  evaluated_[infeed] =
      Literal::CreateFromShapeWithUnknownLeafArrays(infeed->shape());
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleConstant(const HloInstruction*) {
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleReshape(const HloInstruction* reshape) {
  TF_ASSIGN_OR_RETURN(evaluated_[reshape],
                      GetEvaluatedLiteralFor(reshape->operand(0))
                          .Reshape(reshape->shape().dimensions()));
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleTranspose(const HloInstruction* transpose) {
  evaluated_[transpose] = GetEvaluatedLiteralFor(transpose->operand(0))
                              .Transpose(transpose->dimensions());
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleConcatenate(
    const HloInstruction* concatenate) {
  absl::Span<HloInstruction* const> operands(concatenate->operands());
  // The result concatenate dimension is going to be the sum of all
  // concatenate dimensions of the operands taking part of the operation.
  const Shape& reference_shape = operands[0]->shape();
  CHECK(reference_shape.IsArray());
  const int64_t rank = reference_shape.rank();
  const int64_t concat_dim = concatenate->dimensions()[0];
  CHECK_GE(concat_dim, 0);
  CHECK_LT(concat_dim, rank);

  DimensionVector concat_dimensions(reference_shape.dimensions().begin(),
                                    reference_shape.dimensions().end());

  for (int64_t i = 1; i < operands.size(); ++i) {
    const Shape& operand_shape = operands[i]->shape();
    CHECK(operand_shape.IsArray());
    // Accumulate the concat dimension from all tensors taking part to the
    // operation.
    concat_dimensions[concat_dim] +=
        ShapeUtil::GetDimension(operand_shape, concat_dim);
  }

  auto result_literal = LiteralUtil::CreateFromDimensions(
      reference_shape.element_type(), concat_dimensions);
  DimensionVector source_indices(rank, 0);
  DimensionVector dest_indices(concat_dimensions.size(), 0);

  for (auto operand : operands) {
    const Shape& operand_shape = operand->shape();
    TF_RETURN_IF_ERROR(result_literal.CopySliceFrom(
        GetEvaluatedLiteralFor(operand), source_indices, dest_indices,
        operand_shape.dimensions()));
    dest_indices[concat_dim] +=
        ShapeUtil::GetDimension(operand_shape, concat_dim);
  }

  evaluated_[concatenate] = std::move(result_literal);
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleCompare(const HloInstruction* compare) {
  ComparisonDirection direction = compare->comparison_direction();
  ComparisonOrder order = compare->comparison_order();
  auto lhs = compare->operand(0);
  auto rhs = compare->operand(1);
  DCHECK(ShapeUtil::SameDimensions(compare->shape(), rhs->shape()) &&
         ShapeUtil::SameDimensions(lhs->shape(), rhs->shape()));

  TF_RET_CHECK(lhs->shape().element_type() == rhs->shape().element_type());
  auto element_type = lhs->shape().element_type();
  Comparison comparison(direction, element_type, order);

  const Literal& lhs_literal = GetEvaluatedLiteralFor(lhs);
  const Literal& rhs_literal = GetEvaluatedLiteralFor(rhs);
  // Note here we switch on the operand's type.
  return primitive_util::PrimitiveTypeSwitch<absl::Status>(
      [&](auto primitive_type_constant) -> absl::Status {
        if constexpr (primitive_util::IsArrayType(primitive_type_constant)) {
          using NativeT = primitive_util::NativeTypeOf<primitive_type_constant>;
          TF_ASSIGN_OR_RETURN(evaluated_[compare],
                              Compare<NativeT>(compare->shape(), comparison,
                                               lhs_literal, rhs_literal));
          return absl::OkStatus();
        }
        LOG(FATAL) << "HandleCompare: unknown primitive type: "
                   << PrimitiveType_Name(element_type);
      },
      element_type);
}

absl::Status HloEvaluator::HandleTuple(const HloInstruction* tuple) {
  std::vector<const Literal*> operand_literals;
  std::vector<Literal> operand_literal_values;
  if (!visitor_shape_index_.empty()) {
    // We only need to evaluate tuple at visitor_shape_index_. The other
    // operands might not have been evaluated, so mark the other operands as
    // undetermined.
    int64_t tuple_index = visitor_shape_index_.front();
    operand_literal_values.resize(tuple->operand_count());
    for (int operand_index = 0; operand_index < tuple->operand_count();
         ++operand_index) {
      if (operand_index == tuple_index) {
        operand_literals.push_back(
            &GetEvaluatedLiteralFor(tuple->operand(operand_index)));
      } else {
        operand_literal_values[operand_index] =
            Literal::CreateFromShapeWithUndeterminedLeafArrays(
                ShapeUtil::GetSubshape(tuple->shape(), {operand_index}));
        operand_literals.push_back(&operand_literal_values[operand_index]);
      }
    }
  } else {
    for (auto operand : tuple->operands()) {
      operand_literals.push_back(&GetEvaluatedLiteralFor(operand));
    }
  }

  // Inline part of LiteralUtil::MakeTuple() that avoids creating the leaf
  // buffers; these buffers can be extremely large.
  std::vector<const Shape*> element_shapes;
  element_shapes.reserve(operand_literals.size());
  for (const auto* element : operand_literals) {
    element_shapes.push_back(&element->shape());
  }
  Literal new_result = Literal::CreateFromShapeWithUndeterminedLeafArrays(
      ShapeUtil::MakeTupleShapeWithPtrs(element_shapes));
  for (int i = 0, end = operand_literals.size(); i < end; ++i) {
    TF_RETURN_IF_ERROR(
        new_result.CopyFrom(*operand_literals[i], /*dest_shape_index=*/{i}));
  }

  if (evaluated_.contains(tuple)) {
    CHECK(new_result.IsDetermined(visitor_shape_index_));
    TF_RETURN_IF_ERROR(
        evaluated_[tuple].CopyFrom(std::move(new_result),
                                   /*dest_shape_index=*/visitor_shape_index_,
                                   /*src_shape_index=*/visitor_shape_index_));
  } else {
    evaluated_[tuple] = std::move(new_result);
  }
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleFft(const HloInstruction* fft) {
  return absl::UnimplementedError("Fft is not implemented.");
  // const Literal& input_literal = GetEvaluatedLiteralFor(fft->operand(0));
  // Literal output_literal = Literal::CreateFromShape(fft->shape());

  // FftTransform<complex128> transform(fft);
  // TF_RETURN_IF_ERROR(transform.ComputeFft(fft, input_literal,
  // &output_literal)); evaluated_[fft] = std::move(output_literal);

  // return absl::OkStatus();
}

absl::Status HloEvaluator::HandleGather(const HloInstruction* gather) {
  // TODO(chokobole): Implement this. Dependency: GatherDimensionNumbers
  return absl::UnimplementedError("HandleGather not implemented");
}

absl::Status HloEvaluator::HandleScatter(const HloInstruction* hlo) {
  // TODO(chokobole): Implement this. Dependency: HloScatterInstruction
  return absl::UnimplementedError("HandleScatter not implemented");
}

absl::Status HloEvaluator::HandleBroadcast(const HloInstruction* broadcast) {
  const Literal& operand = GetEvaluatedLiteralFor(broadcast->operand(0));
  TF_RET_CHECK(broadcast->shape().element_type() ==
               operand.shape().element_type())
      << " broadcast from a different data type is not supported";
  TF_RET_CHECK(broadcast->dimensions().size() == operand.shape().rank())
      << "broadcast dimensions is of size: " << broadcast->dimensions().size()
      << " and rank of operand_to_broadcast is: " << operand.shape().rank();
  // Checks that operand's dimensions are the same as the broadcast's
  // dimensions along the dimensions to be broadcasted.
  for (int64_t i = 0; i < broadcast->dimensions().size(); ++i) {
    auto operand_dim_size = operand.shape().dimensions(i);
    auto broadcast_dim_size =
        broadcast->shape().dimensions(broadcast->dimensions(i));
    TF_RET_CHECK(operand_dim_size == broadcast_dim_size) << absl::StreamFormat(
        "Operand dimension %d is broadcast to output dimension %d, but the "
        "sizes of these two dims do not match (%d vs %d): %s",
        i, broadcast->dimensions(i), operand_dim_size, broadcast_dim_size,
        broadcast->ToString());
  }

  TF_ASSIGN_OR_RETURN(
      evaluated_[broadcast],
      operand.Broadcast(broadcast->shape(), broadcast->dimensions()));

  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleAfterAll(const HloInstruction* after_all) {
  evaluated_[after_all] = LiteralUtil::CreateToken();
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleAddDependency(
    const HloInstruction* add_dependency) {
  // AddDependency just forwards its zero-th operand.
  evaluated_[add_dependency] =
      GetEvaluatedLiteralFor(add_dependency->operand(0)).Clone();
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleGetTupleElement(
    const HloInstruction* get_tuple_element) {
  const auto result_shape = get_tuple_element->shape();
  const int64_t index = get_tuple_element->tuple_index();

  auto operand = get_tuple_element->operand(0);
  TF_ASSIGN_OR_RETURN(
      auto inferred_return_shape,
      ShapeInference::InferGetTupleElementShape(operand->shape(), index));
  TF_RET_CHECK(ShapeUtil::Compatible(result_shape, inferred_return_shape))
      << "return shape set to: " << ShapeUtil::HumanString(result_shape)
      << " but is inferred to be: "
      << ShapeUtil::HumanString(inferred_return_shape);

  const Literal& operand_tuple_literal = GetEvaluatedLiteralFor(operand);

  evaluated_[get_tuple_element] =
      Literal(ShapeUtil::GetTupleElementShape(operand->shape(), index));
  return evaluated_[get_tuple_element].CopyFrom(operand_tuple_literal,
                                                /*dest_shape_index=*/{},
                                                /*src_shape_index=*/{index});
}

absl::Status HloEvaluator::HandleCopy(const HloInstruction* copy) {
  // If only the element type is different, try converting the literal.
  if (copy->shape().element_type() !=
      copy->operand(0)->shape().element_type()) {
    TF_ASSIGN_OR_RETURN(Literal result,
                        GetEvaluatedLiteralFor(copy->operand(0))
                            .Convert(copy->shape().element_type()));
    TF_RET_CHECK(ShapeUtil::Compatible(copy->shape(), result.shape()));
    evaluated_[copy] = std::move(result);
  } else {
    TF_RET_CHECK(
        ShapeUtil::Compatible(copy->shape(), copy->operand(0)->shape()));
    evaluated_[copy] = GetEvaluatedLiteralFor(copy->operand(0)).Clone();
  }
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleAsyncStart(const HloInstruction* async_start) {
  std::vector<const Literal*> arg_literals;
  arg_literals.reserve(async_start->operands().size());
  for (auto operand : async_start->operands()) {
    const Literal& arg_literal = GetEvaluatedLiteralFor(operand);
    arg_literals.push_back(&arg_literal);
  }

  std::unique_ptr<HloEvaluator> embedded_evaluator =
      CreateEmbedded(max_loop_iterations_);
  embedded_evaluator->set_dynamic_dimension_inference(
      dynamic_dimension_inference_);
  TF_ASSIGN_OR_RETURN(
      Literal result,
      embedded_evaluator->Evaluate(*async_start->async_wrapped_computation(),
                                   arg_literals));

  evaluated_[async_start] = Literal(async_start->shape());
  // Copy the operand values to the index {0, i} of the output.
  for (int i = 0; i < arg_literals.size(); ++i) {
    TF_RETURN_IF_ERROR(evaluated_[async_start].CopyFrom(
        *arg_literals[i], /*dest_shape_index=*/{0, i},
        /*src_shape_index=*/{}));
  }
  // Move the output value to the index {1} of the output.
  TF_RETURN_IF_ERROR(evaluated_[async_start].MoveFrom(
      std::move(result), /*dest_shape_index=*/{1}));

  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleAsyncUpdate(
    const HloInstruction* async_update) {
  const Literal& operand_tuple_literal =
      GetEvaluatedLiteralFor(async_update->operand(0));
  evaluated_[async_update] = Literal(async_update->shape());
  TF_RETURN_IF_ERROR(evaluated_[async_update].CopyFrom(operand_tuple_literal,
                                                       /*dest_shape_index=*/{},
                                                       /*src_shape_index=*/{}));
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleAsyncDone(const HloInstruction* async_done) {
  const Literal& operand_tuple_literal =
      GetEvaluatedLiteralFor(async_done->operand(0));
  evaluated_[async_done] = Literal(async_done->shape());
  TF_RETURN_IF_ERROR(evaluated_[async_done].CopyFrom(operand_tuple_literal,
                                                     /*dest_shape_index=*/{},
                                                     /*src_shape_index=*/{1}));
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleCopyStart(const HloInstruction* copy_start) {
  if (copy_start->user_count() != 1 ||
      copy_start->users().at(0)->opcode() != HloOpcode::kCopyDone) {
    return absl::FailedPreconditionError(
        absl::StrCat("Cannot evaluate a kCopyStart that doesn't have a single "
                     "kCopyDone user. Instruction: ",
                     copy_start->ToString()));
  }

  // The context in index {2} is undefined, but since we can't represent
  // undefined values using a Literal, we just use 0. This should be safe though
  // since we ensure that the only user of a kCopyStart is a kCopyDone which
  // consumes the context. Also note that MakeTuple copies its arguments, so
  // this is memory-safe.
  const Literal context_literal = LiteralUtil::CreateR0<uint32_t>(0);
  evaluated_[copy_start] = LiteralUtil::MakeTuple(
      {&GetEvaluatedLiteralFor(copy_start->operand(0)),
       &GetEvaluatedLiteralFor(copy_start->operand(0)), &context_literal});
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleCopyDone(const HloInstruction* copy_done) {
  const HloInstruction* operand = copy_done->operand(0);
  if (operand->opcode() != HloOpcode::kCopyStart) {
    return absl::FailedPreconditionError(
        absl::StrCat("Cannot evaluate a kCopyDone that doesn't have a "
                     "kCopyStart as operand. Instruction: ",
                     copy_done->ToString()));
  }

  const Literal& operand_tuple_literal = GetEvaluatedLiteralFor(operand);
  evaluated_[copy_done] =
      Literal(ShapeUtil::GetTupleElementShape(operand->shape(), /*index=*/0));
  TF_RETURN_IF_ERROR(evaluated_[copy_done].CopyFrom(operand_tuple_literal,
                                                    /*dest_shape_index=*/{},
                                                    /*src_shape_index=*/{0}));
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleCall(const HloInstruction* call) {
  auto* computation = call->to_apply();
  auto operands = call->operands();

  std::vector<const Literal*> arg_literals;
  arg_literals.reserve(operands.size());
  for (auto operand : operands) {
    const Literal& arg_literal = GetEvaluatedLiteralFor(operand);
    arg_literals.push_back(&arg_literal);
  }

  std::unique_ptr<HloEvaluator> embedded_evaluator =
      CreateEmbedded(max_loop_iterations_);
  embedded_evaluator->set_dynamic_dimension_inference(
      dynamic_dimension_inference_);
  TF_ASSIGN_OR_RETURN(Literal result,
                      embedded_evaluator->Evaluate(*computation, arg_literals));

  evaluated_[call] = std::move(result);
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleFusion(const HloInstruction* fusion) {
  HloModuleConfig config;
  // Attach cloned computation to an empty HLO module so the existing ones are
  // not modified.
  HloModule empty_hlo_module("EmptyModuleForFusion", config,
                             std::make_unique<CompilationEnvironments>(
                                 fusion->GetModule()->comp_envs()));
  HloCloneContext context(&empty_hlo_module);
  auto cloned_fused_computation =
      fusion->fused_instructions_computation()->Clone(
          /*suffix=*/"clone_with_layout", &context);
  for (auto* instruction : cloned_fused_computation->instructions()) {
    if (!LayoutUtil::HasLayout(instruction->shape())) {
      LayoutUtil::SetToDefaultLayout(instruction->mutable_shape());
    }
  }
  auto readded_computation =
      empty_hlo_module.AddEntryComputation(std::move(cloned_fused_computation));

  auto operands = fusion->operands();
  std::vector<const Literal*> arg_literals;
  arg_literals.reserve(operands.size());
  for (auto operand : operands) {
    const Literal& arg_literal = GetEvaluatedLiteralFor(operand);
    arg_literals.push_back(&arg_literal);
  }

  std::unique_ptr<HloEvaluator> embedded_evaluator =
      CreateEmbedded(max_loop_iterations_);
  embedded_evaluator->set_dynamic_dimension_inference(
      dynamic_dimension_inference_);
  TF_ASSIGN_OR_RETURN(Literal result, embedded_evaluator->Evaluate(
                                          *readded_computation, arg_literals));

  evaluated_[fusion] = std::move(result);
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleConditional(
    const HloInstruction* conditional) {
  const auto& branch_index_literal =
      GetEvaluatedLiteralFor(conditional->operand(0));
  int branch_index;
  if (conditional->operand(0)->shape().element_type() == PRED) {
    branch_index = branch_index_literal.Get<bool>({}) ? 0 : 1;
  } else {
    branch_index = branch_index_literal.Get<int32_t>({});
    if (branch_index < 0 || branch_index >= conditional->branch_count()) {
      branch_index = conditional->branch_count() - 1;
    }
  }
  const auto& branch_computation_arg =
      GetEvaluatedLiteralFor(conditional->operand(1 + branch_index));

  std::unique_ptr<HloEvaluator> embedded_evaluator =
      CreateEmbedded(max_loop_iterations_);
  embedded_evaluator->set_dynamic_dimension_inference(
      dynamic_dimension_inference_);
  TF_ASSIGN_OR_RETURN(Literal result,
                      embedded_evaluator->Evaluate(
                          *conditional->branch_computation(branch_index),
                          {&branch_computation_arg}));

  evaluated_[conditional] = std::move(result);
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleConvert(const HloInstruction* convert) {
  const HloInstruction* operand = convert->operand(0);
  TF_RET_CHECK(ShapeUtil::SameDimensions(operand->shape(), convert->shape()));
  TF_ASSIGN_OR_RETURN(Literal result, GetEvaluatedLiteralFor(operand).Convert(
                                          convert->shape().element_type()));
  evaluated_[convert] = std::move(result);
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleDynamicSlice(
    const HloInstruction* dynamic_slice) {
  auto operand = dynamic_slice->operand(0);
  auto start_indices = dynamic_slice->operand(1);
  auto result_shape = dynamic_slice->shape();
  TF_ASSIGN_OR_RETURN(
      auto inferred_return_shape,
      ShapeInference::InferDynamicSliceShape(
          operand->shape(),
          Cast<HloDynamicSliceInstruction>(dynamic_slice)->index_shapes(),
          dynamic_slice->dynamic_slice_sizes()));
  TF_RET_CHECK(ShapeUtil::Compatible(result_shape, inferred_return_shape))
      << "return shape is set to: " << ShapeUtil::HumanString(result_shape)
      << " but is inferred to be: "
      << ShapeUtil::HumanString(inferred_return_shape);
  TF_RET_CHECK(
      primitive_util::IsIntegralType(start_indices->shape().element_type()));

  const Literal& operand_literal = GetEvaluatedLiteralFor(operand);

  std::vector<int64_t> start =
      GetS64Indices(absl::MakeConstSpan(dynamic_slice->operands()).subspan(1));

  // Clamp the start indices so the slice is in-bounds w.r.t the operand.
  for (int64_t i = 0; i < start.size(); ++i) {
    start[i] = std::min<int64_t>(
        std::max(int64_t{0}, start[i]),
        operand_literal.shape().dimensions(i) - result_shape.dimensions(i));
  }

  std::vector<int64_t> operand_index(start.size());
  Literal result(result_shape);
  const size_t element_byte_size =
      primitive_util::ByteWidth(result_shape.element_type());
  auto* operand_base = static_cast<const char*>(operand_literal.untyped_data());
  auto func = [&](void* dest, absl::Span<const int64_t> result_index) {
    for (int64_t i = 0; i < operand_index.size(); ++i) {
      CHECK_GE(result_index[i] + start[i], 0);
      operand_index[i] = result_index[i] + start[i];
    }

    auto* src = operand_base + (element_byte_size *
                                IndexUtil::MultidimensionalIndexToLinearIndex(
                                    operand_literal.shape(), operand_index));

    std::memcpy(dest, src, element_byte_size);
    return true;
  };
  TF_RETURN_IF_ERROR(result.PopulateInplace(func));
  evaluated_[dynamic_slice] = std::move(result);
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleDynamicUpdateSlice(const HloInstruction* dus) {
  auto operand = dus->operand(0);
  auto update = dus->operand(1);
  auto start_indices = dus->operand(2);
  auto result_shape = dus->shape();
  TF_ASSIGN_OR_RETURN(
      auto inferred_return_shape,
      ShapeInference::InferDynamicUpdateSliceShape(
          operand->shape(), update->shape(),
          Cast<HloDynamicUpdateSliceInstruction>(dus)->index_shapes()));
  TF_RET_CHECK(ShapeUtil::Compatible(result_shape, inferred_return_shape))
      << "return shape is set to: " << ShapeUtil::HumanString(result_shape)
      << " but is inferred to be: "
      << ShapeUtil::HumanString(inferred_return_shape);
  TF_RET_CHECK(
      primitive_util::IsIntegralType(start_indices->shape().element_type()));
  TF_RET_CHECK(ShapeUtil::Compatible(result_shape, operand->shape()));

  const Literal& operand_literal = GetEvaluatedLiteralFor(operand);
  const Literal& update_literal = GetEvaluatedLiteralFor(update);

  auto result = operand_literal.Clone();
  const auto rank = result.shape().rank();
  std::vector<int64_t> start =
      GetS64Indices(absl::MakeConstSpan(dus->operands()).subspan(2));

  // Clamp the update start indices so the slice is in-bounds w.r.t the
  // operand.
  for (int64_t i = 0; i < rank; ++i) {
    start[i] = std::min<int64_t>(
        std::max<int64_t>(0, start[i]),
        result.shape().dimensions(i) - update_literal.shape().dimensions(i));
  }
  std::vector<int64_t> result_index(rank, 0);

  auto func = [&](absl::Span<const int64_t> update_index) {
    std::transform(update_index.begin(), update_index.end(), start.begin(),
                   result_index.begin(), std::plus<int64_t>());
    result.CopyElementFrom(update_literal, update_index, result_index);
    return true;
  };

  std::vector<int64_t> base(update_literal.shape().dimensions_size(), 0);
  std::vector<int64_t> step(update_literal.shape().dimensions_size(), 1);
  ShapeUtil::ForEachIndexNoStatus(update_literal.shape(), base,
                                  update_literal.shape().dimensions(), step,
                                  func);
  evaluated_[dus] = std::move(result);

  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleSelect(const HloInstruction* select) {
  const auto& pred = GetEvaluatedLiteralFor(select->operand(0));
  const auto& on_true = GetEvaluatedLiteralFor(select->operand(1));
  const auto& on_false = GetEvaluatedLiteralFor(select->operand(2));

  // If predicate is of scalar type, no element-wise selection would be needed.
  if (ShapeUtil::IsScalar(pred.shape())) {
    if (pred.Get<bool>({})) {
      evaluated_[select] = on_true.Clone();
    } else {
      evaluated_[select] = on_false.Clone();
    }
    return absl::OkStatus();
  }

  return DefaultAction(select);
}

namespace {

absl::StatusOr<Literal> CreateScalarLiteral(int64_t value,
                                            PrimitiveType element_type) {
  return primitive_util::PrimitiveTypeSwitch<absl::StatusOr<Literal>>(
      [&](auto primitive_type_constant) -> absl::StatusOr<Literal> {
        if constexpr (primitive_util::IsIntegralType(primitive_type_constant)) {
          return LiteralUtil::CreateR0(
              static_cast<NativeTypeOf<primitive_type_constant>>(value));
        }
        return absl::InvalidArgumentError("Unsupported element type.");
      },
      element_type);
}

// Parses the while loop if it matches one of the known patterns. Returns the
// value of the loop induction variable after the loop execution if the loop is
// static.
absl::StatusOr<Literal> TryParseAndEvaluateWhileInductionVar(
    const HloInstruction* while_hlo) {
  std::optional<ParsedWhileLoop> parsed_while_loop =
      PatternMatchParseWhileLoop(while_hlo, /*precomputed_analyses=*/{});
  if (!parsed_while_loop.has_value() || parsed_while_loop->is_dynamic()) {
    return absl::FailedPreconditionError(
        "Cannot evaluate a while loop's induction variable since the loop "
        "does not match a known loop pattern or the loop is not static.");
  }
  int64_t induction_var_value =
      parsed_while_loop->static_while_loop->induction_var_init_value +
      parsed_while_loop->static_while_loop->trip_count *
          parsed_while_loop->static_while_loop->step_size;
  Shape result_shape = while_hlo->shape().tuple_shapes(
      parsed_while_loop->static_while_loop->induction_var_index);
  TF_ASSIGN_OR_RETURN(
      Literal result,
      CreateScalarLiteral(induction_var_value, result_shape.element_type()));
  std::vector<Literal*> while_result_element_ptrs;
  while_result_element_ptrs.reserve(while_hlo->shape().tuple_shapes_size());
  std::vector<Literal> while_result_elements(
      while_hlo->shape().tuple_shapes_size());
  for (int i = 0; i < while_hlo->shape().tuple_shapes_size(); ++i) {
    if (i == parsed_while_loop->static_while_loop->induction_var_index) {
      while_result_element_ptrs.push_back(&result);
    } else {
      const Shape& shape = while_hlo->shape().tuple_shapes(i);
      while_result_elements[i] =
          Literal::CreateFromShapeWithUnknownLeafArrays(shape);
      while_result_element_ptrs.push_back(&while_result_elements[i]);
    }
  }
  return LiteralUtil::MakeTuple(while_result_element_ptrs);
}

}  // namespace

absl::Status HloEvaluator::HandleWhile(const HloInstruction* while_hlo) {
  const HloComputation* cond_comp = while_hlo->while_condition();
  const HloComputation* body_comp = while_hlo->while_body();
  // Initialize the loop carried valued with the input to the While instruction.
  auto lcv = GetEvaluatedLiteralFor(while_hlo->operand(0)).Clone();
  if (!lcv.IsKnown()) {
    std::optional<ParsedWhileLoop> parsed_while_loop =
        PatternMatchParseWhileLoop(while_hlo,
                                   /*precomputed_analyses=*/{});
    evaluated_[while_hlo] =
        Literal::CreateFromShapeWithUnknownLeafArrays(while_hlo->shape());
    if (!parsed_while_loop.has_value() || parsed_while_loop->is_dynamic() ||
        visitor_shape_index_.size() != 1 ||
        parsed_while_loop->static_while_loop->induction_var_index !=
            visitor_shape_index_[0]) {
      return absl::OkStatus();
    }
    Shape induction_var_shape =
        ShapeUtil::GetSubshape(while_hlo->shape(), visitor_shape_index_);
    int64_t trip_count = parsed_while_loop->static_while_loop->trip_count;
    TF_ASSIGN_OR_RETURN(
        Literal induction_var_val,
        CreateScalarLiteral(trip_count, induction_var_shape.element_type()));
    TF_RETURN_IF_ERROR(evaluated_[while_hlo].CopyFrom(
        induction_var_val, /*dest_shape_index=*/visitor_shape_index_,
        /*src_shape_index=*/{}));
    return absl::OkStatus();
  }
  bool keep_going = true;
  int64_t iteration_count = 0;
  std::unique_ptr<HloEvaluator> cond_evaluator =
      CreateEmbedded(max_loop_iterations_);
  cond_evaluator->set_dynamic_dimension_inference(dynamic_dimension_inference_);
  std::unique_ptr<HloEvaluator> loop_body_evaluator =
      CreateEmbedded(max_loop_iterations_);
  loop_body_evaluator->set_dynamic_dimension_inference(
      dynamic_dimension_inference_);
  while (keep_going) {
    if (max_loop_iterations_ >= 0 && iteration_count++ > max_loop_iterations_) {
      absl::StatusOr<Literal> result =
          TryParseAndEvaluateWhileInductionVar(while_hlo);
      if (result.ok()) {
        lcv = std::move(result).value();
        break;
      } else {
        return absl::InvalidArgumentError(
            absl::StrFormat("Loop %s exceeded loop iteration limit (%d).",
                            while_hlo->name(), max_loop_iterations_));
      }
    }
    TF_ASSIGN_OR_RETURN(auto cond_val,
                        cond_evaluator->Evaluate(*cond_comp, {&lcv}));
    keep_going = cond_val.GetFirstElement<bool>();
    if (keep_going) {
      TF_ASSIGN_OR_RETURN(auto body_val,
                          loop_body_evaluator->Evaluate(*body_comp, {&lcv}));
      VLOG(3) << "Loop iteration result: " << body_val.ToString();
      lcv = std::move(body_val);
      cond_evaluator->ResetVisitStates();
      loop_body_evaluator->ResetVisitStates();
    }
  }
  evaluated_[while_hlo] = std::move(lcv);
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleReverse(const HloInstruction* reverse) {
  const Shape& result_shape = reverse->shape();
  const auto reverse_dimensions = reverse->dimensions();

  auto operand = reverse->operand(0);
  TF_ASSIGN_OR_RETURN(
      auto inferred_return_shape,
      ShapeInference::InferReverseShape(operand->shape(), reverse_dimensions));

  TF_RET_CHECK(ShapeUtil::Compatible(result_shape, inferred_return_shape))
      << "return shape set to: " << ShapeUtil::HumanString(result_shape)
      << " but is inferred to be: "
      << ShapeUtil::HumanString(inferred_return_shape);

  const Literal& operand_literal = GetEvaluatedLiteralFor(operand);
  Literal result(result_shape);
  const size_t element_byte_size =
      primitive_util::ByteWidth(result_shape.element_type());
  auto* operand_base = static_cast<const char*>(operand_literal.untyped_data());
  TF_RETURN_IF_ERROR(result.PopulateInplaceParallel(
      [&](void* dest, absl::Span<const int64_t> out_index, int) {
        std::vector<int64_t> from_index(out_index.begin(), out_index.end());
        for (const int64_t dim : reverse_dimensions) {
          from_index[dim] = result_shape.dimensions(dim) - 1 - out_index[dim];
        }
        auto* src =
            operand_base +
            (element_byte_size * IndexUtil::MultidimensionalIndexToLinearIndex(
                                     operand_literal.shape(), from_index));
        std::memcpy(dest, src, element_byte_size);
      }));

  evaluated_[reverse] = std::move(result);
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleBitReverse(const HloInstruction* bit_reverse) {
  const Shape& result_shape = bit_reverse->shape();
  const auto bit_reverse_dimensions = bit_reverse->dimensions();

  auto operand = bit_reverse->operand(0);
  TF_ASSIGN_OR_RETURN(auto inferred_return_shape,
                      ShapeInference::InferBitReverseShape(
                          operand->shape(), bit_reverse_dimensions));

  TF_RET_CHECK(ShapeUtil::Compatible(result_shape, inferred_return_shape))
      << "return shape set to: " << ShapeUtil::HumanString(result_shape)
      << " but is inferred to be: "
      << ShapeUtil::HumanString(inferred_return_shape);

  const Literal& operand_literal = GetEvaluatedLiteralFor(operand);
  evaluated_[bit_reverse] = operand_literal.BitReverse(bit_reverse_dimensions);
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleSlice(const HloInstruction* slice) {
  auto operand = slice->operand(0);
  const Shape& shape = slice->shape();
  TF_ASSIGN_OR_RETURN(auto inferred_return_shape,
                      ShapeInference::InferSliceShape(
                          operand->shape(), slice->slice_starts(),
                          slice->slice_limits(), slice->slice_strides()));
  TF_RET_CHECK(ShapeUtil::Compatible(shape, inferred_return_shape))
      << "return shape set to: " << ShapeUtil::HumanString(shape)
      << " but is inferred to be: "
      << ShapeUtil::HumanString(inferred_return_shape);

  const int64_t rank = operand->shape().rank();
  const Literal& operand_literal = GetEvaluatedLiteralFor(operand);
  const size_t element_byte_size =
      primitive_util::ByteWidth(shape.element_type());
  auto* operand_base = static_cast<const char*>(operand_literal.untyped_data());
  auto func = [&](void* dest, absl::Span<const int64_t> out_index, int) {
    DimensionVector operand_index(rank);
    for (int64_t i = 0; i < rank; ++i) {
      operand_index[i] =
          slice->slice_starts(i) + out_index[i] * slice->slice_strides(i);
    }
    auto* src = operand_base + (element_byte_size *
                                IndexUtil::MultidimensionalIndexToLinearIndex(
                                    operand_literal.shape(), operand_index));
    std::memcpy(dest, src, element_byte_size);
  };

  Literal result(shape);
  TF_RETURN_IF_ERROR(result.PopulateInplaceParallel(func));
  evaluated_[slice] = std::move(result);
  return absl::OkStatus();
}

namespace {

template <typename NativeT>
Literal ExtractLiteralFromIndexPositions(const Literal& from,
                                         absl::Span<int64_t const> indices) {
  // We use a InlinedVector here because we need to convert it to an
  // absl::Span later, and this would not work with std::vector<bool>.
  absl::InlinedVector<NativeT, 10> values;
  for (int64_t index : indices) {
    values.push_back(from.Get<NativeT>({index}));
  }
  return LiteralUtil::CreateR1<NativeT>(values);
}

absl::StatusOr<Literal> ExtractFromIndexPositions(
    const Literal& from, absl::Span<int64_t const> indices) {
  PrimitiveType type = from.shape().element_type();
  return primitive_util::PrimitiveTypeSwitch<absl::StatusOr<Literal>>(
      [&](auto primitive_type_constant) -> absl::StatusOr<Literal> {
        if constexpr (primitive_util::IsArrayType(primitive_type_constant)) {
          return ExtractLiteralFromIndexPositions<
              NativeTypeOf<primitive_type_constant>>(from, indices);
        }
        return absl::InvalidArgumentError(absl::StrFormat(
            "Unsupported type for Sort: %s", PrimitiveType_Name(type)));
      },
      type);
}

}  // namespace

absl::Status HloEvaluator::HandleSort(const HloInstruction* sort) {
  TF_RET_CHECK(sort->operand_count() >= 1)
      << "Expected at least 1 operand for sort";
  for (int64_t i = 1; i < sort->operand_count(); ++i) {
    TF_RET_CHECK(ShapeUtil::SameDimensions(sort->operand(0)->shape(),
                                           sort->operand(i)->shape()))
        << "All Sort operands must have the same dimensions";
  }

  if (VLOG_IS_ON(3)) {
    for (int64_t i = 0; i < sort->operand_count(); ++i) {
      VLOG(3) << "HandleSort operand " << i << " literal: "
              << GetEvaluatedLiteralFor(sort->operand(i)).ToString();
    }
  }
  Shape key_shape = sort->operand(0)->shape();
  auto rank = key_shape.rank();
  std::vector<Literal> result_literals;
  result_literals.reserve(sort->operand_count());
  for (int64_t i = 0; i < sort->operand_count(); ++i) {
    result_literals.emplace_back(sort->operand(i)->shape());
  }
  std::vector<int64_t> zero_base(rank, 0);
  std::vector<int64_t> increment(rank, 1);
  int64_t sort_dim = sort->dimensions(0);
  int64_t sort_dim_elements = key_shape.dimensions(sort_dim);
  TF_RET_CHECK(sort_dim >= 0 && sort_dim < increment.size())
      << "Unexpected out-of-bound sort dimension " << sort_dim
      << " accessing increment of size " << increment.size();
  increment[sort_dim] = sort_dim_elements;

  auto comparator =
      [sort](absl::Span<const Literal> literals_to_sort, int64_t a, int64_t b,
             HloEvaluator* embedded_evaluator) -> absl::StatusOr<bool> {
    absl::InlinedVector<Literal, 8> literals;
    literals.reserve(2 * sort->operand_count());
    for (int64_t i = 0; i < sort->operand_count(); ++i) {
      literals.push_back(
          LiteralUtil::GetScalarLiteral(literals_to_sort[i], {a}));
      literals.push_back(
          LiteralUtil::GetScalarLiteral(literals_to_sort[i], {b}));
    }
    absl::InlinedVector<const Literal*, 8> literal_ptrs;
    absl::c_transform(literals, std::back_inserter(literal_ptrs),
                      [](const Literal& literal) { return &literal; });

    TF_ASSIGN_OR_RETURN(
        auto computed_result,
        embedded_evaluator->Evaluate(*sort->to_apply(), literal_ptrs));
    // Clear visit states so that we can use the evaluator again
    // on the same computation.
    embedded_evaluator->ResetVisitStates();
    return computed_result.Get<bool>({});
  };
  auto less_than =
      [&comparator](absl::Span<const Literal> literals_to_sort, int64_t a,
                    int64_t b,
                    HloEvaluator* embedded_evaluator) -> absl::StatusOr<bool> {
    TF_ASSIGN_OR_RETURN(bool a_is_smaller,
                        comparator(literals_to_sort, a, b, embedded_evaluator));
#ifndef NDEBUG
    // Let's see if the comparator violates strict weak ordering.
    // N.B. This does not test transitivity.
    TF_ASSIGN_OR_RETURN(bool b_is_smaller,
                        comparator(literals_to_sort, b, a, embedded_evaluator));
    TF_RET_CHECK(!(b_is_smaller && a_is_smaller));
    TF_ASSIGN_OR_RETURN(bool b_is_reflexive,
                        comparator(literals_to_sort, b, b, embedded_evaluator));
    TF_RET_CHECK(!b_is_reflexive);
    TF_ASSIGN_OR_RETURN(bool a_is_reflexive,
                        comparator(literals_to_sort, a, a, embedded_evaluator));
    TF_RET_CHECK(!a_is_reflexive);
#endif
    return a_is_smaller;
  };
  std::function<absl::Status(absl::Span<const Literal>, absl::Span<int64_t>,
                             absl::Span<int64_t>, absl::Span<int64_t>,
                             std::vector<int64_t>&, HloEvaluator*)>
      merge = [&](absl::Span<const Literal> literals_to_sort,
                  absl::Span<int64_t> lhs, absl::Span<int64_t> rhs,
                  absl::Span<int64_t> output, std::vector<int64_t>& tmp,
                  HloEvaluator* embedded_evaluator) -> absl::Status {
    tmp.clear();
    tmp.reserve(output.size());
    // Keep picking between elements.
    while (!lhs.empty() && !rhs.empty()) {
      // If rhs < lhs, pick rhs. Otherwise, pick lhs. This should ensure
      // stability as lhs comes first in the array.
      TF_ASSIGN_OR_RETURN(bool rhs_is_smaller,
                          less_than(literals_to_sort, rhs.front(), lhs.front(),
                                    embedded_evaluator));
      if (rhs_is_smaller) {
        tmp.push_back(rhs.front());
        rhs.remove_prefix(1);
      } else {
        tmp.push_back(lhs.front());
        lhs.remove_prefix(1);
      }
    }
    // At least one of the two input arrays are now empty, we need to copy
    // the remaining elements.
    absl::c_copy(lhs, std::back_inserter(tmp));
    absl::c_copy(rhs, std::back_inserter(tmp));
    absl::c_copy(tmp, output.begin());
    return absl::OkStatus();
  };
  auto* env = tsl::Env::Default();
  const int max_parallelism = tsl::port::MaxParallelism();
  constexpr size_t kMinElementsPerThread{1024};
  const size_t useful_parallelism = std::min<size_t>(
      sort_dim_elements / kMinElementsPerThread, max_parallelism);
  const size_t work_per_thread = useful_parallelism > 1
                                     ? sort_dim_elements / useful_parallelism
                                     : std::numeric_limits<size_t>::max();
  std::function<absl::Status(absl::Span<const Literal>, absl::Span<int64_t>,
                             std::vector<int64_t>*, HloEvaluator*)>
      mergesort = [&merge, &mergesort, &less_than, this, env, work_per_thread](
                      absl::Span<const Literal> literals_to_sort,
                      absl::Span<int64_t> to_sort,
                      std::vector<int64_t>* scratch,
                      HloEvaluator* embedded_evaluator) -> absl::Status {
    // Base case: inputs with 0 or 1 elements are already sorted.
    if (to_sort.size() < 2) {
      return absl::OkStatus();
    }
    size_t halfway = to_sort.size() / 2;
    auto lhs = to_sort.subspan(/*pos=*/0, halfway);
    auto rhs = to_sort.subspan(/*pos=*/halfway);

    // Allocate an evaluator if we never got one, we will reuse an
    // allocator so long as we are not moving it between threads.
    std::unique_ptr<HloEvaluator> thread_local_embedded_evaluator;
    if (embedded_evaluator == nullptr) {
      thread_local_embedded_evaluator = CreateEmbedded(max_loop_iterations_);
      embedded_evaluator = thread_local_embedded_evaluator.get();
    }

    constexpr size_t kMinElementsForMergesort{9};
    if (to_sort.size() >= kMinElementsForMergesort) {
      std::unique_ptr<std::vector<int64_t>> thread_local_scratch;
      if (!scratch) {
        thread_local_scratch = std::make_unique<std::vector<int64_t>>();
        scratch = thread_local_scratch.get();
      }
      // Overlap sorting the LHS with the RHS if we have enough work to
      // do. The recursive call for to `mergesort(rhs)` will potentially
      // create more threads.
      absl::Status lhs_status;
      if (to_sort.size() >= work_per_thread) {
        std::unique_ptr<tsl::Thread> thread = absl::WrapUnique(env->StartThread(
            tsl::ThreadOptions(), "XLA_mergesort",
            [literals_to_sort, lhs, &mergesort, &lhs_status] {
              lhs_status = mergesort(literals_to_sort, lhs, nullptr, nullptr);
            }));
        TF_RETURN_IF_ERROR(
            mergesort(literals_to_sort, rhs, scratch, embedded_evaluator));
        // Here, `thread` will run its destructor ensuring that it is done
        // sorting `lhs`.
        thread.reset();
      } else {
        TF_RETURN_IF_ERROR(
            mergesort(literals_to_sort, rhs, scratch, embedded_evaluator));
        lhs_status =
            mergesort(literals_to_sort, lhs, scratch, embedded_evaluator);
      }
      TF_RETURN_IF_ERROR(lhs_status);
      TF_RETURN_IF_ERROR(merge(literals_to_sort, lhs, rhs, to_sort, *scratch,
                               embedded_evaluator));
    } else {
      // Do an insertion sort. Values to the left of `i` are sorted.
      // Any values larger than it in will be moved past `i`. Binary
      // search in [0, i) looking for the smallest value larger than `i`
      // which we will call `ub`. By induction, [ub, i) are all larger
      // than `i`.
      for (auto i = to_sort.begin(); i != to_sort.end(); ++i) {
        auto len = i - to_sort.begin();
        auto ub = to_sort.begin();
        auto needle = *i;
        while (len != 0) {
          auto half_len = len / 2;
          auto midpoint = ub + half_len;
          TF_ASSIGN_OR_RETURN(bool is_smaller,
                              less_than(literals_to_sort, needle, *midpoint,
                                        embedded_evaluator));
          if (is_smaller) {
            // Our needle is smaller than the midpoint, we need to shrink
            // the range by trimming the rightmost portion of it. We can't
            // exclude the midpoint value yet.
            len = half_len;
          } else {
            // Our needle is at least as big as the midpoint but we want
            // something larger, we can exclude the midpoint.
            ub = midpoint + 1;
            len -= half_len + 1;
          }
        }
        // Shift values larger than `i` to the right by 1 and insert `i`
        // in the new gap. Now the sorted range is [0, i].
        std::rotate(ub, i, i + 1);
      }
    }
    return absl::OkStatus();
  };

  // Iterate through each dimension except 'sort_dim'.
  TF_RETURN_IF_ERROR(ShapeUtil::ForEachIndexWithStatus(
      key_shape, zero_base, key_shape.dimensions(), increment,
      [&](absl::Span<const int64_t> indices) -> absl::StatusOr<bool> {
        // Extract a slice from each operand literal that corresponds to
        // exactly the row in dimension 'sort_dim'.
        std::vector<int64_t> limit_indices(indices.begin(), indices.end());
        absl::c_for_each(limit_indices, [](int64_t& index) { ++index; });
        limit_indices[sort_dim] = sort_dim_elements;
        std::vector<Literal> literals_to_sort;
        literals_to_sort.reserve(sort->operand_count());
        for (int64_t i = 0; i < sort->operand_count(); ++i) {
          TF_ASSIGN_OR_RETURN(auto literal_to_sort,
                              GetEvaluatedLiteralFor(sort->operand(i))
                                  .Slice(indices, limit_indices)
                                  .Reshape({sort_dim_elements}));
          literals_to_sort.push_back(std::move(literal_to_sort));
        }
        std::vector<int64_t> indices_to_sort(sort_dim_elements);
        std::iota(indices_to_sort.begin(), indices_to_sort.end(), 0);
        TF_RETURN_IF_ERROR(mergesort(literals_to_sort,
                                     absl::MakeSpan(indices_to_sort), nullptr,
                                     nullptr));
        std::vector<int64_t> slice_dimensions(rank, 1);
        slice_dimensions[sort_dim] = sort_dim_elements;
        std::vector<int64_t> start_indices(rank, 0);
        for (int64_t i = 0; i < sort->operand_count(); ++i) {
          TF_ASSIGN_OR_RETURN(
              Literal sorted_literal,
              ExtractFromIndexPositions(literals_to_sort[i], indices_to_sort));
          TF_ASSIGN_OR_RETURN(auto sorted_literal_reshaped,
                              sorted_literal.Reshape(slice_dimensions));
          TF_RETURN_IF_ERROR(result_literals[i].CopySliceFrom(
              sorted_literal_reshaped, start_indices, indices,
              slice_dimensions));
        }
        return true;
      }));

  if (sort->operand_count() == 1) {
    evaluated_[sort] = std::move(result_literals[0]);
  } else {
    std::vector<const Literal*> literal_ptrs;
    absl::c_transform(result_literals, std::back_inserter(literal_ptrs),
                      [](const Literal& literal) { return &literal; });

    Literal result_tuple = LiteralUtil::MakeTuple(literal_ptrs);
    VLOG(3) << "HandleSort result_tuple: " << result_tuple.ToString();

    evaluated_[sort] = std::move(result_tuple);
  }
  return absl::OkStatus();
}

namespace {

// Run a single step of an inner loop while running reduction, which applies
// the user-provided computation on the accumulator and the output element
// (until the reduction is completed, the output element is also used as
// an accumulator).
absl::StatusOr<bool> PerformReductionStep(
    bool is_tuple, absl::Span<const int64_t> input_index,
    absl::Span<const int64_t> output_index,
    absl::Span<const Literal* const> input_args, absl::Span<Literal> results,
    HloComputation* computation, HloEvaluator* embedded_evaluator) {
  int num_args = results.size();

  absl::InlinedVector<Literal, 1> arg_values;
  arg_values.reserve(num_args);
  absl::InlinedVector<Literal, 1> accumulators;
  accumulators.reserve(num_args);
  for (int64_t i = 0; i < num_args; ++i) {
    arg_values.emplace_back(
        ShapeUtil::MakeShape(input_args[i]->shape().element_type(), {}));
    accumulators.emplace_back(
        ShapeUtil::MakeShape(input_args[i]->shape().element_type(), {}));

    arg_values[i].CopyElementFrom(*input_args[i], input_index, {});
    accumulators[i].CopyElementFrom(results[i], output_index, {});
  }

  // Evaluate computation with specified literal operands.
  absl::InlinedVector<Literal*, 2> embedded_operands;
  for (Literal& accumulator : accumulators) {
    embedded_operands.push_back(&accumulator);
  }
  for (Literal& local_input : arg_values) {
    embedded_operands.push_back(&local_input);
  }

  TF_ASSIGN_OR_RETURN(
      Literal computed_result,
      embedded_evaluator->Evaluate(*computation, embedded_operands));

  // Clear visit states so that we can use the evaluator again on the same
  // computation.
  embedded_evaluator->ResetVisitStates();

  if (is_tuple) {
    std::vector<Literal> computed_results = computed_result.DecomposeTuple();
    for (int64_t i = 0; i < num_args; ++i) {
      results[i].CopyElementFrom(computed_results[i], {}, output_index);
    }
  } else {
    results[0].CopyElementFrom(computed_result, {}, output_index);
  }

  return true;
}

absl::StatusOr<bool> GenerateReduceOutputElement(
    bool is_tuple, absl::Span<const int64_t> output_index,

    absl::Span<const Literal* const> init_values,
    absl::Span<const Literal* const> input_args, absl::Span<Literal> results,

    HloComputation* function, HloEvaluator* embedded_evaluator,

    absl::Span<const int64_t> arg_dim_steps,
    absl::Span<const int64_t> arg_dim_counts,
    absl::Span<const int64_t> result_to_arg_index) {
  const Shape& arg_shape = input_args[0]->shape();
  absl::Span<const int64_t> arg_dimensions = arg_shape.dimensions();
  std::vector<int64_t> base(arg_dimensions.size());
  for (int64_t i = 0; i < output_index.size(); ++i) {
    base[result_to_arg_index[i]] = output_index[i];
  }

  for (int64_t i = 0; i < results.size(); ++i) {
    results[i].CopyElementFrom(*init_values[i], {}, output_index);
  }

  // Iterates only over reduced shape, as counts and steps are set to zero
  // for all non-reduced dimensions.
  TF_RETURN_IF_ERROR(ShapeUtil::ForEachIndexWithStatus(
      arg_shape, base, arg_dim_counts, arg_dim_steps,
      [&](absl::Span<const int64_t> input_index) {
        return PerformReductionStep(is_tuple, input_index, output_index,
                                    input_args, results, function,
                                    embedded_evaluator);
      }));
  return true;
}

}  // namespace

absl::Status HloEvaluator::HandleReduce(const HloInstruction* hlo) {
  const HloReduceInstruction* reduce = Cast<HloReduceInstruction>(hlo);
  int64_t num_args = reduce->inputs().size();
  absl::Span<const int64_t> dimensions_to_reduce(reduce->dimensions());
  HloComputation* function = reduce->to_apply();

  absl::InlinedVector<const Shape*, 1> operand_shapes;
  for (const HloInstruction* operand : reduce->operands()) {
    operand_shapes.push_back(&operand->shape());
  }
  TF_ASSIGN_OR_RETURN(auto inferred_return_shape,
                      ShapeInference::InferReduceShape(
                          operand_shapes, dimensions_to_reduce,
                          /*to_apply=*/function->ComputeProgramShape()));
  TF_RET_CHECK(ShapeUtil::Compatible(reduce->shape(), inferred_return_shape))
      << "return shape is set to: " << ShapeUtil::HumanString(reduce->shape())
      << " but is inferred to be: "
      << ShapeUtil::HumanString(inferred_return_shape);

  absl::InlinedVector<const Literal*, 1> input_args(num_args);
  absl::InlinedVector<const Literal*, 1> init_values(num_args);
  for (int64_t i = 0; i < num_args; ++i) {
    input_args[i] = &GetEvaluatedLiteralFor(reduce->inputs()[i]);
    VLOG(3) << "HandleReduce arg_literal: " << input_args[i]->ToString();
    init_values[i] = &GetEvaluatedLiteralFor(reduce->init_values()[i]);
    VLOG(3) << "HandleReduce init_literal: " << init_values[i]->ToString();
    TF_RET_CHECK(ShapeUtil::IsScalar(init_values[i]->shape()));
  }

  // All args and results have the same dimensions, so pick an arbitrary one.
  const Shape& arg_shape = input_args[0]->shape();
  const Shape& out_shape = inferred_return_shape;
  bool is_tuple = out_shape.IsTuple();
  const Shape& output_shape = inferred_return_shape.IsTuple()
                                  ? inferred_return_shape.tuple_shapes(0)
                                  : inferred_return_shape;

  absl::Span<const int64_t> arg_dimensions = arg_shape.dimensions();

  // All increments are set to 0.
  std::vector<int64_t> arg_dim_steps(arg_dimensions.size());

  // All counts are set to 0.
  std::vector<int64_t> arg_dim_counts(arg_dimensions.size());

  // Set steps and counts for reduced dimensions.
  // This avoids iterating over non-reduced dimensions, as their step
  // and count is set to zero.
  for (const int64_t dim : dimensions_to_reduce) {
    arg_dim_steps[dim] = 1;
    arg_dim_counts[dim] = arg_dimensions[dim];
  }

  // Map each dimension in the result to a dimension in arg that isn't
  // being reduced.
  std::vector<int64_t> result_to_arg_index;
  for (int64_t i = 0; i < arg_dimensions.size(); ++i) {
    if (arg_dim_steps[i] == 0) {
      result_to_arg_index.push_back(i);
    }
  }

  const int num_threads = ShapeUtil::GetForEachIndexParallelThreadCount() + 1;
  std::vector<std::unique_ptr<HloEvaluator>> embedded_evaluators;
  embedded_evaluators.reserve(num_threads);
  for (int i = 0; i < num_threads; ++i) {
    embedded_evaluators.push_back(CreateEmbedded(max_loop_iterations_));
  }

  absl::InlinedVector<Literal, 1> results(num_args);
  for (int64_t i = 0; i < num_args; ++i) {
    results[i] = Literal(is_tuple ? out_shape.tuple_shapes(i) : out_shape);
  }

  TF_RETURN_IF_ERROR(ShapeUtil::ForEachIndexParallelWithStatus(
      output_shape, [&](absl::Span<const int64_t> output_index, int thread_id) {
        return GenerateReduceOutputElement(
            is_tuple, output_index, init_values, input_args,
            absl::Span<Literal>(results), function,
            embedded_evaluators[thread_id + 1].get(), arg_dim_steps,
            arg_dim_counts, result_to_arg_index);
      }));

  if (is_tuple) {
    Literal tuple_result(inferred_return_shape);
    for (int64_t i = 0; i < num_args; ++i) {
      CHECK_OK(tuple_result.MoveFrom(std::move(results[i]), {i}));
    }
    evaluated_[reduce] = std::move(tuple_result);
  } else {
    CHECK_EQ(results.size(), 1);
    evaluated_[reduce] = std::move(results[0]);
  }
  if (!ShapeUtil::Compatible(reduce->shape(), inferred_return_shape)) {
    TF_ASSIGN_OR_RETURN(evaluated_[reduce],
                        evaluated_[reduce].ConvertToShape(reduce->shape()));
  }
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleMap(const HloInstruction* map) {
  auto operands = map->operands();
  const HloComputation* computation = map->to_apply();

  Literal result(map->shape());

  HloEvaluator embedded_evaluator(max_loop_iterations_);
  TF_RETURN_IF_ERROR(
      Apply<PopulateImpl>(result, [&](absl::Span<const int64_t> multi_index) {
        std::vector<Literal> arg_literals;
        arg_literals.reserve(operands.size());

        // Construct scalar literal parameters to be passed to the map
        // computation.
        for (auto operand : operands) {
          const Literal& arg_literal = GetEvaluatedLiteralFor(operand);
          arg_literals.push_back(
              LiteralUtil::GetScalarLiteral(arg_literal, multi_index));
        }

        Literal computed_result =
            embedded_evaluator.Evaluate(*computation, arg_literals).value();
        // Clear visit states so that the we can use the evaluate again on
        // the same computation.
        embedded_evaluator.ResetVisitStates();

        return computed_result;
      }));
  evaluated_[map] = std::move(result);
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleCustomCall(const HloInstruction* custom_call) {
  if (!custom_call_handler_) {
    // No handler is registered; this means custom-calls are not allowed.
    return DefaultAction(custom_call);
  }

  // Evaluate input operands so the handler has access to the operand data.
  std::vector<const Literal*> operands;
  operands.reserve(custom_call->operand_count());
  for (const HloInstruction* operand : custom_call->operands()) {
    operands.push_back(&GetEvaluatedLiteralFor(operand));
  }

  // Synchronously issue the handler to populate the instruction output literal.
  TF_ASSIGN_OR_RETURN(
      auto output, custom_call_handler_(custom_call, absl::MakeSpan(operands)));

  evaluated_[custom_call] = std::move(output);
  return absl::OkStatus();
}

absl::Status HloEvaluator::Preprocess(const HloInstruction* hlo) {
  VLOG(3) << "About to visit HLO: " << hlo->ToString();
  if (!enable_partial_evaluation_) {
    for (const HloInstruction* operand : hlo->operands()) {
      if (!IsAlreadyEvaluated(operand) ||
          !GetEvaluatedLiteralFor(operand).IsKnown()) {
        return absl::FailedPreconditionError(
            "Failed to evaluate instruction since its operands are unknown "
            "or undetermined and partial evaluation is not enabled.");
      }
    }
  }
  return ShapeUtil::ValidateShapeWithOptionalLayout(hlo->shape());
}

absl::Status HloEvaluator::Postprocess(const HloInstruction* hlo) {
  VLOG(3) << "Finished visiting " << hlo->ToString()
          << "; evaluated value is: " << GetEvaluatedLiteralFor(hlo).ToString();
  // Out of convenience the literal may have been produced with a different
  // layout. Relayout as indicated by the HLO instruction.
  auto evaluated_shape = GetEvaluatedLiteralFor(hlo).shape();
  Shape hlo_shape = hlo->shape();
  if (hlo_shape.IsArray() && !hlo_shape.has_layout()) {
    *hlo_shape.mutable_layout() =
        LayoutUtil::GetDefaultLayoutForShape(hlo_shape);
  }
  if (evaluated_shape.has_layout() && hlo_shape.has_layout() &&
      !Layout::Equal().MinorToMajorOnly()(evaluated_shape.layout(),
                                          hlo_shape.layout())) {
    evaluated_.at(hlo) = evaluated_.at(hlo).Relayout(hlo_shape);
  }
  return absl::OkStatus();
}

}  // namespace zkx
