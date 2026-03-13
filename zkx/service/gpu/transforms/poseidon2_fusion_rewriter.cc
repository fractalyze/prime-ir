/* Copyright 2026 The ZKX Authors.

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

#include "zkx/service/gpu/transforms/poseidon2_fusion_rewriter.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"

#include "xla/tsl/platform/statusor.h"
#include "zkx/hlo/evaluator/hlo_evaluator.h"
#include "zkx/hlo/ir/hlo_computation.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/ir/hlo_instructions.h"
#include "zkx/hlo/ir/hlo_opcode.h"
#include "zkx/literal.h"
#include "zkx/literal_util.h"
#include "zkx/primitive_util.h"
#include "zkx/service/gpu/backend_configs.pb.h"
#include "zkx/service/poseidon2_reference.h"

namespace zkx::gpu {
namespace {

// Poseidon2 while-loop pattern: 3 consecutive kWhile instructions representing
// external-init, internal, and external-terminal rounds.
struct Poseidon2WhilePattern {
  HloInstruction* ext_init_while;
  HloInstruction* internal_while;
  HloInstruction* ext_term_while;
  // Pre-while instructions (initial MDS) to include in the fusion.
  HloInstruction* state_param;
  // Extracted configuration.
  int64_t width;
  int ext_rounds;
  int int_rounds;
  int sbox_degree;
  PrimitiveType field_type;
};

// Extracts the trip count from a while instruction's condition computation.
// Looks for: compare(get-tuple-element(param, 0), constant(N)), direction=LT.
// Returns the constant N, or -1 if the pattern doesn't match.
int ExtractTripCount(const HloInstruction* while_instr) {
  const HloComputation* cond = while_instr->while_condition();
  const HloInstruction* root = cond->root_instruction();

  if (root->opcode() != HloOpcode::kCompare) return -1;
  if (root->comparison_direction() != ComparisonDirection::kLt) return -1;

  // One operand should be get-tuple-element(param, 0), other should be
  // constant.
  const HloInstruction* lhs = root->operand(0);
  const HloInstruction* rhs = root->operand(1);

  const HloInstruction* limit_op = nullptr;

  if (lhs->opcode() == HloOpcode::kGetTupleElement &&
      rhs->opcode() == HloOpcode::kConstant) {
    limit_op = rhs;
  } else if (rhs->opcode() == HloOpcode::kGetTupleElement &&
             lhs->opcode() == HloOpcode::kConstant) {
    limit_op = lhs;
  } else {
    return -1;
  }

  if (!limit_op->literal().shape().IsInteger()) return -1;
  if (limit_op->literal().shape().rank() != 0) return -1;

  return limit_op->literal().Get<int32_t>({});
}

// Returns true if the instruction is a kWhile operating on field-typed state.
// The while tuple should contain at least (s32[], field_type[width]).
bool IsFieldWhile(const HloInstruction* instr, int64_t* width,
                  PrimitiveType* field_type) {
  if (instr->opcode() != HloOpcode::kWhile) return false;

  const Shape& shape = instr->shape();
  if (!shape.IsTuple()) return false;

  // Look for a field-typed array element in the tuple.
  for (int i = 0; i < shape.tuple_shapes_size(); ++i) {
    const Shape& elem = shape.tuple_shapes(i);
    if (elem.rank() == 1 && primitive_util::IsFieldType(elem.element_type())) {
      *width = elem.dimensions(0);
      *field_type = elem.element_type();
      return true;
    }
  }
  return false;
}

// Checks if while_b's state input comes from while_a's state output.
// Pattern: get-tuple-element(while_a, state_idx) → tuple → while_b.
bool WhilesAreChained(const HloInstruction* while_a,
                      const HloInstruction* while_b) {
  // while_b's init is a tuple. One of its elements should be
  // get-tuple-element(while_a, ...).
  const HloInstruction* init_b = while_b->operand(0);
  if (init_b->opcode() != HloOpcode::kTuple) return false;

  for (int i = 0; i < init_b->operand_count(); ++i) {
    const HloInstruction* elem = init_b->operand(i);
    // Handle copy(get-tuple-element(...)) pattern from XLA.
    if (elem->opcode() == HloOpcode::kCopy) {
      elem = elem->operand(0);
    }
    if (elem->opcode() == HloOpcode::kGetTupleElement &&
        elem->operand(0) == while_a) {
      return true;
    }
  }
  return false;
}

// Finds the Poseidon2 while-loop pattern in a computation.
std::optional<Poseidon2WhilePattern> FindPoseidon2WhilePattern(
    HloComputation* computation) {
  // Collect all kWhile instructions operating on field-typed state.
  struct WhileInfo {
    HloInstruction* instr;
    int64_t width;
    PrimitiveType field_type;
    int trip_count;
  };
  std::vector<WhileInfo> field_whiles;

  for (HloInstruction* instr : computation->MakeInstructionPostOrder()) {
    int64_t width;
    PrimitiveType field_type;
    if (IsFieldWhile(instr, &width, &field_type)) {
      int trip_count = ExtractTripCount(instr);
      if (trip_count > 0) {
        field_whiles.push_back({instr, width, field_type, trip_count});
      }
    }
  }

  if (field_whiles.size() < 3) return std::nullopt;

  // Look for 3 consecutive whiles forming ext→int→ext pattern:
  // - while[i] and while[i+2] have the same trip count (external rounds)
  // - while[i+1] has a different trip count (internal rounds)
  // - All 3 have the same width and field type
  // - They form a data dependency chain
  for (size_t i = 0; i + 2 < field_whiles.size(); ++i) {
    const auto& w0 = field_whiles[i];
    const auto& w1 = field_whiles[i + 1];
    const auto& w2 = field_whiles[i + 2];

    if (w0.width != w1.width || w1.width != w2.width) continue;
    if (w0.field_type != w1.field_type || w1.field_type != w2.field_type)
      continue;
    if (w0.trip_count != w2.trip_count) continue;
    if (w0.trip_count == w1.trip_count) continue;  // ext != int rounds

    // Verify data dependency chain.
    if (!WhilesAreChained(w0.instr, w1.instr)) continue;
    if (!WhilesAreChained(w1.instr, w2.instr)) continue;

    // Find the state parameter that feeds into the first while.
    // Walk backward through the dependency graph using BFS to handle
    // arbitrary instruction chains (copies, reshapes, MDS ops, etc.).
    HloInstruction* state_param = nullptr;
    {
      absl::flat_hash_set<const HloInstruction*> visited;
      std::vector<const HloInstruction*> worklist;
      worklist.push_back(w0.instr->operand(0));
      while (!worklist.empty() && !state_param) {
        const HloInstruction* cur = worklist.back();
        worklist.pop_back();
        if (!visited.insert(cur).second) continue;
        if (cur->opcode() == HloOpcode::kParameter) {
          state_param = const_cast<HloInstruction*>(cur);
          break;
        }
        if (cur->opcode() == HloOpcode::kConstant) continue;
        for (const HloInstruction* op : cur->operands()) {
          worklist.push_back(op);
        }
      }
    }

    Poseidon2WhilePattern pattern;
    pattern.ext_init_while = w0.instr;
    pattern.internal_while = w1.instr;
    pattern.ext_term_while = w2.instr;
    pattern.state_param = state_param;
    pattern.width = w0.width;
    pattern.ext_rounds = w0.trip_count;
    pattern.int_rounds = w1.trip_count;
    pattern.sbox_degree = 0;  // Inferred during verification.
    pattern.field_type = w0.field_type;

    VLOG(1) << "Poseidon2FusionRewriter: found while pattern"
            << " width=" << pattern.width
            << " ext_rounds=" << pattern.ext_rounds
            << " int_rounds=" << pattern.int_rounds << " field_type="
            << primitive_util::LowercasePrimitiveTypeName(pattern.field_type);

    return pattern;
  }

  return std::nullopt;
}

// Finds a kConstant instruction in a computation matching the given shape.
const HloInstruction* FindConstantByShape(const HloComputation* computation,
                                          const Shape& target_shape) {
  for (const HloInstruction* instr : computation->instructions()) {
    if (instr->opcode() == HloOpcode::kConstant &&
        ShapeUtil::Equal(instr->shape(), target_shape)) {
      return instr;
    }
  }
  return nullptr;
}

// Verifies the detected while-loop pattern against a reference Poseidon2
// implementation. Temporarily sets the computation root to the Poseidon2
// output, evaluates with HloEvaluator on a test input, and compares against
// the C++ reference. Eliminates false positives from structural matching.
bool VerifyPoseidon2Pattern(HloComputation* computation,
                            const Poseidon2WhilePattern& pattern,
                            const HloInstruction* original_root,
                            const Literal& ext_init_rc_lit,
                            const Literal& int_rc_lit,
                            const Literal& ext_term_rc_lit,
                            const Literal& diag_lit) {
  // Temporarily rewire the computation root to the Poseidon2 output so
  // HloEvaluator returns it. Restore after verification.
  HloInstruction* saved_root = computation->root_instruction();
  bool root_changed = false;
  if (saved_root != original_root) {
    computation->set_root_instruction(
        const_cast<HloInstruction*>(original_root));
    root_changed = true;
  }

  auto verify_typed = [&](auto type_tag) -> bool {
    using NativeT = typename decltype(type_tag)::value_type;

    std::vector<NativeT> test_values;
    test_values.reserve(pattern.width);
    for (int i = 0; i < pattern.width; ++i) {
      test_values.push_back(NativeT(i + 1));
    }
    Literal test_input = LiteralUtil::CreateR1<NativeT>(test_values);

    // Compute reference result.
    Literal ref_result = ComputeReferencePoseidon2<NativeT>(
        test_input, ext_init_rc_lit, int_rc_lit, ext_term_rc_lit, diag_lit,
        pattern.width, pattern.ext_rounds, pattern.int_rounds,
        pattern.sbox_degree);

    // Evaluate the computation with HloEvaluator.
    int max_iters =
        pattern.ext_rounds + pattern.int_rounds + pattern.ext_rounds + 10;
    HloEvaluator evaluator(max_iters);
    const Literal* input_ptrs[] = {&test_input};
    auto eval_result = evaluator.Evaluate(*computation, input_ptrs);
    if (!eval_result.ok()) {
      VLOG(1) << "Poseidon2FusionRewriter: HloEvaluator failed: "
              << eval_result.status();
      return false;
    }

    // Extract the field state from the eval result. If the root is a while
    // (tuple output), find the field-typed element via ShapeIndex.
    ShapeIndex state_index;
    if (eval_result->shape().IsTuple()) {
      for (int idx = 0; idx < eval_result->shape().tuple_shapes_size(); ++idx) {
        const Shape& elem = eval_result->shape().tuple_shapes(idx);
        if (elem.rank() == 1 && elem.element_type() == pattern.field_type) {
          state_index = {idx};
          break;
        }
      }
    }

    // Compare results element by element.
    for (int i = 0; i < pattern.width; ++i) {
      if (ref_result.Get<NativeT>({i}) !=
          eval_result->Get<NativeT>({i}, state_index)) {
        VLOG(1) << "Poseidon2FusionRewriter: verification mismatch at index "
                << i;
        return false;
      }
    }

    VLOG(1) << "Poseidon2FusionRewriter: verification passed";
    return true;
  };

  bool verified = primitive_util::FieldTypeSwitch<bool>(
      [&](auto primitive_type_constant) -> bool {
        if constexpr (primitive_util::IsFieldType(primitive_type_constant())) {
          using NativeT =
              primitive_util::NativeTypeOf<primitive_type_constant()>;
          struct Tag {
            using value_type = NativeT;
          };
          return verify_typed(Tag{});
        }
        return false;
      },
      pattern.field_type);

  // Restore original root.
  if (root_changed) {
    computation->set_root_instruction(saved_root);
  }

  return verified;
}

// Creates the Poseidon2 custom fusion.
absl::StatusOr<bool> CreatePoseidon2Fusion(
    HloModule* module, HloComputation* computation,
    const Poseidon2WhilePattern& pattern) {
  if (!pattern.state_param) return false;

  // Find the output: get-tuple-element from the last while, extracting the
  // state.
  HloInstruction* original_root = nullptr;
  for (HloInstruction* user : pattern.ext_term_while->users()) {
    if (user->opcode() == HloOpcode::kGetTupleElement) {
      const Shape& elem_shape = user->shape();
      if (elem_shape.rank() == 1 &&
          elem_shape.element_type() == pattern.field_type) {
        original_root = user;
        break;
      }
    }
  }
  if (!original_root) {
    // The while output might be the computation root directly.
    if (computation->root_instruction() == pattern.ext_term_while) {
      original_root = pattern.ext_term_while;
    } else {
      return false;
    }
  }

  Shape output_shape =
      ShapeUtil::MakeShape(pattern.field_type, {pattern.width});

  // Extract constants from while bodies for the fusion operands.
  // Look for kConstant instructions in each while body.
  Shape ext_rc_shape = ShapeUtil::MakeShape(
      pattern.field_type, {pattern.ext_rounds, pattern.width});
  Shape int_rc_shape =
      ShapeUtil::MakeShape(pattern.field_type, {pattern.int_rounds});
  Shape diag_shape = ShapeUtil::MakeShape(pattern.field_type, {pattern.width});

  // Find constants in while bodies.
  const HloInstruction* ext_init_rc_const =
      FindConstantByShape(pattern.ext_init_while->while_body(), ext_rc_shape);
  const HloInstruction* int_rc_const =
      FindConstantByShape(pattern.internal_while->while_body(), int_rc_shape);
  const HloInstruction* ext_term_rc_const =
      FindConstantByShape(pattern.ext_term_while->while_body(), ext_rc_shape);
  const HloInstruction* diag_const =
      FindConstantByShape(pattern.internal_while->while_body(), diag_shape);

  if (!ext_init_rc_const || !int_rc_const || !ext_term_rc_const ||
      !diag_const) {
    VLOG(1) << "Poseidon2FusionRewriter: could not extract all constants"
            << " ext_init_rc=" << (ext_init_rc_const != nullptr)
            << " int_rc=" << (int_rc_const != nullptr)
            << " ext_term_rc=" << (ext_term_rc_const != nullptr)
            << " diag=" << (diag_const != nullptr);
    return false;
  }

  // Infer S-box degree by trying known Poseidon2 degrees (3, 5, 7) and
  // verifying against a reference implementation. JAX lowers integer_pow to
  // a multiply chain, so we cannot read the degree directly from HLO ops.
  Poseidon2WhilePattern verified_pattern = pattern;
  bool verified = false;
  for (int degree : {7, 3, 5}) {
    verified_pattern.sbox_degree = degree;
    if (VerifyPoseidon2Pattern(
            computation, verified_pattern, original_root,
            ext_init_rc_const->literal(), int_rc_const->literal(),
            ext_term_rc_const->literal(), diag_const->literal())) {
      VLOG(1) << "Poseidon2FusionRewriter: verified sbox_degree=" << degree;
      verified = true;
      break;
    }
  }
  if (!verified) {
    VLOG(1) << "Poseidon2FusionRewriter: verification failed for all known "
            << "sbox degrees, skipping fusion";
    return false;
  }

  // Create constant instructions in the entry computation for the fusion
  // operands.
  HloInstruction* ext_init_rc_op = computation->AddInstruction(
      HloInstruction::CreateConstant(ext_init_rc_const->literal().Clone()));
  HloInstruction* int_rc_op = computation->AddInstruction(
      HloInstruction::CreateConstant(int_rc_const->literal().Clone()));
  HloInstruction* ext_term_rc_op = computation->AddInstruction(
      HloInstruction::CreateConstant(ext_term_rc_const->literal().Clone()));
  HloInstruction* diag_op = computation->AddInstruction(
      HloInstruction::CreateConstant(diag_const->literal().Clone()));

  // Build fusion body with all 5 parameters referenced.
  //
  // The custom emitter (Poseidon2Fusion::EmitEntryFunction) replaces this body
  // entirely and uses all 5 parameters. However, HLO passes (DCE, buffer
  // analysis, scheduling) inspect the body and crash on unused parameters.
  // We chain all parameters into the root via dummy ops to keep them alive.
  //
  // TODO(chokobole): Consider teaching HLO passes to skip kCustom fusion
  // bodies, which would let us use a simple `ROOT = parameter(0)` placeholder
  // here.
  HloComputation::Builder builder("poseidon2_permutation");
  HloInstruction* p0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, output_shape, "state"));
  HloInstruction* p1 = builder.AddInstruction(HloInstruction::CreateParameter(
      1, ext_init_rc_op->shape(), "ext_init_rc"));
  HloInstruction* p2 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, int_rc_op->shape(), "int_rc"));
  HloInstruction* p3 = builder.AddInstruction(HloInstruction::CreateParameter(
      3, ext_term_rc_op->shape(), "ext_term_rc"));
  HloInstruction* p4 = builder.AddInstruction(
      HloInstruction::CreateParameter(4, diag_op->shape(), "diag"));

  // p4 (diag) has the same shape as p0 (state) → add directly.
  HloInstruction* acc = builder.AddInstruction(
      HloInstruction::CreateBinary(output_shape, HloOpcode::kAdd, p0, p4));

  // p1, p3 ([ext_rounds, width]) → slice first row → reshape to [width].
  auto slice_row = [&](HloInstruction* tensor_2d) -> HloInstruction* {
    Shape row_2d = ShapeUtil::MakeShape(pattern.field_type, {1, pattern.width});
    HloInstruction* sliced = builder.AddInstruction(
        HloInstruction::CreateSlice(row_2d, tensor_2d,
                                    /*start_indices=*/{0, 0},
                                    /*limit_indices=*/{1, pattern.width},
                                    /*strides=*/{1, 1}));
    return builder.AddInstruction(
        HloInstruction::CreateReshape(output_shape, sliced));
  };
  acc = builder.AddInstruction(HloInstruction::CreateBinary(
      output_shape, HloOpcode::kAdd, acc, slice_row(p1)));
  acc = builder.AddInstruction(HloInstruction::CreateBinary(
      output_shape, HloOpcode::kAdd, acc, slice_row(p3)));

  // p2 ([int_rounds]) → slice [0:1] → reshape to scalar → broadcast.
  Shape scalar_shape = ShapeUtil::MakeShape(pattern.field_type, {});
  Shape one_shape = ShapeUtil::MakeShape(pattern.field_type, {1});
  HloInstruction* p2_slice =
      builder.AddInstruction(HloInstruction::CreateSlice(one_shape, p2,
                                                         /*start_indices=*/{0},
                                                         /*limit_indices=*/{1},
                                                         /*strides=*/{1}));
  HloInstruction* p2_scalar = builder.AddInstruction(
      HloInstruction::CreateReshape(scalar_shape, p2_slice));
  HloInstruction* p2_bcast = builder.AddInstruction(
      HloInstruction::CreateBroadcast(output_shape, p2_scalar, {}));
  acc = builder.AddInstruction(HloInstruction::CreateBinary(
      output_shape, HloOpcode::kAdd, acc, p2_bcast));

  HloComputation* body =
      module->AddComputationAndUnifyNamesAndIds(builder.Build(acc), false);

  // Create the fusion instruction with 5 operands.
  HloInstruction* fusion =
      computation->AddInstruction(HloInstruction::CreateFusion(
          output_shape, HloInstruction::FusionKind::kCustom,
          {pattern.state_param, ext_init_rc_op, int_rc_op, ext_term_rc_op,
           diag_op},
          body));
  module->SetAndUniquifyInstrName(fusion, "poseidon2_permutation");

  // Set backend config with parameterized name.
  GpuBackendConfig gpu_config;
  FusionBackendConfig& backend_config =
      *gpu_config.mutable_fusion_backend_config();
  backend_config.set_kind("__custom_fusion");
  CustomFusionConfig config;
  // Encode configuration in the name: "poseidon2:WIDTH:EXT:INT:SBOX"
  config.set_name(absl::StrCat(
      "poseidon2:", verified_pattern.width, ":", verified_pattern.ext_rounds,
      ":", verified_pattern.int_rounds, ":", verified_pattern.sbox_degree));
  *backend_config.mutable_custom_fusion_config() = config;
  TF_RETURN_IF_ERROR(fusion->set_backend_config(std::move(gpu_config)));

  // Replace uses of the original root with the fusion output.
  TF_RETURN_IF_ERROR(computation->ReplaceInstruction(original_root, fusion));

  VLOG(1) << "Poseidon2FusionRewriter: created poseidon2 fusion"
          << " (width=" << verified_pattern.width
          << " ext=" << verified_pattern.ext_rounds
          << " int=" << verified_pattern.int_rounds
          << " sbox=" << verified_pattern.sbox_degree << ")";
  return true;
}

}  // namespace

absl::StatusOr<bool> Poseidon2FusionRewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<std::string_view>& execution_threads) {
  bool changed = false;

  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    auto pattern = FindPoseidon2WhilePattern(computation);
    if (!pattern.has_value()) continue;

    VLOG(2) << "Poseidon2FusionRewriter: found Poseidon2 pattern in "
            << computation->name();

    TF_ASSIGN_OR_RETURN(bool fused,
                        CreatePoseidon2Fusion(module, computation, *pattern));
    changed |= fused;
  }

  // Remove while body/condition computations that are no longer reachable
  // after replacing the 3 while loops with a single custom fusion. Without
  // this, later passes (priority-fusion, DCE) process the dead computations
  // and can crash during parameter removal.
  if (changed) {
    TF_RETURN_IF_ERROR(module->RemoveUnusedComputations());
  }

  return changed;
}

}  // namespace zkx::gpu
