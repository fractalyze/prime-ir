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

#ifndef ZKX_BACKENDS_GPU_CODEGEN_EMITTERS_POSEIDON2_H_
#define ZKX_BACKENDS_GPU_CODEGEN_EMITTERS_POSEIDON2_H_

#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"

#include "zkx/backends/gpu/codegen/emitters/emitter_base.h"
#include "zkx/codegen/emitters/computation_partitioner.h"
#include "zkx/hlo/analysis/indexing_map.h"
#include "zkx/hlo/ir/hlo_instructions.h"
#include "zkx/service/gpu/hlo_fusion_analysis.h"
#include "zkx/service/gpu/launch_dimensions.h"

namespace zkx::gpu {

// Extracted Poseidon2 configuration from the fusion name.
struct Poseidon2Config {
  int64_t width;            // State width (e.g., 16, 24)
  int num_external_rounds;  // External rounds per phase (typically 4)
  int num_internal_rounds;  // Internal rounds (13, 20, 21, or 23)
  int sbox_degree;          // S-box exponent (3 or 7)
};

// Parses Poseidon2Config from a fusion config name string.
// Format: "poseidon2:WIDTH:EXT:INT:SBOX"
std::optional<Poseidon2Config> ParsePoseidon2Config(std::string_view name);

// ─── Scalar Poseidon2 helpers (1-thread-per-hash, no shuffles) ───────────────
// Exported for reuse by merkle_compress and future merkle tree fusions.

// ARC (Add Round Constant): state += rc.
mlir::Value EmitArc(mlir::ImplicitLocOpBuilder& b, mlir::Value state,
                    mlir::Value rc);

// S-box: x^(sbox_degree).
mlir::Value EmitSbox(mlir::ImplicitLocOpBuilder& b, mlir::Value x,
                     int sbox_degree);

// Scalar external diffusion: M4 = circ(2,3,1,1) within groups of 4, then
// column sums across groups. ~48 field::AddOp, 0 shuffles, 0 multiplies.
void EmitScalarExtDiffusion(mlir::ImplicitLocOpBuilder& b,
                            std::vector<mlir::Value>& state);

// Scalar internal diffusion: state[i] = diag[i] * state[i] + sum(state).
void EmitScalarIntDiffusion(mlir::ImplicitLocOpBuilder& b,
                            std::vector<mlir::Value>& state,
                            const std::vector<mlir::Value>& diag);

// Full scalar Poseidon2 permutation on state[WIDTH].
// Uses scf.for loops for rounds to minimize PTX code size.
void EmitScalarPoseidon2(mlir::ImplicitLocOpBuilder& b,
                         std::vector<mlir::Value>& state,
                         mlir::Value ext_init_rc, mlir::Value int_rc,
                         mlir::Value ext_term_rc,
                         const std::vector<mlir::Value>& diag, int ext_rounds,
                         int int_rounds, int sbox_degree);

// Emit diagonal values, preferring compile-time field::ConstantOp from the
// literal. Falls back to tensor loads inside command buffers (kParameter).
//
// Unlike round constants (loaded from tensors at runtime via scf.for),
// diagonal values are inlined as constants when possible because:
// 1. They are loop-invariant — same values in every internal round.
// 2. prime-ir canonicalize can strength-reduce power-of-2 multiplies
//    (e.g., diag=2 → add(x,x), diag=4 → shift), eliminating field::MulOp.
// 3. Only WIDTH values (16), not hundreds like round constants.
std::vector<mlir::Value> EmitDiagConstants(mlir::ImplicitLocOpBuilder& b,
                                           mlir::func::FuncOp entry_function,
                                           const HloFusionInstruction& fusion,
                                           int64_t diag_operand_idx,
                                           int64_t width);

// Custom fusion emitter for the Poseidon2 permutation.
//
// Emits a scalar GPU kernel where 1 thread computes the full permutation
// with all state elements in registers. No warp shuffles.
//
// Constants (round constants, diagonal) are read from fusion operands at
// emission time and emitted as immediate field constants in registers.
//
// Launch dimensions: (1, 1) — single thread for a single permutation.
class Poseidon2Fusion : public EmitterBase {
 public:
  explicit Poseidon2Fusion(const HloFusionAnalysis& analysis);

  LaunchDimensions launch_dimensions() const override;

  std::optional<IndexingMap> ComputeThreadIdToOutputIndexing(
      int64_t root_index, mlir::MLIRContext* ctx) const override;

  std::optional<IndexingMap> ComputeThreadIdToInputIndexing(
      int64_t root_index, int64_t hero_operand_index,
      mlir::MLIRContext* ctx) const override;

 protected:
  absl::Status EmitEntryFunction(
      const emitters::PartitionedComputations& computations,
      const emitters::CallTargetProvider& call_targets,
      mlir::func::FuncOp entry_function,
      const HloFusionInstruction& fusion) const override;

 private:
  const HloFusionAnalysis& analysis_;
  Poseidon2Config config_;
};

}  // namespace zkx::gpu

#endif  // ZKX_BACKENDS_GPU_CODEGEN_EMITTERS_POSEIDON2_H_
