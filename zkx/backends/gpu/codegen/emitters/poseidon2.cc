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

#include "zkx/backends/gpu/codegen/emitters/poseidon2.h"

#include <cstring>
#include <string>

#include "absl/log/check.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "llvm/ADT/APInt.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ValueRange.h"

#include "prime_ir/Dialect/Field/IR/FieldOps.h"

namespace zkx::gpu {

namespace field = mlir::prime_ir::field;

using mlir::ImplicitLocOpBuilder;
using mlir::Value;
using mlir::ValueRange;

// ─── Exported scalar helpers ─────────────────────────────────────────────────

// ARC (Add Round Constant): state += rc.
Value EmitArc(ImplicitLocOpBuilder& b, Value state, Value rc) {
  return b.create<field::AddOp>(state, rc);
}

// S-box: x^(sbox_degree). prime-ir's ConvertPowUI handles the decomposition
// into optimal square-and-multiply chains.
Value EmitSbox(ImplicitLocOpBuilder& b, Value x, int sbox_degree) {
  auto exp = b.create<mlir::arith::ConstantIntOp>(sbox_degree, 32);
  return b.create<field::PowUIOp>(x, exp);
}

// Scalar external diffusion: M4 = circ(2,3,1,1) within groups of 4, then
// column sums across 4 groups.
//
// SP1's externalLinearLayer logic, implemented with field::AddOp.
// ~48 field::AddOp, 0 shuffles, 0 multiplies.
void EmitScalarExtDiffusion(ImplicitLocOpBuilder& b,
                            std::vector<Value>& state) {
  const int width = state.size();
  CHECK_EQ(width % 4, 0);
  int num_groups = width / 4;

  // Phase 1: M4 = circ(2,3,1,1) within each group of 4.
  std::vector<Value> m4(width);
  for (int g = 0; g < num_groups; ++g) {
    int base = g * 4;
    Value t01 = b.create<field::AddOp>(state[base + 0], state[base + 1]);
    Value t23 = b.create<field::AddOp>(state[base + 2], state[base + 3]);
    Value t0123 = b.create<field::AddOp>(t01, t23);

    // M4 = circ(2,3,1,1): m4[k] = t0123 + s[k] + 2*s[(k+1)%4].
    // m4[0] = 2*s0 + 3*s1 + s2 + s3
    Value double_s1 = b.create<field::AddOp>(state[base + 1], state[base + 1]);
    m4[base + 0] = b.create<field::AddOp>(t0123, state[base + 0]);
    m4[base + 0] = b.create<field::AddOp>(m4[base + 0], double_s1);

    // m4[1] = s0 + 2*s1 + 3*s2 + s3
    Value double_s2 = b.create<field::AddOp>(state[base + 2], state[base + 2]);
    m4[base + 1] = b.create<field::AddOp>(t0123, state[base + 1]);
    m4[base + 1] = b.create<field::AddOp>(m4[base + 1], double_s2);

    // m4[2] = s0 + s1 + 2*s2 + 3*s3
    Value double_s3 = b.create<field::AddOp>(state[base + 3], state[base + 3]);
    m4[base + 2] = b.create<field::AddOp>(t0123, state[base + 2]);
    m4[base + 2] = b.create<field::AddOp>(m4[base + 2], double_s3);

    // m4[3] = 3*s0 + s1 + s2 + 2*s3
    Value double_s0 = b.create<field::AddOp>(state[base + 0], state[base + 0]);
    m4[base + 3] = b.create<field::AddOp>(t0123, state[base + 3]);
    m4[base + 3] = b.create<field::AddOp>(m4[base + 3], double_s0);
  }

  // Phase 2: column sums across groups.
  std::vector<Value> col(4);
  for (int k = 0; k < 4; ++k) {
    col[k] = m4[k];
    for (int g = 1; g < num_groups; ++g) {
      col[k] = b.create<field::AddOp>(col[k], m4[g * 4 + k]);
    }
  }

  // Phase 3: output = m4 + col.
  for (int i = 0; i < width; ++i) {
    state[i] = b.create<field::AddOp>(m4[i], col[i % 4]);
  }
}

// Scalar internal diffusion: state[i] = diag[i] * state[i] + sum(state).
void EmitScalarIntDiffusion(ImplicitLocOpBuilder& b, std::vector<Value>& state,
                            const std::vector<Value>& diag) {
  const int width = state.size();

  // Compute sum of all state elements.
  Value sum = state[0];
  for (int i = 1; i < width; ++i) {
    sum = b.create<field::AddOp>(sum, state[i]);
  }

  // state[i] = diag[i] * state[i] + sum.
  for (int i = 0; i < width; ++i) {
    Value diag_product = b.create<field::MulOp>(diag[i], state[i]);
    state[i] = b.create<field::AddOp>(diag_product, sum);
  }
}

std::vector<Value> EmitDiagConstants(ImplicitLocOpBuilder& b,
                                     mlir::func::FuncOp entry_function,
                                     const HloFusionInstruction& fusion,
                                     int64_t diag_operand_idx, int64_t width) {
  auto elem_type = mlir::cast<mlir::RankedTensorType>(
                       entry_function.getArgument(0).getType())
                       .getElementType();

  // Prefer inline field::ConstantOp from the literal (enables strength
  // reduction). Inside command buffers, the constant is wrapped as kParameter;
  // fall back to tensor loads.
  if (fusion.operand(diag_operand_idx)->opcode() == HloOpcode::kConstant) {
    const void* raw_data =
        fusion.operand(diag_operand_idx)->literal().untyped_data();
    auto pf_type = mlir::cast<field::PrimeFieldType>(elem_type);
    int bit_width = pf_type.getStorageType().getIntOrFloatBitWidth();
    int byte_width = bit_width / 8;
    const auto* raw = static_cast<const char*>(raw_data);

    std::vector<Value> diag(width);
    for (int64_t i = 0; i < width; ++i) {
      uint64_t val = 0;
      std::memcpy(&val, raw + i * byte_width, byte_width);
      diag[i] =
          field::createFieldConstant(pf_type, b, llvm::APInt(bit_width, val));
    }
    return diag;
  }

  // Fallback: load from tensor argument (inside command buffer).
  Value diag_tensor = entry_function.getArgument(diag_operand_idx);
  std::vector<Value> diag(width);
  for (int64_t i = 0; i < width; ++i) {
    auto idx = b.create<mlir::arith::ConstantIndexOp>(i);
    diag[i] = b.create<mlir::tensor::ExtractOp>(diag_tensor, ValueRange{idx});
  }
  return diag;
}

namespace {

// Emit Poseidon2 external rounds using scf.for to reduce PTX size.
// state[WIDTH] is updated in-place via loop-carried values.
void EmitExtRoundsLoop(ImplicitLocOpBuilder& b, std::vector<Value>& state,
                       Value rc_tensor, int num_rounds, int width,
                       int sbox_degree, int rc_base_offset) {
  auto lb_val = b.create<mlir::arith::ConstantIndexOp>(0);
  auto ub_val = b.create<mlir::arith::ConstantIndexOp>(num_rounds);
  auto step_val = b.create<mlir::arith::ConstantIndexOp>(1);
  auto width_idx = b.create<mlir::arith::ConstantIndexOp>(width);
  auto base_offset = b.create<mlir::arith::ConstantIndexOp>(rc_base_offset);

  llvm::SmallVector<Value> init(state.begin(), state.end());

  auto loop = b.create<mlir::scf::ForOp>(
      lb_val, ub_val, step_val, init,
      [&](mlir::OpBuilder& loop_b, mlir::Location loc, Value iv,
          ValueRange iter_args) {
        ImplicitLocOpBuilder lb(loc, loop_b);

        std::vector<Value> s(width);
        for (int i = 0; i < width; ++i) s[i] = iter_args[i];

        // Flat index base: (rc_base_offset + iv) * width
        Value round_idx = lb.create<mlir::arith::AddIOp>(base_offset, iv);
        Value flat_base = lb.create<mlir::arith::MulIOp>(round_idx, width_idx);

        // ARC + S-box for each element.
        for (int i = 0; i < width; ++i) {
          auto off = lb.create<mlir::arith::ConstantIndexOp>(i);
          Value rc_idx = lb.create<mlir::arith::AddIOp>(flat_base, off);
          Value rc =
              lb.create<mlir::tensor::ExtractOp>(rc_tensor, ValueRange{rc_idx});
          s[i] = EmitArc(lb, s[i], rc);
          s[i] = EmitSbox(lb, s[i], sbox_degree);
        }

        EmitScalarExtDiffusion(lb, s);

        llvm::SmallVector<Value> results(s.begin(), s.end());
        lb.create<mlir::scf::YieldOp>(results);
      });
  for (int i = 0; i < width; ++i) state[i] = loop.getResult(i);
}

// Emit Poseidon2 internal rounds using scf.for.
void EmitIntRoundsLoop(ImplicitLocOpBuilder& b, std::vector<Value>& state,
                       Value int_rc_tensor, int num_rounds,
                       const std::vector<Value>& diag, int sbox_degree) {
  const int width = state.size();

  auto lb_val = b.create<mlir::arith::ConstantIndexOp>(0);
  auto ub_val = b.create<mlir::arith::ConstantIndexOp>(num_rounds);
  auto step_val = b.create<mlir::arith::ConstantIndexOp>(1);

  llvm::SmallVector<Value> init(state.begin(), state.end());

  auto loop = b.create<mlir::scf::ForOp>(
      lb_val, ub_val, step_val, init,
      [&](mlir::OpBuilder& loop_b, mlir::Location loc, Value iv,
          ValueRange iter_args) {
        ImplicitLocOpBuilder lb(loc, loop_b);

        std::vector<Value> s(width);
        for (int i = 0; i < width; ++i) s[i] = iter_args[i];

        // ARC + S-box on state[0] only.
        Value rc =
            lb.create<mlir::tensor::ExtractOp>(int_rc_tensor, ValueRange{iv});
        s[0] = EmitArc(lb, s[0], rc);
        s[0] = EmitSbox(lb, s[0], sbox_degree);

        EmitScalarIntDiffusion(lb, s, diag);

        llvm::SmallVector<Value> results(s.begin(), s.end());
        lb.create<mlir::scf::YieldOp>(results);
      });
  for (int i = 0; i < width; ++i) state[i] = loop.getResult(i);
}

}  // namespace

// Full scalar Poseidon2 permutation on state[WIDTH].
// Uses scf.for loops for rounds to minimize PTX code size.
void EmitScalarPoseidon2(ImplicitLocOpBuilder& b, std::vector<Value>& state,
                         Value ext_init_rc_tensor, Value int_rc_tensor,
                         Value ext_term_rc_tensor,
                         const std::vector<Value>& diag, int ext_rounds,
                         int int_rounds, int sbox_degree) {
  const int width = state.size();

  // Initial external diffusion.
  EmitScalarExtDiffusion(b, state);

  // Initial external rounds (scf.for).
  EmitExtRoundsLoop(b, state, ext_init_rc_tensor, ext_rounds, width,
                    sbox_degree, /*rc_base_offset=*/0);

  // Internal rounds (scf.for).
  EmitIntRoundsLoop(b, state, int_rc_tensor, int_rounds, diag, sbox_degree);

  // Terminal external rounds (scf.for).
  EmitExtRoundsLoop(b, state, ext_term_rc_tensor, ext_rounds, width,
                    sbox_degree, /*rc_base_offset=*/0);
}

// ─── Config parsing ──────────────────────────────────────────────────────────

std::optional<Poseidon2Config> ParsePoseidon2Config(std::string_view name) {
  // Format: "poseidon2:WIDTH:EXT:INT:SBOX"
  std::vector<std::string_view> parts = absl::StrSplit(name, ':');
  if (parts.size() != 5 || parts[0] != "poseidon2") return std::nullopt;

  Poseidon2Config config;
  if (!absl::SimpleAtoi(parts[1], &config.width)) return std::nullopt;
  if (!absl::SimpleAtoi(parts[2], &config.num_external_rounds))
    return std::nullopt;
  if (!absl::SimpleAtoi(parts[3], &config.num_internal_rounds))
    return std::nullopt;
  if (!absl::SimpleAtoi(parts[4], &config.sbox_degree)) return std::nullopt;

  return config;
}

// ─── Poseidon2Fusion ─────────────────────────────────────────────────────────

Poseidon2Fusion::Poseidon2Fusion(const HloFusionAnalysis& analysis)
    : analysis_(analysis) {
  const auto& backend_config = analysis_.fusion_backend_config();
  const std::string& config_name = backend_config.custom_fusion_config().name();

  auto parsed = ParsePoseidon2Config(config_name);
  CHECK(parsed.has_value()) << "Invalid Poseidon2 config: " << config_name;
  config_ = *parsed;
}

LaunchDimensions Poseidon2Fusion::launch_dimensions() const {
  return LaunchDimensions(/*block_x_count=*/1,
                          /*thread_x_count_per_block=*/1);
}

std::optional<IndexingMap> Poseidon2Fusion::ComputeThreadIdToOutputIndexing(
    int64_t root_index, mlir::MLIRContext* ctx) const {
  return std::nullopt;
}

std::optional<IndexingMap> Poseidon2Fusion::ComputeThreadIdToInputIndexing(
    int64_t root_index, int64_t hero_operand_index,
    mlir::MLIRContext* ctx) const {
  return std::nullopt;
}

absl::Status Poseidon2Fusion::EmitEntryFunction(
    const emitters::PartitionedComputations& computations,
    const emitters::CallTargetProvider& call_targets,
    mlir::func::FuncOp entry_function,
    const HloFusionInstruction& fusion) const {
  ImplicitLocOpBuilder b(entry_function.getLoc(), entry_function);
  b.setInsertionPointToStart(entry_function.addEntryBlock());

  const int64_t width = config_.width;
  const int ext_rounds = config_.num_external_rounds;
  const int int_rounds = config_.num_internal_rounds;
  const int sbox_degree = config_.sbox_degree;

  // Argument layout: [state, ext_init_rc, int_rc, ext_term_rc, diag, output]
  Value state_tensor = entry_function.getArgument(0);
  Value ext_init_rc_tensor = entry_function.getArgument(1);
  Value int_rc_tensor = entry_function.getArgument(2);
  Value ext_term_rc_tensor = entry_function.getArgument(3);
  // diag_tensor (arg 4) unused — diagonal emitted as field::ConstantOp below.
  Value output_tensor = entry_function.getArgument(5);

  // Load all width elements into registers.
  std::vector<Value> state(width);
  for (int64_t i = 0; i < width; ++i) {
    auto idx = b.create<mlir::arith::ConstantIndexOp>(i);
    state[i] = b.create<mlir::tensor::ExtractOp>(state_tensor,
                                                 ValueRange{idx.getResult()});
  }

  // Emit diagonal as inline field::ConstantOp for strength reduction.
  std::vector<Value> diag = EmitDiagConstants(b, entry_function, fusion,
                                              /*diag_operand_idx=*/4, width);

  // Run scalar Poseidon2 permutation.
  EmitScalarPoseidon2(b, state, ext_init_rc_tensor, int_rc_tensor,
                      ext_term_rc_tensor, diag, ext_rounds, int_rounds,
                      sbox_degree);

  // Store results back.
  Value result = output_tensor;
  for (int64_t i = 0; i < width; ++i) {
    auto idx = b.create<mlir::arith::ConstantIndexOp>(i);
    result = b.create<mlir::tensor::InsertOp>(state[i], result,
                                              ValueRange{idx.getResult()});
  }

  b.create<mlir::func::ReturnOp>(ValueRange{result});

  return absl::OkStatus();
}

}  // namespace zkx::gpu
