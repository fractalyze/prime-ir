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

#ifndef ZKX_BACKENDS_CPU_CODEGEN_CPU_KERNEL_EMITTER_H_
#define ZKX_BACKENDS_CPU_CODEGEN_CPU_KERNEL_EMITTER_H_

#include <cstdint>
#include <optional>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/MLIRContext.h"

#include "zkx/codegen/emitter_loc_op_builder.h"
#include "zkx/codegen/kernel_emitter.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/service/buffer_assignment.h"

namespace zkx::cpu {

class CpuKernelEmitter final : public KernelEmitter {
 public:
  struct PassFlag {
    bool enable_one_shot_bufferize = false;
    bool enable_buffer_results_to_out_params = true;
    bool enable_poly_to_field = false;
    bool enable_tensor_ext_to_tensor = false;
    bool enable_elliptic_curve_to_field = false;
    bool enable_field_to_arith = false;
    bool enable_elliptic_curve_to_llvm = false;
    bool enable_ext_field_to_llvm = false;
    bool enable_lower_affine = false;
    bool enable_elementwise_to_linalg = false;
    bool enable_tensor_to_linalg = false;
    bool enable_linalg_generalize_named_ops = false;
    bool enable_linalg_to_parallel_loops = false;
    bool enable_scf_to_cf = false;
    bool enable_expand_strided_metadata = false;
    bool enable_finalize_memref_to_llvm = false;
  };

  // Multi-dimensional partitioning info from backend_config.
  // When the outermost dimension alone cannot satisfy the target parallelism,
  // additional dimensions are partitioned (e.g., shape [4,64,1024] with
  // partitions=[4,4] → dim0×dim1 = 16-way parallelism).
  struct ParallelPartition {
    int64_t total_partitions;  // Product of all per-dim partition counts.
    // Per-dimension info, outer-to-inner order:
    llvm::SmallVector<int64_t> dim_indices;     // Logical dim indices.
    llvm::SmallVector<int64_t> dim_partitions;  // Per-dim partition counts.
    llvm::SmallVector<int64_t> dim_full_sizes;  // Original dim sizes.
    llvm::SmallVector<int64_t> dim_min_sizes;   // Min sizes across operands.
  };

  CpuKernelEmitter(mlir::MLIRContext* context, const HloInstruction* instr,
                   const BufferAssignment* buffer_assignment);

  absl::StatusOr<KernelDefinition> EmitKernelDefinition() override;

  // TODO(chokobole): Since CPU code generation is handled in CpuKernelEmitter,
  // EmitComparator() is currently implemented here for code reuse. However,
  // a comparator is not a kernel, so this logic should be refactored and moved
  // to a more appropriate location.
  absl::StatusOr<std::unique_ptr<KernelSource>> EmitComparator(
      const HloComputation* comparator);

 private:
  absl::StatusOr<KernelDefinition> EmitKernelDefinitionImpl(
      std::string_view kernel_name);

  absl::StatusOr<llvm::SmallVector<mlir::Type>> MakeFuncArguments() const;
  absl::StatusOr<llvm::SmallVector<mlir::Type>> MakeFuncReturnTypes() const;

  absl::StatusOr<absl::flat_hash_map<const HloInstruction*, mlir::Value>>
  EmitOperands(EmitterLocOpBuilder& b, mlir::Block* entry_block) const;

  absl::Status EmitEpilog(EmitterLocOpBuilder& b, mlir::Block* entry_block,
                          mlir::MemRefType ret_type, mlir::Value result) const;

  absl::StatusOr<mlir::Value> EmitOp(
      const HloInstruction* instr, EmitterLocOpBuilder& b,
      absl::flat_hash_map<const HloInstruction*, mlir::Value>& values);

  void EmitOpInToApply(
      EmitterLocOpBuilder& b,
      absl::flat_hash_map<const HloInstruction*, mlir::Value>& values,
      const HloInstruction* instr);

  absl::StatusOr<mlir::Value> EmitConstantOp(const HloInstruction* instr,
                                             EmitterLocOpBuilder& b);

  absl::StatusOr<mlir::Value> EmitFftOp(
      const HloInstruction* instr, EmitterLocOpBuilder& b, mlir::Value value,
      mlir::Value twiddle_factor = mlir::Value());

  absl::StatusOr<mlir::Value> EmitFusionOp(const HloInstruction* instr,
                                           EmitterLocOpBuilder& b,
                                           mlir::ValueRange operands);

  absl::StatusOr<mlir::Value> EmitMsmOp(const HloInstruction* instr,
                                        EmitterLocOpBuilder& b,
                                        mlir::Value scalars, mlir::Value bases);

  absl::StatusOr<mlir::Value> EmitPairingCheckOp(const HloInstruction* instr,
                                                 EmitterLocOpBuilder& b,
                                                 mlir::Value g1_points,
                                                 mlir::Value g2_points);

  absl::StatusOr<mlir::Value> EmitBroadcastOp(
      const HloInstruction* instr, EmitterLocOpBuilder& b, mlir::Value input,
      absl::Span<const int64_t> source_dimensions);

  absl::StatusOr<mlir::Value> EmitConcatenateOp(const HloInstruction* instr,
                                                EmitterLocOpBuilder& b,
                                                mlir::ValueRange inputs);

  absl::StatusOr<mlir::Value> EmitDotOp(const HloInstruction* instr,
                                        EmitterLocOpBuilder& b, mlir::Value lhs,
                                        mlir::Value rhs);

  absl::StatusOr<mlir::Value> EmitDynamicSliceOp(
      const HloInstruction* instr, EmitterLocOpBuilder& b, mlir::Value input,
      mlir::ValueRange start_indices);

  absl::StatusOr<mlir::Value> EmitGatherOp(const HloInstruction* instr,
                                           EmitterLocOpBuilder& b,
                                           mlir::Value operand,
                                           mlir::Value start_indices);

  llvm::SmallVector<mlir::Value> LookUpGatherStartValues(
      EmitterLocOpBuilder& b, mlir::Value start_indices_tensor,
      llvm::ArrayRef<mlir::Value> batch_idx, int64_t index_vector_dim,
      const Shape& indices_shape,
      absl::Span<const int64_t> start_index_map) const;

  llvm::SmallVector<mlir::Value> ComputeGatherOperandIndices(
      EmitterLocOpBuilder& b, mlir::ValueRange output_indices,
      llvm::ArrayRef<mlir::Value> start_vals, const Shape& operand_shape,
      absl::Span<const int64_t> offset_dims,
      absl::Span<const int64_t> slice_sizes,
      absl::Span<const int64_t> start_index_map,
      const absl::flat_hash_set<int64_t>& collapsed_set) const;

  absl::StatusOr<mlir::Value> EmitDynamicUpdateSliceOp(
      const HloInstruction* instr, EmitterLocOpBuilder& b, mlir::Value dest,
      mlir::Value update, mlir::ValueRange start_indices);

  absl::StatusOr<mlir::Value> EmitIotaOp(const HloInstruction* instr,
                                         EmitterLocOpBuilder& b);

  absl::StatusOr<mlir::Value> EmitMapOp(const HloInstruction* instr,
                                        EmitterLocOpBuilder& b,
                                        mlir::ValueRange inputs);

  absl::StatusOr<mlir::Value> EmitPadOp(const HloInstruction* instr,
                                        EmitterLocOpBuilder& b,
                                        mlir::Value input,
                                        mlir::Value padding_value);

  absl::StatusOr<mlir::Value> EmitReduceOp(const HloInstruction* instr,
                                           EmitterLocOpBuilder& b,
                                           mlir::ValueRange inputs,
                                           mlir::ValueRange inits);

  absl::StatusOr<mlir::Value> EmitReduceWindowOp(const HloInstruction* instr,
                                                 EmitterLocOpBuilder& b,
                                                 mlir::ValueRange inputs,
                                                 mlir::ValueRange inits);

  absl::StatusOr<mlir::Value> EmitReshapeOp(const HloInstruction* instr,
                                            EmitterLocOpBuilder& b,
                                            mlir::Value value);

  absl::StatusOr<mlir::Value> EmitBitReverseOp(const HloInstruction* instr,
                                               EmitterLocOpBuilder& b,
                                               mlir::Value value);

  absl::StatusOr<mlir::Value> EmitReverseOp(const HloInstruction* instr,
                                            EmitterLocOpBuilder& b,
                                            mlir::Value value);

  absl::StatusOr<mlir::Value> EmitSliceOp(const HloInstruction* instr,
                                          EmitterLocOpBuilder& b,
                                          mlir::Value value);

  absl::StatusOr<mlir::Value> EmitTransposeOp(const HloInstruction* instr,
                                              EmitterLocOpBuilder& b,
                                              mlir::Value value);

  absl::StatusOr<mlir::Value> EmitIntegerConstantOp(const HloInstruction* instr,
                                                    EmitterLocOpBuilder& b);

  absl::StatusOr<mlir::Value> EmitFieldConstantOp(const HloInstruction* instr,
                                                  EmitterLocOpBuilder& b);

  // Helper to create a zero-initialized tensor for reduction operations
  mlir::Value CreateZeroInitializedTensor(EmitterLocOpBuilder& b,
                                          mlir::RankedTensorType result_type,
                                          bool is_field);

  // Dense matrix-vector multiplication using linalg::MatvecOp
  absl::StatusOr<mlir::Value> EmitDenseMatvecOp(const HloInstruction* instr,
                                                EmitterLocOpBuilder& b,
                                                mlir::Value lhs,
                                                mlir::Value rhs);

  // Dense vector-vector dot product using linalg::DotOp
  absl::StatusOr<mlir::Value> EmitDenseVecVecDotOp(const HloInstruction* instr,
                                                   EmitterLocOpBuilder& b,
                                                   mlir::Value lhs,
                                                   mlir::Value rhs);

  // Dense matrix-matrix multiplication using linalg::MatmulOp
  absl::StatusOr<mlir::Value> EmitDenseMatmulOp(const HloInstruction* instr,
                                                EmitterLocOpBuilder& b,
                                                mlir::Value lhs,
                                                mlir::Value rhs);

  // Enables lowering passes for ZK types (field, EC point) in pass_flag_.
  void EnableZkTypePasses(PrimitiveType element_type);

  // Returns a shape with all partitioned dimensions reduced to chunk size
  // when parallel partitioning is active, or the original shape otherwise.
  Shape GetChunkShape(const Shape& shape) const;

  // Get the scaled shape for a fusion body instruction.
  // All partitioned dimensions with size >= min_dim_size are scaled down by
  // their respective partition counts. Smaller shapes (broadcast sources)
  // are returned unchanged.
  Shape GetScaledFusionShape(const Shape& original_shape) const;

  mlir::MLIRContext* const mlir_context_;
  const HloInstruction* const instr_;

  const BufferAssignment* const buffer_assignment_;

  // Set when the instruction has parallel partitioning info from
  // ParallelTaskAssigner and the kernel can be parallelized.
  std::optional<ParallelPartition> partition_;

  mutable PassFlag pass_flag_;
};

}  // namespace zkx::cpu

#endif  // ZKX_BACKENDS_CPU_CODEGEN_CPU_KERNEL_EMITTER_H_
