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

#ifndef ZKX_SERVICE_ELEMENTAL_IR_EMITTER_H_
#define ZKX_SERVICE_ELEMENTAL_IR_EMITTER_H_

#include <string_view>

#include "absl/status/statusor.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"

#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/service/buffer_assignment.h"

namespace zkx {

class ElementalIrEmitter {
 public:
  struct PassFlag {
    bool enable_sparsification_and_bufferization = false;
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
    bool enable_linalg_to_parallel_loops = false;
    bool enable_scf_to_cf = false;
    bool enable_expand_strided_metadata = false;
    bool enable_finalize_memref_to_llvm = false;
#ifdef ZKX_HAS_OPENMP
    bool enable_omp = true;
#else
    bool enable_omp = false;
#endif
  };

  ElementalIrEmitter(mlir::ImplicitLocOpBuilder* b, mlir::MLIRContext* context,
                     const BufferAssignment* buffer_assignment)
      : b_(b), context_(context), buffer_assignment_(buffer_assignment) {}

  virtual ~ElementalIrEmitter() = default;

  mlir::ImplicitLocOpBuilder* builder() const { return b_; }
  mlir::MLIRContext* context() const { return context_; }

  // Returns which ops invalidate the cache of emitted instructions by creating
  // a new BasicBlock and setting the insertion point to the newly created
  // BasicBlock. We can only reuse cached values if they were emitted in the
  // same BasicBlock as the current BasicBlock.
  static bool OpInvalidatesCache(const HloInstruction* hlo);

 protected:
  virtual absl::StatusOr<llvm::SmallVector<mlir::Value>> EmitThreadLocalCall(
      const HloComputation& callee, mlir::ValueRange parameters,
      std::string_view name, bool is_reducer);

 private:
  absl::StatusOr<mlir::Value> EmitUnaryOp(const HloInstruction* op,
                                          mlir::Value operand_value);
  absl::StatusOr<mlir::Value> EmitIntegerUnaryOp(const HloInstruction* op,
                                                 mlir::Value operand_value,
                                                 bool is_signed);
  absl::StatusOr<mlir::Value> EmitFieldUnaryOp(const HloInstruction* op,
                                               mlir::Value operand_value);
  absl::StatusOr<mlir::Value> EmitEcPointUnaryOp(const HloInstruction* op,
                                                 mlir::Value operand_value);

  absl::StatusOr<mlir::Value> EmitBinaryOp(const HloInstruction* op,
                                           mlir::Value lhs_value,
                                           mlir::Value rhs_value);
  absl::StatusOr<mlir::Value> EmitIntegerBinaryOp(const HloInstruction* op,
                                                  mlir::Value lhs_value,
                                                  mlir::Value rhs_value,
                                                  bool is_signed);
  absl::StatusOr<mlir::Value> EmitFieldBinaryOp(const HloInstruction* op,
                                                mlir::Value lhs_value,
                                                mlir::Value rhs_value);
  absl::StatusOr<mlir::Value> EmitEcPointBinaryOp(const HloInstruction* op,
                                                  mlir::Value lhs_value,
                                                  mlir::Value rhs_value);

  absl::StatusOr<mlir::Value> EmitTernaryOp(const HloInstruction* op,
                                            mlir::Value value1,
                                            mlir::Value value2,
                                            mlir::Value value3);
  absl::StatusOr<mlir::Value> EmitIntegerTernaryOp(const HloInstruction* op,
                                                   mlir::Value value1,
                                                   mlir::Value value2,
                                                   mlir::Value value3,
                                                   bool is_signed);
  absl::StatusOr<mlir::Value> EmitFieldTernaryOp(const HloInstruction* op,
                                                 mlir::Value value1,
                                                 mlir::Value value2,
                                                 mlir::Value value3);
  absl::StatusOr<mlir::Value> EmitEcPointTernaryOp(const HloInstruction* op,
                                                   mlir::Value value1,
                                                   mlir::Value value2,
                                                   mlir::Value value3);

  mlir::Value EmitIntegerDivide(mlir::Value lhs, mlir::Value rhs,
                                bool is_signed);
  mlir::Value EmitIntegerRemainder(mlir::Value lhs, mlir::Value rhs,
                                   bool is_signed);
  mlir::Value EmitIntegerMax(mlir::Value lhs_value, mlir::Value rhs_value,
                             bool is_signed);
  mlir::Value EmitIntegerMin(mlir::Value lhs_value, mlir::Value rhs_value,
                             bool is_signed);

  mlir::Value EmitFieldCompare(ComparisonDirection direction, mlir::Value lhs,
                               mlir::Value rhs);
  mlir::Value EmitFieldMax(mlir::Value lhs, mlir::Value rhs);
  mlir::Value EmitFieldMin(mlir::Value lhs, mlir::Value rhs);

  absl::StatusOr<mlir::Value> EmitElementalConcatenate(
      const HloInstruction* instr, mlir::ValueRange inputs);
  absl::StatusOr<mlir::Value> EmitElementalDynamicSlice(
      const HloInstruction* instr, mlir::ValueRange indices);
  absl::StatusOr<mlir::Value> EmitElementalBroadcast(
      const HloInstruction* instr, mlir::Value input,
      absl::Span<const int64_t> source_dimensions);
  absl::StatusOr<mlir::Value> EmitElementalGather(const HloInstruction* instr,
                                                  mlir::ValueRange operands);
  absl::StatusOr<mlir::Value> EmitElementalDynamicSlice(
      const HloInstruction* instr, mlir::Value input,
      mlir::ValueRange start_indices);
  absl::StatusOr<mlir::Value> EmitElementalDynamicUpdateSlice(
      const HloInstruction* instr, mlir::Value dest, mlir::Value update,
      mlir::ValueRange start_indices);
  absl::StatusOr<mlir::Value> EmitElementalIota(const HloInstruction* instr);

  absl::StatusOr<mlir::Value> EmitElementalMap(const HloInstruction* instr,
                                               mlir::ValueRange inputs);
  absl::StatusOr<mlir::Value> EmitElementalPad(const HloInstruction* instr,
                                               mlir::Value input,
                                               mlir::Value padding_value);
  absl::StatusOr<mlir::Value> EmitElementalDot(const HloInstruction* instr,
                                               mlir::Value lhs,
                                               mlir::Value rhs);
  absl::StatusOr<mlir::Value> EmitElementalMatvec(const HloInstruction* instr,
                                                  mlir::Value lhs,
                                                  mlir::Value rhs);
  absl::StatusOr<mlir::Value> EmitElementalReshape(const HloInstruction* instr,
                                                   mlir::Value value);
  absl::StatusOr<mlir::Value> EmitElementalReverse(const HloInstruction* instr,
                                                   mlir::Value value);
  absl::StatusOr<mlir::Value> EmitElementalSlice(const HloInstruction* instr,
                                                 mlir::Value value);
  absl::StatusOr<mlir::Value> EmitElementalTranspose(
      const HloInstruction* instr, mlir::Value value);
  absl::StatusOr<mlir::Value> EmitElementalReduce(const HloInstruction* instr,
                                                  mlir::ValueRange inputs,
                                                  mlir::ValueRange inits);

  mlir::Value EmitAccumResult(mlir::ValueRange accum_values, bool is_variadic);

  mlir::ImplicitLocOpBuilder* const b_;              // not owned
  mlir::MLIRContext* const context_;                 // not owned
  const BufferAssignment* const buffer_assignment_;  // not owned

  mutable PassFlag pass_flag_;

  friend class ElementalIrEmitterForTests;
};

// Allow to instantiate IR emitter in tests.
class ElementalIrEmitterForTests : public ElementalIrEmitter {
 public:
  ElementalIrEmitterForTests(mlir::ImplicitLocOpBuilder* builder,
                             mlir::MLIRContext* context,
                             const BufferAssignment* buffer_assignment)
      : ElementalIrEmitter(builder, context, buffer_assignment) {}

 private:
  absl::StatusOr<llvm::SmallVector<mlir::Value>> EmitThreadLocalCall(
      const HloComputation& callee, mlir::ValueRange parameters,
      std::string_view name, bool is_reducer) override;
};

}  // namespace zkx

#endif  // ZKX_SERVICE_ELEMENTAL_IR_EMITTER_H_
