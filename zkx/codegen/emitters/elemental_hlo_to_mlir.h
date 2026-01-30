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
#ifndef ZKX_CODEGEN_EMITTERS_ELEMENTAL_HLO_TO_MLIR_H_
#define ZKX_CODEGEN_EMITTERS_ELEMENTAL_HLO_TO_MLIR_H_

#include <stdint.h>

#include <functional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"

#include "zkx/codegen/emitters/computation_partitioner.h"
#include "zkx/hlo/analysis/indexing_map.h"
#include "zkx/hlo/ir/hlo_instruction.h"

namespace zkx::emitters {

using OperandProvider =
    std::function<absl::StatusOr<llvm::SmallVector<mlir::Value, 1>>(
        const HloInstruction* instr, int index, mlir::ValueRange indices)>;

// Emits MLIR to produce the value of a parameter. The parameter must be located
// outside the subgraph. By default, the caller subgraph will be determined by
// searching in 'computation' for the subgraph that contains 'instr'. If
// 'instr' does not belong to 'computation', the caller subgraph can be passed
// directly.
mlir::ValueRange ProvideParameter(
    const PartitionedComputation& computation, const HloInstruction* instr,
    int operand_index, mlir::ValueRange indices,
    const CallTargetProvider& call_target_provider, mlir::func::FuncOp this_fn,
    mlir::ImplicitLocOpBuilder& builder,
    const PartitionedComputation::Subgraph* caller = nullptr);

// Emits MLIR to produce the values of a range of parameters. The parameters
// must all be scalars. The parameters are all evaluated at the same indices.
llvm::SmallVector<mlir::Value, 2> ProvideParameterRange(
    const PartitionedComputation& computation, const HloInstruction* instr,
    int start, int num, mlir::ValueRange indices,
    const CallTargetProvider& call_target_provider, mlir::func::FuncOp this_fn,
    mlir::ImplicitLocOpBuilder& builder);

// Converts a function (subgraph) to an MLIR function producing one element of
// the result. The function must have the correct interface.
absl::Status SubgraphToMlirFunction(
    const PartitionedComputation& computation,
    const PartitionedComputation::Subgraph& subgraph, mlir::func::FuncOp& func,
    const CallTargetProvider& call_target_provider);

// Creates an `apply_indexing` op for the given map.
llvm::SmallVector<mlir::Value, 3> ApplyIndexing(const IndexingMap& map,
                                                mlir::ValueRange dims,
                                                mlir::ValueRange symbols,
                                                mlir::ImplicitLocOpBuilder& b);

// Checks all the constraints and dimension ranges in the map.
mlir::Value CheckConstraints(const IndexingMap& map, mlir::ValueRange dims,
                             mlir::ValueRange symbols,
                             mlir::ImplicitLocOpBuilder& b);

// Emits a loop nest over the entire domain of the indexing_map at a point
// `dim_values`.
// If `vectorize` is set, the loop essentially turns into multiple independent
// loops, and the results of all the loops are returned as a vector. The last
// symbol dimension is used as the vectorized dimension.
// If `vectorize` is set:
// - the body will still be called with scalars and should return scalars.
// - the loop for the last symbol in `indexing_map` will be vectorized
// - the symbol range should be [0, 2] or [0, 4] for vectorization to work.
//   [0, 1] is supported and will have no effect. The lower bound must be 0.
// - all scalar results of `EmitLoopNest` will become vectors instead. Scalar
//   inits will be initialized with a vector splat. Passing a vector init is
//   supported.
// - Tensor arguments and results are unaffected.
// TODO(chokobole): Implement EmitLoopNest.

// Same as EmitLoopNest, but uses zkx_gpu.loop.
mlir::ValueRange EmitZkxLoopOp(
    mlir::ImplicitLocOpBuilder& b, mlir::ValueRange dim_values,
    mlir::ValueRange iter_args_inits, const IndexingMap& indexing_map,
    mlir::function_ref<llvm::SmallVector<mlir::Value>(
        mlir::ImplicitLocOpBuilder& nested_b, mlir::ValueRange ivs,
        mlir::ValueRange map_results, mlir::ValueRange iter_args)>
        create_body,
    bool vectorize = false);

// Populates `lbs`, `ubs` and `steps` with the loop bounds from `indexing_map`.
void GetLoopBoundsFromIndexingMap(mlir::ImplicitLocOpBuilder& b,
                                  const IndexingMap& indexing_map,
                                  llvm::SmallVectorImpl<mlir::Value>& lbs,
                                  llvm::SmallVectorImpl<mlir::Value>& ubs,
                                  llvm::SmallVectorImpl<mlir::Value>& steps);

// Clamps `index` to [0, high] boundaries.
mlir::Value ClampIndex(mlir::Value index, bool is_unsigned, int64_t high,
                       mlir::ImplicitLocOpBuilder& b);

}  // namespace zkx::emitters

#endif  // ZKX_CODEGEN_EMITTERS_ELEMENTAL_HLO_TO_MLIR_H_
