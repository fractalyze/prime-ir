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

#ifndef ZKX_SERVICE_CPU_PARALLEL_TASK_ASSIGNMENT_H_
#define ZKX_SERVICE_CPU_PARALLEL_TASK_ASSIGNMENT_H_

#include <cstdint>
#include <memory>
#include <string_view>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"

#include "zkx/hlo/ir/hlo_computation.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/hlo/pass/hlo_pass_interface.h"
#include "zkx/service/hlo_cost_analysis.h"

namespace zkx::cpu {

// Simple interface for different parallel cost model implementations.
class ParallelCostModel {
 public:
  virtual ~ParallelCostModel() = default;
  virtual int64_t GetParallelTaskCount(HloInstruction* instruction) = 0;
};

// ParallelTaskAssignment computes parallel task counts for HLOs in 'module'.
class ParallelTaskAssignment {
 public:
  // 'max_parallelism': the maximum parallel task count per instruction.
  // 'shape_size': shape size function used by HloCostAnalysis during parallel
  //               task assignment.
  // 'module': the containing HloModule.
  //
  // NOTE: ZKX removes the TargetMachineFeatures parameter from XLA because
  // PotentiallyImplementedAsEigenConvolution (its only consumer) does not exist
  // in the ZKX CPU backend.
  ParallelTaskAssignment(int64_t max_parallelism,
                         const HloCostAnalysis::ShapeSizeFunction& shape_size,
                         HloModule* module);
  ~ParallelTaskAssignment() = default;

  // Computes and returns the target parallel task count for 'instruction'.
  int64_t GetTargetParallelTaskCount(HloInstruction* instruction);

 private:
  std::unique_ptr<ParallelCostModel> cost_model_;
};

// ParallelTaskAssigner computes target parallel task counts for all HLOs
// in the module, then assigns parallel task counts to HLOs in the entry
// computation, or to HLOs in embedded computations invoked by (potentially
// nested) kWhile, kConditional, or kCall instructions.
//
// Each parallelized HLO is outlined into its own kCall computation (XLA-style),
// with outer_dimension_partitions set on the outlined root's BackendConfig.
// Only outermost-dimension partitioning is used: the target is clamped to the
// outermost dimension size before ShapePartitionAssigner::Run().
class ParallelTaskAssigner : public HloModulePass {
 public:
  // 'max_parallelism': the maximum parallel task count per instruction.
  // 'shape_size': shape size function used by HloCostAnalysis during parallel
  //               task assignment.
  ParallelTaskAssigner(int64_t max_parallelism,
                       const HloCostAnalysis::ShapeSizeFunction& shape_size)
      : max_parallelism_(max_parallelism), shape_size_function_(shape_size) {}
  ~ParallelTaskAssigner() override = default;

  std::string_view name() const override {
    return "cpu-parallel-task-assigner";
  }

  // Run parallel task assigner on computations with specified
  // `execution_threads` in 'module'. By default, all `execution_threads` are
  // included. Returns true if the computation was changed, false otherwise.
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<std::string_view>& execution_threads) override;

 private:
  using HloToParallelTasks =
      absl::flat_hash_map<const HloInstruction*, int64_t>;

  // Assigns target parallel tasks from 'hlo_to_parallel_tasks' to HLOs in
  // 'module'.
  // Returns true if the computation was changed, false otherwise.
  bool AssignParallelTasks(HloModule* module,
                           const HloToParallelTasks& hlo_to_parallel_tasks);
  bool AssignParallelTasksHelper(
      HloModule* module, HloComputation* computation,
      const HloToParallelTasks& hlo_to_parallel_tasks);

  // Computes target parallel task counts (returned in 'parallel_task_counts')
  // for parallelizable instructions in 'module'.
  void ComputeTargetParallelTasks(HloModule* module,
                                  HloToParallelTasks* hlo_to_parallel_tasks);

  int64_t max_parallelism_;
  HloCostAnalysis::ShapeSizeFunction shape_size_function_;
};

}  // namespace zkx::cpu

#endif  // ZKX_SERVICE_CPU_PARALLEL_TASK_ASSIGNMENT_H_
