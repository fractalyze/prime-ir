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

#include "zkx/service/cpu/parallel_task_assignment.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"

#include "xla/tsl/platform/cpu_info.h"
#include "zkx/base/logging.h"
#include "zkx/hlo/ir/hlo_casting_utils.h"
#include "zkx/hlo/ir/hlo_instructions.h"
#include "zkx/hlo/ir/hlo_opcode.h"
#include "zkx/service/cpu/backend_config.pb.h"
#include "zkx/service/cpu/dynamic_update_slice_util.h"
#include "zkx/service/cpu/shape_partition.h"

namespace zkx::cpu {

namespace {

// Returns true if a fusion computation contains any instruction that cannot
// be safely partitioned along the outermost dimension. Only elementwise ops,
// constants, parameters, bitcasts, broadcasts, reshapes, and safe
// slices/concatenates are allowed.
bool FusionContainsUnsafePartitionOp(const HloInstruction* fusion,
                                     int64_t outermost_dim) {
  for (const auto* instr : fusion->fused_instructions()) {
    switch (instr->opcode()) {
      case HloOpcode::kParameter:
      case HloOpcode::kConstant:
      case HloOpcode::kBitcastConvert:
      case HloOpcode::kBroadcast:
      case HloOpcode::kTuple:
      case HloOpcode::kGetTupleElement:
        continue;
      case HloOpcode::kBitcast:
      case HloOpcode::kReshape: {
        // Bitcasts/reshapes that change the partition dimension size are
        // unsafe: subsequent slices/concatenates may operate on remapped
        // dimensions, making per-partition scaling incorrect.
        const Shape& in_shape = instr->operand(0)->shape();
        const Shape& out_shape = instr->shape();
        if (outermost_dim < in_shape.rank() &&
            outermost_dim < out_shape.rank() &&
            in_shape.dimensions(outermost_dim) !=
                out_shape.dimensions(outermost_dim)) {
          VLOG(3) << "Fusion " << fusion->name()
                  << " has bitcast/reshape changing partition dim: "
                  << instr->ToString();
          return true;
        }
        continue;
      }
      case HloOpcode::kSlice: {
        const Shape& operand_shape = instr->operand(0)->shape();
        if (operand_shape.rank() == 0) continue;
        if (outermost_dim >= operand_shape.rank()) continue;
        if (instr->slice_starts(outermost_dim) != 0 ||
            instr->slice_limits(outermost_dim) !=
                operand_shape.dimensions(outermost_dim)) {
          VLOG(3) << "Fusion " << fusion->name()
                  << " has slice along partition dim: " << instr->ToString();
          return true;
        }
        continue;
      }
      case HloOpcode::kConcatenate:
        if (instr->concatenate_dimension() == outermost_dim) {
          VLOG(3) << "Fusion " << fusion->name()
                  << " has concat along partition dim: " << instr->ToString();
          return true;
        }
        continue;
      default:
        if (instr->IsElementwise()) continue;
        VLOG(3) << "Fusion " << fusion->name()
                << " contains unsafe op for partitioning: "
                << HloOpcodeString(instr->opcode());
        return true;
    }
  }
  return false;
}

}  // namespace

class SimpleCostModel : public ParallelCostModel {
 public:
  SimpleCostModel(int64_t max_parallelism,
                  const HloCostAnalysis::ShapeSizeFunction& shape_size)
      : max_parallelism_(max_parallelism), shape_size_(shape_size) {}
  ~SimpleCostModel() override = default;

  int64_t GetParallelTaskCount(HloInstruction* instruction) override {
    // Simple cost model based on hlo size and typical L2 cache size.
    const int64_t instruction_cost = shape_size_(instruction->shape());
    const int64_t min_cost_per_thread = 256LL << 10;  // 256KB L2 Cache size.
    // Return target parallel task count in [1, max_parallelism_].
    return std::min(
        max_parallelism_,
        std::max(int64_t{1}, instruction_cost / min_cost_per_thread));
  }

 private:
  const int64_t max_parallelism_;
  const HloCostAnalysis::ShapeSizeFunction shape_size_;
};

class DefaultCostModel : public ParallelCostModel {
 public:
  DefaultCostModel(int64_t max_parallelism,
                   const HloCostAnalysis::ShapeSizeFunction& shape_size,
                   std::unique_ptr<HloCostAnalysis> cost_analysis)
      : max_parallelism_(max_parallelism),
        shape_size_(shape_size),
        cost_analysis_(std::move(cost_analysis)) {}
  ~DefaultCostModel() override = default;

  int64_t GetParallelTaskCount(HloInstruction* instruction) override {
    // Parameters for parallel task count computation.
    int64_t instruction_cost;
    int64_t min_cost_per_thread;
    int64_t max_parallelism;
    // Calculate flops-to-bytes-ratio for 'instruction'.
    const int64_t bytes_accessed =
        std::max(int64_t{1}, cost_analysis_->bytes_accessed(*instruction));
    const float flops_to_bytes_ratio =
        cost_analysis_->flop_count(*instruction) /
        static_cast<float>(bytes_accessed);
    // Check for I/O bound instructions.
    if (flops_to_bytes_ratio <= 1.0) {
      // Limit max parallelism for I/O bound instructions by assuming a
      // sub-linear scaling function (fit based on empirical benchmark results).
      // TODO(b/29630486) Develop system bandwidth model.
      // TODO(chokobole): ZK field ops report 0 flops, causing
      // misclassification as I/O-bound. Use max parallelism until
      // HloCostAnalysis supports ZK ops.
      // max_parallelism = std::min<int64_t>(
      //     max_parallelism_,
      //     std::ceil(std::sqrt(tsl::port::MaxParallelism())));
      max_parallelism = max_parallelism_;
      // Use bytes accessed cost and L2 cache size min per-thread cost.
      instruction_cost = bytes_accessed;
      min_cost_per_thread = 256LL << 10;  // 256KB L2 Cache size.
    } else {
      // Use max parallelism for compute bound instructions.
      max_parallelism = max_parallelism_;
      // Calculate the instruction cost in cycles.
      // TODO(b/29630486) Improve on this linear cost model.
      // Consider making 'min_cost_per_thread' be a function of the target
      // bandwidth limit for instructions with low arithmetic complexity.
      instruction_cost = 1 * cost_analysis_->flop_count(*instruction) +
                         10 * cost_analysis_->bytes_accessed(*instruction);
      // Minimum per-thread cost is 100us of work on a 2GHz core.
      min_cost_per_thread = 100000;
    }
    // Return target parallel task count in [1, max_parallelism_].
    return std::min(
        max_parallelism,
        std::max(int64_t{1}, instruction_cost / min_cost_per_thread));
  }

 private:
  const int64_t max_parallelism_;
  const HloCostAnalysis::ShapeSizeFunction shape_size_;
  const std::unique_ptr<HloCostAnalysis> cost_analysis_;
};

ParallelTaskAssignment::ParallelTaskAssignment(
    int64_t max_parallelism,
    const HloCostAnalysis::ShapeSizeFunction& shape_size, HloModule* module) {
  VLOG(1) << "ParallelTaskAssignment max_parallelism: " << max_parallelism;
  // Run cost analysis on 'module'.
  auto cost_analysis = std::make_unique<HloCostAnalysis>(shape_size);
  HloComputation* computation = module->entry_computation();
  absl::Status status =
      computation->root_instruction()->Accept(cost_analysis.get());
  if (status.ok()) {
    // Set default cost model based on 'cost_analysis'.
    cost_model_ = std::make_unique<DefaultCostModel>(
        max_parallelism, shape_size, std::move(cost_analysis));
  } else {
    // Fall back to a simple cost model based on hlo size and L2 cache size.
    // Note that HloCostAnalysis can returns an error status (likely because
    // HLOs like CustomCall are not yet implemented in the HloCostAnalysis).
    cost_model_ =
        std::make_unique<SimpleCostModel>(max_parallelism, shape_size);
  }
}

int64_t ParallelTaskAssignment::GetTargetParallelTaskCount(
    HloInstruction* instruction) {
  // Currently, we do not assign parallel tasks to instructions with at least
  // one of the following properties:
  // *) Internal threading (library calls to kDot, kCustomCall).
  // *) Emit custom loops (kSelectAndScatter).
  // *) Operations that are not thread safe (like infeed and rng).
  // *) Tuple-shaped.
  // *) Operations that might be implemented as an in-place
  //    dynamic-update-slice, because we can't know how many output elements
  //    they will write (out-of-place will touch the whole output buffer, while
  //    in-place will only touch the updated elements).
  // TODO(b/27458679) Parallelize instructions which are skipped here.
  auto opcode = instruction->opcode();
  if (llvm_ir::MayBeImplementedAsInPlaceDynamicUpdateSlice(instruction) ||
      instruction->shape().IsTuple() || opcode == HloOpcode::kConstant) {
    return 1;
  }

  // Only allow instructions that can be trivially parallelized (where all
  // outputs can be computed independently of each other).
  //
  // NOTE: XLA additionally checks PotentiallyImplementedAsEigenConvolution for
  // kConvolution. This is unavailable in the ZKX CPU backend.
  if (instruction->IsElementwise() || instruction->IsLoopFusion() ||
      opcode == HloOpcode::kBroadcast || opcode == HloOpcode::kConcatenate ||
      opcode == HloOpcode::kDynamicSlice ||
      opcode == HloOpcode::kDynamicUpdateSlice ||
      opcode == HloOpcode::kGather || opcode == HloOpcode::kIota ||
      opcode == HloOpcode::kPad || opcode == HloOpcode::kReduce ||
      opcode == HloOpcode::kReduceWindow || opcode == HloOpcode::kReshape ||
      opcode == HloOpcode::kReverse || opcode == HloOpcode::kSlice ||
      opcode == HloOpcode::kTranspose) {
    // Check fusions for ops unsafe for outermost-dimension partitioning.
    if (opcode == HloOpcode::kFusion) {
      const Shape& shape = instruction->shape();
      if (shape.IsArray() && shape.rank() > 0) {
        int64_t outermost_dim = shape.layout().minor_to_major(shape.rank() - 1);
        if (FusionContainsUnsafePartitionOp(instruction, outermost_dim)) {
          VLOG(2) << "Skipping fusion with unsafe partitioning op: "
                  << instruction->name();
          return 1;
        }
      }
    }
    return cost_model_->GetParallelTaskCount(instruction);
  }

  return 1;
}

absl::StatusOr<bool> ParallelTaskAssigner::Run(
    HloModule* module,
    const absl::flat_hash_set<std::string_view>& execution_threads) {
  ZKX_VLOG_LINES(2, "ParallelTaskAssigner ENTRY");
  ZKX_VLOG_LINES(3, module->ToString());
  // Compute target parallel task counts for all instructions in 'module'.
  HloToParallelTasks hlo_to_parallel_tasks;
  ComputeTargetParallelTasks(module, &hlo_to_parallel_tasks);

  // Assign parallel tasks to target specific instructions in 'module'.
  // TODO(b/27458679) Support inter-op parallelism.
  bool changed = AssignParallelTasks(module, hlo_to_parallel_tasks);

  ZKX_VLOG_LINES(2, "ParallelTaskAssigner EXIT");
  ZKX_VLOG_LINES(3, module->ToString());
  return changed;
}

bool ParallelTaskAssigner::AssignParallelTasks(
    HloModule* module, const HloToParallelTasks& hlo_to_parallel_tasks) {
  return AssignParallelTasksHelper(module, module->entry_computation(),
                                   hlo_to_parallel_tasks);
}

bool ParallelTaskAssigner::AssignParallelTasksHelper(
    HloModule* module, HloComputation* computation,
    const HloToParallelTasks& hlo_to_parallel_tasks) {
  bool changed = false;
  // Snapshot set of instructions because assignment may modify the set.
  std::vector<HloInstruction*> instructions(computation->instructions().begin(),
                                            computation->instructions().end());
  for (auto* instruction : instructions) {
    // Assign parallel tasks to sub-computations for While and Call HLOs.
    // TODO(b/27458679) Evaluate alternative intra-op parallelism placement,
    // and support other callable computations like reduce.
    if (instruction->opcode() == HloOpcode::kWhile) {
      changed |= AssignParallelTasksHelper(module, instruction->while_body(),
                                           hlo_to_parallel_tasks);
      continue;
    } else if (instruction->opcode() == HloOpcode::kCall) {
      changed |= AssignParallelTasksHelper(module, instruction->to_apply(),
                                           hlo_to_parallel_tasks);
      continue;
    }
    // Skip if no parallel tasks were computed in first pass.
    auto it = hlo_to_parallel_tasks.find(instruction);
    if (it == hlo_to_parallel_tasks.end()) {
      continue;
    }
    // Get target parallel task count computed for 'instruction'.
    const int64_t target_parallel_task_count = (*it).second;

    const Shape& shape = instruction->shape();
    if (!shape.IsArray()) {
      continue;
    }

    // Assign feasible dimension partitions (based on actual dimension sizes).
    // ShapePartitionAssigner may spread partitions across multiple outer dims
    // when the outermost dim alone cannot satisfy the target.
    auto dim_partition_counts =
        ShapePartitionAssigner(shape).Run(target_parallel_task_count);

    // For fusions, verify that inner partition dims are safe. The outermost dim
    // was already checked by GetTargetParallelTaskCount, but inner dims need
    // their own safety check against FusionContainsUnsafePartitionOp.
    if (instruction->opcode() == HloOpcode::kFusion &&
        dim_partition_counts.size() > 1) {
      bool inner_unsafe = false;
      for (int i = 1; i < static_cast<int>(dim_partition_counts.size()); ++i) {
        if (dim_partition_counts[i] > 1) {
          int64_t dim = shape.layout().minor_to_major(shape.rank() - 1 - i);
          if (FusionContainsUnsafePartitionOp(instruction, dim)) {
            inner_unsafe = true;
            break;
          }
        }
      }
      if (inner_unsafe) {
        // Fallback: outermost-only partitioning.
        int64_t outermost_size =
            shape.dimensions(shape.layout().minor_to_major(shape.rank() - 1));
        int64_t outermost_target =
            std::min(target_parallel_task_count, outermost_size);
        dim_partition_counts = {outermost_target};
      }
    }
    const int64_t total_partition_count =
        ShapePartitionAssigner::GetTotalPartitionCount(dim_partition_counts);
    if (total_partition_count <= 1) {
      // Feasible partition calculation resulting in no partitioning, so skip.
      continue;
    }

    // Outline instruction into a kCall computation.
    auto call_or = module->OutlineExpressionFromComputation(
        {instruction}, absl::StrCat("parallel_", instruction->name()),
        computation);
    CHECK_OK(call_or.status());
    auto* call = call_or.value();

    // Set partition config on the outlined root.
    auto* new_root = call->to_apply()->root_instruction();
    BackendConfig backend_config;
    absl::c_copy(dim_partition_counts,
                 google::protobuf::RepeatedFieldBackInserter(
                     backend_config.mutable_outer_dimension_partitions()));
    CHECK_OK(new_root->set_backend_config(backend_config));

    VLOG(2) << "Assigned parallel task count: " << total_partition_count
            << " to instruction: " << new_root->name() << " (outlined from "
            << call->name() << ")";
    changed = true;
  }
  return changed;
}

void ParallelTaskAssigner::ComputeTargetParallelTasks(
    HloModule* module, HloToParallelTasks* hlo_to_parallel_tasks) {
  ParallelTaskAssignment parallel_task_assignment(max_parallelism_,
                                                  shape_size_function_, module);

  // Compute parallel task counts for all instructions in 'module'.
  for (auto* computation : module->MakeNonfusionComputations()) {
    for (auto* instruction : computation->instructions()) {
      // Query ParallelTaskAssignment for target parallel task count.
      const int64_t target_parallel_task_count =
          parallel_task_assignment.GetTargetParallelTaskCount(instruction);
      if (target_parallel_task_count > 1) {
        hlo_to_parallel_tasks->insert(
            {instruction, target_parallel_task_count});
      }
    }
  }
}

}  // namespace zkx::cpu
