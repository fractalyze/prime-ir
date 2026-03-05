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

#include <cstdint>
#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "gtest/gtest.h"

#include "xla/tsl/platform/statusor.h"
#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/hlo/ir/hlo_opcode.h"
#include "zkx/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "zkx/service/cpu/backend_config.pb.h"
#include "zkx/service/cpu/cpu_executable.h"
#include "zkx/service/hlo_cost_analysis.h"

namespace zkx::cpu {
namespace {

class ParallelTaskAssignmentTest : public HloHardwareIndependentTestBase {
 protected:
  // Use any value larger than 2 since we only test whether a module is
  // parallelized or not
  static constexpr int max_parallelism_ = 10;

  absl::StatusOr<bool> RunParallelTaskAssigner(HloModule* module) {
    return ParallelTaskAssigner(max_parallelism_, shape_size_func_).Run(module);
  }

  const HloCostAnalysis::ShapeSizeFunction shape_size_func_ =
      CpuExecutable::ShapeSizeBytes;
};

TEST_F(ParallelTaskAssignmentTest, ReduceWindowParallelized) {
  constexpr char hlo_string[] = R"(
  HloModule m
    add {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      ROOT add = s32[] add(lhs, rhs)
    }

    ENTRY e {
      p0 = s32[512,256] parameter(0)
      p1 = s32[] parameter(1)
      ROOT reduce-window = s32[16,256] reduce-window(p0, p1),
          window={size=32x1 stride=32x1}, to_apply=add
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunParallelTaskAssigner(m.get()));
  EXPECT_TRUE(changed);

  // With outlining, the reduce-window should be wrapped in a kCall.
  auto* call = FindInstruction(m.get(), HloOpcode::kCall);
  ASSERT_NE(call, nullptr);
  auto* root = call->to_apply()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(auto backend_config,
                          root->backend_config<BackendConfig>());
  EXPECT_EQ(backend_config.outer_dimension_partitions_size(), 1);
  EXPECT_EQ(backend_config.outer_dimension_partitions(0), 2);
}

TEST_F(ParallelTaskAssignmentTest, DotOperationNotParallelized) {
  const std::string hlo_string = R"(
    HloModule TestTaskParallel_Dot
    ENTRY Dot {
      dot_lhs = s32[196614,2]{1,0} parameter(0)
      dot_rhs = s32[2,1]{1,0} parameter(1)
      ROOT dot = s32[196614,1]{1,0} dot(dot_lhs, dot_rhs),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunParallelTaskAssigner(m.get()));
  EXPECT_FALSE(changed);
}

TEST_F(ParallelTaskAssignmentTest,
       FusedComputationWithDotOperationNotParallelized) {
  const std::string hlo_string = R"(
    HloModule TestTaskParallel_DotNestedInFusedComp
    fused_computation.0 {
      parameter.0 = s32[196614,2]{1,0} parameter(0)
      parameter.0.1 = s32[2,1]{1,0} parameter(1)
      parameter.0.2 = s32[196614,1]{1,0} parameter(2)
      dot.0 = s32[196614,1]{1,0} dot(parameter.0, parameter.0.1),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
      ROOT add.0 = s32[196614,1]{1,0} add(dot.0, parameter.0.2)

    }
    ENTRY DotNestedInFusedComp {
      parameter = s32[196614,2]{1,0} parameter(0)
      parameter.1 = s32[2,1]{1,0} parameter(1)
      parameter.2 = s32[196614,1]{1,0} parameter(2)
      ROOT fusion = s32[196614,1]{1,0} fusion(parameter, parameter.1,
        parameter.2), kind=kOutput, calls=fused_computation.0
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunParallelTaskAssigner(m.get()));
  EXPECT_FALSE(changed);
}

// NOTE: XLA's RngOperationNotParallelized test is omitted because kRng does
// not exist in ZKX.

TEST_F(ParallelTaskAssignmentTest, InfeedOutfeedOperationNotParallelized) {
  const std::string hlo_string = R"(
    HloModule TestTaskParallel_infeed_outfeed
    ENTRY InfeedOutfeed {
      token0 = token[] after-all()
      infeed0 = (u32[12345678,2]{1,0}, token[]) infeed(token0)
      infeed0.data = u32[12345678,2]{1,0} get-tuple-element((u32[12345678,2]{1,0}, token[]) infeed0), index=0
      ROOT outfeed0 = token[] outfeed(infeed0.data, token0)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunParallelTaskAssigner(m.get()));
  EXPECT_FALSE(changed);
}

TEST_F(ParallelTaskAssignmentTest, InPlaceDynamicUpdateSliceNotParallelized) {
  // A dynamic-update-slice within a while loop.  This construction is an easy
  // way to make a DUS which can be run "in-place" (i.e. the input and output
  // are the same buffer, and running the DUS only writes to the updated
  // elements).
  const std::string hlo_string = R"(
  HloModule test

  body {
    zero = s32[] constant(0)
    one = s32[] constant(1)
    ten = s32[] constant(10)
    loop_carry = (s32[], u32[1,100], u32[10000,100]) parameter(0)
    i = s32[] get-tuple-element(loop_carry), index=0
    i_plus_ten = s32[] add(i, ten)
    update = u32[1,100] get-tuple-element(loop_carry), index=1
    data = u32[10000,100] get-tuple-element(loop_carry), index=2
    new_data = u32[10000,100] dynamic-update-slice(data, update, i_plus_ten, zero)
    new_i = s32[] add(i, one)
    ROOT tuple = (s32[], u32[1,100], u32[10000,100]) tuple(new_i, update, new_data)
  }

  cond {
    loop_carry = (s32[], u32[1,100], u32[10000,100]) parameter(0)
    two = s32[] constant(2)
    i = s32[] get-tuple-element(loop_carry), index=0
    ROOT less-than = pred[] compare(i, two), direction=LT
  }

  ENTRY test {
    zero = s32[] constant(0)
    initial_i = s32[] parameter(0)
    update = u32[1,100] parameter(1)
    data = u32[10000,100] parameter(2)
    tuple = (s32[], u32[1,100], u32[10000,100]) tuple(initial_i, update, data)
    ROOT while = (s32[], u32[1,100], u32[10000,100]) while(tuple), condition=cond, body=body
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunParallelTaskAssigner(m.get()));
  EXPECT_FALSE(changed);
}

TEST_F(ParallelTaskAssignmentTest, AllReduceNotParallelized) {
  constexpr char hlo_string[] = R"(
  HloModule TestTaskParallel_allreduce
    add {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      ROOT add = s32[] add(lhs, rhs)
    }

    ENTRY CRS {
      input = s32[1234567] parameter(0)
      ROOT crs = s32[1234567] all-reduce(input), replica_groups={}, to_apply=add
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunParallelTaskAssigner(m.get()));
  EXPECT_FALSE(changed);
}

TEST_F(ParallelTaskAssignmentTest, ConstantNotParallelized) {
  constexpr char hlo_string[] = R"(
  HloModule TestTaskParallel_constant
    ENTRY const {
      ROOT constant = s32[1234567] constant({...})
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunParallelTaskAssigner(m.get()));
  EXPECT_FALSE(changed);
}

TEST_F(ParallelTaskAssignmentTest, LoopFusionParallelized) {
  constexpr char hlo_string[] = R"(
  HloModule m
    fused_computation {
      p0 = s32[131072] parameter(0)
      p1 = s32[131072] parameter(1)
      ROOT add = s32[131072] add(p0, p1)
    }

    ENTRY e {
      a = s32[131072] parameter(0)
      b = s32[131072] parameter(1)
      ROOT fusion = s32[131072] fusion(a, b), kind=kLoop, calls=fused_computation
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunParallelTaskAssigner(m.get()));
  EXPECT_TRUE(changed);

  // The fusion should be outlined into a kCall.
  auto* call = FindInstruction(m.get(), HloOpcode::kCall);
  ASSERT_NE(call, nullptr);
  auto* root = call->to_apply()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(auto backend_config,
                          root->backend_config<BackendConfig>());
  EXPECT_GT(backend_config.outer_dimension_partitions_size(), 0);
}

TEST_F(ParallelTaskAssignmentTest, LoopFusionWithTransposeNotParallelized) {
  constexpr char hlo_string[] = R"(
  HloModule m
    fused_computation {
      p0 = s32[512,256]{1,0} parameter(0)
      ROOT transpose = s32[256,512]{1,0} transpose(p0), dimensions={1,0}
    }

    ENTRY e {
      a = s32[512,256]{1,0} parameter(0)
      ROOT fusion = s32[256,512]{1,0} fusion(a), kind=kLoop, calls=fused_computation
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN([[maybe_unused]] bool changed,
                          RunParallelTaskAssigner(m.get()));
  // Transpose is allowed by the opcode filter, but the cost model should
  // determine whether it's worth parallelizing. With outlining, safety
  // checking inside fusions is not needed.
  // Either way, no multi-dim checks are performed.
}

}  // namespace
}  // namespace zkx::cpu
