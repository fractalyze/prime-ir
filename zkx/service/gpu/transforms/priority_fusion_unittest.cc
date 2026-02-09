/* Copyright 2023 The OpenXLA Authors.
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

#include "zkx/service/gpu/transforms/priority_fusion.h"

#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "xla/tsl/platform/statusor.h"
#include "zkx/hlo/ir/hlo_computation.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/testlib/pattern_matcher_gmock.h"
#include "zkx/service/gpu/gpu_device_info_for_tests.h"
#include "zkx/service/gpu/hlo_fusion_analysis.h"
#include "zkx/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "zkx/service/pattern_matcher.h"
#include "zkx/stream_executor/device_description.h"
#include "zkx/tests/hlo_test_base.h"

namespace m = ::zkx::match;

namespace zkx::gpu {
namespace {

// NOTE: These tests use s32 instead of f32/bf16 because ZKX doesn't have
// floating-point types - it's designed for zero-knowledge proof computations.

class PriorityFusionTest : public HloTestBase {
 public:
  se::DeviceDescription device_info_ = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  PriorityFusion priority_fusion_{
      /*thread_pool=*/nullptr, device_info_,
      GpuHloCostAnalysis::Options{.count_multiple_input_accesses = true}};
};

TEST_F(PriorityFusionTest, FuseWithSharedArgument) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    ENTRY main {
      %p0 = s32[] parameter(0)
      %p1 = s32[] parameter(1)
      %subtract = s32[] subtract(%p0, %p1)
      %compare = pred[] compare(%subtract, %subtract), direction=NE
      %add = s32[] add(%p0, %p1)
      %abs = s32[] abs(%subtract)
      ROOT %select = s32[] select(%compare, %add, %abs)
    })")
                    .value();

  EXPECT_TRUE(priority_fusion_.Run(module.get()).value());

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Fusion()));
  EXPECT_EQ(root->fusion_kind(), HloInstruction::FusionKind::kLoop);
}

TEST_F(PriorityFusionTest, DoNotChangeReductionFusionToLoopFusion) {
  // Regression test for epilogue fusion of slice into a reduction. The fusion
  // kind for the reduction fusion is intentionally chosen to be set to kLoop,
  // as we cannot rely on reductions always having fusion kind kInput.
  auto module = *ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    add {
      rhs.407 = s32[] parameter(1)
      lhs.407 = s32[] parameter(0)
      ROOT add.24451 = s32[] add(lhs.407, rhs.407)
    }

    fused_computation {
      p0 = s32[16,64]{1,0} parameter(0)
      zero = s32[] constant(0)
      ROOT reduce = s32[16]{0} reduce(p0, zero), dimensions={1}, to_apply=add
    }

    ENTRY main {
      param0 = s32[16,64]{1,0} parameter(0)
      fusion = s32[16]{0} fusion(param0), kind=kLoop, calls=fused_computation
      ROOT slice = s32[8]{0} slice(fusion), slice={[0:8]}
    })");
  EXPECT_FALSE(priority_fusion_.Run(module.get()).value());
}

TEST_F(PriorityFusionTest, DontFuseIntoFirstOperandOfScatter) {
  auto module = *ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    add {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      ROOT add = s32[] add(lhs, rhs)
    }

    ENTRY FuseIntoScatter {
      p0 = s32[3,3] parameter(0)
      operand = s32[3,3] add(p0, p0)
      p1 = s32[2,1] parameter(1)
      indices = s32[2,1] add(p1, p1)
      p2 = s32[2,3] parameter(2)
      updates = s32[2,3] add(p2, p2)
      scatter = s32[3,3] scatter(operand, indices, updates),
          to_apply=add,
          update_window_dims={1},
          inserted_window_dims={0},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1
      ROOT add = s32[3,3] add(scatter, scatter)
    })");

  EXPECT_TRUE(priority_fusion_.Run(module.get()).value());

  HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* fusion = nullptr;
  ASSERT_THAT(root, GmockMatch(m::Add(m::Fusion(&fusion), m::Fusion())));
  EXPECT_EQ(fusion->fusion_kind(), HloInstruction::FusionKind::kInput);
  EXPECT_THAT(fusion->fused_expression_root(),
              GmockMatch(m::Scatter(m::Parameter(), m::Add(), m::Add())));
}

// This test is similar to DontFuseIntoFirstOperandOfScatter, but PriorityFusion
// has a separate run to fuse constants. Fusing anything into a scatter fusion
// will fail in the emitter.
TEST_F(PriorityFusionTest, DontFuseConstantIntoFirstOperandOfScatter) {
  auto module = *ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    add {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      ROOT add = s32[] add(lhs, rhs)
    }

    ENTRY FuseIntoScatter {
      operand = s32[1] constant({0})
      indices = s32[24,1] parameter(0)
      constant = s32[] constant(1)
      updates = s32[24,1] broadcast(constant)
      ROOT scatter = s32[1] scatter(operand, indices, updates),
          to_apply=add,
          update_window_dims={1},
          inserted_window_dims={},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1
    })");

  EXPECT_TRUE(priority_fusion_.Run(module.get()).value());

  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_THAT(root, GmockMatch(m::Fusion(m::Constant(), m::Parameter())));
  EXPECT_EQ(root->fusion_kind(), HloInstruction::FusionKind::kInput);
  EXPECT_THAT(root->fused_expression_root(),
              GmockMatch(m::Scatter(m::Parameter(), m::Parameter(),
                                    m::Broadcast(m::Constant()))));
}

TEST_F(PriorityFusionTest, DoNotFuseIntoRoot) {
  auto module = *ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    ENTRY %main (p.0: u32[2], p.1: u32[]) -> u32[2] {
      %p.0 = u32[2]{0} parameter(0)
      %p.1 = u32[] parameter(1)
      ROOT %broadcast = u32[2]{0} broadcast(u32[] %p.1), dimensions={}, sharding={replicated}
      %add = u32[2]{0} add(u32[2]{0} %p.0, u32[2]{0} %broadcast)
      %tuple.1 = (u32[2]{0}) tuple(u32[2]{0} %add)
      %token.0 = token[] after-all()
      %outfeed.6 = token[] outfeed((u32[2]{0}) %tuple.1, token[] %token.0), outfeed_shape=(u32[2]{0}), sharding={maximal device=0}
    })");

  EXPECT_FALSE(priority_fusion_.Run(module.get()).value());
}

TEST_F(PriorityFusionTest, DoNotFuseInsideReducer) {
  auto module = *ParseAndReturnVerifiedModule(R"(
    %reducer {
      p0 = s32[] parameter(0)
      p1 = s32[] parameter(1)
      add = s32[] add(p0, p1)
      ROOT max = s32[] maximum(add, p0)
    }

    %fused_reduce {
      p0 = s32[256] parameter(0)
      p1 = s32[] parameter(1)
      ROOT reduce = s32[] reduce(p0, p1), dimensions={0}, to_apply=%reducer
    }

    ENTRY fusion {
      p0 = s32[256] parameter(0)
      p1 = s32[] parameter(1)
      ROOT %reduce = s32[] fusion(p0, p1), kind=kInput, calls=fused_reduce
    }
  )");
  EXPECT_FALSE(priority_fusion_.Run(module.get()).value());
}

}  // namespace
}  // namespace zkx::gpu
