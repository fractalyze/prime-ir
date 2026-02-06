/* Copyright 2024 The OpenXLA Authors.
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

#include "zkx/service/gpu/model/gpu_performance_model_base.h"

#include <string_view>

#include "gtest/gtest.h"

#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "zkx/hlo/ir/hlo_computation.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/service/gpu/gpu_device_info_for_tests.h"
#include "zkx/service/gpu/hlo_fusion_analysis.h"
#include "zkx/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "zkx/stream_executor/device_description.h"
#include "zkx/tests/hlo_test_base.h"

namespace zkx::gpu {
namespace {

class GpuPerformanceModelBaseTest : public HloTestBase {
 public:
  GpuHloCostAnalysis::Options options_{.count_multiple_input_accesses = true};
  // The reference times in the test cases below are measured
  // on A6000 by profiling the execution of the HLOs.
  se::DeviceDescription device_info_{TestGpuDeviceInfo::RTXA6000DeviceInfo()};
  // TODO(batzor): Pass device_info_ to GpuHloCostAnalysis once it supports
  // the two-argument constructor (matching XLA's interface).
  GpuHloCostAnalysis analysis_{options_};

  GpuPerformanceModelBaseTest() : HloTestBase() {}
};

TEST_F(GpuPerformanceModelBaseTest, SharedOperandBytesAccessed_InPlaceDUS) {
  std::string_view hlo_string = R"(
HloModule m

ENTRY entry_computation {
  param_0 = s32[8,16] parameter(0)
  param_1 = s32[4,4] parameter(1)
  c_0 = s32[] constant(0)
  negate = s32[4,4] negate(param_1)
  ROOT dynamic-update-slice = s32[8,16] dynamic-update-slice(param_0, negate, c_0, c_0)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto computation = module->entry_computation();
  TF_ASSERT_OK(computation->Accept(&analysis_));

  auto dus_consumer = computation->root_instruction();
  auto negate_producer = dus_consumer->mutable_operand(1);

  auto get_shared_operand_bytes_accessed = [&](const HloInstruction* operand) {
    return GpuPerformanceModelBase::GetSharedOperandBytesAccessed(
        &analysis_, negate_producer, dus_consumer, operand);
  };

  EXPECT_EQ(get_shared_operand_bytes_accessed(dus_consumer->operand(0)), 0);
  EXPECT_EQ(get_shared_operand_bytes_accessed(negate_producer->operand(0)), 64);
}

TEST_F(GpuPerformanceModelBaseTest, SharedOperandBytesAccessed_DUS) {
  std::string_view hlo_string = R"(
HloModule m

ENTRY entry_computation {
  param_0 = s32[8,16] parameter(0)
  param_1 = s32[4,4] parameter(1)
  c_0 = s32[] constant(0)
  negate = s32[8,16] negate(param_0)
  ROOT dynamic-update-slice = s32[8,16] dynamic-update-slice(negate, param_1, c_0, c_0)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto computation = module->entry_computation();
  TF_ASSERT_OK(computation->Accept(&analysis_));

  auto dus_consumer = computation->root_instruction();
  auto negate_producer = dus_consumer->mutable_operand(0);

  auto get_shared_operand_bytes_accessed = [&](const HloInstruction* operand) {
    return GpuPerformanceModelBase::GetSharedOperandBytesAccessed(
        &analysis_, negate_producer, dus_consumer, operand);
  };

  EXPECT_EQ(get_shared_operand_bytes_accessed(dus_consumer->operand(1)), 64);
  EXPECT_EQ(get_shared_operand_bytes_accessed(negate_producer->operand(0)),
            448);
}

// This test documents current behaviour. See comments below how the correct
// result should look like.
TEST_F(GpuPerformanceModelBaseTest,
       ReduceBroadcastedDim_IncorrectBytesAccessed) {
  std::string_view hlo_string = R"(
HloModule m

add {
  p0 = s32[] parameter(0)
  p1 = s32[] parameter(1)
  ROOT add = s32[] add(p0, p1)
}

f1 {
  p0 = s32[128] parameter(0)
  c0 = s32[] constant(0)
  broadcast = s32[128,256] broadcast(p0), dimensions={0}
  ROOT reduce = s32[128] reduce(broadcast, c0), dimensions={1}, to_apply=add
}

ENTRY entry_computation {
  param_0 = s32[128] parameter(0)
  param_1 = s32[4,4] parameter(1)
  ROOT fusion = s32[128] fusion(param_0), kind=kLoop, calls=f1
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto computation = module->entry_computation();
  TF_ASSERT_OK(computation->Accept(&analysis_));

  auto root = computation->root_instruction();

  // Cost Model estimates that input element we be re-read in reduce. Each
  // element of reduce output needs only one input element. Bytes accessed
  // should be 4*128=512.
  EXPECT_EQ(GpuPerformanceModelBase::GetOperandBytesAccessed(&analysis_, root,
                                                             root->operand(0)),
            /*4*128*256=*/131072);
}

// This test documents current behaviour. See comments below how the correct
// result should look like.
TEST_F(GpuPerformanceModelBaseTest, ElementwiseBitcast_IncorrectBytesAccessed) {
  std::string_view hlo_string = R"(
HloModule m

f1 {
  p0 = s32[128] parameter(0)
  bitcast.1 = s32[8,16] bitcast(p0)
  negate = s32[128] negate(p0)
  bitcast.2 = s32[8,16] bitcast(negate)
  ROOT add = s32[8,16] add(bitcast.1, bitcast.2)
}

ENTRY entry_computation {
  param_0 = s32[128] parameter(0)
  ROOT fusion = s32[8,16] fusion(param_0), kind=kLoop, calls=f1
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto computation = module->entry_computation();
  TF_ASSERT_OK(computation->Accept(&analysis_));

  auto root = computation->root_instruction();

  // Bitcast breaks the chain of elementwise utilization even if the bitcast
  // doesn't change physical layout. Each element of `param_0` should be read
  // only once, but Cost Model estimates that it will be accessed twice. Bytes
  // accessed should be 4*128=512.
  EXPECT_EQ(GpuPerformanceModelBase::GetOperandBytesAccessed(&analysis_, root,
                                                             root->operand(0)),
            /*2*4*128=*/1024);
}

TEST_F(GpuPerformanceModelBaseTest, EstimateFusionLaunchDimensions_LoopFusion) {
  std::string_view hlo_string = R"(
HloModule m

f1 {
  p0 = s32[8,16,128] parameter(0)
  negate = s32[8,16,128] negate(p0)
  ROOT add = s32[8,16,128] add(p0, negate)
}

ENTRY entry_computation {
  param_0 = s32[8,16,128] parameter(0)
  ROOT fusion = s32[8,16,128] fusion(param_0), kind=kLoop, calls=f1
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto fusion_analysis = HloFusionAnalysis::Create(
      *module->entry_computation()->root_instruction(), device_info_);
  auto launch_dimensions =
      GpuPerformanceModelBase::EstimateFusionLaunchDimensions(fusion_analysis);

  EXPECT_EQ(launch_dimensions.num_blocks(), 128);
  EXPECT_EQ(launch_dimensions.num_threads_per_block(), 128);
}

// NOTE: TritonSoftMax and CuDNN fusion tests from XLA are not ported
// because Triton and cuDNN are not applicable to ZKX.

}  // namespace
}  // namespace zkx::gpu
