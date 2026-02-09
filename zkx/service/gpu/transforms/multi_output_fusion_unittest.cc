/* Copyright 2018 The OpenXLA Authors.
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

#include "zkx/service/gpu/transforms/multi_output_fusion.h"

#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

#include "absl/strings/str_cat.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "zkx/hlo/ir/hlo_computation.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/hlo/ir/hlo_opcode.h"
#include "zkx/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "zkx/hlo/testlib/pattern_matcher_gmock.h"
#include "zkx/service/gpu/gpu_device_info_for_tests.h"
#include "zkx/service/gpu/gpu_fusible.h"
#include "zkx/service/hlo_cost_analysis.h"
#include "zkx/service/pattern_matcher.h"
#include "zkx/shape_util.h"

namespace zkx::gpu {

namespace m = ::zkx::match;

class MultiOutputFusionTest : public HloHardwareIndependentTestBase {
 public:
  MultiOutputFusion mof_{TestGpuDeviceInfo::RTXA6000DeviceInfo(),
                         HloCostAnalysis::DefaultShapeSize};

  void CheckMultiOutputFusion(std::string_view hlo,
                              std::optional<std::string_view> expected) {
    RunAndFilecheckHloRewrite(
        hlo,
        MultiOutputFusion{TestGpuDeviceInfo::RTXA6000DeviceInfo(),
                          HloCostAnalysis::DefaultShapeSize},
        expected);
  }
};

const char kModulePrefix[] = R"(
    HloModule test_module

    scalar_add_computation {
      scalar_lhs.0 = s32[] parameter(0)
      scalar_rhs.0 = s32[] parameter(1)
      ROOT add.0 = s32[] add(scalar_lhs.0, scalar_rhs.0)
    }
    scalar_mul_computation {
      scalar_lhs.1 = s32[] parameter(0)
      scalar_rhs.1 = s32[] parameter(1)
      ROOT mul.1 = s32[] multiply(scalar_lhs.1, scalar_rhs.1)
    })";

static int64_t CountMultiOutputFusions(const HloModule* module) {
  int multi_output_fusion_count = 0;
  for (auto* computation : module->MakeNonfusionComputations()) {
    for (auto* instr : computation->instructions()) {
      if (instr->IsMultiOutputFusion()) {
        multi_output_fusion_count++;
      }
    }
  }
  return multi_output_fusion_count;
}

TEST_F(MultiOutputFusionTest, MultiOutputFusionSiblingReduceAndReduceFusion) {
  // Fusion with reduce instruction root and a sibling reduce instruction
  // sharing the same input param.
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation {
      p1.1 = s32[128,512,28,28]{3,2,1,0} parameter(1)
      mul = s32[128,512,28,28]{3,2,1,0} multiply(p1.1, p1.1)
      const.1 = s32[] parameter(0)
      ROOT reduce.1 = s32[512]{0} reduce(mul, const.1), dimensions={0,2,3}, to_apply=scalar_add_computation
    }

    ENTRY entry {
      p0 = s32[] parameter(0)
      p1 = s32[128,512,28,28]{3,2,1,0} parameter(1)
      const.2 = s32[] constant(1)
      fusion = s32[512] fusion(p0, p1), kind=kInput, calls=fused_computation
      reduce.2 = s32[512]{0} reduce(p1, const.2), dimensions={0,2,3}, to_apply=scalar_add_computation
      ROOT root = (s32[512]{0}, s32[512]{0}) tuple(fusion, reduce.2)
    })"))
                    .value();
  ASSERT_TRUE(mof_.Run(module.get()).value());
  SCOPED_TRACE(module->ToString());
  const HloInstruction* fusion =
      module->entry_computation()->root_instruction()->operand(0)->operand(0);
  ASSERT_TRUE(fusion->IsMultiOutputFusion());
  EXPECT_THAT(fusion->fused_expression_root(),
              GmockMatch(m::Tuple(m::Reduce(), m::Reduce())));
}

TEST_F(MultiOutputFusionTest, MultiOutputFusionDifferentReduceInputShapes) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation_1 {
      p1.1 = s32[6400]{0} parameter(1)
      mul = s32[6400]{0} multiply(p1.1, p1.1)
      const.1 = s32[] parameter(0)
      ROOT reduce.1 = s32[] reduce(mul, const.1), dimensions={0}, to_apply=scalar_add_computation
    }

    fused_computation_2 {
      p1.2 = s32[6400]{0} parameter(1)
      r1 = s32[64,100]{0,1} reshape(p1.2)
      const.2 = s32[] parameter(0)
      ROOT reduce.2 = s32[] reduce(r1, const.2), dimensions={1,0}, to_apply=scalar_mul_computation
    }

    ENTRY entry {
      p0 = s32[] parameter(0)
      p1 = s32[6400]{0} parameter(1)
      fusion.1 = s32[] fusion(p0, p1), kind=kInput, calls=fused_computation_1
      fusion.2 = s32[] fusion(p0, p1), kind=kInput, calls=fused_computation_2
      ROOT root = (s32[], s32[]) tuple(fusion.1, fusion.2)
    })"))
                    .value();
  ASSERT_FALSE(mof_.Run(module.get()).value());
}

TEST_F(MultiOutputFusionTest, ReduceMofDifferentTypes) {
  // Fusion with reduce instruction root and a sibling reduce instruction
  // sharing the same input param.
  const char* hlo = R"(
HloModule module

scalar_add_computation {
  scalar_lhs.1 = s32[] parameter(0)
  scalar_rhs.1 = s32[] parameter(1)
  ROOT add.1 = s32[] add(scalar_lhs.1, scalar_rhs.1)
}

scalar_add_computation_s16 {
  scalar_lhs.0 = s16[] parameter(0)
  scalar_rhs.0 = s16[] parameter(1)
  ROOT add.0 = s16[] add(scalar_lhs.0, scalar_rhs.0)
}

fused_computation {
  param_0.2 = s32[128,512,28,28]{3,2,1,0} parameter(0)
  c.1 = s16[128,512,28,28]{3,2,1,0} convert(param_0.2)
  const.0 = s16[] constant(0)
  ROOT reduce.0 = s16[512]{0} reduce(c.1, const.0), dimensions={0,2,3}, to_apply=scalar_add_computation_s16
}

ENTRY entry {
  p0 = s32[] parameter(0)
  p1 = s32[128,512,28,28]{3,2,1,0} parameter(1)
  const.2 = s32[] constant(0)
  reduce.1 = s32[512]{0} reduce(p1, const.2), dimensions={0,2,3}, to_apply=scalar_add_computation
  fusion = s16[512]{0} fusion(p1), kind=kInput, calls=fused_computation
  ROOT root = (s32[512]{0}, s16[512]{0}) tuple(reduce.1, fusion)
})";

  CheckMultiOutputFusion(hlo, R"(
// CHECK: %fused_computation
// CHECK-NEXT:   [[param_0_2_0:%[^ ]+]] = s32[128,512,28,28]{3,2,1,0} parameter(0)
// CHECK-NEXT:   [[c_1_1:%[^ ]+]] = s16[128,512,28,28]{3,2,1,0} convert([[param_0_2_0]])
// CHECK-NEXT:   [[const_0_2:%[^ ]+]] = s16[] constant(0)
// CHECK-NEXT:   [[reduce_0_3:%[^ ]+]] = s16[512]{0} reduce([[c_1_1]], [[const_0_2]]), dimensions={0,2,3}, to_apply=[[scalar_add_computation_s16_4:%[^ ]+]]
// CHECK-NEXT:   [[param_1_5:%[^ ]+]] = s32[] parameter(1)
// CHECK-NEXT:   [[reduce_2_6:%[^ ]+]] = s32[512]{0} reduce([[param_0_2_0]], [[param_1_5]]), dimensions={0,2,3}, to_apply=[[scalar_add_computation_7:%[^ ]+]]
// CHECK-NEXT:   ROOT [[tuple_8:%[^ ]+]] = (s16[512]{0}, s32[512]{0}) tuple([[reduce_0_3]], [[reduce_2_6]])
// CHECK:   [[fusion_9:%[^ ]+]] = (s16[512]{0}, s32[512]{0}) fusion([[p1_10:%[^ ]+]], [[const_2_11:%[^ ]+]]), kind=kInput, calls=[[fused_computation_12:%[^ ]+]]
)");
}

TEST_F(MultiOutputFusionTest, MultiOutputFusionDifferentReduceOutputShapes) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation_1 {
      p1.1 = s32[10,10]{1,0} parameter(1)
      mul = s32[10,10]{1,0} multiply(p1.1, p1.1)
      const.1 = s32[] parameter(0)
      ROOT reduce.1 = s32[] reduce(mul, const.1), dimensions={0,1}, to_apply=scalar_add_computation
    }

    fused_computation_2 {
      p1.2 = s32[10,10]{1,0} parameter(1)
      const.2 = s32[] parameter(0)
      ROOT reduce.2 = s32[10]{0} reduce(p1.2, const.2), dimensions={0}, to_apply=scalar_mul_computation
    }

    ENTRY entry {
      p0 = s32[] parameter(0)
      p1.3 = s32[10,10]{1,0} parameter(1)
      fusion.1 = s32[] fusion(p0, p1.3), kind=kInput, calls=fused_computation_1
      p2 = s32[] parameter(2)
      fusion.2 = s32[10]{0} fusion(p2, p1.3), kind=kInput, calls=fused_computation_2
      ROOT root = (s32[], s32[10]{0}) tuple(fusion.1, fusion.2)
    })"))
                    .value();
  ASSERT_FALSE(mof_.Run(module.get()).value());
}

TEST_F(MultiOutputFusionTest, MultiOutputFusionSiblingReduceFusions) {
  // Two sibling fusions with reduce instruction roots sharing the same input
  // param.
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation_1 {
      p1.1 = s32[128,512,28,28]{3,2,1,0} parameter(1)
      mul = s32[128,512,28,28]{3,2,1,0} multiply(p1.1, p1.1)
      const.1 = s32[] parameter(0)
      ROOT reduce.1 = s32[512]{0} reduce(mul, const.1), dimensions={0,2,3}, to_apply=scalar_add_computation
    }

    fused_computation_2 {
      p1.2 = s32[128,512,28,28]{3,2,1,0} parameter(1)
      const.2 = s32[] parameter(0)
      ROOT reduce.2 = s32[512]{0} reduce(p1.2, const.2), dimensions={0,2,3}, to_apply=scalar_add_computation
    }

    ENTRY entry {
      p0 = s32[] parameter(0)
      p1 = s32[128,512,28,28]{3,2,1,0} parameter(1)
      fusion.1 = s32[512] fusion(p0, p1), kind=kInput, calls=fused_computation_1
      fusion.2 = s32[512] fusion(p0, p1), kind=kInput, calls=fused_computation_2
      ROOT root = (s32[512]{0}, s32[512]{0}) tuple(fusion.1, fusion.2)
    })"))
                    .value();
  ASSERT_TRUE(mof_.Run(module.get()).value());
  SCOPED_TRACE(module->ToString());
  const HloInstruction* fusion =
      module->entry_computation()->root_instruction()->operand(0)->operand(0);
  ASSERT_TRUE(fusion->IsMultiOutputFusion());
  EXPECT_THAT(fusion->fused_expression_root(),
              GmockMatch(m::Tuple(m::Reduce(), m::Reduce())));
}

TEST_F(MultiOutputFusionTest, MultiOutputFusionNoSiblingFusionForCommonScalar) {
  // Two sibling fusions with bitcast roots sharing the same scalar input param.
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation_1 {
      param_0.87 = s16[32,4096,16384]{2,1,0} parameter(0)
      param_1.4620 = s32[] parameter(1)
      constant_3949 = s32[] constant(0)
      compare.1026 = pred[] compare(param_1.4620, constant_3949), direction=LT
      constant_5437 = s32[] constant(32)
      add.6859 = s32[] add(param_1.4620, constant_5437)
      select.1599 = s32[] select(compare.1026, add.6859, param_1.4620)
      dynamic-slice.59 = s16[1,4096,16384]{2,1,0} dynamic-slice(param_0.87, select.1599, constant_3949, constant_3949), dynamic_slice_sizes={1,4096,16384}
      ROOT bitcast.41089 = s16[4096,16384]{1,0} bitcast(dynamic-slice.59)
    }

    fused_computation_2 {
      param_0 = s16[32,4096,16384]{2,1,0} parameter(0)
      param_1 = s32[] parameter(1)
      constant = s32[] constant(0)
      compare = pred[] compare(param_1, constant), direction=LT
      constant.32 = s32[] constant(32)
      add = s32[] add(param_1, constant.32)
      select = s32[] select(compare, add, param_1)
      dynamic-slice = s16[1,4096,16384]{2,1,0} dynamic-slice(param_0, select, constant, constant), dynamic_slice_sizes={1,4096,16384}
      ROOT bitcast.41087 = s16[4096,16384]{1,0} bitcast(dynamic-slice)
    }

    ENTRY entry {
      p0 = s32[] parameter(0)
      p1 = s16[32,4096,16384]{2,1,0} parameter(1)
      p2 = s16[32,4096,16384]{2,1,0} parameter(2)
      fusion.1 = s16[4096,16384]{1,0} fusion(p1, p0), kind=kLoop, calls=fused_computation_1
      fusion.2 = s16[4096,16384]{1,0} fusion(p2, p0), kind=kLoop, calls=fused_computation_2
      ROOT root = (s16[4096,16384]{1,0}, s16[4096,16384]{1,0}) tuple(fusion.1, fusion.2)
    })"))
                    .value();
  ASSERT_FALSE(mof_.Run(module.get()).value());
}

TEST_F(MultiOutputFusionTest,
       MultiOutputFusionSiblingReduceAndReduceMultiOutputFusion) {
  // Multi-output fusion with two reduce instructions root and a sibling reduce
  // instruction sharing the same input param.
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation (p0: s32[128,512,28,28]) -> (s32[512], s32[512]) {
      const.1 = s32[] constant(1)
      p0.1 = s32[128,512,28,28]{3,2,1,0} parameter(0)
      mul = s32[128,512,28,28]{3,2,1,0} multiply(s32[128,512,28,28]{3,2,1,0} p0.1, s32[128,512,28,28]{3,2,1,0} p0.1)
      reduce.1 = s32[512]{0} reduce(s32[128,512,28,28]{3,2,1,0} mul, s32[] const.1), dimensions={0,2,3}, to_apply=scalar_add_computation
      reduce.2 = s32[512]{0} reduce(s32[128,512,28,28]{3,2,1,0} p0.1, s32[] const.1), dimensions={0,2,3}, to_apply=scalar_add_computation
      ROOT tuple = (s32[512]{0}, s32[512]{0}) tuple(s32[512]{0} reduce.1, s32[512]{0} reduce.2)
    }

    ENTRY entry (p0: s32[128,512,28,28]) -> (s32[512], s32[512], s32[512]) {
      p0 = s32[128,512,28,28]{3,2,1,0} parameter(0)
      const = s32[] constant(1)
      fusion = (s32[512]{0}, s32[512]{0}) fusion(s32[128,512,28,28]{3,2,1,0} p0), kind=kInput, calls=fused_computation
      get-tuple-element = s32[512]{0} get-tuple-element((s32[512]{0}, s32[512]{0}) fusion), index=0
      get-tuple-element.1 = s32[512]{0} get-tuple-element((s32[512]{0}, s32[512]{0}) fusion), index=1
      reduce.3 = s32[512]{0} reduce(p0, const), dimensions={0,2,3}, to_apply=scalar_add_computation
      ROOT root = (s32[512]{0}, s32[512]{0}, s32[512]{0}) tuple(s32[512]{0} get-tuple-element, s32[512]{0} get-tuple-element.1, s32[512]{0} reduce.3)
    })"))
                    .value();
  ASSERT_TRUE(mof_.Run(module.get()).value());
  SCOPED_TRACE(module->ToString());
  const HloInstruction* fusion =
      module->entry_computation()->root_instruction()->operand(0)->operand(0);
  ASSERT_TRUE(fusion->IsMultiOutputFusion());
  EXPECT_THAT(fusion->fused_expression_root(),
              GmockMatch(m::Tuple(m::Reduce(), m::Reduce(), m::Reduce())));
}

TEST_F(MultiOutputFusionTest,
       MultiOutputFusionSiblingFusionCheckAgainstReduceOperand) {
  // Verify that if we already have a multi-output fusion that we prefer to pick
  // a reduce op from its operands for checking shape compatibility.
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation_1 {
      p1.1 = s32[10,10]{1,0} parameter(1)
      mul = s32[10,10]{1,0} multiply(p1.1, p1.1)
      const.1 = s32[] parameter(0)
      reduce.1 = s32[] reduce(p1.1, const.1), dimensions={0,1}, to_apply=scalar_add_computation
      ROOT tuple = (s32[10,10], s32[]) tuple(mul, reduce.1)
    }

    fused_computation_2 {
      p1.2 = s32[10,10]{1,0} parameter(1)
      const.2 = s32[] parameter(0)
      ROOT reduce.2 = s32[10] reduce(p1.2, const.2), dimensions={0}, to_apply=scalar_mul_computation
    }

    ENTRY entry {
      p0 = s32[] parameter(0)
      p1 = s32[10,10]{1,0} parameter(1)
      p2 = s32[] parameter(2)
      fusion.1 = (s32[10,10], s32[]) fusion(p0, p1), kind=kInput, calls=fused_computation_1
      get-tuple-element.1 = s32[10,10] get-tuple-element((s32[10,10], s32[]) fusion.1), index=0
      get-tuple-element.2 = s32[] get-tuple-element((s32[10,10], s32[]) fusion.1), index=1
      fusion.2 = s32[10] fusion(p2, p1), kind=kInput, calls=fused_computation_2
      ROOT root = (s32[10,10], s32[], s32[10]) tuple(get-tuple-element.1, get-tuple-element.2, fusion.2)
    })"))
                    .value();
  ASSERT_FALSE(mof_.Run(module.get()).value());
}

TEST_F(MultiOutputFusionTest, LoopVariadicReductionFusions) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation.94 {
      tmp_0 = s32[] parameter(0)
      tmp_1 = s32[] parameter(1)
      tmp_2 = pred[] compare(tmp_0, tmp_1), direction=GE
      tmp_3 = s32[] select(tmp_2, tmp_0, tmp_1)
      tmp_4 = pred[] compare(tmp_0, tmp_1), direction=EQ
      tmp_5 = s32[] parameter(2)
      tmp_6 = s32[] parameter(3)
      tmp_7 = s32[] minimum(tmp_5, tmp_6)
      tmp_8 = s32[] select(tmp_2, tmp_5, tmp_6)
      tmp_9 = s32[] select(tmp_4, tmp_7, tmp_8)
      ROOT tmp_10 = (s32[], s32[]) tuple(tmp_3, tmp_9)
    }

    minmax_func.1536 {
      tmp_0 = s32[] parameter(0)
      tmp_1 = s32[] parameter(2)
      tmp_2 = s32[] parameter(1)
      tmp_3 = s32[] parameter(3)
      ROOT tmp_4 = (s32[], s32[]) fusion(tmp_0, tmp_1, tmp_2, tmp_3), kind=kLoop, calls=fused_computation.94
    }

    fused_computation {
      tmp_0 = s32[554112,10]{1,0} parameter(0)
      tmp_1 = s32[554112,10]{1,0} iota(), iota_dimension=1
      tmp_2 = s32[] constant(-2147483648)
      tmp_3 = s32[] constant(0)
      ROOT tmp_4 = (s32[554112]{0}, s32[554112]{0}) reduce(tmp_0, tmp_1, tmp_2, tmp_3), dimensions={1}, to_apply=minmax_func.1536
    }

    fused_computation2 {
      tmp_0 = s32[554112,10]{1,0} parameter(0)
      tmp_1 = s32[554112,10]{1,0} iota(), iota_dimension=1
      tmp_2 = s32[] constant(2147483647)
      tmp_3 = s32[] constant(1)
      ROOT tmp_4 = (s32[554112]{0}, s32[554112]{0}) reduce(tmp_0, tmp_1, tmp_2, tmp_3), dimensions={1}, to_apply=minmax_func.1536
    }

    ENTRY e {
      tmp_0 = s32[554112,10]{1,0} parameter(0)
      tmp_1 = (s32[554112]{0}, s32[554112]{0}) fusion(tmp_0), kind=kLoop, calls=fused_computation
      tmp_2 = s32[554112]{0} get-tuple-element(tmp_1), index=1
      tmp_4 = (s32[554112]{0}, s32[554112]{0}) fusion(tmp_0), kind=kLoop, calls=fused_computation2
      tmp_5 = s32[554112]{0} get-tuple-element(tmp_4), index=1
      ROOT tmp_6 = s32[554112]{0} add(tmp_2, tmp_5)
    })"))
                    .value();
  EXPECT_FALSE(mof_.Run(module.get()).value());
}

TEST_F(MultiOutputFusionTest, InputVariadicReductionFusions) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation.1117 {
      param_0.2433 = s32[] parameter(0)
      param_1.2571 = s32[] parameter(1)
      compare.1770 = pred[] compare(param_0.2433, param_1.2571), direction=LE
      select.682 = s32[] select(compare.1770, param_0.2433, param_1.2571)
      compare.1303.clone.1 = pred[] compare(param_0.2433, param_1.2571), direction=EQ
      param_2.6460 = s32[] parameter(2)
      param_3.6755 = s32[] parameter(3)
      minimum.633.clone.1 = s32[] minimum(param_2.6460, param_3.6755)
      select.398.clone.1 = s32[] select(compare.1770, param_2.6460, param_3.6755)
      select.397.clone.1 = s32[] select(compare.1303.clone.1, minimum.633.clone.1, select.398.clone.1)
      ROOT tuple.151 = (s32[], s32[]) tuple(select.682, select.397.clone.1)
    }

    minmax_func.223 {
      lhs_value.224 = s32[] parameter(0)
      rhs_value.226 = s32[] parameter(2)
      lhs_index.225 = s32[] parameter(1)
      rhs_index.227 = s32[] parameter(3)
      ROOT fusion.1117 = (s32[], s32[]) fusion(lhs_value.224, rhs_value.226, lhs_index.225, rhs_index.227), kind=kLoop, calls=fused_computation.1117
    }

    fused_computation.73 {
      bitcast.86661 = s32[3,1024,300]{2,1,0} parameter(0)
      iota.734 = s32[3,1,1024,300]{3,2,1,0} iota(), iota_dimension=3
      bitcast.97555 = s32[3,1024,300]{2,1,0} bitcast(iota.734)
      constant_3917 = s32[] constant(2147483647)
      constant_3918 = s32[] constant(0)
      ROOT reduce.1069 = (s32[3,1024]{1,0}, s32[3,1024]{1,0}) reduce(bitcast.86661, bitcast.97555, constant_3917, constant_3918), dimensions={2}, to_apply=minmax_func.223
    }

    fused_computation.84 {
      bitcast.86676 = s32[3,1024,300]{2,1,0} parameter(0)
      iota.732 = s32[3,1,1024,300]{3,2,1,0} iota(), iota_dimension=3
      bitcast.97553 = s32[3,1024,300]{2,1,0} bitcast(iota.732)
      constant_3915 = s32[] constant(2147483647)
      constant_3916 = s32[] constant(0)
      ROOT reduce.1070 = (s32[3,1024]{1,0}, s32[3,1024]{1,0}) reduce(bitcast.86676, bitcast.97553, constant_3915, constant_3916), dimensions={2}, to_apply=minmax_func.223
    }

    ENTRY e {
      p0 = s32[3,1024,300]{2,1,0} parameter(0)
      fusion.84 = (s32[3,1024]{1,0}, s32[3,1024]{1,0}) fusion(p0), kind=kInput, calls=fused_computation.84
      gte.391 = s32[3,1024]{1,0} get-tuple-element(fusion.84), index=1
      fusion.73 = (s32[3,1024]{1,0}, s32[3,1024]{1,0}) fusion(p0), kind=kInput, calls=fused_computation.73
      gte.393 = s32[3,1024]{1,0} get-tuple-element(fusion.73), index=1
      ROOT r = s32[3,1024]{1,0} add(gte.391, gte.393)
    })"))
                    .value();
  EXPECT_TRUE(mof_.Run(module.get()).value());
  EXPECT_EQ(module->entry_computation()->parameter_instruction(0)->user_count(),
            1);
  const HloInstruction* fusion =
      module->entry_computation()->parameter_instruction(0)->users()[0];
  EXPECT_THAT(fusion, GmockMatch(m::Fusion()));
  EXPECT_THAT(fusion->fused_expression_root(),
              GmockMatch(m::Tuple(m::Reduce(), m::Reduce())));
}

TEST_F(MultiOutputFusionTest, MultiOutputFusionTwoLoops) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation_1 {
      p0.1 = s32[6400]{0} parameter(0)
      ROOT mul = s32[6400]{0} multiply(p0.1, p0.1)
    }

    fused_computation_2 {
      p0.2 = s32[6400]{0} parameter(0)
      const.2 = s32[] constant(1)
      broadcast = s32[6400]{0} broadcast(const.2), dimensions={}
      ROOT div = s32[6400]{0} divide(p0.2, broadcast)
    }

    ENTRY entry {
      p0 = s32[6400]{0} parameter(0)
      fusion.1 = s32[6400]{0} fusion(p0), kind=kLoop, calls=fused_computation_1
      fusion.2 = s32[6400]{0} fusion(p0), kind=kLoop, calls=fused_computation_2
      ROOT root = (s32[6400]{0}, s32[6400]{0}) tuple(fusion.1, fusion.2)
    })"))
                    .value();
  ASSERT_TRUE(mof_.Run(module.get()).value());
  SCOPED_TRACE(module->ToString());
  const HloInstruction* fusion =
      module->entry_computation()->root_instruction()->operand(0)->operand(0);
  ASSERT_TRUE(fusion->IsMultiOutputFusion());
  EXPECT_THAT(fusion->fused_expression_root(),
              GmockMatch(m::Tuple(m::Multiply(), m::Divide())));
}

TEST_F(MultiOutputFusionTest, MultiOutputFusionLoopElementwise) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation_1 {
      p0.1 = s32[6400]{0} parameter(0)
      ROOT mul = s32[6400]{0} multiply(p0.1, p0.1)
    }

    ENTRY entry {
      p0 = s32[6400]{0} parameter(0)
      fusion.1 = s32[6400]{0} fusion(p0), kind=kLoop, calls=fused_computation_1
      const.2 = s32[] constant(1)
      broadcast = s32[6400]{0} broadcast(const.2), dimensions={}
      div = s32[6400]{0} divide(p0, broadcast)
      ROOT root = (s32[6400]{0}, s32[6400]{0}) tuple(fusion.1, div)
    })"))
                    .value();
  ASSERT_TRUE(mof_.Run(module.get()).value());
  SCOPED_TRACE(module->ToString());
  const HloInstruction* fusion =
      module->entry_computation()->root_instruction()->operand(0)->operand(0);
  ASSERT_TRUE(fusion->IsMultiOutputFusion());
  EXPECT_THAT(fusion->fused_expression_root(),
              GmockMatch(m::Tuple(m::Multiply(), m::Divide())));
}

TEST_F(MultiOutputFusionTest, MultiOutputFusionSiblingLoopsDifferentShapes) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation_1 {
      p0.1 = s32[8,1,5,16,1,2]{5,4,3,2,1,0} parameter(0)
      ROOT mul = s32[8,1,5,16,1,2]{5,4,3,2,1,0} multiply(p0.1, p0.1)
    }

    fused_computation_2 {
      p0.2 = s32[8,1,5,16,1,2]{5,4,3,2,1,0} parameter(0)
      const.2 = s32[] constant(0)
      ROOT reduce = s32[1,5,1,2]{3,2,1,0} reduce(p0.2, const.2), dimensions={0,3}, to_apply=scalar_add_computation
    }

    ENTRY entry {
      p0 = s32[8,1,5,16,1,2]{5,4,3,2,1,0} parameter(0)
      fusion.1 = s32[8,1,5,16,1,2]{5,4,3,2,1,0} fusion(p0), kind=kLoop, calls=fused_computation_1
      fusion.2 = s32[1,5,1,2]{3,2,1,0} fusion(p0), kind=kLoop, calls=fused_computation_2
      ROOT root = (s32[8,1,5,16,1,2]{5,4,3,2,1,0}, s32[1,5,1,2]{3,2,1,0}) tuple(fusion.1, fusion.2)
    })"))
                    .value();
  ASSERT_FALSE(mof_.Run(module.get()).value());
}

TEST_F(MultiOutputFusionTest, MultiOutputFusionSiblingLoopAndMultiOutputLoop) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation_1 {
      p0.1 = s32[8,1,5,16,1,1]{5,4,3,2,1,0} parameter(0)
      mul = s32[8,1,5,16,1,1]{5,4,3,2,1,0} multiply(p0.1, p0.1)
      neg = s32[8,1,5,16,1,1]{5,4,3,2,1,0} negate(p0.1)
      ROOT tuple = (s32[8,1,5,16,1,1]{5,4,3,2,1,0},
        s32[8,1,5,16,1,1]{5,4,3,2,1,0}) tuple(mul, neg)
    }

    fused_computation_2 {
      p0.2 = s32[8,1,5,16,1,1]{5,4,3,2,1,0} parameter(0)
      const.2 = s32[] constant(0)
      broadcast = s32[8,1,5,16,1,1]{5,4,3,2,1,0} broadcast(const.2),
        dimensions={}
      ROOT add = s32[8,1,5,16,1,1]{5,4,3,2,1,0} add(p0.2, broadcast)
    }

    ENTRY entry {
      p0 = s32[8,1,5,16,1,1]{5,4,3,2,1,0} parameter(0)
      fusion.1 = (s32[8,1,5,16,1,1]{5,4,3,2,1,0},
        s32[8,1,5,16,1,1]{5,4,3,2,1,0}) fusion(p0), kind=kLoop,
        calls=fused_computation_1
      fusion.2 = s32[8,1,5,16,1,1]{5,4,3,2,1,0} fusion(p0), kind=kLoop,
        calls=fused_computation_2
      gte0 = s32[8,1,5,16,1,1]{5,4,3,2,1,0} get-tuple-element(fusion.1), index=0
      gte1 = s32[8,1,5,16,1,1]{5,4,3,2,1,0} get-tuple-element(fusion.1), index=1
      ROOT root = (s32[8,1,5,16,1,1]{5,4,3,2,1,0},
        s32[8,1,5,16,1,1]{5,4,3,2,1,0}, s32[8,1,5,16,1,1]{5,4,3,2,1,0})
        tuple(gte0, gte1, fusion.2)
    })"))
                    .value();
  ASSERT_TRUE(mof_.Run(module.get()).value());
  SCOPED_TRACE(module->ToString());
  const HloInstruction* fusion =
      module->entry_computation()->root_instruction()->operand(0)->operand(0);
  ASSERT_TRUE(fusion->IsMultiOutputFusion());
  EXPECT_THAT(fusion->fused_expression_root(),
              GmockMatch(m::Tuple(m::Multiply(), m::Negate(), m::Add())));
}

TEST_F(MultiOutputFusionTest,
       MultiOutputFusionSiblingMultiOutputLoopAndMultiOutputLoop) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation_1 {
      p0.1 = s32[8,16]{1,0} parameter(0)
      mul = s32[8,16]{1,0} multiply(p0.1, p0.1)
      neg = s32[8,16]{1,0} negate(p0.1)
      ROOT tuple = (s32[8,16]{1,0}, s32[8,16]{1,0}) tuple(mul, neg)
    }

    fused_computation_2 {
      p0.2 = s32[8,16]{1,0} parameter(0)
      const.2 = s32[] constant(0)
      broadcast = s32[8,16]{1,0} broadcast(const.2),
        dimensions={}
      add = s32[8,16]{1,0} add(p0.2, broadcast)
      ROOT tuple.1 = (s32[8,16]{1,0}, s32[8,16]{1,0}) tuple(add, broadcast)
    }

    ENTRY entry {
      p0 = s32[8,16]{1,0} parameter(0)
      fusion.1 = (s32[8,16]{1,0}, s32[8,16]{1,0}) fusion(p0), kind=kLoop,
        calls=fused_computation_1
      fusion.2 = (s32[8,16]{1,0}, s32[8,16]{1,0}) fusion(p0), kind=kLoop,
        calls=fused_computation_2
      gte0 = s32[8,16]{1,0} get-tuple-element(fusion.1), index=0
      gte1 = s32[8,16]{1,0} get-tuple-element(fusion.1), index=1
      gte2 = s32[8,16]{1,0} get-tuple-element(fusion.2), index=0
      gte3 = s32[8,16]{1,0} get-tuple-element(fusion.2), index=1
      ROOT root = (s32[8,16]{1,0}, s32[8,16]{1,0}, s32[8,16]{1,0},
        s32[8,16]{1,0})
        tuple(gte0, gte1, gte2, gte3)
    })"))
                    .value();
  ASSERT_TRUE(mof_.Run(module.get()).value());
  SCOPED_TRACE(module->ToString());
  const HloInstruction* fusion =
      module->entry_computation()->root_instruction()->operand(0)->operand(0);
  ASSERT_TRUE(fusion->IsMultiOutputFusion());
  EXPECT_THAT(fusion->fused_expression_root(),
              GmockMatch(m::Tuple(m::Multiply(), m::Negate(), m::Add(),
                                  m::Broadcast())));
}

TEST_F(MultiOutputFusionTest,
       MultiOutputFusionSiblingLoopAndMultiOutputLoopDifferentShapes) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation_1 {
      p0.1 = s32[8,1,5,16,1,2]{5,4,3,2,1,0} parameter(0)
      mul = s32[8,1,5,16,1,2]{5,4,3,2,1,0} multiply(p0.1, p0.1)
      neg = s32[8,1,5,16,1,2]{5,4,3,2,1,0} negate(p0.1)
      ROOT tuple = (s32[8,1,5,16,1,2]{5,4,3,2,1,0},
        s32[8,1,5,16,1,2]{5,4,3,2,1,0}) tuple(mul, neg)
    }

    fused_computation_2 {
      p0.2 = s32[8,1,5,16,1,2]{5,4,3,2,1,0} parameter(0)
      const.2 = s32[] constant(0)
      ROOT reduce = s32[1,5,1,2]{3,2,1,0} reduce(p0.2, const.2),
        dimensions={0,3}, to_apply=scalar_add_computation
    }

    ENTRY entry {
      p0 = s32[8,1,5,16,1,2]{5,4,3,2,1,0} parameter(0)
      fusion.1 = (s32[8,1,5,16,1,2]{5,4,3,2,1,0},
        s32[8,1,5,16,1,2]{5,4,3,2,1,0}) fusion(p0), kind=kLoop,
        calls=fused_computation_1
      fusion.2 = s32[1,5,1,2]{3,2,1,0} fusion(p0), kind=kLoop,
        calls=fused_computation_2
      gte0 = s32[8,1,5,16,1,2]{5,4,3,2,1,0} get-tuple-element(fusion.1), index=0
      gte1 = s32[8,1,5,16,1,2]{5,4,3,2,1,0} get-tuple-element(fusion.1), index=1
      ROOT root = (s32[8,1,5,16,1,2]{5,4,3,2,1,0},
        s32[8,1,5,16,1,2]{5,4,3,2,1,0}, s32[1,5,1,2]{3,2,1,0})
        tuple(gte0, gte1, fusion.2)
    })"))
                    .value();
  ASSERT_FALSE(mof_.Run(module.get()).value());
}

TEST_F(MultiOutputFusionTest, SiblingFusionBitcastAndLoopFusionNotFused) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule test

fused_computation_1 {
  p0.1 = s32[2048,16000]{1,0} parameter(0)
  bitcast = s32[2048,1,16000]{2,1,0} bitcast(p0.1)
  ROOT neg = s32[2048,1,16000]{2,1,0} negate(bitcast)
}

ENTRY main {
  param_0 = s32[2048,16000]{1,0} parameter(0)
  fusion = s32[2048,1,16000]{2,1,0} fusion(param_0), kind=kLoop, calls=fused_computation_1
  bitcast = s32[16000,1,2048]{2,1,0} bitcast(param_0)
  ROOT tuple.143 = (s32[16000,1,2048]{2,1,0}, s32[2048,1,16000]{2,1,0}) tuple(bitcast, fusion)
})")
                    .value();
  EXPECT_FALSE(mof_.Run(module.get()).value());
}

TEST_F(MultiOutputFusionTest,
       ProducerConsumerFusionBitcastAndElementwiseNotFused) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule test

ENTRY main {
  param_0 = s32[2048,16000]{1,0} parameter(0)
  convert = s16[2048,16000]{1,0} convert(param_0)
  bitcast = s16[16000,1,2048]{2,1,0} bitcast(convert)
  ROOT tuple.143 = (s16[16000,1,2048]{2,1,0}, s16[2048,16000]{1,0}) tuple(bitcast, convert)
})")
                    .value();
  EXPECT_FALSE(mof_.Run(module.get()).value());
}

TEST_F(MultiOutputFusionTest, ProducerConsumerFusionElementwiseAndReduce) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    ENTRY reduce {
      p0 = s32[32,32,32]{2,1,0} parameter(0)
      c0 = s32[] constant(0)
      neg = s32[32,32,32]{2,1,0} negate(p0)
      reduce = s32[32,32]{1,0} reduce(neg, c0), dimensions={2},
        to_apply=scalar_add_computation
      ROOT root = (s32[32,32]{1,0}, s32[32,32,32]{2,1,0}) tuple(reduce, neg)
    })"))
                    .value();
  ASSERT_TRUE(mof_.Run(module.get()).value());
  SCOPED_TRACE(module->ToString());
  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* fusion = nullptr;
  ASSERT_THAT(root, GmockMatch(m::Tuple(m::GetTupleElement(m::Fusion(&fusion)),
                                        m::GetTupleElement())));
  ASSERT_TRUE(fusion->IsMultiOutputFusion());
  EXPECT_THAT(fusion->fused_expression_root(),
              GmockMatch(m::Tuple(m::Reduce(), m::Negate())));
}

TEST_F(MultiOutputFusionTest, ProducerConsumerFusionLoopFusionAndReduce) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_add {
      p0.1 = s32[32,32,32]{2,1,0} parameter(0)
      p1.1 = s32[32,32,32]{2,1,0} parameter(1)
      ROOT add = s32[32,32,32]{2,1,0} add(p0.1, p1.1)
    }

    ENTRY reduce {
      p0 = s32[32,32,32]{2,1,0} parameter(0)
      p1 = s32[32,32,32]{2,1,0} parameter(1)
      c0 = s32[] constant(0)
      add = s32[32,32,32]{2,1,0} fusion(p0, p1), kind=kLoop, calls=fused_add
      reduce = s32[32,32]{1,0} reduce(add, c0), dimensions={2},
        to_apply=scalar_add_computation
      ROOT root = (s32[32,32]{1,0}, s32[32,32,32]{2,1,0}) tuple(reduce, add)
    })"))
                    .value();
  ASSERT_TRUE(mof_.Run(module.get()).value());
  SCOPED_TRACE(module->ToString());
  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* fusion = nullptr;
  ASSERT_THAT(root, GmockMatch(m::Tuple(m::GetTupleElement(m::Fusion(&fusion)),
                                        m::GetTupleElement())));
  ASSERT_TRUE(fusion->IsMultiOutputFusion());
  EXPECT_THAT(fusion->fused_expression_root(),
              GmockMatch(m::Tuple(m::Reduce(), m::Add())));
}

TEST_F(MultiOutputFusionTest, ProducerConsumerFusionLoopFusionAndReduceFusion) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_select {
      p1.1 = s32[32,32,32]{2,1,0} parameter(1)
      c0 = s32[] constant(0)
      broadcast = s32[32,32,32]{2,1,0} broadcast(s32[] c0), dimensions={}
      greater-than = pred[32,32,32]{2,1,0} compare(s32[32,32,32]{2,1,0} p1.1,
        s32[32,32,32]{2,1,0} broadcast), direction=GT
      p0.1 = s32[32,32,32]{2,1,0} parameter(0)
      ROOT select = s32[32,32,32]{2,1,0} select(pred[32,32,32]{2,1,0}
        greater-than, s32[32,32,32]{2,1,0} p0.1, s32[32,32,32]{2,1,0} broadcast)
    }

    fused_reduce {
      p0.2 = s32[32,32,32]{2,1,0} parameter(0)
      c1 = s32[] constant(0)
      r1 = s32[32,32]{1,0} reduce(p0.2, c1), dimensions={2},
        to_apply=scalar_add_computation
      mul = s32[32,32,32]{2,1,0} multiply(p0.2, p0.2)
      r2 = s32[32,32]{1,0} reduce(mul, c1), dimensions={2},
        to_apply=scalar_add_computation
      ROOT tuple = (s32[32,32]{1,0}, s32[32,32]{1,0}) tuple(r1, r2)
    }

    ENTRY reduce {
      p0 = s32[32,32,32]{2,1,0} parameter(0)
      p1 = s32[32,32,32]{2,1,0} parameter(1)
      select = s32[32,32,32]{2,1,0} fusion(p0, p1), kind=kLoop, calls=fused_select
      fusion = (s32[32,32]{1,0}, s32[32,32]{1,0}) fusion(select), kind=kInput,
        calls=fused_reduce
      gte0 = s32[32,32]{1,0} get-tuple-element(fusion), index=0
      gte1 = s32[32,32]{1,0} get-tuple-element(fusion), index=1
      ROOT root = (s32[32,32]{1,0}, s32[32,32]{1,0}, s32[32,32,32]{2,1,0})
        tuple(gte1, gte1, select)
    })"))
                    .value();
  ASSERT_TRUE(mof_.Run(module.get()).value());
  SCOPED_TRACE(module->ToString());
  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* fusion = nullptr;
  ASSERT_THAT(root,
              GmockMatch(m::Tuple(m::GetTupleElement(m::Fusion(&fusion)),
                                  m::GetTupleElement(), m::GetTupleElement())));
  ASSERT_TRUE(fusion->IsMultiOutputFusion());
  EXPECT_THAT(fusion->fused_expression_root(),
              GmockMatch(m::Tuple(m::Reduce(), m::Reduce(), m::Select())));
}

TEST_F(MultiOutputFusionTest, ProducerConsumerFusionDoNotFuseLoopReduceFusion) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_element_wise {
      p0.1 = s32[2,2,2]{2,1,0} parameter(0)
      p1.1 = s32[2,2,2]{2,1,0} parameter(1)
      ROOT root = s32[2,2,2]{2,1,0} add(p0.1, p1.1)
    }

    fused_reduce {
      p0.2 = s32[2,2,2]{2,1,0} parameter(0)
      mul = s32[2,2,2]{2,1,0} multiply(s32[2,2,2]{2,1,0} p0.2,
        s32[2,2,2]{2,1,0} p0.2)
      broadcast = s32[2,2,2,2]{3,2,1,0} broadcast(mul), dimensions={3,2,1}
      c1 = s32[] constant(0)
      ROOT reduce = s32[2,2]{1,0} reduce(s32[2,2,2,2]{3,2,1,0} broadcast,
        s32[] c1), dimensions={1,3}, to_apply=scalar_add_computation
    }

    ENTRY reduce {
      p0 = s32[2,2,2]{2,1,0} parameter(0)
      p1 = s32[2,2,2]{2,1,0} parameter(1)
      element_wise = s32[2,2,2]{2,1,0} fusion(p0, p1), kind=kLoop, calls=fused_element_wise
      fusion = s32[2,2]{1,0} fusion(element_wise), kind=kLoop, calls=fused_reduce
      ROOT root = (s32[2,2]{1,0}, s32[2,2,2]{2,1,0}) tuple(fusion, element_wise)
    })"))
                    .value();
  ASSERT_FALSE(mof_.Run(module.get()).value());
}

TEST_F(MultiOutputFusionTest,
       ProducerConsumerFusionS16LoopFusionAndReduceFusion) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_select {
      p1.1 = s16[32,32,32]{2,1,0} parameter(1)
      c0 = s16[] constant(0)
      broadcast = s16[32,32,32]{2,1,0} broadcast(s16[] c0), dimensions={}
      greater-than = pred[32,32,32]{2,1,0} compare(s16[32,32,32]{2,1,0} p1.1,
        s16[32,32,32]{2,1,0} broadcast), direction=GT
      p0.1 = s16[32,32,32]{2,1,0} parameter(0)
      ROOT select = s16[32,32,32]{2,1,0} select(pred[32,32,32]{2,1,0}
        greater-than, s16[32,32,32]{2,1,0} p0.1, s16[32,32,32]{2,1,0} broadcast)
    }
    fused_reduce {
      p0.2 = s16[32,32,32]{2,1,0} parameter(0)
      convert = s32[32,32,32]{2,1,0} convert(p0.2)
      c1 = s32[] constant(0)
      r1 = s32[32,32]{1,0} reduce(convert, c1), dimensions={2},
        to_apply=scalar_add_computation
      mul = s32[32,32,32]{2,1,0} multiply(convert, convert)
      r2 = s32[32,32]{1,0} reduce(mul, c1), dimensions={2},
        to_apply=scalar_add_computation
      ROOT tuple = (s32[32,32]{1,0}, s32[32,32]{1,0}) tuple(r1, r2)
    }
    ENTRY reduce {
      p0 = s16[32,32,32]{2,1,0} parameter(0)
      p1 = s16[32,32,32]{2,1,0} parameter(1)
      select = s16[32,32,32]{2,1,0} fusion(p0, p1), kind=kLoop, calls=fused_select
      fusion = (s32[32,32]{1,0}, s32[32,32]{1,0}) fusion(select), kind=kInput,
        calls=fused_reduce
      gte0 = s32[32,32]{1,0} get-tuple-element(fusion), index=0
      gte1 = s32[32,32]{1,0} get-tuple-element(fusion), index=1
      ROOT root = (s32[32,32]{1,0}, s32[32,32]{1,0}, s16[32,32,32]{2,1,0})
        tuple(gte1, gte1, select)
    })"))
                    .value();
  ASSERT_TRUE(mof_.Run(module.get()).value());
  SCOPED_TRACE(module->ToString());
  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* fusion = nullptr;
  ASSERT_THAT(root,
              GmockMatch(m::Tuple(m::GetTupleElement(m::Fusion(&fusion)),
                                  m::GetTupleElement(), m::GetTupleElement())));
  ASSERT_TRUE(fusion->IsMultiOutputFusion());
  EXPECT_THAT(fusion->fused_expression_root(),
              GmockMatch(m::Tuple(m::Reduce(), m::Reduce(), m::Select())));
}

TEST_F(MultiOutputFusionTest,
       ProducerConsumerFusionReduceUnfriendlyLoopFusion) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    mixed_input_layouts_computation {
      p0.1 = s16[128,32,32,1024]{3,2,1,0} parameter(0)
      p1.1 = s16[128,1024,32,32]{3,2,1,0} parameter(1)
      transpose = s16[128,32,32,1024]{3,2,1,0} transpose(p1.1), dimensions={0,2,3,1}
      c0 = s16[] constant(0)
      broadcast = s16[128,32,32,1024]{3,2,1,0} broadcast(c0), dimensions={}
      greater-than = pred[128,32,32,1024]{3,2,1,0} compare(transpose, broadcast), direction=GT
      ROOT root = s16[128,32,32,1024]{3,2,1,0} select(greater-than, p0.1, broadcast)
    }
    fused_reduce {
      p0.2 = s16[128,32,32,1024]{3,2,1,0} parameter(0)
      convert = s32[128,32,32,1024]{3,2,1,0} convert(p0.2)
      c0.2 = s32[] constant(0)
      ROOT reduce = s32[1024]{0} reduce(convert, c0.2), dimensions={0,1,2}, to_apply=scalar_add_computation
    }
    ENTRY reduce {
      p0 = s16[128,32,32,1024]{3,2,1,0} parameter(0)
      p1 = s16[128,1024,32,32]{3,2,1,0} parameter(1)
      loop_fusion = s16[128,32,32,1024]{3,2,1,0} fusion(p0, p1), kind=kLoop, calls=mixed_input_layouts_computation
      reduce_fusion = s32[1024]{0} fusion(loop_fusion), kind=kInput, calls=fused_reduce
      ROOT root = (s32[1024]{0}, s16[128,32,32,1024]{3,2,1,0}) tuple(reduce_fusion, loop_fusion)
    })"))
                    .value();
  ASSERT_FALSE(mof_.Run(module.get()).value());
}

TEST_F(MultiOutputFusionTest, ProducerConsumerFusionAvoidsCycles) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_add {
      p0 = s32[32,32,32]{2,1,0} parameter(0)
      p1 = s32[32,32,32]{2,1,0} parameter(1)
      ROOT add = s32[32,32,32]{2,1,0} add(p0, p1)
    }

    fused_mul {
      p2 = s32[64,64,64]{2,1,0} parameter(0)
      p3 = s32[64,64,64]{2,1,0} parameter(1)
      ROOT multiply = s32[64,64,64]{2,1,0} multiply(p2, p3)
    }

    fused_reduce_1 {
      p4 = s32[32,32,32]{2,1,0} parameter(0)
      p5 = s32[64,64,64]{2,1,0} parameter(1)
      slice = s32[32,32,32]{2,1,0} slice(p5), slice={[0:32], [0:32], [0:32]}
      add = s32[32,32,32]{2,1,0} add(p4, slice)
      c0 = s32[] constant(0)
      ROOT r1 = s32[32,32]{1,0} reduce(add, c0), dimensions={2},
        to_apply=scalar_add_computation
    }

    fused_reduce_2 {
      p6 = s32[32,32,32]{2,1,0} parameter(0)
      p7 = s32[64,64,64]{2,1,0} parameter(1)
      c0 = s32[] constant(0)
      pad = s32[64,64,64]{2,1,0} pad(p6, c0), padding=16_16x16_16x16_16
      mul = s32[64,64,64]{2,1,0} multiply(pad, p7)
      ROOT r1 = s32[64,64]{1,0} reduce(mul, c0), dimensions={2},
        to_apply=scalar_add_computation
    }

    ENTRY reduce {
      p8 = s32[32,32,32]{2,1,0} parameter(0)
      p9 = s32[64,64,64]{2,1,0} parameter(1)
      // `add` and `mul` can be multi-output fused with `reduce1` and `reduce2`,
      // respectively. However, both isn't possible, because multi-output fusion
      // will introduce an extra dependency from `neg` to `abs` or vice versa.
      // Hence, the second multi-output fusion would introduce a cycle.
      add = s32[32,32,32]{2,1,0} fusion(p8, p8), kind=kLoop, calls=fused_add
      mul = s32[64,64,64]{2,1,0} fusion(p9, p9), kind=kLoop, calls=fused_mul

      reduce1 = s32[32,32]{1,0} fusion(add, mul), kind=kInput,
          calls=fused_reduce_1
      reduce2 = s32[64,64]{1,0} fusion(add, mul), kind=kInput,
          calls=fused_reduce_2
      ROOT root = (s32[32,32,32]{2,1,0}, s32[32,32]{1,0}, s32[64,64]{1,0},
                   s32[64,64,64]{2,1,0}) tuple(add, reduce1, reduce2, mul)
    })"))
                    .value();
  ASSERT_TRUE(mof_.Run(module.get()).value());
  SCOPED_TRACE(module->ToString());
  EXPECT_EQ(1, CountMultiOutputFusions(module.get()));
}

TEST_F(MultiOutputFusionTest, PreferFuseProducerIntoFusionConsumer) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_add {
      p0 = s32[32,32,32]{2,1,0} parameter(0)
      p1 = s32[32,32,32]{2,1,0} parameter(1)
      ROOT add = s32[32,32,32]{2,1,0} add(p0, p1)
    }
    fused_reduce {
      p0 = s32[32,32,32]{2,1,0} parameter(0)
      p1 = s32[64,64,64]{2,1,0} parameter(1)
      slice = s32[32,32,32]{2,1,0} slice(p1), slice={[0:32], [0:32], [0:32]}
      add = s32[32,32,32]{2,1,0} add(p0, slice)
      c0 = s32[] constant(0)
      ROOT r1 = s32[32,32]{1,0} reduce(add, c0), dimensions={2},
        to_apply=scalar_add_computation
    }
    ENTRY reduce {
      p0 = s32[32,32,32]{2,1,0} parameter(0)
      p1 = s32[64,64,64]{2,1,0} parameter(1)
      add = s32[32,32,32]{2,1,0} fusion(p0, p0), kind=kLoop, calls=fused_add
      c0 = s32[] constant(0)
      reduce2 = s32[32,32]{1,0} reduce(add, c0), dimensions={2},
        to_apply=scalar_add_computation
      reduce = s32[32,32]{1,0} fusion(add, p1), kind=kInput, calls=fused_reduce
      ROOT root = (s32[32,32,32]{2,1,0}, s32[32,32]{1,0}, s32[32,32]{1,0})
                  tuple(add, reduce, reduce2)
    })"))
                    .value();
  ASSERT_TRUE(mof_.Run(module.get()).value());
  SCOPED_TRACE(module->ToString());
  int multi_output_fusion_count = 0;
  for (auto* computation : module->MakeNonfusionComputations()) {
    for (auto* instr : computation->instructions()) {
      if (instr->IsMultiOutputFusion()) {
        multi_output_fusion_count++;
      }
    }
  }
  EXPECT_EQ(1, multi_output_fusion_count);
}

// Check that we limit the number of operands to fusions we create.
TEST_F(MultiOutputFusionTest, AvoidsLargeFusion) {
  constexpr int64_t kNumParams = 200;
  ASSERT_GT(kNumParams, MaxOperandsAndOutputsPerFusion());

  // Compute
  //   p0 * p1,
  //   p0 * p1 + p1 * p2
  //   p0 * p1 + p1 * p2 + p2 * p3
  //   ...
  // where each of the (pi * pj)'s is represented as a fusion node so that
  // multi-output fusion will pay attention to it.
  auto module = CreateNewVerifiedModule();
  HloComputation::Builder b(TestName());
  Shape shape = ShapeUtil::MakeShape(S32, {10, 100});

  std::vector<HloInstruction*> params;
  for (int64_t i = 0; i < kNumParams; ++i) {
    params.push_back(
        b.AddInstruction(HloInstruction::CreateParameter(i, shape, "p")));
  }

  // Creates a fusion node that calculates x*y.
  auto make_fusion = [&](HloInstruction* x, HloInstruction* y) {
    HloComputation::Builder sub_builder("subcomp");
    auto* p0 = sub_builder.AddInstruction(
        HloInstruction::CreateParameter(0, shape, "p"));
    auto* p1 = sub_builder.AddInstruction(
        HloInstruction::CreateParameter(1, shape, "p"));
    sub_builder.AddInstruction(
        HloInstruction::CreateBinary(shape, HloOpcode::kMultiply, p0, p1));
    HloComputation* subcomp =
        module->AddEmbeddedComputation(sub_builder.Build());
    return HloInstruction::CreateFusion(
        shape, HloInstruction::FusionKind::kLoop, {x, y}, subcomp);
  };

  auto* sum = b.AddInstruction(make_fusion(params[0], params[1]));
  for (int64_t i = 2; i < kNumParams; ++i) {
    sum = b.AddInstruction(HloInstruction::CreateBinary(
        shape, HloOpcode::kAdd, sum,
        b.AddInstruction(make_fusion(params[i - 1], params[i]))));
  }
  auto computation = module->AddEntryComputation(b.Build());
  EXPECT_TRUE(mof_.Run(module.get()).value());
  SCOPED_TRACE(module->ToString());
  for (const HloInstruction* instr : computation->instructions()) {
    EXPECT_LE(instr->operand_count() + ShapeUtil::SubshapeCount(instr->shape()),
              MaxOperandsAndOutputsPerFusion())
        << instr->ToString();
  }
}

TEST_F(MultiOutputFusionTest, MultiOutputFusionDUS) {
  auto module = ParseAndReturnVerifiedModule(R"(HloModule dus_mof
    fusion.1 {
      p.0 = s16[50,96,1024]{2,1,0} parameter(0)
      p.1 = s16[1,96,1024]{2,1,0} parameter(1)
      c.0 = s32[3]{0} constant({0, 0, 0})
      ROOT %dynamic-update-slice = s16[50,96,1024]{2,1,0} dynamic-update-slice(p.0, p.1, c.0)
    }

    fusion.2 {
      p.0 = s16[50,96,1024]{2,1,0} parameter(0)
      p.1 = s16[1,96,1024]{2,1,0} parameter(1)
      c.0 = s32[3]{0} constant({0, 0, 0})
      ROOT %dynamic-update-slice = s16[50,96,1024]{2,1,0} dynamic-update-slice(p.0, p.1, c.0)
    }

    ENTRY entry {
      p.00 = s16[50,96,1024]{2,1,0} parameter(0)
      p.01 = s16[50,96,1024]{2,1,0} parameter(1)
      p.1 = s16[1,96,1024]{2,1,0} parameter(2)

      f1 = s16[50,96,1024] fusion(p.00, p.1), kind=kLoop, calls=fusion.1
      f2 = s16[50,96,1024] fusion(p.01, p.1), kind=kLoop, calls=fusion.2
      ROOT tuple = (s16[50,96,1024],s16[50,96,1024]) tuple(f1, f2)
    })")
                    .value();
  ASSERT_FALSE(mof_.Run(module.get()).value());
}

// Check that we don't fuse too many reductions together.
TEST_F(MultiOutputFusionTest, SharedMemoryBudget) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation0 {
      p0 = s32[64,64] parameter(0)
      p1 = s32[64,64] parameter(1)
      p2 = s32[] parameter(2)
      add = s32[64,64] add(p0, p1)
      ROOT reduce = s32[64] reduce(s32[64,64] add, s32[] p2), dimensions={0},
        to_apply=scalar_add_computation
    }
    fused_computation1 {
      p0 = s32[64,64] parameter(0)
      p1 = s32[64,64] parameter(1)
      p2 = s32[] parameter(2)
      add = s32[64,64] add(p0, p1)
      ROOT reduce = s32[64] reduce(s32[64,64] add, s32[] p2), dimensions={0},
        to_apply=scalar_add_computation
    }
    fused_computation2 {
      p0 = s32[64,64] parameter(0)
      p1 = s32[64,64] parameter(1)
      p2 = s32[] parameter(2)
      add = s32[64,64] add(p0, p1)
      ROOT reduce = s32[64] reduce(s32[64,64] add, s32[] p2), dimensions={0},
        to_apply=scalar_add_computation
    }
    fused_computation3 {
      p0 = s32[64,64] parameter(0)
      p1 = s32[64,64] parameter(1)
      p2 = s32[] parameter(2)
      add = s32[64,64] add(p0, p1)
      ROOT reduce = s32[64] reduce(s32[64,64] add, s32[] p2), dimensions={0},
        to_apply=scalar_add_computation
    }
    fused_computation4 {
      p0 = s32[64,64] parameter(0)
      p1 = s32[64,64] parameter(1)
      p2 = s32[] parameter(2)
      add = s32[64,64] add(p0, p1)
      ROOT reduce = s32[64] reduce(s32[64,64] add, s32[] p2), dimensions={0},
        to_apply=scalar_add_computation
    }
    fused_computation5 {
      p0 = s32[64,64] parameter(0)
      p1 = s32[64,64] parameter(1)
      p2 = s32[] parameter(2)
      add = s32[64,64] add(p0, p1)
      ROOT reduce = s32[64] reduce(s32[64,64] add, s32[] p2), dimensions={0},
        to_apply=scalar_add_computation
    }
    fused_computation6 {
      p0 = s32[64,64] parameter(0)
      p1 = s32[64,64] parameter(1)
      p2 = s32[] parameter(2)
      add = s32[64,64] add(p0, p1)
      ROOT reduce = s32[64] reduce(s32[64,64] add, s32[] p2), dimensions={0},
        to_apply=scalar_add_computation
    }
    fused_computation7 {
      p0 = s32[64,64] parameter(0)
      p1 = s32[64,64] parameter(1)
      p2 = s32[] parameter(2)
      add = s32[64,64] add(p0, p1)
      ROOT reduce = s32[64] reduce(s32[64,64] add, s32[] p2), dimensions={0},
        to_apply=scalar_add_computation
    }
    fused_computation8 {
      p0 = s32[64,64] parameter(0)
      p1 = s32[64,64] parameter(1)
      p2 = s32[] parameter(2)
      add = s32[64,64] add(p0, p1)
      ROOT reduce = s32[64] reduce(s32[64,64] add, s32[] p2), dimensions={0},
        to_apply=scalar_add_computation
    }
    fused_computation9 {
      p0 = s32[64,64] parameter(0)
      p1 = s32[64,64] parameter(1)
      p2 = s32[] parameter(2)
      add = s32[64,64] add(p0, p1)
      ROOT reduce = s32[64] reduce(s32[64,64] add, s32[] p2), dimensions={0},
        to_apply=scalar_add_computation
    }
    ENTRY computation {
      zero = s32[] constant(0)
      param0 = s32[64,64] parameter(0)
      param1 = s32[64,64] parameter(1)
      param2 = s32[64,64] parameter(2)
      param3 = s32[64,64] parameter(3)
      param4 = s32[64,64] parameter(4)
      param5 = s32[64,64] parameter(5)
      param6 = s32[64,64] parameter(6)
      param7 = s32[64,64] parameter(7)
      param8 = s32[64,64] parameter(8)
      param9 = s32[64,64] parameter(9)
      out0 = s32[64] fusion(param0, param1, zero), kind=kInput, calls=fused_computation0
      out1 = s32[64] fusion(param1, param2, zero), kind=kInput, calls=fused_computation1
      out2 = s32[64] fusion(param2, param3, zero), kind=kInput, calls=fused_computation2
      out3 = s32[64] fusion(param3, param4, zero), kind=kInput, calls=fused_computation3
      out4 = s32[64] fusion(param4, param5, zero), kind=kInput, calls=fused_computation4
      out5 = s32[64] fusion(param5, param6, zero), kind=kInput, calls=fused_computation5
      out6 = s32[64] fusion(param6, param7, zero), kind=kInput, calls=fused_computation6
      out7 = s32[64] fusion(param7, param8, zero), kind=kInput, calls=fused_computation7
      out8 = s32[64] fusion(param8, param9, zero), kind=kInput, calls=fused_computation8
      out9 = s32[64] fusion(param9, param0, zero), kind=kInput, calls=fused_computation9
      ROOT out = (s32[64], s32[64], s32[64], s32[64], s32[64], s32[64], s32[64], s32[64], s32[64], s32[64]) tuple(s32[64] out0, s32[64] out1, s32[64] out2, s32[64] out3, s32[64] out4, s32[64] out5, s32[64] out6, s32[64] out7, s32[64] out8, s32[64] out9)
    }
  )"))
                    .value();
  ASSERT_TRUE(mof_.Run(module.get()).value());

  EXPECT_EQ(5, CountMultiOutputFusions(module.get()));
}

TEST_F(MultiOutputFusionTest, DoNotGroupTooManyReductions) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation0 {
      p0 = s32[64,64] parameter(0)
      p1 = s32[64,64] parameter(1)
      p2 = s32[] parameter(2)
      add = s32[64,64] add(p0, p1)
      ROOT reduce = s32[64] reduce(s32[64,64] add, s32[] p2), dimensions={1},
        to_apply=scalar_add_computation
    }
    fused_computation1 {
      p0 = s32[64,64] parameter(0)
      p1 = s32[64,64] parameter(1)
      p2 = s32[] parameter(2)
      add = s32[64,64] add(p0, p1)
      ROOT reduce = s32[64] reduce(s32[64,64] add, s32[] p2), dimensions={1},
        to_apply=scalar_add_computation
    }
    fused_computation2 {
      p0 = s32[64,64] parameter(0)
      p1 = s32[64,64] parameter(1)
      p2 = s32[] parameter(2)
      add = s32[64,64] add(p0, p1)
      ROOT reduce = s32[64] reduce(s32[64,64] add, s32[] p2), dimensions={1},
        to_apply=scalar_add_computation
    }
    fused_computation3 {
      p0 = s32[64,64] parameter(0)
      p1 = s32[64,64] parameter(1)
      p2 = s32[] parameter(2)
      add = s32[64,64] add(p0, p1)
      ROOT reduce = s32[64] reduce(s32[64,64] add, s32[] p2), dimensions={1},
        to_apply=scalar_add_computation
    }
    fused_computation4 {
      p0 = s32[64,64] parameter(0)
      p1 = s32[64,64] parameter(1)
      p2 = s32[] parameter(2)
      add = s32[64,64] add(p0, p1)
      ROOT reduce = s32[64] reduce(s32[64,64] add, s32[] p2), dimensions={1},
        to_apply=scalar_add_computation
    }
    fused_computation5 {
      p0 = s32[64,64] parameter(0)
      p1 = s32[64,64] parameter(1)
      p2 = s32[] parameter(2)
      add = s32[64,64] add(p0, p1)
      ROOT reduce = s32[64] reduce(s32[64,64] add, s32[] p2), dimensions={1},
        to_apply=scalar_add_computation
    }
    fused_computation6 {
      p0 = s32[64,64] parameter(0)
      p1 = s32[64,64] parameter(1)
      p2 = s32[] parameter(2)
      add = s32[64,64] add(p0, p1)
      ROOT reduce = s32[64] reduce(s32[64,64] add, s32[] p2), dimensions={1},
        to_apply=scalar_add_computation
    }
    fused_computation7 {
      p0 = s32[64,64] parameter(0)
      p1 = s32[64,64] parameter(1)
      p2 = s32[] parameter(2)
      add = s32[64,64] add(p0, p1)
      ROOT reduce = s32[64] reduce(s32[64,64] add, s32[] p2), dimensions={1},
        to_apply=scalar_add_computation
    }
    fused_computation8 {
      p0 = s32[64,64] parameter(0)
      p1 = s32[64,64] parameter(1)
      p2 = s32[] parameter(2)
      add = s32[64,64] add(p0, p1)
      ROOT reduce = s32[64] reduce(s32[64,64] add, s32[] p2), dimensions={1},
        to_apply=scalar_add_computation
    }
    fused_computation9 {
      p0 = s32[64,64] parameter(0)
      p1 = s32[64,64] parameter(1)
      p2 = s32[] parameter(2)
      add = s32[64,64] add(p0, p1)
      ROOT reduce = s32[64] reduce(s32[64,64] add, s32[] p2), dimensions={1},
        to_apply=scalar_add_computation
    }
    ENTRY computation {
      zero = s32[] constant(0)
      param0 = s32[64,64] parameter(0)
      param1 = s32[64,64] parameter(1)
      param2 = s32[64,64] parameter(2)
      param3 = s32[64,64] parameter(3)
      param4 = s32[64,64] parameter(4)
      param5 = s32[64,64] parameter(5)
      param6 = s32[64,64] parameter(6)
      param7 = s32[64,64] parameter(7)
      param8 = s32[64,64] parameter(8)
      param9 = s32[64,64] parameter(9)
      out0 = s32[64] fusion(param0, param1, zero), kind=kInput, calls=fused_computation0
      out1 = s32[64] fusion(param1, param2, zero), kind=kInput, calls=fused_computation1
      out2 = s32[64] fusion(param2, param3, zero), kind=kInput, calls=fused_computation2
      out3 = s32[64] fusion(param3, param4, zero), kind=kInput, calls=fused_computation3
      out4 = s32[64] fusion(param4, param5, zero), kind=kInput, calls=fused_computation4
      out5 = s32[64] fusion(param5, param6, zero), kind=kInput, calls=fused_computation5
      out6 = s32[64] fusion(param6, param7, zero), kind=kInput, calls=fused_computation6
      out7 = s32[64] fusion(param7, param8, zero), kind=kInput, calls=fused_computation7
      out8 = s32[64] fusion(param8, param9, zero), kind=kInput, calls=fused_computation8
      out9 = s32[64] fusion(param9, param0, zero), kind=kInput, calls=fused_computation9
      ROOT out = (s32[64], s32[64], s32[64], s32[64], s32[64], s32[64], s32[64], s32[64], s32[64], s32[64]) tuple(s32[64] out0, s32[64] out1, s32[64] out2, s32[64] out3, s32[64] out4, s32[64] out5, s32[64] out6, s32[64] out7, s32[64] out8, s32[64] out9)
    }
  )"))
                    .value();
  ASSERT_TRUE(mof_.Run(module.get()).value());

  EXPECT_EQ(2, CountMultiOutputFusions(module.get()));
}

TEST_F(MultiOutputFusionTest, NoFusionToAvoidUsingTooMuchSharedMemory) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule xla_computation_update_step.10931

%scalar_add_computation.1 (scalar_lhs.1: s64[], scalar_rhs.1: s64[]) -> s64[] {
  %scalar_lhs.1 = s64[] parameter(0)
  %scalar_rhs.1 = s64[] parameter(1)
  ROOT %add.1257 = s64[] add(s64[] %scalar_lhs.1, s64[] %scalar_rhs.1)
}

%fused_computation.1 (param_0.8: s64[64,64], param_1.11: s64[64,64], param_2.9: s64[64,64]) -> (s64[64], s64[64]) {
  %param_0.8 = s64[64,64]{1,0} parameter(0)
  %param_1.11 = s64[64,64]{1,0} parameter(1)
  %multiply.2 = s64[64,64]{1,0} multiply(s64[64,64]{1,0} %param_0.8, s64[64,64]{1,0} %param_1.11)
  %constant_5217.3 = s64[] constant(0)
  %broadcast.1 = s64[64,64]{1,0} broadcast(s64[] %constant_5217.3), dimensions={}
  %multiply.0 = s64[64,64]{1,0} multiply(s64[64,64]{1,0} %multiply.2, s64[64,64]{1,0} %broadcast.1)
  %reduce.0 = s64[64]{0} reduce(s64[64,64]{1,0} %multiply.0, s64[] %constant_5217.3), dimensions={0}, to_apply=%scalar_add_computation.1
  %param_2.9 = s64[64,64]{1,0} parameter(2)
  %multiply.1514.clone.0.clone.1 = s64[64,64]{1,0} multiply(s64[64,64]{1,0} %param_2.9, s64[64,64]{1,0} %param_1.11)
  %constant_5217.1.clone.1 = s64[] constant(0)
  %broadcast.0.clone.1 = s64[64,64]{1,0} broadcast(s64[] %constant_5217.1.clone.1), dimensions={}
  %multiply.1341.clone.0.clone.1 = s64[64,64]{1,0} multiply(s64[64,64]{1,0} %multiply.1514.clone.0.clone.1, s64[64,64]{1,0} %broadcast.0.clone.1)
  %reduce.630.clone.0.clone.1 = s64[64]{0} reduce(s64[64,64]{1,0} %multiply.1341.clone.0.clone.1, s64[] %constant_5217.1.clone.1), dimensions={0}, to_apply=%scalar_add_computation.1
  ROOT %tuple = (s64[64]{0}, s64[64]{0}) tuple(s64[64]{0} %reduce.0, s64[64]{0} %reduce.630.clone.0.clone.1)
}

%primitive_computation_add__1.6426 (parameter.6427: s64[], parameter.6428: s64[]) -> s64[] {
  %parameter.6427 = s64[] parameter(0)
  %parameter.6428 = s64[] parameter(1)
  ROOT %add.6429 = s64[] add(s64[] %parameter.6427, s64[] %parameter.6428)
}

%fused_computation.2 (param_0.7: s64[64,64], param_1.9: s64[64,64]) -> s64[64] {
  %param_0.7 = s64[64,64]{1,0} parameter(0)
  %param_1.9 = s64[64,64]{1,0} parameter(1)
  %multiply.1 = s64[64,64]{1,0} multiply(s64[64,64]{1,0} %param_0.7, s64[64,64]{1,0} %param_1.9)
  %constant_5217.2 = s64[] constant(0)
  ROOT %reduce.740.clone.0 = s64[64]{0} reduce(s64[64,64]{1,0} %multiply.1, s64[] %constant_5217.2), dimensions={0}, to_apply=%primitive_computation_add__1.6426
}

ENTRY %reproducer (param_0.1090: s64[64,64], param_1.1377: s64[64,64], param_2.1948: s64[64,64]) -> (s64[64], s64[64], s64[64]) {
  %param_0.1090 = s64[64,64]{1,0} parameter(0)
  %param_1.1377 = s64[64,64]{1,0} parameter(1)
  %param_2.1948 = s64[64,64]{1,0} parameter(2)
  %fusion.1 = (s64[64]{0}, s64[64]{0}) fusion(s64[64,64]{1,0} %param_0.1090, s64[64,64]{1,0} %param_1.1377, s64[64,64]{1,0} %param_2.1948), kind=kInput, calls=%fused_computation.1
  %get-tuple-element = s64[64]{0} get-tuple-element((s64[64]{0}, s64[64]{0}) %fusion.1), index=0
  %fusion.2 = s64[64]{0} fusion(s64[64,64]{1,0} %param_0.1090, s64[64,64]{1,0} %param_1.1377), kind=kInput, calls=%fused_computation.2
  %get-tuple-element.1 = s64[64]{0} get-tuple-element((s64[64]{0}, s64[64]{0}) %fusion.1), index=1
  ROOT %tuple.428 = (s64[64]{0}, s64[64]{0}, s64[64]{0}) tuple(s64[64]{0} %get-tuple-element, s64[64]{0} %fusion.2, s64[64]{0} %get-tuple-element.1)
}
  )")
                    .value();
  EXPECT_FALSE(mof_.Run(module.get()).value());
}

TEST_F(MultiOutputFusionTest, NoProblemWithCodeDuplication) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule module

and.reduce_sub_computation {
  x = pred[] parameter(0)
  y = pred[] parameter(1)
  ROOT and = pred[] and(x, y)
}

fused_computation.1 {
  param_4.658 = s32[2,20,256]{2,0,1} parameter(4)
  slice.1385 = s32[2,1,256]{2,0,1} slice(param_4.658), slice={[0:2], [11:12], [0:256]}
  constant.6847 = s32[] constant(0)
  broadcast.4823 = s32[3]{0} broadcast(constant.6847), dimensions={}
  param_9.415 = s32[3]{0} parameter(9)
  compare.700 = pred[3]{0} compare(broadcast.4823, param_9.415), direction=LE
  constant.6846 = pred[] constant(true)
  reduce.221 = pred[] reduce(compare.700, constant.6846), dimensions={0}, to_apply=and.reduce_sub_computation
  broadcast.2933 = pred[2,1,256]{2,0,1} broadcast(reduce.221), dimensions={}
  param_5.528 = s32[2,512]{1,0} parameter(5)
  slice.1384 = s32[2,256]{1,0} slice(param_5.528), slice={[0:2], [0:256]}
  bitcast.341 = s32[2,1,256]{2,0,1} bitcast(slice.1384)
  constant.5418 = s32[] constant(0)
  broadcast.3227 = s32[2,1,256]{2,0,1} broadcast(constant.5418), dimensions={}
  select.173 = s32[2,1,256]{2,0,1} select(broadcast.2933, bitcast.341, broadcast.3227)
  add.573 = s32[2,1,256]{2,0,1} add(slice.1385, select.173)
  param_0.299 = s32[] parameter(0)
  constant.5157 = s32[] constant(11)
  dynamic-update-slice.189 = s32[2,20,256]{2,0,1} dynamic-update-slice(param_4.658, add.573, param_0.299, constant.5157, param_0.299)
  slice.1383 = s32[2,1,256]{2,0,1} slice(dynamic-update-slice.189), slice={[0:2], [10:11], [0:256]}
  constant.6800 = s32[] constant(0)
  broadcast.4803 = s32[3]{0} broadcast(constant.6800), dimensions={}
  param_8.484 = s32[3]{0} parameter(8)
  compare.681 = pred[3]{0} compare(broadcast.4803, param_8.484), direction=LE
  constant.6798 = pred[] constant(true)
  reduce.203 = pred[] reduce(compare.681, constant.6798), dimensions={0}, to_apply=and.reduce_sub_computation
  broadcast.2932 = pred[2,1,256]{2,0,1} broadcast(reduce.203), dimensions={}
  param_3.1169 = s32[2,512]{1,0} parameter(3)
  slice.1382 = s32[2,256]{1,0} slice(param_3.1169), slice={[0:2], [0:256]}
  bitcast.340 = s32[2,1,256]{2,0,1} bitcast(slice.1382)
  select.172 = s32[2,1,256]{2,0,1} select(broadcast.2932, bitcast.340, broadcast.3227)
  add.572 = s32[2,1,256]{2,0,1} add(slice.1383, select.172)
  constant.5154 = s32[] constant(10)
  dynamic-update-slice.188 = s32[2,20,256]{2,0,1} dynamic-update-slice(dynamic-update-slice.189, add.572, param_0.299, constant.5154, param_0.299)
  slice.1381 = s32[2,1,256]{2,0,1} slice(dynamic-update-slice.188), slice={[0:2], [9:10], [0:256]}
  constant.6794 = s32[] constant(0)
  broadcast.4801 = s32[3]{0} broadcast(constant.6794), dimensions={}
  param_7.478 = s32[3]{0} parameter(7)
  compare.679 = pred[3]{0} compare(broadcast.4801, param_7.478), direction=LE
  constant.6793 = pred[] constant(true)
  reduce.201 = pred[] reduce(compare.679, constant.6793), dimensions={0}, to_apply=and.reduce_sub_computation
  broadcast.2930 = pred[2,1,256]{2,0,1} broadcast(reduce.201), dimensions={}
  param_2.1685 = s32[2,512]{1,0} parameter(2)
  slice.1380 = s32[2,256]{1,0} slice(param_2.1685), slice={[0:2], [0:256]}
  bitcast.339 = s32[2,1,256]{2,0,1} bitcast(slice.1380)
  select.171 = s32[2,1,256]{2,0,1} select(broadcast.2930, bitcast.339, broadcast.3227)
  add.571 = s32[2,1,256]{2,0,1} add(slice.1381, select.171)
  constant.5153 = s32[] constant(9)
  dynamic-update-slice.187 = s32[2,20,256]{2,0,1} dynamic-update-slice(dynamic-update-slice.188, add.571, param_0.299, constant.5153, param_0.299)
  slice.1379 = s32[2,1,256]{2,0,1} slice(dynamic-update-slice.187), slice={[0:2], [8:9], [0:256]}
  constant.6788 = s32[] constant(0)
  broadcast.4799 = s32[3]{0} broadcast(constant.6788), dimensions={}
  param_6.495 = s32[3]{0} parameter(6)
  compare.677 = pred[3]{0} compare(broadcast.4799, param_6.495), direction=LE
  constant.6786 = pred[] constant(true)
  reduce.199 = pred[] reduce(compare.677, constant.6786), dimensions={0}, to_apply=and.reduce_sub_computation
  broadcast.2929 = pred[2,1,256]{2,0,1} broadcast(reduce.199), dimensions={}
  param_1.1408 = s32[2,512]{1,0} parameter(1)
  slice.1378 = s32[2,256]{1,0} slice(param_1.1408), slice={[0:2], [0:256]}
  bitcast.338 = s32[2,1,256]{2,0,1} bitcast(slice.1378)
  select.170 = s32[2,1,256]{2,0,1} select(broadcast.2929, bitcast.338, broadcast.3227)
  add.570 = s32[2,1,256]{2,0,1} add(slice.1379, select.170)
  constant.5152 = s32[] constant(8)
  ROOT dynamic-update-slice.186 = s32[2,20,256]{2,0,1} dynamic-update-slice(dynamic-update-slice.187, add.570, param_0.299, constant.5152, param_0.299)
}

fused_computation.2 {
  param_4.655 = s32[2,20,256]{2,0,1} parameter(4)
  slice.1369 = s32[2,1,256]{2,0,1} slice(param_4.655), slice={[0:2], [7:8], [0:256]}
  param_6.483 = pred[] parameter(6)
  broadcast.2927 = pred[2,1,256]{2,0,1} broadcast(param_6.483), dimensions={}
  param_5.525 = s32[2,512]{1,0} parameter(5)
  slice.1368 = s32[2,256]{1,0} slice(param_5.525), slice={[0:2], [0:256]}
  bitcast.333 = s32[2,1,256]{2,0,1} bitcast(slice.1368)
  constant.5415 = s32[] constant(0)
  broadcast.3225 = s32[2,1,256]{2,0,1} broadcast(constant.5415), dimensions={}
  select.161 = s32[2,1,256]{2,0,1} select(broadcast.2927, bitcast.333, broadcast.3225)
  add.549 = s32[2,1,256]{2,0,1} add(slice.1369, select.161)
  param_0.265 = s32[] parameter(0)
  constant.5151 = s32[] constant(7)
  dynamic-update-slice.185 = s32[2,20,256]{2,0,1} dynamic-update-slice(param_4.655, add.549, param_0.265, constant.5151, param_0.265)
  slice.1367 = s32[2,1,256]{2,0,1} slice(dynamic-update-slice.185), slice={[0:2], [6:7], [0:256]}
  constant.6782 = s32[] constant(0)
  broadcast.4797 = s32[3]{0} broadcast(constant.6782), dimensions={}
  param_9.391 = s32[3]{0} parameter(9)
  compare.675 = pred[3]{0} compare(broadcast.4797, param_9.391), direction=LE
  constant.6781 = pred[] constant(true)
  reduce.197 = pred[] reduce(compare.675, constant.6781), dimensions={0}, to_apply=and.reduce_sub_computation
  broadcast.2926 = pred[2,1,256]{2,0,1} broadcast(reduce.197), dimensions={}
  param_3.1167 = s32[2,512]{1,0} parameter(3)
  slice.1366 = s32[2,256]{1,0} slice(param_3.1167), slice={[0:2], [0:256]}
  bitcast.332 = s32[2,1,256]{2,0,1} bitcast(slice.1366)
  select.160 = s32[2,1,256]{2,0,1} select(broadcast.2926, bitcast.332, broadcast.3225)
  add.548 = s32[2,1,256]{2,0,1} add(slice.1367, select.160)
  constant.5150 = s32[] constant(6)
  dynamic-update-slice.184 = s32[2,20,256]{2,0,1} dynamic-update-slice(dynamic-update-slice.185, add.548, param_0.265, constant.5150, param_0.265)
  slice.1365 = s32[2,1,256]{2,0,1} slice(dynamic-update-slice.184), slice={[0:2], [5:6], [0:256]}
  constant.6776 = s32[] constant(0)
  broadcast.4794 = s32[3]{0} broadcast(constant.6776), dimensions={}
  param_8.464 = s32[3]{0} parameter(8)
  compare.673 = pred[3]{0} compare(broadcast.4794, param_8.464), direction=LE
  constant.6775 = pred[] constant(true)
  reduce.195 = pred[] reduce(compare.673, constant.6775), dimensions={0}, to_apply=and.reduce_sub_computation
  broadcast.2925 = pred[2,1,256]{2,0,1} broadcast(reduce.195), dimensions={}
  param_2.1684 = s32[2,512]{1,0} parameter(2)
  slice.1364 = s32[2,256]{1,0} slice(param_2.1684), slice={[0:2], [0:256]}
  bitcast.331 = s32[2,1,256]{2,0,1} bitcast(slice.1364)
  select.159 = s32[2,1,256]{2,0,1} select(broadcast.2925, bitcast.331, broadcast.3225)
  add.547 = s32[2,1,256]{2,0,1} add(slice.1365, select.159)
  constant.5149 = s32[] constant(5)
  dynamic-update-slice.183 = s32[2,20,256]{2,0,1} dynamic-update-slice(dynamic-update-slice.184, add.547, param_0.265, constant.5149, param_0.265)
  slice.1363 = s32[2,1,256]{2,0,1} slice(dynamic-update-slice.183), slice={[0:2], [4:5], [0:256]}
  constant.6770 = s32[] constant(0)
  broadcast.4792 = s32[3]{0} broadcast(constant.6770), dimensions={}
  param_7.458 = s32[3]{0} parameter(7)
  compare.671 = pred[3]{0} compare(broadcast.4792, param_7.458), direction=LE
  constant.6769 = pred[] constant(true)
  reduce.193 = pred[] reduce(compare.671, constant.6769), dimensions={0}, to_apply=and.reduce_sub_computation
  broadcast.2924 = pred[2,1,256]{2,0,1} broadcast(reduce.193), dimensions={}
  param_1.1405 = s32[2,512]{1,0} parameter(1)
  slice.1362 = s32[2,256]{1,0} slice(param_1.1405), slice={[0:2], [0:256]}
  bitcast.330 = s32[2,1,256]{2,0,1} bitcast(slice.1362)
  select.158 = s32[2,1,256]{2,0,1} select(broadcast.2924, bitcast.330, broadcast.3225)
  add.546 = s32[2,1,256]{2,0,1} add(slice.1363, select.158)
  constant.5148 = s32[] constant(4)
  ROOT dynamic-update-slice.182 = s32[2,20,256]{2,0,1} dynamic-update-slice(dynamic-update-slice.183, add.546, param_0.265, constant.5148, param_0.265)
}

ENTRY main {
  param_0.0 = s32[] parameter(0)
  param_1.0 = s32[2,512]{1,0} parameter(1)
  param_2.0 = s32[2,512]{1,0} parameter(2)
  param_3.0 = s32[2,512]{1,0} parameter(3)
  param_4.0 = s32[2,20,256]{2,1,0} parameter(4)
  param_5.0 = s32[2,512]{1,0} parameter(5)
  param_6.0 = s32[3]{0} parameter(6)
  param_7.0 = s32[3]{0} parameter(7)
  param_8.0 = s32[3]{0} parameter(8)
  param_9.0 = s32[3]{0} parameter(9)
  fusion.1 = s32[2,20,256]{2,0,1} fusion(param_0.0, param_1.0, param_2.0, param_3.0, param_4.0, param_5.0, param_6.0, param_7.0, param_8.0, param_9.0), kind=kLoop, calls=fused_computation.1
  param_10 = pred[] parameter(10)
  fusion.2 = s32[2,20,256]{2,0,1} fusion(param_0.0, param_1.0, param_2.0, param_3.0, fusion.1, param_5.0, param_10, param_7.0, param_8.0, param_9.0), kind=kLoop, calls=fused_computation.2
  ROOT root = (s32[2,20,256]{2,0,1}, s32[2,20,256]{2,0,1}) tuple(fusion.1, fusion.2)
}
  )")
                    .value();
  EXPECT_TRUE(mof_.Run(module.get()).value());
}

TEST_F(MultiOutputFusionTest, DoNotFuseRoot) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule module

no_op {
  arg_empty_tuple = () parameter(0)
  ROOT tuple = () tuple()
}

fused_computation {
  param_0 = s16[] parameter(0)
  ROOT convert = s32[] convert(param_0)
}

ENTRY main {
  param_0 = s16[] parameter(0)
  fusion = s32[] fusion(param_0), kind=kLoop, calls=fused_computation
  tuple = () tuple()
  conditional = () conditional(fusion, tuple, tuple), branch_computations={no_op, no_op}
  constant = s16[] constant(1)
  ROOT root = s16[] add(param_0, constant)
}
  )")
                    .value();
  EXPECT_FALSE(mof_.Run(module.get()).value());
}

TEST_F(MultiOutputFusionTest, CostBasedNoMerge) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule m

region_3.63 {
  Arg_0.64 = s32[] parameter(0)
  Arg_1.65 = s32[] parameter(1)
  ROOT add.66 = s32[] add(Arg_0.64, Arg_1.65)
}

fused_computation.29 {
  param_0.161 = s32[5,32,32,1]{3,2,1,0} parameter(0)
  multiply.208 = s32[5,32,32,1]{3,2,1,0} multiply(param_0.161, param_0.161)
  bitcast.67 = s32[5,32,32]{2,1,0} bitcast(multiply.208)
  constant.265 = s32[] constant(0)
  reduce-window.81 = s32[5,30,31]{2,1,0} reduce-window(bitcast.67, constant.265), window={size=1x3x2}, to_apply=region_3.63
  constant.264 = s32[] constant(1)
  broadcast.204 = s32[5,30,31]{2,1,0} broadcast(constant.264), dimensions={}
  multiply.205 = s32[5,30,31]{2,1,0} multiply(reduce-window.81, broadcast.204)
  constant.263 = s32[] constant(0)
  reduce-window.80 = s32[5,30,31]{2,1,0} reduce-window(multiply.205, constant.263), window={size=1x2x3 pad=0_0x0_1x1_1}, to_apply=region_3.63
  constant.262 = s32[] constant(1)
  broadcast.201 = s32[5,30,31]{2,1,0} broadcast(constant.262), dimensions={}
  multiply.204 = s32[5,30,31]{2,1,0} multiply(reduce-window.80, broadcast.201)
  constant.261 = s32[] constant(0)
  reduce-window.78 = s32[5,30,31]{2,1,0} reduce-window(multiply.204, constant.261), window={size=1x1x2 pad=0_0x0_0x0_1}, to_apply=region_3.63
  constant.113 = s32[] constant(1)
  broadcast.137 = s32[5,30,31]{2,1,0} broadcast(constant.113), dimensions={}
  multiply.125 = s32[5,30,31]{2,1,0} multiply(reduce-window.78, broadcast.137)
  constant.114 = s32[] constant(0)
  ROOT reduce-window.17 = s32[5,30,31]{2,1,0} reduce-window(multiply.125, constant.114), window={size=1x2x1 pad=0_0x0_1x0_0}, to_apply=region_3.63
}

fused_computation.15 {
  constant.108 = s32[] constant(1)
  broadcast.105 = s32[5,5,30,31]{3,2,1,0} broadcast(constant.108), dimensions={}
  param_3.126 = s32[5,30,31]{2,1,0} parameter(3)
  constant.295 = s32[] constant(1)
  broadcast.234 = s32[5,30,31]{2,1,0} broadcast(constant.295), dimensions={}
  multiply.242 = s32[5,30,31]{2,1,0} multiply(param_3.126, broadcast.234)
  broadcast.233 = s32[5,5,30,31]{3,2,1,0} broadcast(multiply.242), dimensions={0,2,3}
  param_2.154 = s32[5,30,31]{2,1,0} parameter(2)
  multiply.241 = s32[5,30,31]{2,1,0} multiply(param_2.154, broadcast.234)
  broadcast.232 = s32[5,5,30,31]{3,2,1,0} broadcast(multiply.241), dimensions={1,2,3}
  multiply.240 = s32[5,5,30,31]{3,2,1,0} multiply(broadcast.233, broadcast.232)
  param_1.188 = s32[5,5,30,31]{3,2,1,0} parameter(1)
  constant.294 = s32[] constant(1)
  broadcast.231 = s32[5,5,30,31]{3,2,1,0} broadcast(constant.294), dimensions={}
  multiply.239 = s32[5,5,30,31]{3,2,1,0} multiply(param_1.188, broadcast.231)
  param_0.164 = s32[5,5,30,31]{3,2,1,0} parameter(0)
  add.19 = s32[5,5,30,31]{3,2,1,0} add(multiply.239, param_0.164)
  constant.293 = s32[] constant(0)
  reduce-window.90 = s32[5,5,30,31]{3,2,1,0} reduce-window(add.19, constant.293), window={size=1x1x1x2 pad=0_0x0_0x0_0x0_1}, to_apply=region_3.63
  constant.292 = s32[] constant(1)
  broadcast.230 = s32[5,5,30,31]{3,2,1,0} broadcast(constant.292), dimensions={}
  multiply.238 = s32[5,5,30,31]{3,2,1,0} multiply(reduce-window.90, broadcast.230)
  constant.291 = s32[] constant(0)
  reduce-window.89 = s32[5,5,30,31]{3,2,1,0} reduce-window(multiply.238, constant.291), window={size=1x1x2x1 pad=0_0x0_0x0_1x0_0}, to_apply=region_3.63
  constant.290 = s32[] constant(1)
  broadcast.229 = s32[5,5,30,31]{3,2,1,0} broadcast(constant.290), dimensions={}
  multiply.237 = s32[5,5,30,31]{3,2,1,0} multiply(reduce-window.89, broadcast.229)
  multiply.236 = s32[5,5,30,31]{3,2,1,0} multiply(multiply.237, multiply.237)
  subtract.10 = s32[5,5,30,31]{3,2,1,0} subtract(multiply.240, multiply.236)
  constant.289 = s32[] constant(0)
  broadcast.228 = s32[5,5,30,31]{3,2,1,0} broadcast(constant.289), dimensions={}
  maximum.6 = s32[5,5,30,31]{3,2,1,0} maximum(subtract.10, broadcast.228)
  negate.6 = s32[5,5,30,31]{3,2,1,0} negate(maximum.6)
  constant.110 = s32[] constant(0)
  broadcast.107 = s32[5,5,30,31]{3,2,1,0} broadcast(constant.110), dimensions={}
  compare.4 = pred[5,5,30,31]{3,2,1,0} compare(negate.6, broadcast.107), direction=EQ
  constant.243 = s32[] constant(1)
  broadcast.193 = s32[5,5,30,31]{3,2,1,0} broadcast(constant.243), dimensions={}
  multiply.194 = s32[5,5,30,31]{3,2,1,0} multiply(param_1.188, broadcast.193)
  add.15 = s32[5,5,30,31]{3,2,1,0} add(multiply.194, param_0.164)
  constant.242 = s32[] constant(0)
  reduce-window.66 = s32[5,5,30,31]{3,2,1,0} reduce-window(add.15, constant.242), window={size=1x1x1x2 pad=0_0x0_0x0_0x0_1}, to_apply=region_3.63
  constant.241 = s32[] constant(1)
  broadcast.192 = s32[5,5,30,31]{3,2,1,0} broadcast(constant.241), dimensions={}
  multiply.193 = s32[5,5,30,31]{3,2,1,0} multiply(reduce-window.66, broadcast.192)
  constant.240 = s32[] constant(0)
  reduce-window.65 = s32[5,5,30,31]{3,2,1,0} reduce-window(multiply.193, constant.240), window={size=1x1x2x1 pad=0_0x0_0x0_1x0_0}, to_apply=region_3.63
  constant.239 = s32[] constant(1)
  broadcast.191 = s32[5,5,30,31]{3,2,1,0} broadcast(constant.239), dimensions={}
  multiply.192 = s32[5,5,30,31]{3,2,1,0} multiply(reduce-window.65, broadcast.191)
  compare.3 = pred[5,5,30,31]{3,2,1,0} compare(multiply.192, broadcast.107), direction=EQ
  and.1 = pred[5,5,30,31]{3,2,1,0} and(compare.4, compare.3)
  constant.109 = s32[] constant(2)
  broadcast.104 = s32[5,5,30,31]{3,2,1,0} broadcast(constant.109), dimensions={}
  subtract.1 = s32[5,5,30,31]{3,2,1,0} subtract(negate.6, multiply.192)
  select.4 = s32[5,5,30,31]{3,2,1,0} select(and.1, broadcast.104, subtract.1)
  constant.107 = s32[] constant(1)
  broadcast.106 = s32[5,5,30,31]{3,2,1,0} broadcast(constant.107), dimensions={}
  multiply.100 = s32[5,5,30,31]{3,2,1,0} multiply(select.4, broadcast.106)
  ROOT subtract.3 = s32[5,5,30,31]{3,2,1,0} subtract(broadcast.105, multiply.100)
}

fused_computation.4 {
  param_0.172 = s32[5,30,31]{2,1,0} parameter(0)
  constant.315 = s32[] constant(1)
  broadcast.242 = s32[5,30,31]{2,1,0} broadcast(constant.315), dimensions={}
  multiply.250 = s32[5,30,31]{2,1,0} multiply(param_0.172, broadcast.242)
  constant.314 = s32[] constant(0)
  reduce-window.100 = s32[5,30,31]{2,1,0} reduce-window(multiply.250, constant.314), window={size=1x3x3 pad=0_0x1_1x1_1}, to_apply=region_3.63
  constant.79 = s32[] constant(1)
  broadcast.85 = s32[5,30,31]{2,1,0} broadcast(constant.79), dimensions={}
  multiply.80 = s32[5,30,31]{2,1,0} multiply(reduce-window.100, broadcast.85)
  constant.81 = s32[] constant(0)
  reduce-window.1 = s32[5,30,31]{2,1,0} reduce-window(multiply.80, constant.81), window={size=1x3x3 pad=0_0x1_1x1_1}, to_apply=region_3.63
  constant.80 = s32[] constant(1)
  broadcast.86 = s32[5,30,31]{2,1,0} broadcast(constant.80), dimensions={}
  multiply.79 = s32[5,30,31]{2,1,0} multiply(reduce-window.1, broadcast.86)
  bitcast.26 = s32[5,930]{1,0} bitcast(multiply.79)
  ROOT reduce.8 = s32[5]{0} reduce(bitcast.26, constant.81), dimensions={1}, to_apply=region_3.63
}

ENTRY e {
  Arg_0.1 = s32[5,32,32,1]{3,2,1,0} parameter(0)
  p1 = s32[5,5,30,31]{3,2,1,0} parameter(1)
  p2 = s32[5,5,30,31]{3,2,1,0} parameter(2)
  p3 = s32[5,30,31]{2,1,0} parameter(3)
  fusion.29 = s32[5,30,31]{2,1,0} fusion(Arg_0.1), kind=kLoop, calls=fused_computation.29
  fusion.15 = s32[5,5,30,31]{3,2,1,0} fusion(p2, p1, p3, fusion.29), kind=kLoop, calls=fused_computation.15
  ROOT fusion.4 = s32[5]{0} fusion(fusion.29), kind=kInput, calls=fused_computation.4
})")
                    .value();
  EXPECT_FALSE(mof_.Run(module.get()).value());
}

TEST_F(MultiOutputFusionTest, NoOverlappingRead) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule module

    fused_computation_1 {
      p0.1 = s32[100,200]{1,0} parameter(0)
      slice.0 = s32[50,100]{1,0} slice(p0.1), slice={[0:50],[0:100]}
      mul = s32[50,100]{1,0} multiply(slice.0, slice.0)
      neg = s32[50,100]{1,0} negate(slice.0)
      ROOT tuple = (s32[50,100]{1,0}, s32[50,100]{1,0}) tuple(mul, neg)
    }

    fused_computation_2 {
      p0.2 = s32[100,200]{1,0} parameter(0)
      slice.1 = s32[50,100]{1,0} slice(p0.2), slice={[0:50],[100:200]}
      const.2 = s32[] constant(0)
      broadcast = s32[50,100]{1,0} broadcast(const.2), dimensions={}
      ROOT add = s32[50,100]{1,0} add(slice.1, broadcast)
    }

    ENTRY entry {
      p0 = s32[100,200]{1,0} parameter(0)
      fusion.1 = (s32[50,100]{1,0}, s32[50,100]{1,0}) fusion(p0), kind=kLoop,
        calls=fused_computation_1
      gte0 = s32[50,100]{1,0} get-tuple-element(fusion.1), index=0
      gte1 = s32[50,100]{1,0} get-tuple-element(fusion.1), index=1
      fusion.2 = s32[50,100]{1,0} fusion(p0), kind=kLoop,
        calls=fused_computation_2
      ROOT root = (s32[50,100]{1,0}, s32[50,100]{1,0}, s32[50,100]{1,0})
        tuple(gte0, gte1, fusion.2)
    })")
                    .value();

  EXPECT_FALSE(mof_.Run(module.get()).value());
}

TEST_F(MultiOutputFusionTest, OverlappingRead) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule module

    fused_computation_1 {
      p0.1 = s32[100,200]{1,0} parameter(0)
      slice.0 = s32[50,100]{1,0} slice(p0.1), slice={[0:50],[50:150]}
      mul = s32[50,100]{1,0} multiply(slice.0, slice.0)
      neg = s32[50,100]{1,0} negate(slice.0)
      ROOT tuple = (s32[50,100]{1,0}, s32[50,100]{1,0}) tuple(mul, neg)
    }

    fused_computation_2 {
      p0.2 = s32[100,200]{1,0} parameter(0)
      slice.1 = s32[50,100]{1,0} slice(p0.2), slice={[30:80],[20:120]}
      const.2 = s32[] constant(0)
      broadcast = s32[50,100]{1,0} broadcast(const.2), dimensions={}
      ROOT add = s32[50,100]{1,0} add(slice.1, broadcast)
    }

    ENTRY entry {
      p0 = s32[100,200]{1,0} parameter(0)
      fusion.1 = (s32[50,100]{1,0}, s32[50,100]{1,0}) fusion(p0), kind=kLoop,
        calls=fused_computation_1
      gte0 = s32[50,100]{1,0} get-tuple-element(fusion.1), index=0
      gte1 = s32[50,100]{1,0} get-tuple-element(fusion.1), index=1
      fusion.2 = s32[50,100]{1,0} fusion(p0), kind=kLoop,
        calls=fused_computation_2
      ROOT root = (s32[50,100]{1,0}, s32[50,100]{1,0}, s32[50,100]{1,0})
        tuple(gte0, gte1, fusion.2)
    })")
                    .value();

  EXPECT_TRUE(mof_.Run(module.get()).value());
}

class TransposeMultiOutputFusionTest : public MultiOutputFusionTest {
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options =
        MultiOutputFusionTest::GetDebugOptionsForTest();
    return debug_options;
  }
};

TEST_F(TransposeMultiOutputFusionTest, MultipleTransposes) {
  const char* hlo = R"(
HloModule module

fused_computation {
  param_0.1 = s32[16,32]{1,0} parameter(0)
  s.1 = s32[16,32]{1,0} negate(param_0.1)
  ROOT t.1 = s32[32,16]{1,0} transpose(s.1), dimensions={1,0}
}

ENTRY main {
  p = s32[16,32]{1,0} parameter(0)
  fusion = s32[32,16]{1,0} fusion(p), kind=kInput, calls=fused_computation
  t1 = s32[32,16]{1,0} transpose(p), dimensions={1,0}
  ROOT t = (s32[32,16]{1,0}, s32[32,16]{1,0}) tuple(fusion, t1)
}
  )";

  CheckMultiOutputFusion(hlo, R"(
// CHECK: %fused_computation (param_0.1: s32[16,32]) -> (s32[32,16], s32[32,16]) {
// CHECK-NEXT:   [[param_0_1_0:%[^ ]+]] = s32[16,32]{1,0} parameter(0)
// CHECK-NEXT:   [[s_1_1:%[^ ]+]] = s32[16,32]{1,0} negate([[param_0_1_0]])
// CHECK-NEXT:   [[c_1_2:%[^ ]+]] = s32[32,16]{1,0} transpose([[s_1_1]]), dimensions={1,0}
// CHECK-NEXT:   [[c1_1_3:%[^ ]+]] = s32[32,16]{1,0} transpose([[param_0_1_0]]), dimensions={1,0}
// CHECK-NEXT:   ROOT [[tuple_4:%[^ ]+]] = (s32[32,16]{1,0}, s32[32,16]{1,0}) tuple([[c_1_2]], [[c1_1_3]])
// CHECK-NEXT: }

// CHECK: [[fusion_0:%[^ ]+]] = (s32[32,16]{1,0}, s32[32,16]{1,0}) fusion([[p_1:%[^ ]+]]), kind=kInput, calls=[[fused_computation_2:%[^ ]+]]
)");
}

TEST_F(TransposeMultiOutputFusionTest, MultipleTransposesDifferentTypes) {
  const char* hlo = R"(
HloModule module

fused_computation {
  param_0.1 = s16[16,32]{1,0} parameter(0)
  s.1 = s8[16,32]{1,0} convert(param_0.1)
  ROOT t.1 = s8[32,16]{1,0} transpose(s.1), dimensions={1,0}
}

ENTRY main {
  p = s16[16,32]{1,0} parameter(0)
  fusion = s8[32,16]{1,0} fusion(p), kind=kInput, calls=fused_computation
  t1 = s16[32,16]{1,0} transpose(p), dimensions={1,0}
  ROOT t = (s8[32,16]{1,0}, s16[32,16]{1,0}) tuple(fusion, t1)
}
  )";

  CheckMultiOutputFusion(hlo, R"(
// CHECK: %fused_computation (param_0.1: s16[16,32]) -> (s8[32,16], s16[32,16]) {
// CHECK-NEXT:   [[param_0_1_0:%[^ ]+]] = s16[16,32]{1,0} parameter(0)
// CHECK-NEXT:   [[s_1_1:%[^ ]+]] = s8[16,32]{1,0} convert([[param_0_1_0]])
// CHECK-NEXT:   [[c_1_2:%[^ ]+]] = s8[32,16]{1,0} transpose([[s_1_1]]), dimensions={1,0}
// CHECK-NEXT:   [[c1_1_3:%[^ ]+]] = s16[32,16]{1,0} transpose([[param_0_1_0]]), dimensions={1,0}
// CHECK-NEXT:   ROOT [[tuple_4:%[^ ]+]] = (s8[32,16]{1,0}, s16[32,16]{1,0}) tuple([[c_1_2]], [[c1_1_3]])
// CHECK:   [[fusion_5:%[^ ]+]] = (s8[32,16]{1,0}, s16[32,16]{1,0}) fusion([[p_6:%[^ ]+]]), kind=kInput, calls=[[fused_computation_7:%[^ ]+]]
)");
}

// Do not group transpose and reduction.
TEST_F(TransposeMultiOutputFusionTest, TiledReduceTranspose) {
  const char* hlo = R"(
HloModule module

add {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = add(lhs, rhs)
}

fused_computation {
  param_0.1 = s32[16,32]{1,0} parameter(0)
  s.1 = s32[16,32]{1,0} negate(param_0.1)
  ROOT t.1 = s32[32,16]{1,0} transpose(s.1), dimensions={1,0}
}

ENTRY main {
  p = s32[16,32]{1,0} parameter(0)
  fusion = s32[32,16]{1,0} fusion(p), kind=kInput, calls=fused_computation
  z = s32[] constant(0)
  r1 = s32[32]{0} reduce(p, z), dimensions={0}, to_apply=add
  ROOT t = (s32[32,16]{1,0}, s32[32]{0}) tuple(fusion, r1)
}
  )";

  CheckMultiOutputFusion(hlo, std::nullopt);
}

// Do not group incompatible transposes.
TEST_F(TransposeMultiOutputFusionTest, IncompatibleTransposes) {
  const char* hlo = R"(
HloModule module

fused_computation {
  param_0.1 = s32[18,16,32]{2,1,0} parameter(0)
  param_1.1 = s32[32,16,18]{2,1,0} parameter(1)
  s.1 = s32[18,16,32]{2,1,0} negate(param_0.1)
  t.1 = s32[32,16,18]{2,1,0} transpose(s.1), dimensions={2,1,0}
  sub.1 = s32[32,16,18]{2,1,0} subtract(t.1, param_1.1)
  neg.1 = s32[32,16,18]{2,1,0} negate(sub.1)
  ROOT add.1 = s32[32,16,18]{2,1,0} add(neg.1, neg.1)
}

fused_computation.2 {
  param_0.2 = s32[18,16,32]{2,1,0} parameter(0)
  s.2 = s32[18,16,32]{2,1,0} negate(param_0.2)
  ROOT t.2 = s32[18,32,16]{2,1,0} transpose(s.2), dimensions={0,2,1}
}

ENTRY main {
  p = s32[18,16,32]{2,1,0} parameter(0)
  p2 = s32[32,16,18]{2,1,0} parameter(1)
  fusion = s32[32,16,18]{2,1,0} fusion(p, p2), kind=kLoop, calls=fused_computation
  fusion2 = s32[18,32,16]{2,1,0} fusion(p), kind=kInput, calls=fused_computation.2
  ROOT t = (s32[32,16,18]{2,1,0}, s32[18,32,16]{2,1,0}) tuple(fusion, fusion2)
}
  )";

  CheckMultiOutputFusion(hlo, std::nullopt);
}

// A variation of the test above, where no CSE was run.
TEST_F(TransposeMultiOutputFusionTest, TransposesNoCSE) {
  const char* hlo = R"(
HloModule module

fused_computation {
  param_0.1 = s32[18,16,32]{2,1,0} parameter(0)
  param_1.1 = s32[32,16,18]{2,1,0} parameter(1)
  s.1 = s32[18,16,32]{2,1,0} negate(param_0.1)
  t.1 = s32[32,16,18]{2,1,0} transpose(s.1), dimensions={2,1,0}
  sub.1 = s32[32,16,18]{2,1,0} subtract(t.1, param_1.1)
  neg.1 = s32[32,16,18]{2,1,0} negate(sub.1)
  neg.2 = s32[32,16,18]{2,1,0} negate(sub.1)
  ROOT add.1 = s32[32,16,18]{2,1,0} add(neg.1, neg.2)
}

fused_computation.2 {
  param_0.2 = s32[18,16,32]{2,1,0} parameter(0)
  s.2 = s32[18,16,32]{2,1,0} negate(param_0.2)
  ROOT t.2 = s32[18,32,16]{2,1,0} transpose(s.2), dimensions={0,2,1}
}

ENTRY main {
  p = s32[18,16,32]{2,1,0} parameter(0)
  p2 = s32[32,16,18]{2,1,0} parameter(1)
  fusion = s32[32,16,18]{2,1,0} fusion(p, p2), kind=kLoop, calls=fused_computation
  fusion2 = s32[18,32,16]{2,1,0} fusion(p), kind=kInput, calls=fused_computation.2
  ROOT t = (s32[32,16,18]{2,1,0}, s32[18,32,16]{2,1,0}) tuple(fusion, fusion2)
}
  )";

  CheckMultiOutputFusion(hlo, std::nullopt);
}

TEST_F(TransposeMultiOutputFusionTest, TransposeAndInput) {
  const char* hlo = R"(
HloModule module

fused_computation {
  param_0.1 = s32[16,32]{1,0} parameter(0)
  s.1 = s32[16,32]{1,0} negate(param_0.1)
  ROOT t.1 = s32[32,16]{1,0} transpose(s.1), dimensions={1,0}
}

ENTRY main {
  p = s32[16,32]{1,0} parameter(0)
  fusion = s32[32,16]{1,0} fusion(p), kind=kInput, calls=fused_computation
  c1 = s32[16,32]{1,0} negate(p)
  ROOT t = (s32[32,16]{1,0}, s32[16,32]{1,0}) tuple(fusion, c1)
}
  )";

  CheckMultiOutputFusion(hlo, R"(
// CHECK: %fused_computation (param_0.1: s32[16,32]) -> (s32[32,16], s32[16,32]) {
// CHECK-NEXT:   [[param_0_1_0:%[^ ]+]] = s32[16,32]{1,0} parameter(0)
// CHECK-NEXT:   [[s_1_1:%[^ ]+]] = s32[16,32]{1,0} negate([[param_0_1_0]])
// CHECK-NEXT:   [[c_1_2:%[^ ]+]] = s32[32,16]{1,0} transpose([[s_1_1]]), dimensions={1,0}
// CHECK-NEXT:   [[c1_1_3:%[^ ]+]] = s32[16,32]{1,0} negate([[param_0_1_0]])
// CHECK-NEXT:   ROOT [[tuple_4:%[^ ]+]] = (s32[32,16]{1,0}, s32[16,32]{1,0}) tuple([[c_1_2]], [[c1_1_3]])
// CHECK-NEXT: }
// CHECK:   [[fusion_0:%[^ ]+]] = (s32[32,16]{1,0}, s32[16,32]{1,0}) fusion([[p_1:%[^ ]+]]), kind=kInput, calls=[[fused_computation_2:%[^ ]+]]
)");
}

TEST_F(TransposeMultiOutputFusionTest, TransposeAndInputEpilogueFusion) {
  const char* hlo = R"(
HloModule module

fused_computation {
  param_0.1 = s32[1,16,32]{2,1,0} parameter(0)
  s.1 = s32[1,16,32]{2,1,0} negate(param_0.1)
  t.1 = s32[1,32,16]{2,1,0} transpose(s.1), dimensions={0,2,1}
  ROOT out = s32[32,16,1]{2,1,0} bitcast(t.1)
}

ENTRY main {
  p = s32[1,16,32]{2,1,0} parameter(0)
  fusion = s32[32,16,1]{2,1,0} fusion(p), kind=kInput, calls=fused_computation
  c1 = s32[1,16,32]{2,1,0} negate(p)
  ROOT t = (s32[32,16,1]{2,1,0}, s32[1,16,32]{2,1,0}) tuple(fusion, c1)
}
  )";

  CheckMultiOutputFusion(hlo, R"(
// CHECK: %fused_computation
// CHECK-NEXT:   [[param_0_1_0:%[^ ]+]] = s32[1,16,32]{2,1,0} parameter(0)
// CHECK-NEXT:   [[s_1_1:%[^ ]+]] = s32[1,16,32]{2,1,0} negate([[param_0_1_0]])
// CHECK-NEXT:   [[c_1_2:%[^ ]+]] = s32[1,32,16]{2,1,0} transpose([[s_1_1]])
// CHECK-NEXT:   [[out_3:%[^ ]+]] = s32[32,16,1]{2,1,0} bitcast([[c_1_2]])
// CHECK-NEXT:   [[c1_1_4:%[^ ]+]] = s32[1,16,32]{2,1,0} negate([[param_0_1_0]])
// CHECK-NEXT:   ROOT [[tuple_5:%[^ ]+]] = (s32[32,16,1]{2,1,0}, s32[1,16,32]{2,1,0}) tuple([[out_3]], [[c1_1_4]])
// CHECK-NEXT: }
// CHECK: [[fusion_0:%[^ ]+]] = (s32[32,16,1]{2,1,0}, s32[1,16,32]{2,1,0}) fusion([[p_1:%[^ ]+]]), kind=kInput, calls=[[fused_computation_2:%[^ ]+]]
)");
}

class ReduceMultiOutputFusionTest : public MultiOutputFusionTest {};

TEST_F(ReduceMultiOutputFusionTest, ReduceAndLoop) {
  const char* hlo = R"(
HloModule module

add {
  a = s32[] parameter(0)
  b = s32[] parameter(1)
  ROOT c = s32[] add(a, b)
}

fused_reduction {
  p = s32[200] parameter(0)
  z = s32[] constant(0)
  e = s32[200] negate(p)
  ROOT r = s32[] reduce(e, z), dimensions={0}, to_apply=add
}

fused_elementwise {
  p = s32[200] parameter(0)
  ROOT r = s32[200] negate(p)
}

ENTRY computation {
  p = s32[200] parameter(0)
  o1 = s32[200] fusion(p), kind=kLoop, calls=fused_elementwise
  o2 = s32[] fusion(p), kind=kInput, calls=fused_reduction
  ROOT out = (s32[200], s32[]) tuple(o1, o2)
}

)";

  CheckMultiOutputFusion(hlo, R"(
// CHECK: %fused_elementwise
// CHECK-NEXT:  [[p_1_0:%[^ ]+]] = s32[200]{0} parameter(0)
// CHECK-NEXT:  [[r_1_1:%[^ ]+]] = s32[200]{0} negate([[p_1_0]])
// CHECK-NEXT:  [[e_2:%[^ ]+]].clone.1 = s32[200]{0} negate([[p_1_0]])
// CHECK-NEXT:  [[z_3:%[^ ]+]].clone.1 = s32[] constant(0)
// CHECK-NEXT:  [[r_4:%[^ ]+]].clone.1 = s32[] reduce([[e_2]].clone.1, [[z_3]].clone.1), dimensions={0}, to_apply=[[add_5:%[^ ]+]]
// CHECK-NEXT:  ROOT [[tuple_6:%[^ ]+]] = (s32[200]{0}, s32[]) tuple([[r_1_1]], [[r_4]].clone.1)
// CHECK-NEXT:}
// CHECK: [[o1_0:%[^ ]+]] = (s32[200]{0}, s32[]) fusion([[p_2_1:%[^ ]+]]), kind=kInput, calls=[[fused_elementwise_2:%[^ ]+]]
  )");
}

TEST_F(ReduceMultiOutputFusionTest, ReduceAndLoopDifferentShape) {
  const char* hlo = R"(
HloModule module

add {
  a = s32[] parameter(0)
  b = s32[] parameter(1)
  ROOT c = s32[] add(a, b)
}

fused_reduction {
  p = s32[10,20] parameter(0)
  z = s32[] constant(0)
  e = s32[10,20] negate(p)
  b = s32[200] bitcast(e)
  ROOT r = s32[] reduce(b, z), dimensions={0}, to_apply=add
}

fused_elementwise {
  p = s32[10,20] parameter(0)
  ROOT r = s32[10,20] negate(p)
}

ENTRY computation {
  p = s32[10,20] parameter(0)
  o1 = s32[10,20] fusion(p), kind=kLoop, calls=fused_elementwise
  o2 = s32[] fusion(p), kind=kInput, calls=fused_reduction
  ROOT out = (s32[10,20], s32[]) tuple(o1, o2)
}
)";

  CheckMultiOutputFusion(hlo, R"(
// CHECK: %fused_elementwise (p.1: s32[10,20]) -> (s32[10,20], s32[]) {
// CHECK-NEXT:   [[p_1_0:%[^ ]+]] = s32[10,20]{1,0} parameter(0)
// CHECK-NEXT:   [[r_1_1:%[^ ]+]] = s32[10,20]{1,0} negate([[p_1_0]])
// CHECK-NEXT:   [[e_2:%[^ ]+]].clone.1 = s32[10,20]{1,0} negate([[p_1_0]])
// CHECK-NEXT:   [[b_1_3:%[^ ]+]].clone.1 = s32[200]{0} bitcast([[e_2]].clone.1)
// CHECK-NEXT:   [[z_4:%[^ ]+]].clone.1 = s32[] constant(0)
// CHECK-NEXT:   [[r_5:%[^ ]+]].clone.1 = s32[] reduce([[b_1_3]].clone.1, [[z_4]].clone.1), dimensions={0}, to_apply=[[add_6:%[^ ]+]]
// CHECK-NEXT:   ROOT [[tuple_7:%[^ ]+]] = (s32[10,20]{1,0}, s32[]) tuple([[r_1_1]], [[r_5]].clone.1)
// CHECK-NEXT: }
  )");
}

TEST_F(ReduceMultiOutputFusionTest, ReduceAndLoopDifferentShapeDifferentType) {
  const char* hlo = R"(
HloModule module, entry_computation_layout={(s16[100,200]{1,0},s32[],s32[])->(s16[100,200]{1,0}, s32[])}

max {
  a = s32[] parameter(0)
  b = s32[] parameter(1)
  ROOT c = s32[] maximum(a, b)
}

fused_computation {
  one_5 = s32[] constant(1)
  one_b.5 = s32[100,200]{1,0} broadcast(one_5), dimensions={}
  param_1.15 = s16[100,200]{1,0} parameter(1)
  c.6 = s32[100,200]{1,0} convert(param_1.15)
  param_0.11 = s32[] parameter(0)
  b.6 = s32[100,200]{1,0} broadcast(param_0.11), dimensions={}
  d.5 = s32[100,200]{1,0} divide(c.6, b.6)
  a.6 = s32[100,200]{1,0} add(one_b.5, d.5)
  bitcast.1 = s32[20000]{0} bitcast(a.6)
  z_1 = s32[] constant(0)
  ROOT r.1 = s32[] reduce(bitcast.1, z_1), dimensions={0}, to_apply=max
}

fused_computation.1 {
  one_3 = s32[] constant(1)
  one_b.3 = s32[100,200]{1,0} broadcast(one_3), dimensions={}
  param_2.7 = s16[100,200]{1,0} parameter(2)
  c.4 = s32[100,200]{1,0} convert(param_2.7)
  param_1.10 = s32[] parameter(1)
  b.4 = s32[100,200]{1,0} broadcast(param_1.10), dimensions={}
  d.3 = s32[100,200]{1,0} divide(c.4, b.4)
  a.4 = s32[100,200]{1,0} add(one_b.3, d.3)
  param_0.8 = s32[] parameter(0)
  output_scale_broadcast.1 = s32[100,200]{1,0} broadcast(param_0.8), dimensions={}
  a_scaled.1 = s32[100,200]{1,0} multiply(a.4, output_scale_broadcast.1)
  ROOT a_scaled_converted.1 = s16[100,200]{1,0} convert(a_scaled.1)
}

ENTRY computation {
  output_scale = s32[] parameter(2)
  input_scale = s32[] parameter(1)
  p = s16[100,200]{1,0} parameter(0)
  fusion.1 = s16[100,200]{1,0} fusion(output_scale, input_scale, p), kind=kLoop, calls=fused_computation.1
  fusion = s32[] fusion(input_scale, p), kind=kInput, calls=fused_computation
  ROOT out = (s16[100,200]{1,0}, s32[]) tuple(fusion.1, fusion)
}
)";

  CheckMultiOutputFusion(hlo, R"(
// CHECK: %fused_computation.1 (param_0.8: s32[], param_1.10: s32[], param_2.7: s16[100,200]) -> (s16[100,200], s32[]) {
// CHECK-NEXT:   [[one_3_0:%[^ ]+]] = s32[] constant(1)
// CHECK-NEXT:   [[one_b_3_1:%[^ ]+]] = s32[100,200]{1,0} broadcast([[one_3_0]]), dimensions={}
// CHECK-NEXT:   [[param_2_7_2:%[^ ]+]] = s16[100,200]{1,0} parameter(2)
// CHECK-NEXT:   [[c_4_3:%[^ ]+]] = s32[100,200]{1,0} convert([[param_2_7_2]])
// CHECK-NEXT:   [[param_1_10_4:%[^ ]+]] = s32[] parameter(1)
// CHECK-NEXT:   [[b_4_5:%[^ ]+]] = s32[100,200]{1,0} broadcast([[param_1_10_4]]), dimensions={}
// CHECK-NEXT:   [[d_3_6:%[^ ]+]] = s32[100,200]{1,0} divide([[c_4_3]], [[b_4_5]])
// CHECK-NEXT:   [[a_4_7:%[^ ]+]] = s32[100,200]{1,0} add([[one_b_3_1]], [[d_3_6]])
// CHECK-NEXT:   [[param_0_8_8:%[^ ]+]] = s32[] parameter(0)
// CHECK-NEXT:   [[output_scale_broadcast_1_9:%[^ ]+]] = s32[100,200]{1,0} broadcast([[param_0_8_8]]), dimensions={}
// CHECK-NEXT:   [[a_scaled_1_10:%[^ ]+]] = s32[100,200]{1,0} multiply([[a_4_7]], [[output_scale_broadcast_1_9]])
// CHECK-NEXT:   [[a_scaled_converted_1_11:%[^ ]+]] = s16[100,200]{1,0} convert([[a_scaled_1_10]])
// CHECK-NEXT:   [[one_5_12:%[^ ]+]].clone.1 = s32[] constant(1)
// CHECK-NEXT:   [[one_b_5_13:%[^ ]+]].clone.1 = s32[100,200]{1,0} broadcast([[one_5_12]].clone.1), dimensions={}
// CHECK-NEXT:   [[c_6_14:%[^ ]+]].clone.1 = s32[100,200]{1,0} convert([[param_2_7_2]])
// CHECK-NEXT:   [[b_6_15:%[^ ]+]].clone.1 = s32[100,200]{1,0} broadcast([[param_1_10_4]]), dimensions={}
// CHECK-NEXT:   [[d_5_16:%[^ ]+]].clone.1 = s32[100,200]{1,0} divide([[c_6_14]].clone.1, [[b_6_15]].clone.1)
// CHECK-NEXT:   [[a_6_17:%[^ ]+]].clone.1 = s32[100,200]{1,0} add([[one_b_5_13]].clone.1, [[d_5_16]].clone.1)
// CHECK-NEXT:   [[bitcast_1_18:%[^ ]+]].clone.1 = s32[20000]{0} bitcast([[a_6_17]].clone.1)
// CHECK-NEXT:   [[z_1_19:%[^ ]+]].clone.1 = s32[] constant(0)
// CHECK-NEXT:   [[r_1_20:%[^ ]+]].clone.1 = s32[] reduce([[bitcast_1_18]].clone.1, [[z_1_19]].clone.1), dimensions={0}, to_apply=[[max_21:%[^ ]+]]
// CHECK-NEXT:   ROOT [[tuple_22:%[^ ]+]] = (s16[100,200]{1,0}, s32[]) tuple([[a_scaled_converted_1_11]], [[r_1_20]].clone.1)
// CHECK-NEXT: }
  )");
}

TEST_F(ReduceMultiOutputFusionTest, GetTupleElementMakeTupleSequence) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    fusion {
      p0 = s32[] parameter(0)
      p1 = s32[32] parameter(1)
      custom-call = (s16[], s32[], u32[]) custom-call(p1), custom_call_target="my_custom_call"
      get-tuple-element.0 = s16[] get-tuple-element(custom-call), index=0
      get-tuple-element.1 = s32[] get-tuple-element(custom-call), index=1
      bitcast = s32[1] bitcast(get-tuple-element.1)
      dynamic-update-slice = s32[32] dynamic-update-slice(p1, bitcast, p0)
      get-tuple-element.2 = u32[] get-tuple-element(custom-call), index=2
      ROOT tuple.30 = (s16[], s32[32], u32[]) tuple(get-tuple-element.0, dynamic-update-slice, get-tuple-element.2)
    }

    ENTRY entry{
      p0 = s32[] parameter(0)
      bitcast = s32[32] bitcast(p0)
      ROOT address_computation.7.0 = (s16[], s32[32], u32[]) fusion(p0, bitcast), kind=kCustom, calls=fusion
    }
  )")
                    .value();

  ASSERT_FALSE(mof_.Run(module.get()).value());
}

}  // namespace zkx::gpu
