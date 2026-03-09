/* Copyright 2019 The OpenXLA Authors.
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

#include "zkx/service/gpu/transforms/reduction_layout_normalizer.h"

#include <optional>
#include <string_view>

#include "absl/status/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "zkx/hlo/testlib/hlo_hardware_independent_test_base.h"

namespace zkx::gpu {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

class ReductionLayoutNormalizerTest : public HloHardwareIndependentTestBase {
 public:
  void CheckReductionLayoutNormalizer(
      std::string_view hlo, std::optional<std::string_view> expected) {
    RunAndFilecheckHloRewrite(hlo, ReductionLayoutNormalizer{}, expected);
  }
};

TEST_F(ReductionLayoutNormalizerTest, LayoutCanonicalizerTest) {
  const char* hlo = R"(
HloModule ReduceWithLayoutChange

add {
  x0 = s32[] parameter(0)
  y0 = s32[] parameter(1)
  ROOT add0 = s32[] add(x0, y0)
}

ENTRY main {
  arg0 = s32[4,5,5,16,12,12,3,3]{2,3,5,4,0,7,6,1}  parameter(0)
  constant0 = s32[] constant(0)
  ROOT reduce0 = s32[4,5,16,12,12]{4,3,2,1,0} reduce(arg0, constant0),
    dimensions={1,6,7}, to_apply=add
}

)";

  CheckReductionLayoutNormalizer(hlo,
                                 R"(
// CHECK:  [[bitcast_0:%[^ ]+]] = s32[5,3,3,4,12,12,16,5]{7,6,5,4,3,2,1,0} bitcast([[arg0_1:%[^ ]+]])
// CHECK:  [[reduce_2:%[^ ]+]] = s32[4,12,12,16,5]{2,1,3,4,0} reduce([[bitcast_0]], [[constant0_3:%[^ ]+]]), dimensions={0,1,2}, to_apply=[[add_4:%[^ ]+]]
// CHECK:  ROOT [[bitcast_1_5:%[^ ]+]] = s32[4,5,16,12,12]{4,3,2,1,0} bitcast([[reduce_2]])
      )");
}

TEST_F(ReductionLayoutNormalizerTest, LayoutCanonicalizerTestVariadic) {
  const char* hlo = R"(
HloModule ReduceWithLayoutChangeVariadic


argmax {
  running_max = s32[] parameter(0)
  running_max_idx = u32[] parameter(1)
  current_value = s32[] parameter(2)
  current_value_idx = u32[] parameter(3)

  current = (s32[], u32[]) tuple(running_max, running_max_idx)
  potential = (s32[], u32[]) tuple(current_value, current_value_idx)

  cmp_code = pred[] compare(current_value, running_max), direction=GT

  new_max = s32[] select(cmp_code, current_value, running_max)
  new_idx = u32[] select(cmp_code, current_value_idx, running_max_idx)

  ROOT out = (s32[], u32[]) tuple(new_max, new_idx)
}

ENTRY main {
  arg0 = s32[4,5,5,16,12,12,3,3]{2,3,5,4,0,7,6,1}  parameter(0)
  idxs = u32[4,5,5,16,12,12,3,3]{2,3,5,4,0,7,6,1}  parameter(1)
  constant0 = s32[] constant(0)
  constant1 = u32[] constant(0)
  ROOT reduce0 = (
      s32[4,5,16,12,12]{4,3,2,1,0},
      u32[4,5,16,12,12]{4,3,2,1,0}
    ) reduce(arg0, idxs, constant0,constant1), dimensions={1,6,7}, to_apply=argmax
}


)";

  CheckReductionLayoutNormalizer(hlo,
                                 R"(
// CHECK:  [[arg0_0:%[^ ]+]] = s32[4,5,5,16,12,12,3,3]{2,3,5,4,0,7,6,1} parameter(0)
// CHECK:  [[bitcast_1:%[^ ]+]] = s32[5,3,3,4,12,12,16,5]{7,6,5,4,3,2,1,0} bitcast([[arg0_0]])
// CHECK:  [[idxs_2:%[^ ]+]] = u32[4,5,5,16,12,12,3,3]{2,3,5,4,0,7,6,1} parameter(1)
// CHECK:  [[bitcast_1_3:%[^ ]+]] = u32[5,3,3,4,12,12,16,5]{7,6,5,4,3,2,1,0} bitcast([[idxs_2]])
// CHECK:  [[reduce_4:%[^ ]+]] = (s32[4,12,12,16,5]{2,1,3,4,0}, u32[4,12,12,16,5]{2,1,3,4,0}) reduce([[bitcast_1]], [[bitcast_1_3]], [[constant0_5:%[^ ]+]], [[constant1_6:%[^ ]+]]), dimensions={0,1,2}, to_apply=[[argmax_7:%[^ ]+]]
// CHECK:  [[get_tuple_element_8:%[^ ]+]] = s32[4,12,12,16,5]{2,1,3,4,0} get-tuple-element([[reduce_4]]), index=0
// CHECK:  [[bitcast_2_9:%[^ ]+]] = s32[4,5,16,12,12]{4,3,2,1,0} bitcast([[get_tuple_element_8]])
// CHECK:  [[get_tuple_element_1_10:%[^ ]+]] = u32[4,12,12,16,5]{2,1,3,4,0} get-tuple-element([[reduce_4]]), index=1
// CHECK:  [[bitcast_3_11:%[^ ]+]] = u32[4,5,16,12,12]{4,3,2,1,0} bitcast([[get_tuple_element_1_10]])
// CHECK:  ROOT [[tuple_12:%[^ ]+]] = (s32[4,5,16,12,12]{4,3,2,1,0}, u32[4,5,16,12,12]{4,3,2,1,0}) tuple([[bitcast_2_9]], [[bitcast_3_11]])
      )");
}

TEST_F(ReductionLayoutNormalizerTest,
       LayoutCanonicalizerTestVariadicDifferentLayouts) {
  const char* hlo = R"(
HloModule ReduceWithLayoutChangeVariadicDifferent

argmax {
  running_max = s32[] parameter(0)
  running_max_idx = u32[] parameter(1)
  current_value = s32[] parameter(2)
  current_value_idx = u32[] parameter(3)

  current = (s32[], u32[]) tuple(running_max, running_max_idx)
  potential = (s32[], u32[]) tuple(current_value, current_value_idx)

  cmp_code = pred[] compare(current_value, running_max), direction=GT

  new_max = s32[] select(cmp_code, current_value, running_max)
  new_idx = u32[] select(cmp_code, current_value_idx, running_max_idx)

  ROOT out = (s32[], u32[]) tuple(new_max, new_idx)
}

ENTRY main {
  arg0 = s32[2,3,4,7]{2,1,0,3}  parameter(0)
  idxs = u32[2,3,4,7]{3,2,1,0}  parameter(1)
  constant0 = s32[] constant(0)
  constant1 = u32[] constant(0)
  ROOT reduce0 = (
      s32[2,3,4]{2,1,0},
      u32[2,3,4]{2,1,0}
    ) reduce(arg0, idxs, constant0,constant1), dimensions={3}, to_apply=argmax
}


)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  ReductionLayoutNormalizer normalizer;
  EXPECT_THAT(normalizer.Run(module.get()),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("Layout assignment")));
}

}  // namespace
}  // namespace zkx::gpu
