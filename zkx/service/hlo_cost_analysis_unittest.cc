/* Copyright 2026 The ZKX Authors.

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

#include "zkx/service/hlo_cost_analysis.h"

#include <cstdint>

#include "gtest/gtest.h"

#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/shape_util.h"
#include "zkx/tests/hlo_test_base.h"

namespace zkx {
namespace {

class HloCostAnalysisTest : public HloTestBase {
 protected:
  HloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction() const {
    return [](const Shape& shape) {
      return ShapeUtil::ByteSizeOf(shape, /*pointer_size=*/8);
    };
  }
};

TEST_F(HloCostAnalysisTest, Add) {
  const char* hlo_text = R"(
    HloModule test_module

    ENTRY main {
      p0 = s32[10] parameter(0)
      p1 = s32[10] parameter(1)
      ROOT add = s32[10] add(p0, p1)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  HloCostAnalysis analysis(ShapeSizeBytesFunction());
  TF_ASSERT_OK(
      module->entry_computation()->root_instruction()->Accept(&analysis));

  // 10 elements, each add is 1 FLOP
  EXPECT_EQ(analysis.flop_count(), 10);
  // Reading 2 inputs + writing 1 output, each 10 * 4 bytes
  EXPECT_EQ(analysis.bytes_accessed(), 10 * 4 * 3);
}

TEST_F(HloCostAnalysisTest, Multiply) {
  const char* hlo_text = R"(
    HloModule test_module

    ENTRY main {
      p0 = s32[100] parameter(0)
      p1 = s32[100] parameter(1)
      ROOT mul = s32[100] multiply(p0, p1)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  HloCostAnalysis analysis(ShapeSizeBytesFunction());
  TF_ASSERT_OK(
      module->entry_computation()->root_instruction()->Accept(&analysis));

  // 100 elements, each multiply is 1 FLOP
  EXPECT_EQ(analysis.flop_count(), 100);
}

TEST_F(HloCostAnalysisTest, Dot) {
  const char* hlo_text = R"(
    HloModule test_module

    ENTRY main {
      p0 = s32[10, 20] parameter(0)
      p1 = s32[20, 30] parameter(1)
      ROOT dot = s32[10, 30] dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  HloCostAnalysis analysis(ShapeSizeBytesFunction());
  TF_ASSERT_OK(
      module->entry_computation()->root_instruction()->Accept(&analysis));

  // Matrix multiply: 10 * 30 output elements, each requires 20 multiply-adds
  // = 10 * 30 * 20 * 2 = 12000 FLOPs
  EXPECT_EQ(analysis.flop_count(), 12000);
}

TEST_F(HloCostAnalysisTest, Reduce) {
  const char* hlo_text = R"(
    HloModule test_module

    add_computation {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      ROOT add = s32[] add(lhs, rhs)
    }

    ENTRY main {
      p0 = s32[100, 10] parameter(0)
      init = s32[] constant(0)
      ROOT reduce = s32[100] reduce(p0, init), dimensions={1}, to_apply=add_computation
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  HloCostAnalysis analysis(ShapeSizeBytesFunction());
  TF_ASSERT_OK(
      module->entry_computation()->root_instruction()->Accept(&analysis));

  // Reduce over dimension of size 10: 100 output elements, each requires 9
  // adds = 100 * 9 = 900 FLOPs
  EXPECT_EQ(analysis.flop_count(), 900);
}

TEST_F(HloCostAnalysisTest, Broadcast) {
  const char* hlo_text = R"(
    HloModule test_module

    ENTRY main {
      p0 = s32[10] parameter(0)
      ROOT broadcast = s32[10, 20] broadcast(p0), dimensions={0}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  HloCostAnalysis analysis(ShapeSizeBytesFunction());
  TF_ASSERT_OK(
      module->entry_computation()->root_instruction()->Accept(&analysis));

  // Broadcast has 0 FLOPs
  EXPECT_EQ(analysis.flop_count(), 0);
}

TEST_F(HloCostAnalysisTest, Reshape) {
  const char* hlo_text = R"(
    HloModule test_module

    ENTRY main {
      p0 = s32[10, 20] parameter(0)
      ROOT reshape = s32[200] reshape(p0)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  HloCostAnalysis analysis(ShapeSizeBytesFunction());
  TF_ASSERT_OK(
      module->entry_computation()->root_instruction()->Accept(&analysis));

  // Reshape has 0 FLOPs
  EXPECT_EQ(analysis.flop_count(), 0);
}

TEST_F(HloCostAnalysisTest, WholeModule) {
  const char* hlo_text = R"(
    HloModule test_module

    ENTRY main {
      p0 = s32[10] parameter(0)
      p1 = s32[10] parameter(1)
      add = s32[10] add(p0, p1)
      ROOT mul = s32[10] multiply(add, p1)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  HloCostAnalysis analysis(ShapeSizeBytesFunction());
  TF_ASSERT_OK(module->entry_computation()->Accept(&analysis));

  // 10 adds + 10 multiplies = 20 FLOPs
  EXPECT_EQ(analysis.flop_count(), 20);
}

}  // namespace
}  // namespace zkx
