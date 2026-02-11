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

#include "zkx/tools/hlo_extractor.h"

#include <memory>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "xla/tsl/platform/statusor.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/hlo/ir/hlo_opcode.h"
#include "zkx/hlo/parser/hlo_parser.h"
#include "zkx/hlo/utils/hlo_matchers.h"
#include "zkx/shape_util.h"
#include "zkx/tests/hlo_test_base.h"
#include "zkx/zkx_data.pb.h"

namespace zkx {
namespace {

namespace op = testing::opcode_matchers;

using HloExtractorTest = HloTestBase;

TEST_F(HloExtractorTest, ExtractTopLevel) {
  const std::string& hlo_string = R"(
HloModule test

ENTRY %entry {
  param.0 = s32[4]{0} parameter(0)
  negate = s32[4]{0} negate(s32[4]{0} param.0)
  ROOT sign = s32[4]{0} sign(s32[4]{0} negate)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));

  {
    auto extracted_module =
        ExtractModule(FindInstruction(hlo_module.get(), "sign"));
    EXPECT_THAT(extracted_module->entry_computation()->root_instruction(),
                op::Sign(op::Negate(op::Parameter(0))));
  }

  {
    auto extracted_module =
        ExtractModule(FindInstruction(hlo_module.get(), "sign"), /*height=*/0);
    EXPECT_THAT(extracted_module->entry_computation()->root_instruction(),
                op::Sign(op::Parameter(0)));
  }

  {
    auto extracted_module = ExtractModule(
        FindInstruction(hlo_module.get(), "negate"), /*height=*/0);
    EXPECT_THAT(extracted_module->entry_computation()->root_instruction(),
                op::Negate(op::Parameter(0)));
  }
}

TEST_F(HloExtractorTest, ExtractDag) {
  const std::string& hlo_string = R"(
HloModule test

ENTRY %entry {
  param.0 = s32[4]{0} parameter(0)
  sign = s32[4]{0} sign(s32[4]{0} param.0)
  negate = s32[4]{0} negate(s32[4]{0} sign)
  negate2 = s32[4]{0} negate(s32[4]{0} negate)
  ROOT add = s32[4]{0} add(s32[4]{0} negate, s32[4]{0} negate2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));

  {
    auto extracted_module =
        ExtractModule(FindInstruction(hlo_module.get(), "negate2"));
    EXPECT_THAT(extracted_module->entry_computation()->root_instruction(),
                op::Negate(op::Negate(op::Sign(op::Parameter(0)))));
  }

  {
    auto extracted_module =
        ExtractModule(FindInstruction(hlo_module.get(), "add"), /*height=*/0);
    EXPECT_THAT(extracted_module->entry_computation()->root_instruction(),
                op::Add(op::Parameter(0), op::Parameter(1)));
  }

  {
    auto extracted_module =
        ExtractModule(FindInstruction(hlo_module.get(), "add"), /*height=*/1);
    EXPECT_THAT(extracted_module->entry_computation()->root_instruction(),
                op::Add(op::Negate(op::Parameter(0)),
                        op::Negate(op::Negate(op::Parameter(0)))));
  }

  {
    auto extracted_module =
        ExtractModule(FindInstruction(hlo_module.get(), "add"), /*height=*/2);
    EXPECT_THAT(extracted_module->entry_computation()->root_instruction(),
                op::Add(op::Negate(op::Sign(op::Parameter(0))),
                        op::Negate(op::Negate(op::Sign(op::Parameter(0))))));
  }
}

TEST_F(HloExtractorTest, ExtractWithConstant) {
  const std::string& hlo_string = R"(
HloModule test

ENTRY %entry {
  p = s32[4]{0} parameter(0)
  negate = s32[4]{0} negate(p)
  c = s32[4]{0} constant({1, 2, 3, 4})
  ROOT add = s32[4]{0} add(negate, c)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));

  {
    auto extracted_module =
        ExtractModule(FindInstruction(hlo_module.get(), "add"), /*height=*/0);
    EXPECT_THAT(extracted_module->entry_computation()->root_instruction(),
                op::Add(op::Parameter(0), op::Parameter(1)));
  }

  {
    auto extracted_module =
        ExtractModule(FindInstruction(hlo_module.get(), "add"), /*height=*/1);
    EXPECT_THAT(extracted_module->entry_computation()->root_instruction(),
                op::Add(op::Negate(op::Parameter(0)), op::Constant()));
  }
}

TEST_F(HloExtractorTest, ExtractFromMultipleComputation) {
  const std::string& hlo_string = R"(
  HloModule axpy_module
    calculate_alpha {
      c.1 = s32[] constant(1)
      c.2 = s32[] constant(2)
      add.0 = s32[] add(c.1, c.2)
      c.3 = s32[] constant(4)
      ROOT ret = s32[] subtract(add.0, c.3)
    }

    ENTRY axpy_computation {
      alpha = s32[] call(), to_apply=calculate_alpha
      broadcast = s32[10] broadcast(alpha), dimensions={}
      x = s32[10] parameter(0)
      ax = s32[10] multiply(broadcast, x)
      y = s32[10] parameter(1)
      ROOT add.1 = s32[10] add(ax, y)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));

  HloInstruction* inst = FindInstruction(hlo_module.get(), "add.0");
  EXPECT_THAT(inst, op::Add());

  auto extract_selector = [&inst](const HloInstruction* hlo_inst) {
    return hlo_inst != inst;
  };

  // Exclude `add.0 = s32[] add(c.1, c.2)` from computation `calculate_alpha`,
  // and replace it with a constant.
  {
    auto replace_type_selector = [](const HloInstruction* hlo_inst) {
      return ReplaceType::kReplaceConst;
    };

    auto extracted_module =
        ExtractModule(hlo_module->entry_computation()->root_instruction(),
                      /*height=*/-1, /*extract_selector=*/extract_selector,
                      /*replace_type_selector=*/replace_type_selector,
                      /*cross_computation=*/true);
    EXPECT_EQ(extracted_module->computation_count(), 2);
    auto calculate_alpha_root_instruction =
        FindComputation(extracted_module.get(), "calculate_alpha")
            ->root_instruction();
    EXPECT_THAT(calculate_alpha_root_instruction,
                op::Subtract(op::Constant(), op::Constant()));
  }

  // Exclude `add.0 = s32[] add(c.1, c.2)` from computation `calculate_alpha`,
  // and replace it with a broadcasted zero.
  {
    auto replace_type_selector = [](const HloInstruction* hlo_inst) {
      return ReplaceType::kReplaceZeroBroadcast;
    };

    auto extracted_module =
        ExtractModule(hlo_module->entry_computation()->root_instruction(),
                      /*height=*/-1, /*extract_selector=*/extract_selector,
                      /*replace_type_selector=*/replace_type_selector,
                      /*cross_computation=*/true);
    EXPECT_EQ(extracted_module->computation_count(), 2);
    auto calculate_alpha_root_instruction =
        FindComputation(extracted_module.get(), "calculate_alpha")
            ->root_instruction();
    EXPECT_THAT(calculate_alpha_root_instruction,
                op::Subtract(op::Broadcast(op::Constant()), op::Constant()));
  }
}

TEST_F(HloExtractorTest, HloSelector) {
  const std::string& hlo_string = R"(
  HloModule axpy_module
    calculate_alpha {
      c.1 = s32[] constant(1)
      c.2 = s32[] constant(2)
      c.3 = s32[] add(c.1, c.2)
      c.4 = s32[] constant(4)
      ROOT ret = s32[] multiply(c.4, c.3)
    }

    ENTRY axpy_computation {
      p.0 = s32[10] parameter(0)
      p.1 = s32[10] parameter(1)
      add.0 = s32[10] add(p.0, p.1)
      alpha = s32[] call(), to_apply=calculate_alpha
      broadcast = s32[10] broadcast(alpha), dimensions={}
      p.2 = s32[10] parameter(2)
      y = s32[10] multiply(broadcast, p.2)
      x = s32[10] subtract(y, add.0)
      p.3 = s32[10] parameter(3)
      ROOT add = s32[10] add(x, p.3)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));

  // Find the kSubtract instruction in ENTRY computation
  HloInstruction* inst =
      FindInstruction(hlo_module.get(), HloOpcode::kSubtract);
  EXPECT_NE(inst, nullptr);
  EXPECT_THAT(inst, op::Subtract(op::Multiply(), op::Add()));

  {
    auto hlo_selector = [](const HloInstruction* hlo_inst) -> bool {
      return hlo_inst->opcode() != HloOpcode::kCall;
    };
    auto extracted_module = ExtractModule(inst, /*height=*/-1, hlo_selector);
    EXPECT_EQ(extracted_module->computation_count(), 1);
    EXPECT_THAT(extracted_module->entry_computation()->root_instruction(),
                op::Subtract(op::Multiply(op::Broadcast(op::Parameter()),
                                          op::Parameter()),
                             op::Add(op::Parameter(), op::Parameter())));
  }

  {
    auto hlo_selector = [](const HloInstruction* hlo_inst) -> bool {
      return hlo_inst->opcode() != HloOpcode::kBroadcast;
    };
    auto extracted_module = ExtractModule(inst, /*height=*/2, hlo_selector);
    EXPECT_EQ(extracted_module->computation_count(), 1);
    EXPECT_THAT(extracted_module->entry_computation()->root_instruction(),
                op::Subtract(op::Multiply(op::Parameter(), op::Parameter()),
                             op::Add(op::Parameter(), op::Parameter())));
  }

  {
    auto hlo_selector = [](const HloInstruction* hlo_inst) -> bool {
      return hlo_inst->opcode() != HloOpcode::kBroadcast;
    };
    auto replace_type_selector =
        [](const HloInstruction* hlo_inst) -> ReplaceType {
      return ReplaceType::kReplaceConst;
    };
    auto extracted_module =
        ExtractModule(inst, /*height=*/2, hlo_selector, replace_type_selector);
    EXPECT_EQ(extracted_module->computation_count(), 1);
    EXPECT_THAT(extracted_module->entry_computation()->root_instruction(),
                op::Subtract(op::Multiply(op::Constant(), op::Parameter()),
                             op::Add(op::Parameter(), op::Parameter())));
  }

  {
    auto hlo_selector = [](const HloInstruction* hlo_inst) -> bool {
      return hlo_inst->opcode() != HloOpcode::kAdd;
    };
    auto extracted_module = ExtractModule(inst, /*height=*/-1, hlo_selector);
    // Here the extracted module should contain Computation: calculate_alpha
    EXPECT_EQ(extracted_module->computation_count(), 2);
    EXPECT_THAT(extracted_module->entry_computation()->root_instruction(),
                op::Subtract(op::Multiply(), op::Parameter()));
  }

  {
    auto hlo_selector = [](const HloInstruction* hlo_inst) -> bool {
      return hlo_inst->opcode() != HloOpcode::kSubtract;
    };
    auto replace_type_selector =
        [](const HloInstruction* hlo_inst) -> ReplaceType {
      return ReplaceType::kReplaceConst;
    };
    auto extracted_module =
        ExtractModule(hlo_module->entry_computation()->root_instruction(),
                      /*height=*/2, hlo_selector, replace_type_selector);
    EXPECT_EQ(extracted_module->computation_count(), 1);
    EXPECT_THAT(extracted_module->entry_computation()->root_instruction(),
                op::Add(op::Constant(), op::Parameter()));
  }

  {
    auto hlo_selector = [](const HloInstruction* hlo_inst) -> bool {
      if (hlo_inst->opcode() != HloOpcode::kBroadcast &&
          hlo_inst->opcode() != HloOpcode::kAdd) {
        return true;
      }
      return false;
    };
    auto replace_type_selector =
        [](const HloInstruction* hlo_inst) -> ReplaceType {
      if (hlo_inst->opcode() == HloOpcode::kBroadcast) {
        return ReplaceType::kReplaceConst;
      }
      return ReplaceType::kReplaceParam;
    };
    auto extracted_module =
        ExtractModule(inst, /*height=*/2, hlo_selector, replace_type_selector);
    EXPECT_EQ(extracted_module->computation_count(), 1);
    EXPECT_THAT(extracted_module->entry_computation()->root_instruction(),
                op::Subtract(op::Multiply(op::Constant(), op::Parameter()),
                             op::Parameter()));
  }
}

TEST_F(HloExtractorTest, ReplaceTupleWithConstant) {
  const std::string& hlo_string = R"(
HloModule test

ENTRY %entry {
  param.0 = s32[4]{0} parameter(0)
  tuple.0 = (s32[4]{0}, s32[4]{0}) tuple(s32[4]{0} param.0, s32[4]{0} param.0)
  negate = s32[4]{0} negate(s32[4]{0} param.0)
  tuple.1 = ((s32[4]{0}, s32[4]{0}), s32[4]{0}) tuple(tuple.0, negate)
  element = s32[4]{0} get-tuple-element(((s32[4]{0}, s32[4]{0}), s32[4]{0}) tuple.1), index=1
  ROOT add = s32[4]{0} add(element, param.0)
}
)";

  // NOTE: This module is intentionally invalid (negate producing a tuple shape)
  // to test tuple replacement logic, so we use unverified parsing.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));

  // Testing kReplaceConst.
  {
    auto hlo_selector = [](const HloInstruction* hlo_inst) -> bool {
      return hlo_inst->opcode() != HloOpcode::kTuple;
    };
    auto replace_type_selector =
        [](const HloInstruction* hlo_inst) -> ReplaceType {
      return ReplaceType::kReplaceConst;
    };
    auto extracted_module =
        ExtractModule(FindInstruction(hlo_module.get(), "add"),
                      /*height=*/-1, hlo_selector, replace_type_selector);
    EXPECT_THAT(extracted_module->entry_computation()->root_instruction(),
                op::Add(op::GetTupleElement(op::Constant()), op::Parameter()));
  }

  // Testing kReplaceZeroBroadcast -- replace a scalar (`element`) with
  // a broadcasted zero.
  {
    auto hlo_selector = [](const HloInstruction* hlo_inst) -> bool {
      return hlo_inst->opcode() != HloOpcode::kGetTupleElement;
    };
    auto replace_type_selector =
        [](const HloInstruction* hlo_inst) -> ReplaceType {
      return ReplaceType::kReplaceZeroBroadcast;
    };
    auto extracted_module =
        ExtractModule(FindInstruction(hlo_module.get(), "add"),
                      /*height=*/-1, hlo_selector, replace_type_selector);
    EXPECT_THAT(extracted_module->entry_computation()->root_instruction(),
                op::Add(op::Broadcast(), op::Parameter()));
  }

  // Testing kReplaceRandomBroadcast -- replace a scalar (`element`) with a
  // broadcasted random constant.
  {
    auto hlo_selector = [](const HloInstruction* hlo_inst) -> bool {
      return hlo_inst->opcode() != HloOpcode::kGetTupleElement;
    };
    auto replace_type_selector =
        [](const HloInstruction* hlo_inst) -> ReplaceType {
      return ReplaceType::kReplaceRandomBroadcast;
    };
    auto extracted_module =
        ExtractModule(FindInstruction(hlo_module.get(), "add"),
                      /*height=*/-1, hlo_selector, replace_type_selector);
    EXPECT_THAT(extracted_module->entry_computation()->root_instruction(),
                op::Add(op::Broadcast(), op::Parameter()));
  }

  // Testing kReplaceZeroBroadcast -- replace a tuple op (`tuple.1`) with zeros.
  {
    auto hlo_selector = [](const HloInstruction* hlo_inst) -> bool {
      return hlo_inst->opcode() != HloOpcode::kTuple;
    };
    auto replace_type_selector =
        [](const HloInstruction* hlo_inst) -> ReplaceType {
      return ReplaceType::kReplaceZeroBroadcast;
    };
    auto extracted_module =
        ExtractModule(FindInstruction(hlo_module.get(), "add"),
                      /*height=*/-1, hlo_selector, replace_type_selector);
    EXPECT_THAT(
        extracted_module->entry_computation()->root_instruction(),
        op::Add(op::GetTupleElement(op::Tuple(op::Tuple(), op::Broadcast())),
                op::Parameter()));
  }

  // Testing kReplaceRandomBroadcast -- replace a tuple op (`tuple.1`) with a
  // broadcasted random constant.
  {
    auto hlo_selector = [](const HloInstruction* hlo_inst) -> bool {
      return hlo_inst->opcode() != HloOpcode::kTuple;
    };
    auto replace_type_selector =
        [](const HloInstruction* hlo_inst) -> ReplaceType {
      return ReplaceType::kReplaceRandomBroadcast;
    };
    auto extracted_module =
        ExtractModule(FindInstruction(hlo_module.get(), "add"),
                      /*height=*/-1, hlo_selector, replace_type_selector);
    EXPECT_THAT(
        extracted_module->entry_computation()->root_instruction(),
        op::Add(op::GetTupleElement(op::Tuple(op::Tuple(), op::Broadcast())),
                op::Parameter()));
  }
}

TEST_F(HloExtractorTest, TestWithCalledComputationsAndFusion) {
  const char* hlo = R"(
  HloModule test
  computation.1 {
    p.0 = s32[] parameter(0)
    p.1 = s32[] parameter(1)
    ROOT tuple = (s32[], s32[]) tuple(p.0, p.1)
  }
  computation.2 {
    p.0 = s32[] parameter(0)
    p.1 = s32[] parameter(1)
    ROOT tuple = (s32[], s32[]) tuple(p.0, p.1)
  }
  ENTRY main {
    p.0 = s32[] parameter(0)
    p.1 = s32[] parameter(1)
    p.2 = s32[] parameter(2)
    call.1 = (s32[], s32[]) call(p.0, p.1), to_apply=computation.1
    fused.1 = (s32[], s32[]) fusion(p.0, p.2), kind=kInput, calls=computation.2
    gte.1 = get-tuple-element(call.1), index=0
    gte.2 = get-tuple-element(fused.1), index=0
    ROOT tuple = tuple(gte.1, gte.2)
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  auto extracted =
      ExtractModule(module->entry_computation()->root_instruction(), -1,
                    nullptr, nullptr, false, true);
  EXPECT_THAT(extracted->entry_computation()->root_instruction(),
              op::Tuple(op::Parameter(0), op::Parameter(0)));
}

TEST_F(HloExtractorTest, TestInvalidModule) {
  constexpr std::string_view hlo = R"(
HloModule main

computation {
  ROOT arg.0 = s32[16] parameter(0)
}

ENTRY main {
  arg.0 = s32[16] parameter(0)
  ROOT call.0 = call(arg.0), to_apply=computation
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));

  HloInstruction* arg0 =
      FindComputation(module.get(), "computation")->root_instruction();
  // Create invalid operand shape.
  *arg0->mutable_shape() = ShapeUtil::MakeShape(S32, {4});

  auto extracted =
      ExtractModule(module->entry_computation()->root_instruction(), -1,
                    nullptr, nullptr, false, false, false);

  // Restore the operand shape to be valid.
  *arg0->mutable_shape() = ShapeUtil::MakeShape(S32, {16});
}

}  // namespace
}  // namespace zkx
