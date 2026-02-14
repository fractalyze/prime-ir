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

#include "zkx/hlo/transforms/host_offload_legalize.h"

#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"

#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "zkx/base/logging.h"
#include "zkx/hlo/ir/hlo_computation.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/hlo/ir/hlo_opcode.h"
#include "zkx/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "zkx/hlo/testlib/pattern_matcher_gmock.h"
#include "zkx/layout_util.h"
#include "zkx/service/host_memory_offload_annotations.h"
#include "zkx/service/pattern_matcher.h"
#include "zkx/shape.h"
#include "zkx/shape_util.h"
#include "zkx/util.h"

namespace zkx {
namespace {

class HostOffloadLegalizeTest : public HloHardwareIndependentTestBase {
 protected:
  static constexpr int64_t kHostMemorySpaceColor{5};

  absl::StatusOr<bool> RunHostOffloadLegalize(HloModule* module) {
    TF_EXPECT_OK(verifier().Run(module).status());
    if (module->has_schedule()) {
      return absl::InternalError("Expected a non-scheduled module");
    }
    HostOffloadLegalize host_offload_legalize(kHostMemorySpaceColor,
                                              /*after_layout=*/true);
    return host_offload_legalize.Run(module);
  }

  void TestShapeHasMemorySpace(const Shape& shape, int64_t memory_space) {
    ASSERT_TRUE(shape.has_layout());
    EXPECT_EQ(shape.layout().memory_space(), memory_space);
  }

  bool HaveRemainingOffloadAnnotations(const HloModule* module) {
    for (const HloComputation* computation : module->computations()) {
      for (const HloInstruction* instruction : computation->instructions()) {
        if (instruction->IsCustomCall(
                {host_memory_offload_annotations::kMoveToHostCustomCallTarget,
                 host_memory_offload_annotations::
                     kMoveToDeviceCustomCallTarget})) {
          return true;
        }
      }
    }
    return false;
  }
};

TEST_F(HostOffloadLegalizeTest, TestWithAsyncCall) {
  const std::string& hlo_string = R"(
HloModule jit_update, entry_computation_layout={(s32[20,3,256,133]{2,3,1,0:T(8,128)S(5)})->(s32[20,3,256,133]{2,1,0,3:T(4,128)}, s32[4096]{0:T(1024)})}

%async_computation {
  %param_0 = s32[20,3,256,133] parameter(0)
  ROOT %offloaded-custom-call = s32[4096] custom-call(%param_0), custom_call_target="HostExecute"
}, execution_thread="host"

ENTRY main {
  %param.246 = s32[20,3,256,133] parameter(0)
  %async-start = ((s32[20,3,256,133]), s32[4096], u32[]) async-start(%param.246), async_execution_thread="host", calls=%async_computation
  %async-done = s32[4096] custom-call-done(%async-start)
  copy.16744 = s32[20,3,256,133]{2,1,0,3:T(4,128)} copy(param.246)
  custom-call.7832 = s32[20,3,256,133]{2,1,0,3:T(4,128)} custom-call(copy.16744), custom_call_target="MoveToDevice"
  ROOT tuple.16745 = (s32[20,3,256,133]{2,1,0,3:T(4,128)}, s32[4096]{0:T(1024)}) tuple(custom-call.7832, %async-done)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloadLegalize(module.get()));
  EXPECT_TRUE(changed);
  HloInstruction* custom_call =
      FindInstruction(module.get(), "custom-call.7832");
  ASSERT_NE(custom_call, nullptr);
  EXPECT_EQ(custom_call->users()[0]->opcode(), HloOpcode::kCopy);
  ZKX_VLOG_LINES(1, module->ToString());
}

TEST_F(HostOffloadLegalizeTest, NoCopyWithOptBarrierMoreElaborate) {
  const std::string& hlo_string = R"(
HloModule jit_f, entry_computation_layout={(s32[16,256]{0,1})->s32[16,256]{1,0}}

ENTRY main.24 {
  Arg_0.1 = s32[16,256]{0,1} parameter(0)
  // Note: negate replaces cosine/sine from XLA (ZKX has no kCos/kSin opcodes).
  negate.4 = s32[16,256]{0,1} negate(Arg_0.1)
  custom-call.5 = s32[16,256]{0,1} custom-call(negate.4), custom_call_target="MoveToHost"
  negate.3 = s32[16,256]{0,1} negate(Arg_0.1)
  negate.7 = s32[16,256]{0,1} negate(negate.3)
  custom-call.8 = s32[16,256]{0,1} custom-call(negate.7), custom_call_target="MoveToHost"
  negate.6 = s32[16,256]{0,1} negate(negate.3)
  negate.9 = s32[16,256]{0,1} negate(negate.6)
  custom-call.10 = s32[16,256]{0,1} custom-call(negate.9), custom_call_target="MoveToHost"
  constant.2 = s32[] constant(1)
  cp = s32[16,256]{1,0} copy(custom-call.8)
  tuple.11 = (s32[16,256]{0,1}, s32[16,256]{1,0}, s32[16,256]{0,1}, s32[]) tuple(custom-call.5, cp, custom-call.10, constant.2)
  opt-barrier.12 = (s32[16,256]{0,1}, s32[16,256]{1,0}, s32[16,256]{0,1}, s32[]) opt-barrier(tuple.11)
  get-tuple-element.16 = s32[] get-tuple-element(opt-barrier.12), index=3
  broadcast.20 = s32[16,256]{0,1} broadcast(get-tuple-element.16), dimensions={}
  get-tuple-element.15 = s32[16,256]{0,1} get-tuple-element(opt-barrier.12), index=2
  custom-call.19 = s32[16,256]{0,1} custom-call(get-tuple-element.15), custom_call_target="MoveToDevice"
  multiply.21 = s32[16,256]{0,1} multiply(broadcast.20, custom-call.19)
  cp2 = s32[16,256]{1,0} copy(multiply.21)
  get-tuple-element.14 = s32[16,256]{1,0} get-tuple-element(opt-barrier.12), index=1
  custom-call.18 = s32[16,256]{1,0} custom-call(get-tuple-element.14), custom_call_target="MoveToDevice"
  multiply.22 = s32[16,256]{1,0} multiply(cp2, custom-call.18)
  get-tuple-element.13 = s32[16,256]{0,1} get-tuple-element(opt-barrier.12), index=0
  custom-call.17 = s32[16,256]{0,1} custom-call(get-tuple-element.13), custom_call_target="MoveToDevice"
  cp3 = s32[16,256]{1,0} copy(custom-call.17)
  ROOT multiply.23 = s32[16,256]{1,0} multiply(multiply.22, cp3)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloadLegalize(module.get()));
  EXPECT_TRUE(changed);
  ZKX_VLOG_LINES(1, module->ToString());

  HloInstruction* custom_call = FindInstruction(module.get(), "custom-call.18");
  ASSERT_NE(custom_call, nullptr);
  EXPECT_EQ(custom_call->users()[0]->opcode(), HloOpcode::kCopy);
  EXPECT_EQ(custom_call->shape().layout(), LayoutUtil::MakeLayout({0, 1}));
  EXPECT_EQ(custom_call->users()[0]->shape().layout(),
            LayoutUtil::MakeLayout({1, 0}));
}

TEST_F(HostOffloadLegalizeTest, XposeCopyOnParameterStreaming) {
  const std::string& hlo_string = R"(
HloModule jit_f, entry_computation_layout={(s32[16,256]{0,1},s32[16,256]{0,1:T(8,128)S(5)})->s32[16,256]{1,0}}

ENTRY main.24 {
  Arg_0.1 = s32[16,256]{0,1} parameter(0)
  Arg_0.2 = s32[16,256]{0,1:T(8,128)} parameter(1)
  cp0 = s32[16,256]{1,0} copy(Arg_0.2)
  // Note: negate replaces cosine/sine from XLA (ZKX has no kCos/kSin opcodes).
  negate.4 = s32[16,256]{0,1} negate(Arg_0.1)
  custom-call.5 = s32[16,256]{0,1} custom-call(negate.4), custom_call_target="MoveToHost"
  negate.3 = s32[16,256]{0,1} negate(Arg_0.1)
  negate.7 = s32[16,256]{0,1} negate(negate.3)
  custom-call.8 = s32[16,256]{0,1} custom-call(negate.7), custom_call_target="MoveToHost"
  constant.2 = s32[] constant(1)
  cp1 = s32[16,256]{1,0} copy(custom-call.8)
  tuple.11 = (s32[16,256]{0,1}, s32[16,256]{1,0}, s32[16,256]{1,0}, s32[]) tuple(custom-call.5, cp1, cp0, constant.2)
  opt-barrier.12 = (s32[16,256]{0,1}, s32[16,256]{1,0}, s32[16,256]{1,0}, s32[]) opt-barrier(tuple.11)
  get-tuple-element.16 = s32[] get-tuple-element(opt-barrier.12), index=3
  broadcast.20 = s32[16,256]{0,1} broadcast(get-tuple-element.16), dimensions={}
  get-tuple-element.15 = s32[16,256]{1,0} get-tuple-element(opt-barrier.12), index=2
  custom-call.19 = s32[16,256]{1,0} custom-call(get-tuple-element.15), custom_call_target="MoveToDevice"
  multiply.21 = s32[16,256]{0,1} multiply(broadcast.20, custom-call.19)
  cp2 = s32[16,256]{1,0} copy(multiply.21)
  get-tuple-element.14 = s32[16,256]{1,0} get-tuple-element(opt-barrier.12), index=1
  custom-call.18 = s32[16,256]{1,0} custom-call(get-tuple-element.14), custom_call_target="MoveToDevice"
  multiply.22 = s32[16,256]{1,0} multiply(cp2, custom-call.18)
  get-tuple-element.13 = s32[16,256]{0,1} get-tuple-element(opt-barrier.12), index=0
  custom-call.17 = s32[16,256]{0,1} custom-call(get-tuple-element.13), custom_call_target="MoveToDevice"
  cp3 = s32[16,256]{1,0} copy(custom-call.17)
  ROOT multiply.23 = s32[16,256]{1,0} multiply(multiply.22, cp3)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloadLegalize(module.get()));
  EXPECT_TRUE(changed);
  ZKX_VLOG_LINES(1, module->ToString());

  HloInstruction* custom_call = FindInstruction(module.get(), "custom-call.18");
  ASSERT_NE(custom_call, nullptr);
  EXPECT_EQ(custom_call->users()[0]->opcode(), HloOpcode::kCopy);
  EXPECT_EQ(custom_call->shape().layout(), LayoutUtil::MakeLayout({0, 1}));
  EXPECT_EQ(custom_call->users()[0]->shape().layout(),
            LayoutUtil::MakeLayout({1, 0}));

  custom_call = FindInstruction(module.get(), "custom-call.19");
  ASSERT_NE(custom_call, nullptr);
  EXPECT_EQ(custom_call->users()[0]->opcode(), HloOpcode::kCopy);
  EXPECT_EQ(custom_call->shape().layout(),
            LayoutUtil::MakeLayout({0, 1}, {}, {}, {}, {Tile{{8, 128}}}));
  EXPECT_EQ(custom_call->users()[0]->shape().layout(),
            LayoutUtil::MakeLayout({1, 0}));
}

TEST_F(HostOffloadLegalizeTest, DUSSameLayoutForOperandAndUpdate_1) {
  const std::string& hlo_string = R"(
HloModule jit_f, entry_computation_layout={(s16[16,512,532]{1,2,0})->s16[1,16,512,532]{2,3,1,0}}

ENTRY main.24 {
  constant_0 = s32[] constant(0)
  cs0 = s16[] constant(0)
  broadcast = s16[20,16,512,532]{3,2,1,0}  broadcast(cs0), dimensions={}
  cp = s16[20,16,512,532]{3,2,1,0} copy(broadcast)
  custom-call.8 = s16[20,16,512,532]{3,2,1,0} custom-call(cp), custom_call_target="MoveToHost"
  copy = s16[20,16,512,532]{2,3,1,0} copy(custom-call.8)
  arg1 = s16[16,512,532]{1,2,0} parameter(0)
  copy.17302 = s16[16,512,532]{2,1,0} copy(arg1)
  bitcast.6100 = s16[1,16,512,532]{3,2,1,0} bitcast(copy.17302)
  copy.20241 = s16[1,16,512,532]{2,3,1,0} copy(bitcast.6100)
  custom-call.6720 = s16[1,16,512,532]{2,3,1,0} custom-call(copy.20241), custom_call_target="MoveToHost"
  dynamic-update-slice.6830 = s16[20,16,512,532]{2,3,1,0} dynamic-update-slice(copy, custom-call.6720, constant_0, constant_0, constant_0, constant_0)
  dynamic_slice_0 = s16[1,16,512,532]{2,3,1,0} dynamic-slice(dynamic-update-slice.6830, constant_0, constant_0, constant_0, constant_0), dynamic_slice_sizes={1,16,512,532}
  ROOT custom_call_0.1 = s16[1,16,512,532]{2,3,1,0} custom-call(dynamic_slice_0), custom_call_target="MoveToDevice"
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloadLegalize(module.get()));
  EXPECT_TRUE(changed);
  ZKX_VLOG_LINES(1, module->ToString());

  HloInstruction* dus =
      FindInstruction(module.get(), "dynamic-update-slice.6830");
  ASSERT_NE(dus, nullptr);
  EXPECT_EQ(dus->operand(0)->shape().layout(),
            dus->operand(1)->shape().layout());
  EXPECT_EQ(dus->shape().layout(), dus->operand(1)->shape().layout());

  const HloInstruction* custom_call =
      module->entry_computation()->root_instruction()->operand(0);
  EXPECT_TRUE(custom_call->IsCustomCall(
      host_memory_offload_annotations::kMoveToDeviceCustomCallTarget));
  EXPECT_EQ(custom_call->users()[0]->opcode(), HloOpcode::kCopy);
  EXPECT_EQ(custom_call->shape().layout(),
            LayoutUtil::MakeLayout({3, 2, 1, 0}));
  EXPECT_EQ(custom_call->users()[0]->shape().layout(),
            LayoutUtil::MakeLayout({2, 3, 1, 0}));
}

TEST_F(HostOffloadLegalizeTest, DUSSameLayoutForOperandAndUpdate_2) {
  const std::string& hlo_string = R"(
HloModule jit_f, entry_computation_layout={(s16[16,512,532]{1,2,0})->s16[1,16,512,532]{2,3,1,0}}

ENTRY main.24 {
  constant_0 = s32[] constant(0)
  cs0 = s16[] constant(0)
  broadcast = s16[20,16,512,532]{3,2,1,0}  broadcast(cs0), dimensions={}
  cp = s16[20,16,512,532]{3,2,1,0} copy(broadcast)
  custom-call.8 = s16[20,16,512,532]{3,2,1,0} custom-call(cp), custom_call_target="MoveToHost"
  copy = s16[20,16,512,532]{2,3,1,0} copy(custom-call.8)
  arg1 = s16[16,512,532]{1,2,0} parameter(0)
  copy.17302 = s16[16,512,532]{2,1,0} copy(arg1)
  custom-call.6720 = s16[16,512,532]{2,1,0} custom-call(copy.17302), custom_call_target="MoveToHost"
  bitcast.6100 = s16[1,16,512,532]{3,2,1,0} bitcast(custom-call.6720)
  copy.20241 = s16[1,16,512,532]{2,3,1,0} copy(bitcast.6100)
  dynamic-update-slice.6830 = s16[20,16,512,532]{2,3,1,0} dynamic-update-slice(copy, copy.20241, constant_0, constant_0, constant_0, constant_0)
  dynamic_slice_0 = s16[1,16,512,532]{2,3,1,0} dynamic-slice(dynamic-update-slice.6830, constant_0, constant_0, constant_0, constant_0), dynamic_slice_sizes={1,16,512,532}
  ROOT custom_call_0.1 = s16[1,16,512,532]{2,3,1,0} custom-call(dynamic_slice_0), custom_call_target="MoveToDevice"
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloadLegalize(module.get()));
  EXPECT_TRUE(changed);
  ZKX_VLOG_LINES(1, module->ToString());

  HloInstruction* dus =
      FindInstruction(module.get(), "dynamic-update-slice.6830");
  ASSERT_NE(dus, nullptr);
  EXPECT_EQ(dus->operand(0)->shape().layout(),
            dus->operand(1)->shape().layout());
  EXPECT_EQ(dus->shape().layout(), dus->operand(1)->shape().layout());

  const HloInstruction* custom_call =
      module->entry_computation()->root_instruction()->operand(0);
  EXPECT_TRUE(custom_call->IsCustomCall(
      host_memory_offload_annotations::kMoveToDeviceCustomCallTarget));
  EXPECT_EQ(custom_call->users()[0]->opcode(), HloOpcode::kCopy);
  EXPECT_EQ(custom_call->shape().layout(),
            LayoutUtil::MakeLayout({3, 2, 1, 0}));
  EXPECT_EQ(custom_call->users()[0]->shape().layout(),
            LayoutUtil::MakeLayout({2, 3, 1, 0}));
}

TEST_F(HostOffloadLegalizeTest, MoveCopyOverBitcast) {
  const std::string& hlo_string = R"(
HloModule jit_f, entry_computation_layout={(s16[1,1,16384,4,256]{4,3,2,1,0:T(4,128)(2,1)S(5)})->s16[1,16384,4,256]{3,1,2,0:T(8,128)(2,1)}}

ENTRY main {
  param = s16[1,1,16384,4,256]{4,3,2,1,0:T(4,128)(2,1)} parameter(0)
  copy = s16[1,1,16384,4,256]{4,2,3,1,0:T(8,128)(2,1)} copy(param)
  bitcast = s16[1,16384,4,256]{3,1,2,0:T(8,128)(2,1)} bitcast(copy)
  custom-call = s16[1,16384,4,256]{3,1,2,0:T(8,128)(2,1)} custom-call(bitcast), custom_call_target="MoveToDevice"
  ROOT add = s16[1,16384,4,256]{3,1,2,0:T(8,128)(2,1)} add(custom-call, custom-call)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloadLegalize(module.get()));

  EXPECT_TRUE(changed);
  ZKX_VLOG_LINES(1, module->ToString());
  HloInstruction* custom_call = FindInstruction(module.get(), "custom-call");
  EXPECT_EQ(custom_call->shape().layout(),
            LayoutUtil::MakeLayout({3, 2, 1, 0}, {}, {}, {},
                                   {Tile{{4, 128}}, Tile{{2, 1}}}));
  EXPECT_EQ(custom_call->users()[0]->opcode(), HloOpcode::kCopy);
  EXPECT_EQ(custom_call->users()[0]->shape().layout(),
            LayoutUtil::MakeLayout({3, 1, 2, 0}, {}, {}, {},
                                   {Tile{{8, 128}}, Tile{{2, 1}}}));
}

TEST_F(HostOffloadLegalizeTest, MoveCopyOverBitcast_2) {
  const std::string& hlo_string = R"(
HloModule jit_f, entry_computation_layout={(s16[1,16384,4,256]{3,2,1,0:T(4,128)(2,1)S(5)})->s16[1,1,16384,4,256]{4,3,1,2,0:T(8,128)(2,1)}}

ENTRY main {
  param = s16[1,16384,4,256]{3,2,1,0:T(4,128)(2,1)} parameter(0)
  copy = s16[1,16384,4,256]{2,3,1,0:T(8,128)(2,1)} copy(param)
  bitcast = s16[1,1,16384,4,256]{4,3,1,2,0:T(8,128)(2,1)} bitcast(copy)
  custom-call = s16[1,1,16384,4,256]{4,3,1,2,0:T(8,128)(2,1)} custom-call(bitcast), custom_call_target="MoveToDevice"
  ROOT add = s16[1,1,16384,4,256]{4,3,1,2,0:T(8,128)(2,1)} add(custom-call, custom-call)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloadLegalize(module.get()));

  EXPECT_TRUE(changed);
  ZKX_VLOG_LINES(1, module->ToString());
  HloInstruction* custom_call = FindInstruction(module.get(), "custom-call");
  EXPECT_EQ(custom_call->shape().layout(),
            LayoutUtil::MakeLayout({4, 3, 2, 1, 0}, {}, {}, {},
                                   {Tile{{4, 128}}, Tile{{2, 1}}}));
  EXPECT_EQ(custom_call->users()[0]->opcode(), HloOpcode::kCopy);
  EXPECT_EQ(custom_call->users()[0]->shape().layout(),
            LayoutUtil::MakeLayout({3, 4, 2, 1, 0}, {}, {}, {},
                                   {Tile{{8, 128}}, Tile{{2, 1}}}));
}

TEST_F(HostOffloadLegalizeTest, MoveCopyUp) {
  const std::string& hlo_string = R"(
HloModule jit_f, entry_computation_layout={(s16[4,4,128,8]{2,0,1,3:T(8,128)(2,1)})->s16[4,4,128,8]{2,3,1,0:T(8,128)(2,1)S(5)}}

ENTRY main {
  param = s16[4,4,128,8]{2,0,1,3:T(8,128)(2,1)} parameter(0)
  custom_call = s16[4,4,128,8]{2,0,1,3:T(8,128)(2,1)} custom-call(param), custom_call_target="MoveToHost"
  ROOT copy = s16[4,4,128,8]{2,3,1,0:T(8,128)(2,1)S(5)} copy(custom_call)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloadLegalize(module.get()));

  ASSERT_TRUE(changed);
  ZKX_VLOG_LINES(1, module->ToString());
  // Check that the custom call and copy have swapped places.
  HloInstruction* custom_call = FindInstruction(module.get(), "custom_call");
  EXPECT_TRUE(custom_call->IsRoot());
  EXPECT_TRUE(custom_call->IsCustomCall(
      host_memory_offload_annotations::kMoveToHostCustomCallTarget));
  const HloInstruction* copy = custom_call->operand(0);
  ASSERT_EQ(copy->opcode(), HloOpcode::kCopy);
  const HloInstruction* param = copy->operand(0);
  EXPECT_EQ(copy->opcode(), HloOpcode::kCopy);
  EXPECT_EQ(param->opcode(), HloOpcode::kParameter);
}

}  // namespace
}  // namespace zkx
