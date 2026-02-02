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

#include "zkx/service/gpu/transforms/layout_assignment.h"

#include <cstdint>
#include <memory>

#include "absl/status/status_matchers.h"
#include "absl/types/span.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "xla/tsl/platform/statusor.h"
#include "zkx/hlo/ir/hlo_computation.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/hlo/ir/hlo_opcode.h"
#include "zkx/hlo/parser/hlo_parser.h"
#include "zkx/hlo/testlib/pattern_matcher_gmock.h"
#include "zkx/layout.h"
#include "zkx/layout_util.h"
#include "zkx/service/computation_layout.h"
#include "zkx/service/pattern_matcher.h"
#include "zkx/shape.h"
#include "zkx/shape_layout.h"
#include "zkx/shape_util.h"
#include "zkx/stream_executor/device_description.h"
#include "zkx/tests/hlo_test_base.h"
#include "zkx/zkx_data.pb.h"

namespace zkx {
namespace gpu {
namespace {

namespace m = ::zkx::match;
using ::absl_testing::IsOkAndHolds;

class LayoutAssignmentTest : public HloTestBase {
 public:
  const se::DeviceDescription& GetDeviceDescription() {
    return backend().default_stream_executor()->GetDeviceDescription();
  }
};

TEST_F(LayoutAssignmentTest, Elementwise) {
  Shape ashape = ShapeUtil::MakeShape(S32, {42, 12});
  Shape ashape_in_row_major(ashape);
  Shape ashape_in_col_major(ashape);
  *ashape_in_row_major.mutable_layout() = LayoutUtil::MakeLayout({1, 0});
  *ashape_in_col_major.mutable_layout() = LayoutUtil::MakeLayout({0, 1});

  // Enumerate all possible combinations of layouts.
  for (const Shape& lhs_shape_with_layout :
       {ashape_in_row_major, ashape_in_col_major}) {
    for (const Shape& rhs_shape_with_layout :
         {ashape_in_row_major, ashape_in_col_major}) {
      for (const Shape& result_shape_with_layout :
           {ashape_in_row_major, ashape_in_col_major}) {
        // GpuLayoutAssignment should assign the same layout to "add" and its
        // two operands.
        auto builder = HloComputation::Builder(TestName());
        auto x = builder.AddInstruction(
            HloInstruction::CreateParameter(0, ashape, "x"));
        auto y = builder.AddInstruction(
            HloInstruction::CreateParameter(1, ashape, "y"));
        auto add = builder.AddInstruction(
            HloInstruction::CreateBinary(ashape, HloOpcode::kAdd, x, y));
        auto module = CreateNewVerifiedModule();
        HloComputation* computation =
            module->AddEntryComputation(builder.Build(add));

        ComputationLayout computation_layout(
            computation->ComputeProgramShape());
        *computation_layout.mutable_parameter_layout(0) =
            ShapeLayout(lhs_shape_with_layout);
        *computation_layout.mutable_parameter_layout(1) =
            ShapeLayout(rhs_shape_with_layout);
        *computation_layout.mutable_result_layout() =
            ShapeLayout(result_shape_with_layout);

        GpuLayoutAssignment layout_assignment(&computation_layout,
                                              GetDeviceDescription());
        EXPECT_THAT(layout_assignment.Run(module.get()), IsOkAndHolds(true));

        for (const HloInstruction* operand : add->operands()) {
          EXPECT_TRUE(LayoutUtil::Equal(add->shape().layout(),
                                        operand->shape().layout()));
        }
      }
    }
  }
}

TEST_F(LayoutAssignmentTest, DotLayoutUnchangedIfValid) {
  const char* hlo_text = R"(
  HloModule DotLayout
  ENTRY dot {
    p0 = s32[5,2,3]{1,2,0} parameter(0)
    p1 = s32[5,3,4]{1,2,0} parameter(1)
    ROOT dot.1330.10585 = s32[5,2,4]{2,1,0} dot(p0, p1),
      lhs_batch_dims={0}, lhs_contracting_dims={2},
      rhs_batch_dims={0}, rhs_contracting_dims={1}
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));

  ComputationLayout computation_layout(
      module->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(&computation_layout,
                                        GetDeviceDescription());
  EXPECT_THAT(layout_assignment.Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Dot(m::Op().WithShape(S32, {5, 2, 3}, {1, 2, 0}),
                                m::Op().WithShape(S32, {5, 3, 4}, {1, 2, 0}))
                             .WithShape(S32, {5, 2, 4}, {2, 1, 0})));
}

TEST_F(LayoutAssignmentTest, DotLayoutSetToDefaultIfDefaultValid) {
  const char* hlo_text = R"(
  HloModule DotLayout
  ENTRY dot {
    p0 = s32[5,3,2] parameter(0)
    p1 = s32[5,4,3]{0,1,2} parameter(1)
    ROOT dot.1330.10585 = s32[5,2,4] dot(p0, p1),
      lhs_batch_dims={0}, lhs_contracting_dims={1},
      rhs_batch_dims={0}, rhs_contracting_dims={2}
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));

  ComputationLayout computation_layout(
      module->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(&computation_layout,
                                        GetDeviceDescription());

  EXPECT_THAT(layout_assignment.Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Dot(m::Op().WithShape(S32, {5, 3, 2}, {2, 1, 0}),
                                m::Op().WithShape(S32, {5, 4, 3}, {2, 1, 0}))
                             .WithShape(S32, {5, 2, 4}, {2, 1, 0})));
}

TEST_F(LayoutAssignmentTest, DotOperandLayoutSetToBatchRowsColsOtherwise) {
  const char* hlo_text = R"(
  HloModule DotLayout
  ENTRY dot {
    p0 = s32[2,3,5]{2,1,0} parameter(0)
    p1 = s32[3,4,5] parameter(1)
    ROOT dot.1330.10585 = s32[5,2,4] dot(p0, p1),
      lhs_batch_dims={2}, lhs_contracting_dims={1},
      rhs_batch_dims={2}, rhs_contracting_dims={0}
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));

  ComputationLayout computation_layout(
      module->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(&computation_layout,
                                        GetDeviceDescription());

  EXPECT_THAT(layout_assignment.Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Dot(m::Op().WithShape(S32, {2, 3, 5}, {0, 1, 2}),
                                m::Op().WithShape(S32, {3, 4, 5}, {1, 0, 2}))));
}

TEST_F(LayoutAssignmentTest, DotOperandInconsistentDimLayouts) {
  const char* hlo_text = R"(
  HloModule DotLayout
  ENTRY dot {
    p0 = s32[5,6,2,3] parameter(0)
    p1 = s32[6,5,3,4] parameter(1)
    ROOT dot.1330.10585 = s32[5,6,2,4] dot(p0, p1),
      lhs_batch_dims={0,1}, lhs_contracting_dims={3},
      rhs_batch_dims={1,0}, rhs_contracting_dims={2}
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));

  ComputationLayout computation_layout(
      module->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(&computation_layout,
                                        GetDeviceDescription());

  EXPECT_THAT(layout_assignment.Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Dot(m::Op().WithShape(S32, {5, 6, 2, 3}, {3, 2, 1, 0}),
                        m::Op().WithShape(S32, {6, 5, 3, 4}, {3, 2, 0, 1}))));
}

TEST_F(LayoutAssignmentTest, TransposedDotLayout) {
  const char* hlo_text = R"(
  HloModule DotLayout
  ENTRY dot {
    p0 = s32[5,2,3] parameter(0)
    p1 = s32[5,3,4,6] parameter(1)
    dot = s32[5,2,4,6] dot(p0, p1),
      lhs_batch_dims={0}, lhs_contracting_dims={2},
      rhs_batch_dims={0}, rhs_contracting_dims={1}
    ROOT out = s32[2,5,4,6] transpose(dot), dimensions={1,0,2,3}
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));

  ComputationLayout computation_layout(
      module->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(&computation_layout,
                                        GetDeviceDescription());

  EXPECT_THAT(layout_assignment.Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Transpose(
                     m::Dot(m::Op().WithShape(S32, {5, 2, 3}, {2, 1, 0}),
                            m::Op().WithShape(S32, {5, 3, 4, 6}, {3, 2, 1, 0}))
                         .WithShape(S32, {5, 2, 4, 6}, {3, 2, 0, 1}))
                     .WithShape(S32, {2, 5, 4, 6}, {3, 2, 1, 0})));
}

TEST_F(LayoutAssignmentTest, TransposedDotOfDotLayout) {
  const char* hlo_text = R"(
  HloModule DotLayout
  ENTRY dot {
    p0 = s32[8,50] parameter(0)
    p1 = s32[2,8,4,4] parameter(1)
    p2 = s32[4,38] parameter(2)
    dot.1 = s32[50,2,4,4]{3,2,1,0} dot(p0, p1),
      lhs_contracting_dims={0}, rhs_contracting_dims={1}
    dot.2 = s32[50,2,4,38]{3,2,1,0} dot(dot.1, p2),
      lhs_contracting_dims={2}, rhs_contracting_dims={0}
    ROOT out = s32[2,50,38,4]{2,3,0,1} transpose(dot.2), dimensions={1,0,3,2}
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));

  ComputationLayout computation_layout(
      module->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(&computation_layout,
                                        GetDeviceDescription());

  EXPECT_THAT(layout_assignment.Run(module.get()), IsOkAndHolds(true));
  // The needed operand layout for the lhs of dot.2 cannot be used as layout
  // for dot.1, so a copy is needed.
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(
          m::Transpose(
              m::Dot(m::Copy(m::Dot(m::Op().WithShape(S32, {8, 50}, {1, 0}),
                                    m::Op().WithShape(S32, {2, 8, 4, 4},
                                                      {3, 2, 0, 1}))
                                 .WithShape(S32, {50, 2, 4, 4}, {3, 2, 1, 0}))
                         .WithShape(S32, {50, 2, 4, 4}, {3, 1, 0, 2}),
                     m::Op().WithShape(S32, {4, 38}, {1, 0}))
                  .WithShape(S32, {50, 2, 4, 38}, {3, 2, 1, 0}))
              .WithShape(S32, {2, 50, 38, 4}, {2, 3, 0, 1})));
}

TEST_F(LayoutAssignmentTest, DotLayoutS8) {
  const char* hlo_text = R"(
  HloModule DotLayout
  ENTRY int8_t {
    p0 = s8[32,64] parameter(0)
    p1 = s8[64,96] parameter(1)
    ROOT out = s32[32,96] dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));

  ComputationLayout computation_layout(
      module->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(&computation_layout,
                                        GetDeviceDescription());

  EXPECT_THAT(layout_assignment.Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Dot(m::Op().WithShape(S8, {32, 64}, {1, 0}),
                                m::Op().WithShape(S8, {64, 96}, {0, 1}))));
}

TEST_F(LayoutAssignmentTest, SameLayoutOnOperandsAndOutputsOfSort) {
  const char* hlo_text = R"(
  HloModule SortLayout

  compare {
    p.0.lhs = s32[] parameter(0)
    p.0.rhs = s32[] parameter(1)
    p.1.lhs = s32[] parameter(2)
    p.1.rhs = s32[] parameter(3)
    ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT
  }

  ENTRY sort {
    keys = s32[3,2]{0,1} constant({{0,1},{0,1},{0,1}})
    values = s32[2,3]{1,0} parameter(0)
    transpose = s32[3,2]{1,0} transpose(values), dimensions={1,0}
    ROOT sort = (s32[3,2]{1,0}, s32[3,2]{1,0}) sort(keys, transpose),
      dimensions={1}, to_apply=compare
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));

  ComputationLayout computation_layout(
      module->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(&computation_layout,
                                        GetDeviceDescription());

  EXPECT_THAT(layout_assignment.Run(module.get()), IsOkAndHolds(true));

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Sort(m::Op().WithShape(S32, {3, 2}, {1, 0}),
                                 m::Op().WithShape(S32, {3, 2}, {1, 0}))));
}

// NOTE: Removed SameLayoutOnOperandsAndOutputsOfCubDeviceRadixSort test
// - CUB DeviceRadixSort is GPU/CUDA-specific.

// NOTE: Removed TopKLayout test - TopK custom call is GPU-specific.

// NOTE: Removed FftLayout test - complex types (c64) are not supported.

TEST_F(LayoutAssignmentTest, CustomCallConstrainedAlias) {
  const char* module_str = R"(
HloModule TestModule

ENTRY entry {
  Arg_0 = s32[2,5,5]{2,1,0} parameter(0)
  Arg_1 = s32[2,5,5]{2,1,0} parameter(1)
  Arg_2 = s32[2,5,5]{2,1,0} parameter(2)
  // NOTE: Removed operand_precision={highest,highest} - ML-specific floating
  // point precision option, not applicable to ZK integer operations.
  dot.0 = s32[2,5,5]{2,1,0} dot(Arg_1, Arg_2), lhs_batch_dims={0}, lhs_contracting_dims={2}, rhs_batch_dims={0}, rhs_contracting_dims={2}
  custom-call.0 = (s32[2,5,5]{1,2,0}, s8[16]{0}, s8[16]{0}) custom-call(Arg_0, dot.0), custom_call_target="dummy_call", operand_layout_constraints={s32[2,5,5]{1,2,0}, s32[2,5,5]{1,2,0}}, output_to_operand_aliasing={{0}: (1, {})}
  ROOT get-tuple-element.0 = s32[2,5,5]{1,2,0} get-tuple-element(custom-call.0), index=0
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(module_str));
  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape());

  GpuLayoutAssignment layout_assignment(&computation_layout,
                                        GetDeviceDescription());

  EXPECT_THAT(layout_assignment.Run(m.get()), IsOkAndHolds(true));

  const HloInstruction* call_0 = FindInstruction(m.get(), "custom-call.0");
  auto expect_layout = [](const Shape& shape,
                          absl::Span<const int64_t> minor_to_major) {
    const Layout expected = LayoutUtil::MakeLayout(minor_to_major);
    EXPECT_TRUE(LayoutUtil::Equal(shape.layout(), expected))
        << "Expected layout " << expected << ", actual " << shape.layout();
  };
  expect_layout(ShapeUtil::GetSubshape(call_0->shape(), {0}), {1, 2, 0});
  expect_layout(call_0->operand(0)->shape(), {1, 2, 0});
  expect_layout(call_0->operand(1)->shape(), {1, 2, 0});
}

TEST_F(LayoutAssignmentTest, MoveToHostCustomCallConstrained) {
  const char* module_str = R"(
HloModule TestModule

ENTRY entry {
  Arg_0 = s32[2,5,5]{2,1,0} parameter(0)
  custom-call.0 = s32[2,5,5] custom-call(Arg_0), custom_call_target="MoveToHost"
  ROOT custom-call.1 = s32[2,5,5]{2, 1, 0} custom-call(custom-call.0), custom_call_target="fixed_call", operand_layout_constraints={s32[2,5,5]{1,2,0}}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(module_str));
  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape());

  GpuLayoutAssignment layout_assignment(&computation_layout,
                                        GetDeviceDescription());

  EXPECT_THAT(layout_assignment.Run(m.get()), IsOkAndHolds(true));

  const HloInstruction* call_0 = FindInstruction(m.get(), "custom-call.0");
  const Layout input_layout = call_0->operand(0)->shape().layout();
  const Layout output_layout = call_0->shape().layout();
  EXPECT_TRUE(LayoutUtil::Equal(input_layout, output_layout))
      << "Expected the same input/output layouts.  Input: " << input_layout
      << ". Output: " << output_layout;
}

TEST_F(LayoutAssignmentTest, MoveToDeviceCustomCallConstrained) {
  const char* module_str = R"(
HloModule TestModule

ENTRY entry {
  Arg_0 = s32[2,5,5]{2,1,0} parameter(0)
  custom-call.0 = s32[2,5,5] custom-call(Arg_0), custom_call_target="MoveToDevice"
  ROOT custom-call.1 = s32[2,5,5]{2, 1, 0} custom-call(custom-call.0), custom_call_target="fixed_call", operand_layout_constraints={s32[2,5,5]{1,2,0}}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(module_str));
  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape());

  GpuLayoutAssignment layout_assignment(&computation_layout,
                                        GetDeviceDescription());

  EXPECT_THAT(layout_assignment.Run(m.get()), IsOkAndHolds(true));

  const HloInstruction* call_0 = FindInstruction(m.get(), "custom-call.0");
  const Layout input_layout = call_0->operand(0)->shape().layout();
  const Layout output_layout = call_0->shape().layout();
  EXPECT_TRUE(LayoutUtil::Equal(input_layout, output_layout))
      << "Expected the same input/output layouts.  Input: " << input_layout
      << ". Output: " << output_layout;
}

// NOTE: Removed CuDNNConvolutionHasNHWCLayoutPostHopper test
// - cuDNN convolutions are GPU/CUDA-specific.

// NOTE: Removed F64CuDNNConvolutionHasNCHWLayoutPostHopper test
// - cuDNN convolutions are GPU/CUDA-specific.

// NOTE: Removed ConvCuDNNF8 test - cuDNN/FP8 is GPU/CUDA-specific.

// NOTE: Removed ConvCuDNNBF16 test - cuDNN/BF16 is GPU/CUDA-specific.

// NOTE: Removed ConvCuDNNFP16 test - cuDNN/FP16 is GPU/CUDA-specific.

// NOTE: Removed ReduceOperandLayout test
// - Requires GetGpuComputeCapability() which is GPU/CUDA-specific.

// NOTE: Removed ReduceOperandLayoutDivisorOfWarpSize test
// - Requires GetGpuComputeCapability() which is GPU/CUDA-specific.

// NOTE: Removed AutoLayoutE4M3ContractingMinorFirst test
// - FP8 types are GPU/ML-specific.

// NOTE: Removed AutoLayoutS4DotContractingMinorLhs test
// - S4/BF16 types and debug options are GPU/ML-specific.

// NOTE: Removed AutoLayoutS4DotContractingMinorRhs test
// - S4/BF16 types and debug options are GPU/ML-specific.

// NOTE: Removed VariadicReduceSameOperandLayout test
// - Requires GetGpuComputeCapability() which is GPU/CUDA-specific.

// NOTE: Removed SendRcvLayout test
// - Requires RunAndFilecheckHloRewrite which depends on GPU backend.

}  // namespace
}  // namespace gpu
}  // namespace zkx
