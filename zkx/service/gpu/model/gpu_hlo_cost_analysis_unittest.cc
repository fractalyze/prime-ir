/* Copyright 2022 The OpenXLA Authors.
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

#include "zkx/service/gpu/model/gpu_hlo_cost_analysis.h"

#include <cstdint>

#include "gtest/gtest.h"

#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/ir/hlo_opcode.h"
#include "zkx/tests/hlo_test_base.h"

namespace zkx {
namespace gpu {

class GpuHloCostAnalysisTest : public HloTestBase {
 public:
  HloCostAnalysis::Options options_{.count_multiple_input_accesses = true};
  GpuHloCostAnalysis analysis_{options_};
  GpuHloCostAnalysisTest() : HloTestBase() {}
};

// TODO(batzor): Implement this test. Dependency: ReduceWindow.
// TEST_F(GpuHloCostAnalysisTest, ReduceWindowWithOverlapsRepeatedReads) {

TEST_F(GpuHloCostAnalysisTest, BroadcastWithRepeats) {
  std::string_view hlo_string = R"(
HloModule m

f {
  p1 = s8[] parameter(0)
  c1 = s8[] constant(0)
  a1 = s8[] add(p1, c1)
  b1 = s8[10000] broadcast(a1), dimensions={}
  b2 = s8[10000] broadcast(c1), dimensions={}
  ROOT r1 = s8[10000] add(b1, b2)
}

ENTRY e {
  p0 = s8[] parameter(0)
  ROOT r0 = s8[10000] fusion(p0), kind=kInput, calls=f
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* root = module->entry_computation()->root_instruction();
  TF_ASSERT_OK(module->entry_computation()->Accept(&analysis_));

  EXPECT_EQ(analysis_.output_bytes_accessed(*root), 10000);
  EXPECT_EQ(analysis_.operand_bytes_accessed(*root, 0), 10000);
  // Operand + output.
  EXPECT_EQ(analysis_.bytes_accessed(*root), 2 * 10000);
  EXPECT_EQ(analysis_.bytes_accessed(), 2 * 10000);
}

TEST_F(GpuHloCostAnalysisTest, WithoutRepeats) {
  std::string_view hlo_string = R"(
HloModule m

f {
  p1 = s8[] parameter(0)
  a1 = s8[] add(p1, p1)
  b1 = s8[10000] broadcast(a1), dimensions={}
  a2 = s8[10000] add(b1, b1)
  slice1 = s8[8000] slice(a2), slice={[0:8000]}
  slice2 = s8[8000] slice(a2), slice={[2000:10000]}
  c = s8[10000] constant({...})
  slicec1 = s8[8000] slice(c), slice={[0:8000]}
  slicec2 = s8[8000] slice(c), slice={[2000:10000]}
  a3 = s8[8000] add(slice1, slice2)
  a4 = s8[8000] add(slicec1, slicec2)
  ROOT a5 = s8[8000] add(a3, a4)
}

ENTRY e {
  p0 = s8[] parameter(0)
  ROOT r0 = s8[8000] fusion(p0), kind=kInput, calls=f
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* root = module->entry_computation()->root_instruction();
  options_.count_multiple_input_accesses = false;
  GpuHloCostAnalysis analysis{options_};
  TF_ASSERT_OK(module->entry_computation()->Accept(&analysis));

  EXPECT_EQ(analysis.output_bytes_accessed(*root), 8000);
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 0), 1);
  // Operand + output + constant.
  EXPECT_EQ(analysis.bytes_accessed(*root), 1 + 8000 + 10000);
  EXPECT_EQ(analysis.bytes_accessed(), 1 + 8000 + 10000);
}

TEST_F(GpuHloCostAnalysisTest, BroadcastFlops) {
  std::string_view hlo_string = R"(
HloModule m

f {
  i0 = s32[1024] iota(), iota_dimension=0
  m0 = s32[1024] add(i0, i0)
  s0 = s32[1024] multiply(m0, m0)
  b0 = s32[1024,1024] broadcast(s0), dimensions={0}
  ROOT r0 = s32[1024,1024] negate(b0)
}

ENTRY e {
  ROOT r = s32[1024,1024] fusion(), kind=kInput, calls=f
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* root = module->entry_computation()->root_instruction();
  TF_ASSERT_OK(module->entry_computation()->Accept(&analysis_));

  auto n_elements = 1024 * 1024;
  EXPECT_EQ(analysis_.output_bytes_accessed(*root), n_elements * 4);
  EXPECT_EQ(analysis_.bytes_accessed(*root), n_elements * 4);
  EXPECT_EQ(analysis_.bytes_accessed(), n_elements * 4);
  EXPECT_EQ(analysis_.flop_count(), n_elements * 3 * 3);
  EXPECT_EQ(analysis_.IrSize(*root), 5);
}

TEST_F(GpuHloCostAnalysisTest, Slice) {
  std::string_view hlo_string = R"(
HloModule m

f {
  p1 = s8[100000000] parameter(0)
  i1 = s8[100000000] iota(), iota_dimension=0
  a1 = s8[100000000] add(p1, i1)
  ROOT r1 = s8[1] slice(a1), slice={[0:1]}
}

ENTRY e {
  p0 = s8[100000000] parameter(0)
  ROOT r0 = s8[1] fusion(p0), kind=kInput, calls=f
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  const HloInstruction* root = module->entry_computation()->root_instruction();
  TF_ASSERT_OK(module->entry_computation()->Accept(&analysis_));

  EXPECT_EQ(analysis_.output_bytes_accessed(*root), 1);
  EXPECT_EQ(analysis_.operand_bytes_accessed(*root, 0), 1);
  EXPECT_EQ(analysis_.bytes_accessed(*root), 2);
  EXPECT_EQ(analysis_.bytes_accessed(), 2);
  EXPECT_EQ(analysis_.IrSize(*root), 4);
}

TEST_F(GpuHloCostAnalysisTest, TwoSlices) {
  std::string_view hlo_string = R"(
HloModule m

f {
  p1 = s8[100] parameter(0)
  i1 = s8[100] iota(), iota_dimension=0
  a1 = s8[100] add(p1, i1)
  slice1 = s8[1] slice(a1), slice={[0:1]}
  slice2 = s8[1] slice(a1), slice={[3:4]}
  ROOT r = s8[1] add(slice1, slice2)
}

ENTRY e {
  p0 = s8[100] parameter(0)
  ROOT r0 = s8[1] fusion(p0), kind=kInput, calls=f
}

)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  const HloInstruction* root = module->entry_computation()->root_instruction();
  TF_ASSERT_OK(module->entry_computation()->Accept(&analysis_));

  EXPECT_EQ(analysis_.output_bytes_accessed(*root), 1);
  EXPECT_EQ(analysis_.operand_bytes_accessed(*root, 0), 2);
  EXPECT_EQ(analysis_.bytes_accessed(*root), 3);
  EXPECT_EQ(analysis_.bytes_accessed(), 3);
  EXPECT_EQ(analysis_.IrSize(*root), 9);
}

TEST_F(GpuHloCostAnalysisTest, MultipleTrivialUsers) {
  std::string_view hlo_string = R"(
HloModule m

f {
  p0 = s8[] parameter(0)
  m0 = s8[] multiply(p0, p0)
  n0 = s8[] negate(p0)
  ROOT a0 = s8[] add(m0, n0)
}

ENTRY e {
  param0 = s8[] parameter(0)
  ROOT r0 = s8[] fusion(param0), kind=kInput, calls=f
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* root = module->entry_computation()->root_instruction();
  TF_ASSERT_OK(module->entry_computation()->Accept(&analysis_));

  // Expect that uses of p0 by different trivial users (m0, n0) can be
  // combined into a single memory access.
  EXPECT_EQ(analysis_.output_bytes_accessed(*root), 1);
  EXPECT_EQ(analysis_.operand_bytes_accessed(*root, 0), 1);
  EXPECT_EQ(analysis_.bytes_accessed(*root), 1 + 1);
  EXPECT_EQ(analysis_.bytes_accessed(), 1 + 1);
  EXPECT_EQ(analysis_.IrSize(*root), 4);
}

TEST_F(GpuHloCostAnalysisTest, MixedUsers) {
  std::string_view hlo_string = R"(
HloModule m

f {
  p0 = s8[10] parameter(0)
  n0 = s8[10] negate(p0)
  m0 = s8[10] multiply(n0, n0)
  a0 = s8[10] add(n0, n0)
  s0 = s8[5] slice(a0), slice={[0:5]}
  svar1 = s8[2] slice(n0), slice={[4:6]}
  n1 = s8[2] negate(svar1)
  ROOT c0 = s8[17] concatenate(s0, m0, n1), dimensions={0}
}

ENTRY e {
  param0 = s8[10] parameter(0)
  ROOT r0 = s8[17] fusion(param0), kind=kInput, calls=f
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* root = module->entry_computation()->root_instruction();
  TF_ASSERT_OK(module->entry_computation()->Accept(&analysis_));

  // Expect that uses of n0 by different trivial users (m0, a0) can be
  // combined into a single memory access, but slices have to be counted
  // separately.
  EXPECT_EQ(analysis_.output_bytes_accessed(*root), 17);
  EXPECT_EQ(analysis_.operand_bytes_accessed(*root, 0), 17);
  EXPECT_EQ(analysis_.bytes_accessed(*root), 17 + 17);
  EXPECT_EQ(analysis_.bytes_accessed(), 17 + 17);
  // There are 2 slice accesses + 1 element-wise from the root.
  EXPECT_EQ(analysis_.IrSize(*root->fused_parameter(0)), 3);
  // Because p0 is only directly used by elementwise n0 their code sizes
  // have to be equal.
  EXPECT_EQ(analysis_.IrSize(*root->fused_parameter(0)),
            analysis_.IrSize(*root->fused_parameter(0)->users()[0]));
  EXPECT_EQ(analysis_.IrSize(*root), 12);
}

TEST_F(GpuHloCostAnalysisTest, FractionalUseRoundingUp) {
  std::string_view hlo_string = R"(
HloModule m

add_s8 {
  lhs = s8[] parameter(0)
  rhs = s8[] parameter(1)
  ROOT add = s8[] add(lhs, rhs)
}

f {
  p0 = s8[] parameter(0)
  b0 = s8[10] broadcast(p0), dimensions={}
  c0 = s8[] constant(0)
  r0 = s8[] reduce(b0, c0), dimensions={0}, to_apply=add_s8
  bitcast0 = s8[1] bitcast(r0)
  i0 = s8[5] iota(), iota_dimension=0
  cat0 = s8[6] concatenate(bitcast0, i0), dimensions={0}
  p1 = s32[] parameter(1)
  ROOT s0 = s8[2] dynamic-slice(cat0, p1), dynamic_slice_sizes={2}
}

ENTRY e {
  p0 = s8[] parameter(0)
  p1 = s32[] parameter(1)
  ROOT r = s8[2] fusion(p0, p1), kind=kInput, calls=f
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* root = module->entry_computation()->root_instruction();
  TF_ASSERT_OK(module->entry_computation()->Accept(&analysis_));

  EXPECT_EQ(analysis_.output_bytes_accessed(*root), 2);
  EXPECT_EQ(analysis_.operand_bytes_accessed(*root, 0), 10);
  EXPECT_EQ(analysis_.operand_bytes_accessed(*root, 1), 4);
  EXPECT_EQ(analysis_.bytes_accessed(*root), 2 + 10 + 4);
  EXPECT_EQ(analysis_.bytes_accessed(), 2 + 10 + 4);
}

TEST_F(GpuHloCostAnalysisTest, LargeConstant) {
  std::string_view hlo_string = R"(
HloModule m

f {
  p0 = s8[1000] parameter(0)
  c0 = s8[1000] constant({...})
  ROOT a0 = s8[1000] add(p0, c0)
}

ENTRY e {
  p0 = s8[1000] parameter(0)
  ROOT r = s8[1000] fusion(p0), kind=kInput, calls=f
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* root = module->entry_computation()->root_instruction();
  TF_ASSERT_OK(module->entry_computation()->Accept(&analysis_));

  EXPECT_EQ(analysis_.output_bytes_accessed(*root), 1000);
  EXPECT_EQ(analysis_.operand_bytes_accessed(*root, 0), 1000);
  // Parameter + output + constant.
  EXPECT_EQ(analysis_.bytes_accessed(*root), 3000);
  EXPECT_EQ(analysis_.bytes_accessed(), 3000);
  EXPECT_EQ(analysis_.IrSize(*root), 3);
}

TEST_F(GpuHloCostAnalysisTest, DynUpdateSliceUsingOperandData) {
  const char* hlo_fusion_module_str = R"(
  HloModule m

  f {
    to_update = s8[3,1,1,1] parameter(0)
    update = s8[1,1,1,1] constant(0)
    a = s32[] constant(0)
    dus = s8[3,1,1,1] dynamic-update-slice(to_update, update, a, a, a, a)
    ROOT _ = s8[3,1,1,1] negate(dus)
  }

  ENTRY _ {
    to_update = s8[3,1,1,1] parameter(0)
    ROOT _ = s8[3,1,1,1] fusion(to_update), kind=kLoop, calls=f
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_fusion_module_str));
  TF_ASSERT_OK(module->entry_computation()->Accept(&analysis_));

  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_EQ(fusion->opcode(), HloOpcode::kFusion);

  // Input size minus update size.
  EXPECT_EQ(analysis_.operand_bytes_accessed(*fusion, 0), 3 - 1);
  EXPECT_EQ(analysis_.output_bytes_accessed(*fusion), 3);
}

TEST_F(GpuHloCostAnalysisTest, DynUpdateSliceNotUsingOperandData) {
  const char* hlo_fusion_module_str = R"(
  HloModule m

  f {
    to_update = s8[3,1,1,1] parameter(0)
    update = s8[1,1,1,1] constant(0)
    a = s32[] constant(0)
    ROOT dus = s8[3,1,1,1] dynamic-update-slice(to_update, update, a, a, a, a)
  }

  ENTRY _ {
    to_update = s8[3,1,1,1] parameter(0)
    ROOT _ = s8[3,1,1,1] fusion(to_update), kind=kLoop, calls=f
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_fusion_module_str));
  TF_ASSERT_OK(module->entry_computation()->Accept(&analysis_));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_EQ(fusion->opcode(), HloOpcode::kFusion);

  EXPECT_EQ(analysis_.operand_bytes_accessed(*fusion, 0), 0);
  EXPECT_EQ(analysis_.output_bytes_accessed(*fusion), 1);
}

TEST_F(GpuHloCostAnalysisTest, CommonElementwiseUseTwoParameters) {
  const char* hlo_fusion_module_str = R"(
  HloModule m

  add {
    p0 = s8[] parameter(0)
    p1 = s8[] parameter(1)
    ROOT _ = s8[] add(p0, p1)
  }

  f {
    p0 = s8[10] parameter(0)
    p1 = s8[10] parameter(1)
    a = s8[10] add(p0, p1)
    c0 = s8[] constant(0)
    r0 = s8[] reduce(a, c0), dimensions={0}, to_apply=add
    c1 = s8[] constant(100)
    r1 = s8[] reduce(a, c1), dimensions={0}, to_apply=add
    ROOT _ = s8[] add(r0, r1)
  }

  ENTRY _ {
    p0 = s8[10] parameter(0)
    p1 = s8[10] parameter(1)
    ROOT _ = s8[] fusion(p0, p1), kind=kLoop, calls=f
  })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_fusion_module_str));
  TF_ASSERT_OK(module->entry_computation()->Accept(&analysis_));
  HloInstruction* fusion = module->entry_computation()->root_instruction();

  EXPECT_EQ(analysis_.CommonElementwiseUtilization(fusion->fused_parameter(0),
                                                   fusion->fused_parameter(1)),
            2.f);
}

TEST_F(GpuHloCostAnalysisTest, CommonElementwiseUseParameterAndRoot) {
  const char* hlo_fusion_module_str = R"(
  HloModule m

  f {
    p0 = s8[10] parameter(0)
    p1 = s8[] parameter(1)
    p1b = s8[10] broadcast(p1)
    a = s8[10] add(p0, p1b)
    ROOT _ = s8[10] negate(a)
  }

  ENTRY _ {
    p0 = s8[10] parameter(0)
    p1 = s8[] parameter(1)
    ROOT _ = s8[10] fusion(p0, p1), kind=kLoop, calls=f
  })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_fusion_module_str));
  TF_ASSERT_OK(module->entry_computation()->Accept(&analysis_));
  HloInstruction* fusion = module->entry_computation()->root_instruction();

  EXPECT_EQ(analysis_.CommonElementwiseUtilization(
                fusion->fused_parameter(0), fusion->fused_expression_root()),
            1.f);
  EXPECT_EQ(analysis_.CommonElementwiseUtilization(
                fusion->fused_parameter(1), fusion->fused_expression_root()),
            0.f);
}

TEST_F(GpuHloCostAnalysisTest,
       CommonElementwiseUseParameterAndRootMultiOutputFusion) {
  const char* hlo_fusion_module_str = R"(
  HloModule m

  f {
    p0 = s8[10] parameter(0)
    p1 = s8[] parameter(1)
    p1b = s8[10] broadcast(p1)
    a = s8[10] add(p0, p1b)
    neg = s8[10] negate(a)
    ROOT _ = (s8[10], s8[10]) tuple(a, neg)
  }

  ENTRY _ {
    p0 = s8[10] parameter(0)
    p1 = s8[] parameter(1)
    ROOT _ = (s8[10], s8[10]) fusion(p0, p1), kind=kLoop, calls=f
  })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_fusion_module_str));
  TF_ASSERT_OK(module->entry_computation()->Accept(&analysis_));
  HloInstruction* fusion = module->entry_computation()->root_instruction();

  EXPECT_EQ(analysis_.CommonElementwiseUtilization(
                fusion->fused_parameter(0), fusion->fused_expression_root()),
            1.f);
  EXPECT_EQ(analysis_.CommonElementwiseUtilization(
                fusion->fused_parameter(1), fusion->fused_expression_root()),
            0.f);
}

TEST_F(GpuHloCostAnalysisTest, Reduce) {
  std::string_view hlo_string = R"(
HloModule m

add {
  param_0 = s32[] parameter(0)
  param_1 = s32[] parameter(1)
  ROOT add.0 = s32[] add(param_0, param_1)
}

ENTRY entry_computation {
  param_0.3 = s32[32,40]{1,0} parameter(0)
  constant = s32[] constant(0)
  ROOT reduce = s32[32]{0} reduce(param_0.3, constant), dimensions={1}, to_apply=add
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK(module->entry_computation()->Accept(&analysis_));
  const HloInstruction* reduce =
      module->entry_computation()->root_instruction();

  int64_t input_bytes_accessed = 4 * 32 * 40;
  int64_t init_bytes_accessed = 4 * 32;
  int64_t output_bytes_accessed = 4 * 32;

  EXPECT_EQ(analysis_.operand_bytes_accessed(*reduce, 0), input_bytes_accessed);
  EXPECT_EQ(analysis_.operand_bytes_accessed(*reduce, 1), init_bytes_accessed);
  EXPECT_EQ(analysis_.output_bytes_accessed(*reduce), output_bytes_accessed);
  EXPECT_EQ(analysis_.bytes_accessed(*reduce),
            input_bytes_accessed + init_bytes_accessed + output_bytes_accessed);
  EXPECT_EQ(analysis_.flop_count(*reduce), 32 * 39 * 3);
}

TEST_F(GpuHloCostAnalysisTest, VariadicReduce) {
  std::string_view hlo_string = R"(
HloModule m

add {
  param_0 = s32[] parameter(0)
  param_1 = s32[] parameter(1)
  param_2 = s32[] parameter(2)
  param_3 = s32[] parameter(3)
  add.0 = s32[] add(param_0, param_2)
  add.1 = s32[] add(param_1, param_3)
  ROOT t = (s32[], s32[]) tuple(add.0, add.1)
}

ENTRY entry_computation {
  param_0.3 = s32[32,40]{1,0} parameter(0)
  param_1.3 = s32[32,40]{1,0} parameter(1)
  param_2.2 = s32[] parameter(2)
  constant = s32[] constant(0)
  ROOT reduce = (s32[32]{0}, s32[32]{0}) reduce(param_0.3, param_1.3, param_2.2, constant), dimensions={1}, to_apply=add
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK(module->entry_computation()->Accept(&analysis_));
  const HloInstruction* reduce =
      module->entry_computation()->root_instruction();

  int64_t input_bytes_accessed = 4 * 32 * 40;
  int64_t init_bytes_accessed = 4 * 32;
  int64_t output_bytes_accessed = 2 * 4 * 32;

  EXPECT_EQ(analysis_.operand_bytes_accessed(*reduce, 0), input_bytes_accessed);
  EXPECT_EQ(analysis_.operand_bytes_accessed(*reduce, 1), input_bytes_accessed);
  EXPECT_EQ(analysis_.operand_bytes_accessed(*reduce, 2), init_bytes_accessed);
  EXPECT_EQ(analysis_.operand_bytes_accessed(*reduce, 3), init_bytes_accessed);
  EXPECT_EQ(analysis_.output_bytes_accessed(*reduce), output_bytes_accessed);
  EXPECT_EQ(analysis_.bytes_accessed(*reduce), 2 * input_bytes_accessed +
                                                   2 * init_bytes_accessed +
                                                   output_bytes_accessed);
  EXPECT_EQ(analysis_.flop_count(*reduce), 32 * 39 * 6);
}

TEST_F(GpuHloCostAnalysisTest, AsyncAllReduce) {
  std::string_view hlo_string = R"(
HloModule m

add {
  param_0 = s32[] parameter(0)
  param_1 = s32[] parameter(1)
  ROOT t = s32[] add(param_0, param_1)
}

ENTRY entry_computation {
  p = s32[4096] parameter(0)
  ar-start = s32[4096] all-reduce-start(p), to_apply=add
  ROOT _ = s32[4096] all-reduce-done(ar-start)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK(module->entry_computation()->Accept(&analysis_));

  const HloInstruction* all_reduce =
      module->entry_computation()->root_instruction()->operand(0);
  EXPECT_EQ(analysis_.BytesTransferred(*all_reduce), 4096 * 4);
  EXPECT_EQ(analysis_.bytes_accessed(*all_reduce), 4096 * 4);
}

TEST_F(GpuHloCostAnalysisTest, AllGather) {
  std::string_view hlo_string = R"(
HloModule m, num_partitions=4

ENTRY entry_computation {
  p = s32[1024] parameter(0)
  ROOT _ = s32[4096] all-gather(p), dimensions={0}, use_global_device_ids=true,
    replica_groups={{0,1,2,3}}, channel_id=1
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK(module->entry_computation()->Accept(&analysis_));

  const HloInstruction* all_gather =
      module->entry_computation()->root_instruction();
  EXPECT_EQ(analysis_.BytesTransferred(*all_gather), 4096 * 4);
  // write = (3 + 4) * 1024 * 4 bytes
  // read = 4 * 1024 * 4 bytes
  EXPECT_EQ(analysis_.bytes_accessed(*all_gather), 45056);
}

TEST_F(GpuHloCostAnalysisTest, AsyncAllGather) {
  std::string_view hlo_string = R"(
HloModule m, num_partitions=4

ENTRY entry_computation {
  p.0 = s32[1024] parameter(0)
  p.1 = s32[512] parameter(1)
  ag-start = ((s32[1024],s32[512]), (s32[4096],s32[2048])) all-gather-start(p.0,p.1),
    dimensions={0}, use_global_device_ids=true, replica_groups={{0,1,2,3}},
    channel_id=1
  ROOT _ = (s32[4096],s32[2048]) all-gather-done(ag-start)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK(module->entry_computation()->Accept(&analysis_));

  const HloInstruction* all_gather =
      module->entry_computation()->root_instruction()->operand(0);
  // Output is (s32[4096], s32[2048]).
  EXPECT_EQ(analysis_.BytesTransferred(*all_gather), 4096 * 4 + 2048 * 4);
  // write = (3 + 4) * (1024 + 512) * 4 bytes
  // read = 4 * (1024 + 512) * 4 bytes
  EXPECT_EQ(analysis_.bytes_accessed(*all_gather), 67584);
}

TEST_F(GpuHloCostAnalysisTest, ReduceScatter) {
  std::string_view hlo_string = R"(
HloModule m, num_partitions=4

add {
  param_0 = s32[] parameter(0)
  param_1 = s32[] parameter(1)
  ROOT t = s32[] add(param_0, param_1)
}

ENTRY entry_computation {
  p = s32[4096] parameter(0)
  ROOT _ = s32[1024] reduce-scatter(p), dimensions={0}, to_apply=add,
      use_global_device_ids=true, replica_groups={{0,1,2,3}}, channel_id=1
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK(module->entry_computation()->Accept(&analysis_));

  const HloInstruction* reduce_scatter =
      module->entry_computation()->root_instruction();
  EXPECT_EQ(analysis_.BytesTransferred(*reduce_scatter), 4096 * 4);
  // read = (3 + 4) * 1024 * 4 bytes
  // write = 4 * 1024 * 4 bytes
  EXPECT_EQ(analysis_.bytes_accessed(*reduce_scatter), 45056);
}

TEST_F(GpuHloCostAnalysisTest, AsyncReduceScatter) {
  std::string_view hlo_string = R"(
HloModule m, num_partitions=4

add {
  param_0 = s32[] parameter(0)
  param_1 = s32[] parameter(1)
  ROOT t = s32[] add(param_0, param_1)
}

async_computation {
  param_3 = s32[4096] parameter(0)
  param_4 = s32[2048] parameter(1)
  ROOT r = (s32[1024],s32[512]) reduce-scatter(param_3,param_4),
    dimensions={0}, to_apply=add, use_global_device_ids=true,
    replica_groups={{0,1,2,3}}, channel_id=1
}

ENTRY entry_computation {
  p.0 = s32[4096] parameter(0)
  p.1 = s32[2048] parameter(1)
  rs-start = ((s32[4096],s32[2048]),(s32[1024],s32[512])) async-start(p.0,p.1),
    calls=async_computation
  ROOT _ = (s32[1024],s32[512]) async-done(rs-start)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK(module->entry_computation()->Accept(&analysis_));

  const HloInstruction* reduce_scatter =
      module->entry_computation()->root_instruction()->operand(0);
  // Output is (s32[1024],s32[512]).
  EXPECT_EQ(analysis_.BytesTransferred(*reduce_scatter), 4096 * 4 + 2048 * 4);
  // read = (3 + 4) * (1024 + 512) * 4 bytes
  // write = 4 * (1024 + 512) * 4 bytes
  EXPECT_EQ(analysis_.bytes_accessed(*reduce_scatter), 67584);
}

// TODO(batzor): Implement this test. Dependency: HloOpProfiles.
// TEST_F(GpuHloCostAnalysisTest, CustomOpProfileIsUsed) {

}  // namespace gpu
}  // namespace zkx
