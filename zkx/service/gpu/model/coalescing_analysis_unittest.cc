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

#include "zkx/service/gpu/model/coalescing_analysis.h"

#include <memory>
#include <string_view>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mlir/IR/MLIRContext.h"

#include "zkx/backends/gpu/codegen/fusion_emitter.h"
#include "zkx/backends/gpu/codegen/fusions.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/ir/hlo_opcode.h"
#include "zkx/hlo/utils/hlo_traversal.h"
#include "zkx/service/gpu/gpu_device_info_for_tests.h"
#include "zkx/service/gpu/hlo_fusion_analysis.h"
#include "zkx/service/hlo_module_config.h"
#include "zkx/shape_util.h"
#include "zkx/stream_executor/device_description.h"
#include "zkx/tests/hlo_test_base.h"

namespace zkx::gpu {
namespace {

using ::testing::ElementsAre;

class CoalescingTest : public HloTestBase {
 public:
  std::vector<bool> IsReadCoalescedPerOperand(std::string_view hlo_string) {
    auto module = ParseAndReturnVerifiedModule(hlo_string).value();
    HloInstruction* root = module->entry_computation()->root_instruction();
    return IsReadCoalescedPerOperand(root);
  }

  std::vector<bool> IsReadCoalescedPerOperand(const HloInstruction* root) {
    auto fusion_adaptor = HloFusionAdaptor::ForInstruction(root);
    auto analysis = HloFusionAnalysis::Create(*root, device_info_);
    auto emitter = GetFusionEmitter(PreBufferAssignmentFusionInfo{analysis});
    auto fusion = dynamic_cast<KernelFusionInterface*>(emitter.get());
    EXPECT_NE(fusion, nullptr);

    CoalescingAnalysis coalescing_analysis(root, root->operands(), analysis,
                                           fusion, &mlir_context_,
                                           /*use_heuristic=*/false);

    std::vector<bool> results;
    for (const HloInstruction* operand : root->operands()) {
      results.push_back(coalescing_analysis.IsReadCoalesced(operand));
    }
    return results;
  }

  bool IsReadCoalescedHeuristic(std::string_view hlo_string) {
    auto module = ParseAndReturnVerifiedModule(hlo_string).value();
    HloInstruction* root = module->entry_computation()->root_instruction();
    auto analysis = HloFusionAnalysis::Create(*root, device_info_);
    return zkx::gpu::IsReadCoalescedHeuristic(
        analysis.GetEmitterFusionKind(), device_info_, root->operand(0), root);
  }

 protected:
  stream_executor::DeviceDescription device_info_ =
      TestGpuDeviceInfo::RTXA6000DeviceInfo();
  mlir::MLIRContext mlir_context_;
};

TEST_F(CoalescingTest, IdentityLayout) {
  std::string_view ir = R"(
    HloModule m
    fusion {
      p0 = s32[100, 200] parameter(0)
      p1 = s32[100, 200] parameter(1)
      ROOT add = s32[100, 200] add(p0, p1)
    }
    ENTRY e {
      p0 = s32[100, 200] parameter(0)
      p1 = s32[100, 200] parameter(1)
      ROOT fusion = s32[100, 200] fusion(p0, p1), kind=kInput, calls=fusion
    }
  )";
  // thread_x to linearized input mapping for thread_x in [0, 31]:
  // Operand 1: (thread_x) -> (thread_x)
  // Operand 2: (thread_x) -> (thread_x)
  EXPECT_THAT(IsReadCoalescedPerOperand(ir), ElementsAre(true, true));
}

TEST_F(CoalescingTest, RhsTransposedLayout) {
  std::string_view ir = R"(
    HloModule m
    fusion {
      p0 = s32[100, 200]{1, 0} parameter(0)
      p1 = s32[100, 200]{0, 1} parameter(1)
      ROOT add = s32[100, 200]{1, 0} add(p0, p1)
    }
    ENTRY e {
      p0 = s32[100, 200]{1, 0} parameter(0)
      p1 = s32[100, 200]{0, 1} parameter(1)
      ROOT fusion = s32[100, 200]{1, 0} fusion(p0, p1), kind=kInput, calls=fusion
    }
  )";
  // thread_x to linearized input mapping for thread_x in [0, 31]:
  // Operand 1: (thread_x) -> (thread_x)
  // Operand 2: (thread_x) -> (thread_x * 100)
  EXPECT_THAT(IsReadCoalescedPerOperand(ir), ElementsAre(true, false));
}

TEST_F(CoalescingTest, OutputTransposedLayout) {
  std::string_view ir = R"(
    HloModule m
    fusion {
      p0 = s32[100, 200]{1, 0} parameter(0)
      p1 = s32[100, 200]{1, 0} parameter(1)
      ROOT add = s32[100, 200]{0, 1} add(p0, p1)
    }
    ENTRY e {
      p0 = s32[100, 200]{1, 0} parameter(0)
      p1 = s32[100, 200]{1, 0} parameter(1)
      ROOT fusion = s32[100, 200]{0, 1} fusion(p0, p1), kind=kInput, calls=fusion
    }
  )";
  // thread_x to linearized input mapping for thread_x in [0, 31]:
  // Operand 1: (thread_x) -> (thread_x * 200)
  // Operand 2: (thread_x) -> (thread_x * 200)
  EXPECT_THAT(IsReadCoalescedPerOperand(ir), ElementsAre(false, false));
}

TEST_F(CoalescingTest, OutputAndLhsTransposedLayout) {
  std::string_view ir = R"(
    HloModule m
    fusion {
      p0 = s32[100, 200]{1, 0} parameter(0)
      p1 = s32[100, 200]{0, 1} parameter(1)
      ROOT add = s32[100, 200]{1, 0} add(p0, p1)
    }
    ENTRY e {
      p0 = s32[100, 200]{1, 0} parameter(0)
      p1 = s32[100, 200]{0, 1} parameter(1)
      ROOT fusion = s32[100, 200]{1, 0} fusion(p0, p1), kind=kInput, calls=fusion
    }
  )";
  // thread_x to linearized input mapping for thread_x in [0, 31]:
  // Operand 1: (thread_x) -> (thread_x)
  // Operand 2: (thread_x) -> (thread_x * 100)
  EXPECT_THAT(IsReadCoalescedPerOperand(ir), ElementsAre(true, false));
}

// TODO(batzor): Implement this. Dependency: TransposeFusion emitter.
// TEST_F(CoalescingTest, Transpose)

TEST_F(CoalescingTest, TransposeOfBroadcastHeuristic) {
  std::string_view ir = R"(
    HloModule module

    fusion {
      input = s32[1, 32, 6400] parameter(0)
      ROOT slice = s32[1, 32, 100] slice(input), slice={[0:1:1], [0:32:1], [0:6400:64]}
    }

    ENTRY entry {
      p0 = s32[32] parameter(0)
      broadcast = s32[1, 6400, 32] broadcast(p0), dimensions={2}
      transpose = s32[1, 32, 6400] transpose(broadcast), dimensions={0, 2, 1}
      ROOT %fusion = s32[1, 32, 100] fusion(transpose), kind=kLoop, calls=fusion
  })";
  EXPECT_TRUE(IsReadCoalescedHeuristic(ir));
}

TEST_F(CoalescingTest, TransposeOfIotaHeuristic) {
  std::string_view ir = R"(
    HloModule module

    fusion {
      p0 = s32[32, 100, 64] parameter(0)
      ROOT slice = s32[32, 100, 1] slice(p0), slice={[0:32:1], [0:100:1], [0:1:1]}
    }

    ENTRY entry {
      iota = s32[100, 64, 32] iota(), iota_dimension=1
      transpose = s32[32, 100, 64] transpose(iota), dimensions={2, 0, 1}
      ROOT %fusion = s32[32, 100, 1] fusion(transpose), kind=kLoop, calls=fusion
  })";
  EXPECT_TRUE(IsReadCoalescedHeuristic(ir));
}

TEST_F(CoalescingTest, TransposeOfAddHeuristic) {
  std::string_view ir = R"(
    HloModule module

    fusion {
      p0 = s32[32, 100, 64] parameter(0)
      ROOT slice = s32[32, 100, 1] slice(p0), slice={[0:32:1], [0:100:1], [0:1:1]}
    }

    ENTRY entry {
      input = s32[100, 64, 32] parameter(0)
      add = s32[100, 64, 32] add(input, input)
      transpose = s32[32, 100, 64] transpose(add), dimensions={2, 0, 1}
      ROOT %fusion = s32[32, 100, 1] fusion(transpose), kind=kLoop, calls=fusion
  })";
  EXPECT_FALSE(IsReadCoalescedHeuristic(ir));
}

// TODO(batzor): Implement this. Dependency: TransposeFusion emitter.
// TEST_F(CoalescingTest, TransposeOnlyOuterDims)

TEST_F(CoalescingTest, PadOp) {
  std::string_view ir = R"(
    HloModule module
    fusion {
      p0 = s32[997, 436] parameter(0)
      p1 = s32[] parameter(1)
      ROOT pad = s32[1024, 512] pad(p0, p1), padding=10_17x24_52
    }
    ENTRY entry {
      p0 = s32[997, 436] parameter(0)
      p1 = s32[] parameter(1)
      ROOT %fusion = s32[1024, 512] fusion(p0, p1), kind=kLoop, calls=fusion
  })";
  // thread_x to linearized input mapping for thread_x in [0, 31]:
  // Operand 1: (thread_x)[s0] -> (thread_x * 4 + s0 - 4384)
  //   for s0 in [0, 3] and thread_x * 4 + s0 in [24, 459]
  // Operand 2: (thread_x) -> ()
  EXPECT_THAT(IsReadCoalescedPerOperand(ir), ElementsAre(true, true));
}

// TODO(batzor): Implement this. Dependency: ReductionFusion emitter.
// TEST_F(CoalescingTest, RowReduction)

// TODO(batzor): Implement this. Dependency: ReductionFusion emitter.
// TEST_F(CoalescingTest, MultiRowReduction)

// TODO(batzor): Implement this. Dependency: ReductionFusion emitter.
// TEST_F(CoalescingTest, ColumnReduction)

// TODO(batzor): Implement this. Dependency: ReductionFusion emitter.
// TEST_F(CoalescingTest, VariadicReduceViaLoopEmitter)

// TODO(batzor): Implement this. Dependency: ReductionFusion emitter.
// TEST_F(CoalescingTest, VariadicReduceViaReductionEmitter)

TEST_F(CoalescingTest, Gather) {
  std::string_view ir = R"(
    HloModule module
    fusion {
      operand = s32[33, 76, 70] parameter(0)
      indices = s32[1806, 2] parameter(1)
      ROOT gather = s32[1806, 7, 8, 4] gather(operand, indices),
        offset_dims={1,2,3}, collapsed_slice_dims={}, start_index_map={0,1},
        index_vector_dim=1, slice_sizes={7,8,4}
    }
    ENTRY entry {
      p0 = s32[33, 76, 70] parameter(0)
      p1 = s32[1806, 2] parameter(1)
      ROOT %fusion = s32[1806, 7, 8, 4] fusion(p0, p1), kind=kLoop, calls=fusion
  })";
  // thread_x to linearized input mapping for thread_x in [0, 31]:
  // Operand 1: (d0)[s0] -> (
  //  (d0 floordiv 8) * 5320 + (d0 mod 8) * 70 + s0 * 70 + 34)
  //  for s0 in [0, 3]
  // Operand 2: (d0)[s0] -> (s0) for s0 in [0, 1].
  EXPECT_THAT(IsReadCoalescedPerOperand(ir), ElementsAre(false, true));
}

TEST_F(CoalescingTest, DynamicSlice) {
  std::string_view ir = R"(
    HloModule module
    fusion {
      %src = s32[2,2,258] parameter(0)
      %of1 = s32[] parameter(1)
      %of2 = s32[] parameter(2)
      %of3 = s32[] parameter(3)
      ROOT %ds = s32[1,2,32] dynamic-slice(s32[2,2,258] %src,
        s32[] %of1, s32[] %of2, s32[] %of3),
        dynamic_slice_sizes={1, 2, 32}
    }
    ENTRY entry {
      %p0 = s32[2,2,258] parameter(0)
      %p1 = s32[] parameter(1)
      %p2 = s32[] parameter(2)
      %p3 = s32[] parameter(3)
      ROOT %fusion = s32[1,2,32] fusion(p0, p1, p2, p3), kind=kLoop, calls=fusion
  })";
  // thread_x to linearized input mapping for thread_x in [0, 31]:
  // Operand 1: (d0) -> (d0).
  EXPECT_THAT(IsReadCoalescedPerOperand(ir),
              ElementsAre(true, true, true, true));
}

TEST_F(CoalescingTest, UnusedParameter) {
  Shape shape = ShapeUtil::MakeShape(S32, {100000});

  auto module = std::make_unique<HloModule>("m", HloModuleConfig{});
  HloComputation::Builder b("b");
  auto p0 = b.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  auto p1 = b.AddInstruction(HloInstruction::CreateParameter(1, shape, "p1"));

  HloComputation::Builder sub_builder("subcomp");
  HloInstruction* p0f = sub_builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "p0f"));
  // p1f is not used.
  HloInstruction* p1f = sub_builder.AddInstruction(
      HloInstruction::CreateParameter(1, shape, "p1f"));
  ASSERT_NE(p1f, nullptr);
  sub_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, p0f));

  HloComputation* subcomp = module->AddEmbeddedComputation(sub_builder.Build());
  auto fusion = HloInstruction::CreateFusion(
      shape, HloInstruction::FusionKind::kLoop, {p0, p1}, subcomp);
  b.AddInstruction(std::move(fusion));
  module->AddEntryComputation(b.Build());

  EXPECT_THAT(IsReadCoalescedPerOperand(
                  module->entry_computation()->root_instruction()),
              ElementsAre(true, true));
}

// TODO(batzor): Implement this. Dependency: ConcatenateFusion emitter.
// TEST_F(CoalescingTest, Param)

// TODO(batzor): Implement this. Dependency: TiledHloInstruction,
// SymbolicTileAnalysis.
// TEST_F(CoalescingForTiledHloTest, TiledReadCoalescedHeuristic_Transpose)

// TODO(batzor): Implement this. Dependency: TiledHloInstruction,
// SymbolicTileAnalysis.
// TEST_F(CoalescingForTiledHloTest,
//        TiledReadCoalescedHeuristic_MaskingIsHandledCorrectly)

// TODO(batzor): Implement this. Dependency: TiledHloInstruction,
// SymbolicTileAnalysis.
// TEST_F(CoalescingForTiledHloTest, RhsTransposedLayout)

// TODO(batzor): Implement this. Dependency: TiledHloInstruction,
// SymbolicTileAnalysis.
// TEST_F(CoalescingForTiledHloTest, SmallDataTypes)

// TODO(batzor): Implement this. Dependency: TiledHloInstruction,
// SymbolicTileAnalysis.
// TEST_F(CoalescingForTiledHloTest,
//        EffectiveBandwidthUtilizationRateIsComputedCorrectlyForTiledMemoryAccess)

}  // namespace
}  // namespace zkx::gpu
