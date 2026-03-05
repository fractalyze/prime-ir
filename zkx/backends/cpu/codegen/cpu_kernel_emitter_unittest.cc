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

// Unit tests for CpuKernelEmitter multi-dimensional partitioning.
// Tests the partition computation, shape scaling, and thread_dim output
// at the emitter level without running the full compiler pipeline.

#include "zkx/backends/cpu/codegen/cpu_kernel_emitter.h"

#include <cstdint>
#include <memory>

#include "gtest/gtest.h"
#include "mlir/IR/MLIRContext.h"

#include "xla/tsl/platform/statusor.h"
#include "zkx/hlo/analysis/hlo_ordering.h"
#include "zkx/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "zkx/service/buffer_assignment.h"
#include "zkx/service/buffer_value.h"
#include "zkx/service/cpu/backend_config.pb.h"
#include "zkx/shape_util.h"

namespace zkx::cpu {
namespace {

class CpuKernelEmitterUnitTest : public HloHardwareIndependentTestBase {
 protected:
  void SetUp() override {
    HloHardwareIndependentTestBase::SetUp();
    mlir_context_ = std::make_unique<mlir::MLIRContext>();
  }

  // Helper: parse HLO, set backend_config on the target instruction,
  // build BufferAssignment, emit kernel, and return thread_dim.x.
  absl::StatusOr<uint64_t> EmitAndGetThreadDimX(
      std::string_view hlo_text, HloOpcode target_opcode,
      const BackendConfig& backend_config) {
    TF_ASSIGN_OR_RETURN(auto module, ParseAndReturnVerifiedModule(hlo_text));

    auto* instr = FindInstruction(module.get(), target_opcode);
    TF_RETURN_IF_ERROR(instr->set_backend_config(backend_config));

    TF_ASSIGN_OR_RETURN(
        auto buffer_assignment,
        BufferAssigner::Run(
            module.get(), std::make_unique<DependencyHloOrdering>(module.get()),
            /*buffer_size=*/
            [](const BufferValue& buf) {
              return ShapeUtil::ByteSizeOf(buf.shape());
            },
            /*color_alignment=*/
            [](LogicalBuffer::Color) -> int64_t { return 1; }));

    CpuKernelEmitter emitter(mlir_context_.get(), instr,
                             buffer_assignment.get());
    TF_ASSIGN_OR_RETURN(auto kernel_def, emitter.EmitKernelDefinition());
    return kernel_def.spec().thread_dim().x;
  }

  std::unique_ptr<mlir::MLIRContext> mlir_context_;
};

// No backend_config → no partitioning → thread_dim.x = 1.
TEST_F(CpuKernelEmitterUnitTest, NoPartitioningReturnsThreadDim1) {
  constexpr char kHlo[] = R"(
    HloModule m
    ENTRY e {
      p0 = s32[1024] parameter(0)
      p1 = s32[1024] parameter(1)
      ROOT add = s32[1024] add(p0, p1)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  auto* add = FindInstruction(module.get(), HloOpcode::kAdd);

  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer_assignment,
      BufferAssigner::Run(
          module.get(), std::make_unique<DependencyHloOrdering>(module.get()),
          [](const BufferValue& buf) {
            return ShapeUtil::ByteSizeOf(buf.shape());
          },
          [](LogicalBuffer::Color) -> int64_t { return 1; }));

  CpuKernelEmitter emitter(mlir_context_.get(), add, buffer_assignment.get());
  TF_ASSERT_OK_AND_ASSIGN(auto kernel_def, emitter.EmitKernelDefinition());

  EXPECT_EQ(kernel_def.spec().thread_dim().x, 1);
}

// Single-dim partitioning: partitions=[4] on s32[256,1024].
// Expected thread_dim.x = 4 (outermost dim 256 ≥ 4).
TEST_F(CpuKernelEmitterUnitTest, SingleDimPartitioning) {
  constexpr char kHlo[] = R"(
    HloModule m
    ENTRY e {
      p0 = s32[256,1024] parameter(0)
      p1 = s32[256,1024] parameter(1)
      ROOT add = s32[256,1024] add(p0, p1)
    }
  )";

  BackendConfig config;
  config.add_outer_dimension_partitions(4);
  TF_ASSERT_OK_AND_ASSIGN(uint64_t td_x,
                          EmitAndGetThreadDimX(kHlo, HloOpcode::kAdd, config));
  EXPECT_EQ(td_x, 4);
}

// Multi-dim partitioning: partitions=[4,4] on s32[4,64,1024].
// dim0=4 partitions=4, dim1=64 partitions=4 → total 4×4=16.
TEST_F(CpuKernelEmitterUnitTest, MultiDimPartition) {
  constexpr char kHlo[] = R"(
    HloModule m
    ENTRY e {
      p0 = s32[4,64,1024] parameter(0)
      p1 = s32[4,64,1024] parameter(1)
      ROOT add = s32[4,64,1024] add(p0, p1)
    }
  )";

  BackendConfig config;
  config.add_outer_dimension_partitions(4);
  config.add_outer_dimension_partitions(4);
  TF_ASSERT_OK_AND_ASSIGN(uint64_t td_x,
                          EmitAndGetThreadDimX(kHlo, HloOpcode::kAdd, config));
  EXPECT_EQ(td_x, 16);
}

// Large outermost dim with high partition count.
// partitions=[8] on s32[256,1024] → LargestFactorAtMost(256,8)=8.
TEST_F(CpuKernelEmitterUnitTest, LargeOutermostDim) {
  constexpr char kHlo[] = R"(
    HloModule m
    ENTRY e {
      p0 = s32[256,1024] parameter(0)
      p1 = s32[256,1024] parameter(1)
      ROOT add = s32[256,1024] add(p0, p1)
    }
  )";

  BackendConfig config;
  config.add_outer_dimension_partitions(8);
  TF_ASSERT_OK_AND_ASSIGN(uint64_t td_x,
                          EmitAndGetThreadDimX(kHlo, HloOpcode::kAdd, config));
  EXPECT_EQ(td_x, 8);
}

// Broadcast operand: s32[1024] broadcast to s32[4,64,1024]. The broadcast
// source has fewer elements and is skipped in min_dim computation.
// partitions=[4] → LargestFactorAtMost(4,4)=4.
TEST_F(CpuKernelEmitterUnitTest, BroadcastOperandSkippedInMinDim) {
  constexpr char kHlo[] = R"(
    HloModule m
    ENTRY e {
      p0 = s32[4,64,1024] parameter(0)
      p1 = s32[1024] parameter(1)
      bcast = s32[4,64,1024] broadcast(p1), dimensions={2}
      ROOT add = s32[4,64,1024] add(p0, bcast)
    }
  )";

  BackendConfig config;
  config.add_outer_dimension_partitions(4);
  TF_ASSERT_OK_AND_ASSIGN(uint64_t td_x,
                          EmitAndGetThreadDimX(kHlo, HloOpcode::kAdd, config));
  EXPECT_EQ(td_x, 4);
}

// Unary elementwise: negate with outermost-dim partition.
TEST_F(CpuKernelEmitterUnitTest, UnaryElementwisePartitioned) {
  constexpr char kHlo[] = R"(
    HloModule m
    ENTRY e {
      p0 = s32[16,1024] parameter(0)
      ROOT neg = s32[16,1024] negate(p0)
    }
  )";

  BackendConfig config;
  config.add_outer_dimension_partitions(4);
  TF_ASSERT_OK_AND_ASSIGN(
      uint64_t td_x, EmitAndGetThreadDimX(kHlo, HloOpcode::kNegate, config));
  EXPECT_EQ(td_x, 4);
}

// Scalar instruction: not eligible for partitioning even with backend_config.
TEST_F(CpuKernelEmitterUnitTest, ScalarNotPartitioned) {
  constexpr char kHlo[] = R"(
    HloModule m
    ENTRY e {
      p0 = s32[] parameter(0)
      p1 = s32[] parameter(1)
      ROOT add = s32[] add(p0, p1)
    }
  )";

  BackendConfig config;
  config.add_outer_dimension_partitions(4);
  TF_ASSERT_OK_AND_ASSIGN(uint64_t td_x,
                          EmitAndGetThreadDimX(kHlo, HloOpcode::kAdd, config));
  EXPECT_EQ(td_x, 1);
}

// Partitions larger than dim size: partitions=[16] on s32[4,1024].
// LargestFactorAtMost(4,16) = 4. Total = 4.
TEST_F(CpuKernelEmitterUnitTest, PartitionLargerThanDimIsClamped) {
  constexpr char kHlo[] = R"(
    HloModule m
    ENTRY e {
      p0 = s32[4,1024] parameter(0)
      p1 = s32[4,1024] parameter(1)
      ROOT add = s32[4,1024] add(p0, p1)
    }
  )";

  BackendConfig config;
  config.add_outer_dimension_partitions(16);
  TF_ASSERT_OK_AND_ASSIGN(uint64_t td_x,
                          EmitAndGetThreadDimX(kHlo, HloOpcode::kAdd, config));
  EXPECT_EQ(td_x, 4);
}

// Prime-sized dimension: partitions=[8] on s32[7,1024].
// LargestFactorAtMost(7,8) = 7. Total = 7.
TEST_F(CpuKernelEmitterUnitTest, PrimeDimPartitioning) {
  constexpr char kHlo[] = R"(
    HloModule m
    ENTRY e {
      p0 = s32[7,1024] parameter(0)
      p1 = s32[7,1024] parameter(1)
      ROOT add = s32[7,1024] add(p0, p1)
    }
  )";

  BackendConfig config;
  config.add_outer_dimension_partitions(8);
  TF_ASSERT_OK_AND_ASSIGN(uint64_t td_x,
                          EmitAndGetThreadDimX(kHlo, HloOpcode::kAdd, config));
  EXPECT_EQ(td_x, 7);
}

}  // namespace
}  // namespace zkx::cpu
