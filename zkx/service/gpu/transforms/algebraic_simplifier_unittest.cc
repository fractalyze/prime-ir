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

#include "zkx/service/gpu/transforms/algebraic_simplifier.h"

#include <string>

#include "gtest/gtest.h"

#include "xla/tsl/platform/statusor.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "zkx/hlo/transforms/simplifiers/algebraic_simplifier.h"
#include "zkx/stream_executor/device_description.h"

namespace zkx::gpu {
namespace {

class GpuAlgebraicSimplifierTest : public HloHardwareIndependentTestBase {
 public:
  se::CudaComputeCapability Ampere() {
    return se::CudaComputeCapability::Ampere();
  }
};

TEST_F(GpuAlgebraicSimplifierTest, VectorVectorDotShouldBeStrengthReduced) {
  const std::string& hlo_string = R"(
HloModule m

ENTRY entry {
  p0 = s32[32, 500] parameter(0)
  p1 = s32[32, 500] parameter(1)
  ROOT dot = s32[32] dot(p0, p1), lhs_batch_dims={0},
    lhs_contracting_dims={1}, rhs_batch_dims={0}, rhs_contracting_dims={1}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  const HloInstruction* dot = module->entry_computation()->root_instruction();
  AlgebraicSimplifierOptions options;
  options.set_enable_dot_strength_reduction(true);
  se::CudaComputeCapability ampere(8, 0);
  GpuAlgebraicSimplifier simplifier(options, ampere);
  GpuAlgebraicSimplifierVisitor visitor(options, ampere, &simplifier);
  EXPECT_TRUE(visitor.ShouldStrengthReduceDotToReduce(dot));
}

TEST_F(GpuAlgebraicSimplifierTest, MatrixVectorDotShouldNotBeStrengthReduced) {
  const std::string& hlo_string = R"(
HloModule m

ENTRY entry {
  p0 = s32[32, 5000, 7000] parameter(0)
  p1 = s32[32, 5000] parameter(1)
  ROOT dot = s32[32,7000] dot(p0, p1), lhs_batch_dims={0},
    lhs_contracting_dims={1}, rhs_batch_dims={0}, rhs_contracting_dims={1}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  const HloInstruction* dot = module->entry_computation()->root_instruction();
  AlgebraicSimplifierOptions options;
  options.set_enable_dot_strength_reduction(true);
  se::CudaComputeCapability ampere(8, 0);
  GpuAlgebraicSimplifier simplifier(options, ampere);
  GpuAlgebraicSimplifierVisitor visitor(options, ampere, &simplifier);
  EXPECT_FALSE(visitor.ShouldStrengthReduceDotToReduce(dot));
}

TEST_F(GpuAlgebraicSimplifierTest, SmallDotShouldBeStrengthReduced) {
  const std::string& hlo_string = R"(
HloModule m

ENTRY entry {
  p0 = s32[32, 50, 70] parameter(0)
  p1 = s32[32, 50] parameter(1)
  ROOT dot = s32[32,70] dot(p0, p1), lhs_batch_dims={0},
    lhs_contracting_dims={1}, rhs_batch_dims={0}, rhs_contracting_dims={1}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  const HloInstruction* dot = module->entry_computation()->root_instruction();
  AlgebraicSimplifierOptions options;
  options.set_enable_dot_strength_reduction(true);
  se::CudaComputeCapability ampere(8, 0);
  GpuAlgebraicSimplifier simplifier(options, ampere);
  GpuAlgebraicSimplifierVisitor visitor(options, ampere, &simplifier);
  EXPECT_TRUE(visitor.ShouldStrengthReduceDotToReduce(dot));
}

TEST_F(GpuAlgebraicSimplifierTest, SmallDotShouldBeStrengthReduced2) {
  const std::string& hlo_string = R"(
HloModule m

ENTRY entry {
  p0 = s32[2000, 3000] parameter(0)
  p1 = s32[2000] parameter(1)
  ROOT dot = s32[3000] dot(p0, p1), lhs_contracting_dims={0},
    rhs_contracting_dims={0}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  const HloInstruction* dot = module->entry_computation()->root_instruction();
  AlgebraicSimplifierOptions options;
  options.set_enable_dot_strength_reduction(true);
  se::CudaComputeCapability ampere(8, 0);
  GpuAlgebraicSimplifier simplifier(options, ampere);
  GpuAlgebraicSimplifierVisitor visitor(options, ampere, &simplifier);
  EXPECT_TRUE(visitor.ShouldStrengthReduceDotToReduce(dot));
}

}  // namespace
}  // namespace zkx::gpu
