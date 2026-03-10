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

#include "zkx/service/gpu/transforms/msm_batch_fusion.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "zkx/hlo/ir/hlo_casting_utils.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/ir/hlo_instructions.h"
#include "zkx/hlo/ir/hlo_opcode.h"
#include "zkx/hlo/testlib/hlo_hardware_independent_test_base.h"

namespace zkx::gpu {
namespace {

class MsmBatchFusionTest : public HloHardwareIndependentTestBase {};

// Two MSMs sharing the same bases should be fused into a batched MSM.
TEST_F(MsmBatchFusionTest, FusesTwoMsmsWithSharedBases) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test

    ENTRY main {
      scalars0 = bn254_sf_mont[1024] parameter(0)
      scalars1 = bn254_sf_mont[1024] parameter(1)
      bases = bn254_g1_affine_mont[1024] parameter(2)
      msm0 = bn254_g1_affine_mont[] msm(scalars0, bases), window_bits=15
      msm1 = bn254_g1_affine_mont[] msm(scalars1, bases), window_bits=15
      ROOT tuple = (bn254_g1_affine_mont[], bn254_g1_affine_mont[]) tuple(msm0, msm1)
    }
  )")
                    .value();

  MsmBatchFusion pass;
  EXPECT_TRUE(pass.Run(module.get()).value());

  // After fusion, both results should come from slices of a batched MSM.
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kTuple);

  // Each tuple element: reshape(slice(batched_msm)).
  HloInstruction* elem0 = root->mutable_operand(0);
  HloInstruction* elem1 = root->mutable_operand(1);
  EXPECT_EQ(elem0->opcode(), HloOpcode::kReshape);
  EXPECT_EQ(elem1->opcode(), HloOpcode::kReshape);

  HloInstruction* slice0 = elem0->mutable_operand(0);
  HloInstruction* slice1 = elem1->mutable_operand(0);
  EXPECT_EQ(slice0->opcode(), HloOpcode::kSlice);
  EXPECT_EQ(slice1->opcode(), HloOpcode::kSlice);

  // Both slices should reference the same batched MSM.
  EXPECT_EQ(slice0->operand(0), slice1->operand(0));
  auto* batched = Cast<HloMsmInstruction>(slice0->mutable_operand(0));
  EXPECT_EQ(batched->opcode(), HloOpcode::kMsm);
  EXPECT_EQ(batched->batch_size(), 2);
  EXPECT_TRUE(batched->are_points_shared());
  EXPECT_EQ(batched->window_bits(), 15);

  // Scalars should be concatenated: [2048].
  EXPECT_EQ(batched->operand(0)->opcode(), HloOpcode::kConcatenate);
  EXPECT_EQ(batched->operand(0)->shape().dimensions(0), 2048);
}

// MSMs with different bases should not be fused.
TEST_F(MsmBatchFusionTest, DoesNotFuseDifferentBases) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test

    ENTRY main {
      scalars0 = bn254_sf_mont[1024] parameter(0)
      scalars1 = bn254_sf_mont[1024] parameter(1)
      bases0 = bn254_g1_affine_mont[1024] parameter(2)
      bases1 = bn254_g1_affine_mont[1024] parameter(3)
      msm0 = bn254_g1_affine_mont[] msm(scalars0, bases0), window_bits=15
      msm1 = bn254_g1_affine_mont[] msm(scalars1, bases1), window_bits=15
      ROOT tuple = (bn254_g1_affine_mont[], bn254_g1_affine_mont[]) tuple(msm0, msm1)
    }
  )")
                    .value();

  MsmBatchFusion pass;
  EXPECT_FALSE(pass.Run(module.get()).value());
}

// MSMs with different window_bits should not be fused.
TEST_F(MsmBatchFusionTest, DoesNotFuseDifferentConfig) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test

    ENTRY main {
      scalars0 = bn254_sf_mont[1024] parameter(0)
      scalars1 = bn254_sf_mont[1024] parameter(1)
      bases = bn254_g1_affine_mont[1024] parameter(2)
      msm0 = bn254_g1_affine_mont[] msm(scalars0, bases), window_bits=15
      msm1 = bn254_g1_affine_mont[] msm(scalars1, bases), window_bits=11
      ROOT tuple = (bn254_g1_affine_mont[], bn254_g1_affine_mont[]) tuple(msm0, msm1)
    }
  )")
                    .value();

  MsmBatchFusion pass;
  EXPECT_FALSE(pass.Run(module.get()).value());
}

// A single MSM should not trigger fusion.
TEST_F(MsmBatchFusionTest, DoesNotFuseSingleMsm) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test

    ENTRY main {
      scalars = bn254_sf_mont[1024] parameter(0)
      bases = bn254_g1_affine_mont[1024] parameter(1)
      ROOT msm = bn254_g1_affine_mont[] msm(scalars, bases), window_bits=15
    }
  )")
                    .value();

  MsmBatchFusion pass;
  EXPECT_FALSE(pass.Run(module.get()).value());
}

// MSMs with data dependency should not be fused.
TEST_F(MsmBatchFusionTest, DoesNotFuseWithDependency) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test

    ENTRY main {
      scalars0 = bn254_sf_mont[1024] parameter(0)
      bases = bn254_g1_affine_mont[1024] parameter(1)
      msm0 = bn254_g1_affine_mont[] msm(scalars0, bases), window_bits=15
      bcast = bn254_g1_affine_mont[1024] broadcast(msm0), dimensions={}
      msm1 = bn254_g1_affine_mont[] msm(scalars0, bcast), window_bits=15
      ROOT tuple = (bn254_g1_affine_mont[], bn254_g1_affine_mont[]) tuple(msm0, msm1)
    }
  )")
                    .value();

  MsmBatchFusion pass;
  // msm1 uses bcast(msm0) as bases, which is different from msm0's bases,
  // so they won't be grouped (different bases operand). No fusion.
  EXPECT_FALSE(pass.Run(module.get()).value());
}

// Three MSMs sharing bases should all be fused into batch_size=3.
TEST_F(MsmBatchFusionTest, FusesThreeMsms) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test

    ENTRY main {
      sc0 = bn254_sf_mont[1024] parameter(0)
      sc1 = bn254_sf_mont[1024] parameter(1)
      sc2 = bn254_sf_mont[1024] parameter(2)
      bases = bn254_g1_affine_mont[1024] parameter(3)
      msm0 = bn254_g1_affine_mont[] msm(sc0, bases), window_bits=15
      msm1 = bn254_g1_affine_mont[] msm(sc1, bases), window_bits=15
      msm2 = bn254_g1_affine_mont[] msm(sc2, bases), window_bits=15
      ROOT tuple = (bn254_g1_affine_mont[], bn254_g1_affine_mont[], bn254_g1_affine_mont[]) tuple(msm0, msm1, msm2)
    }
  )")
                    .value();

  MsmBatchFusion pass;
  EXPECT_TRUE(pass.Run(module.get()).value());

  // Find the batched MSM.
  HloInstruction* root = module->entry_computation()->root_instruction();
  HloInstruction* reshape0 = root->mutable_operand(0);
  HloInstruction* slice0 = reshape0->mutable_operand(0);
  auto* batched = Cast<HloMsmInstruction>(slice0->mutable_operand(0));
  EXPECT_EQ(batched->batch_size(), 3);
  EXPECT_EQ(batched->operand(0)->shape().dimensions(0), 3072);
}

}  // namespace
}  // namespace zkx::gpu
