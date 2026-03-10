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

#include "zkx/service/gpu/transforms/msm_chunk_split.h"

#include <cstdint>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "zkx/hlo/ir/hlo_casting_utils.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/ir/hlo_instructions.h"
#include "zkx/hlo/ir/hlo_opcode.h"
#include "zkx/hlo/testlib/hlo_hardware_independent_test_base.h"

namespace zkx::gpu {
namespace {

class MsmChunkSplitTest : public HloHardwareIndependentTestBase {};

// BN254 Fr = 32 bytes, G1 affine = 64 bytes, G1 projective = 96 bytes.
// For N=2048: c=7, nof_bms=37 → EstimateMsmMemoryBytes ≈ 4.1 MB.
// For N=1024: c=6, nof_bms=43 → EstimateMsmMemoryBytes ≈ 2.4 MB.
// Budget of 3 MB → fits 1024 but not 2048 → splits into 2 chunks.
TEST_F(MsmChunkSplitTest, SplitsLargeMsm) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test

    ENTRY main {
      scalars = bn254_sf_mont[2048] parameter(0)
      bases = bn254_g1_affine_mont[2048] parameter(1)
      ROOT msm = bn254_g1_affine_mont[] msm(scalars, bases), window_bits=15
    }
  )")
                    .value();

  // 3 MB budget → splits 2048 into 2 chunks.
  MsmChunkSplit pass(/*device_memory_bytes=*/3 * 1024 * 1024,
                     /*memory_fraction=*/1.0);
  EXPECT_TRUE(pass.Run(module.get()).value());

  // Root = convert(add(convert(msm0), convert(msm1))).
  // EC add operates in jacobian space, final convert back to affine.
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kConvert);
  EXPECT_EQ(root->shape().element_type(), BN254_G1_AFFINE_MONT);

  const HloInstruction* add = root->operand(0);
  EXPECT_EQ(add->opcode(), HloOpcode::kAdd);
  EXPECT_EQ(add->shape().element_type(), BN254_G1_JACOBIAN_MONT);

  // Both operands are converted from affine to jacobian.
  EXPECT_EQ(add->operand(0)->opcode(), HloOpcode::kConvert);
  EXPECT_EQ(add->operand(1)->opcode(), HloOpcode::kConvert);

  // Each chunk MSM should have sliced operands that sum to 2048.
  auto* chunk0 = Cast<HloMsmInstruction>(add->operand(0)->operand(0));
  auto* chunk1 = Cast<HloMsmInstruction>(add->operand(1)->operand(0));
  int64_t size0 = chunk0->operand(0)->shape().dimensions(0);
  int64_t size1 = chunk1->operand(0)->shape().dimensions(0);
  EXPECT_EQ(size0 + size1, 2048);
  EXPECT_EQ(chunk0->window_bits(), 15);
  EXPECT_EQ(chunk1->window_bits(), 15);
}

// An MSM that fits in memory should not be split.
TEST_F(MsmChunkSplitTest, DoesNotSplitSmallMsm) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test

    ENTRY main {
      scalars = bn254_sf_mont[512] parameter(0)
      bases = bn254_g1_affine_mont[512] parameter(1)
      ROOT msm = bn254_g1_affine_mont[] msm(scalars, bases), window_bits=15
    }
  )")
                    .value();

  // 3 MB budget → 512 should fit easily.
  MsmChunkSplit pass(/*device_memory_bytes=*/3 * 1024 * 1024,
                     /*memory_fraction=*/1.0);
  EXPECT_FALSE(pass.Run(module.get()).value());

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kMsm);
}

// Batched MSMs (batch_size > 1) should be skipped.
TEST_F(MsmChunkSplitTest, SkipsBatchedMsm) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test

    ENTRY main {
      scalars = bn254_sf_mont[4096] parameter(0)
      bases = bn254_g1_affine_mont[2048] parameter(1)
      ROOT msm = bn254_g1_affine_mont[2] msm(scalars, bases), window_bits=15, batch_size=2, are_points_shared=true
    }
  )")
                    .value();

  MsmChunkSplit pass(/*device_memory_bytes=*/3 * 1024 * 1024,
                     /*memory_fraction=*/1.0);
  EXPECT_FALSE(pass.Run(module.get()).value());
}

// Splitting into 3 chunks when size doesn't evenly divide.
TEST_F(MsmChunkSplitTest, SplitsIntoUnevenChunks) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test

    ENTRY main {
      scalars = bn254_sf_mont[3000] parameter(0)
      bases = bn254_g1_affine_mont[3000] parameter(1)
      ROOT msm = bn254_g1_affine_mont[] msm(scalars, bases), window_bits=11
    }
  )")
                    .value();

  // 3 MB budget → max_size < 3000, should split into multiple chunks.
  MsmChunkSplit pass(/*device_memory_bytes=*/3 * 1024 * 1024,
                     /*memory_fraction=*/1.0);
  EXPECT_TRUE(pass.Run(module.get()).value());

  // Root = convert(add(...))
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kConvert);
  EXPECT_EQ(root->shape().element_type(), BN254_G1_AFFINE_MONT);

  const HloInstruction* outer_add = root->operand(0);
  EXPECT_EQ(outer_add->opcode(), HloOpcode::kAdd);
  EXPECT_EQ(outer_add->shape().element_type(), BN254_G1_JACOBIAN_MONT);

  // Third chunk (msm2) is converted to jacobian before the outer add.
  const HloInstruction* chunk2_convert = outer_add->operand(1);
  EXPECT_EQ(chunk2_convert->opcode(), HloOpcode::kConvert);
  EXPECT_EQ(chunk2_convert->shape().element_type(), BN254_G1_JACOBIAN_MONT);
  EXPECT_EQ(chunk2_convert->operand(0)->opcode(), HloOpcode::kMsm);
}

// Precompute factor increases memory usage, causing splits at smaller sizes.
TEST_F(MsmChunkSplitTest, PrecomputeFactorReducesMaxSize) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test

    ENTRY main {
      scalars = bn254_sf_mont[2048] parameter(0)
      bases = bn254_g1_affine_mont[2048] parameter(1)
      ROOT msm = bn254_g1_affine_mont[] msm(scalars, bases), window_bits=15, precompute_factor=4
    }
  )")
                    .value();

  // With precompute_factor=4, points_mem is 4× larger → needs more chunks.
  // N=2048, pf=4: estimated ≈ 3.2 MB. Use 2.5 MB budget to force split.
  MsmChunkSplit pass(/*device_memory_bytes=*/2560 * 1024,
                     /*memory_fraction=*/1.0);
  EXPECT_TRUE(pass.Run(module.get()).value());

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kConvert);
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kAdd);
}

}  // namespace
}  // namespace zkx::gpu
