// Copyright 2026 The PrimeIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

// Guards the Tower<->AES isomorphism tables used by the GFNI (x86) and PMULL
// (ARM) binary-field specializations. The GFNI/ARM multiply computes
//   from_aes( aes_mul( to_aes(a), to_aes(b) ) )
// so the tables MUST implement a field isomorphism between the Fan-Paar
// canonical GF(2^8) tower (the field zk_dtypes multiplies in) and the AES field
// (x^8+x^4+x^3+x+1, 0x11B — what vgf2p8mulb uses). Otherwise the specialized
// paths silently disagree with the portable BinaryFieldToArith lowering. This
// runs pure host arithmetic, so it validates the tables without GFNI hardware.

#include <cstdint>

#include "gtest/gtest.h"
#include "prime_ir/Dialect/Field/Conversions/BinaryFieldToArith/BinaryFieldTables.h"
#include "zk_dtypes/include/field/binary_field.h"

namespace mlir::prime_ir::field {
namespace {

// GF(2^8) tower multiply — the field the codegen targets.
uint8_t towerMul(uint8_t a, uint8_t b) {
  return (zk_dtypes::BinaryFieldT3::FromUnchecked(a) *
          zk_dtypes::BinaryFieldT3::FromUnchecked(b))
      .value();
}

// AES field multiply mod x^8+x^4+x^3+x+1 (0x11B), the vgf2p8mulb field.
uint8_t aesMul(uint8_t a, uint8_t b) {
  uint8_t p = 0;
  for (int i = 0; i < 8; ++i) {
    if (b & 1)
      p ^= a;
    bool hi = a & 0x80;
    a <<= 1;
    if (hi)
      a ^= 0x1b;
    b >>= 1;
  }
  return p;
}

// vgf2p8affineqb semantics with imm8=0: output bit i is the parity of
// (matrix qword byte [7 - i]) AND the input byte. Mirrors the hardware op the
// packed matrix is consumed by.
uint8_t affine(const uint8_t (&m)[8], uint8_t x) {
  uint8_t out = 0;
  for (int i = 0; i < 8; ++i)
    out |= static_cast<uint8_t>(__builtin_parity(m[7 - i] & x) << i);
  return out;
}

// The scalar lookup table must be a field isomorphism tower -> AES.
TEST(BinaryFieldAesTables, LookupIsFieldIsomorphism) {
  bool seen[256] = {false};
  for (int x = 0; x < 256; ++x) {
    ASSERT_FALSE(seen[kTowerToAesLookupTable[x]]) << "not a bijection at " << x;
    seen[kTowerToAesLookupTable[x]] = true;
  }
  ASSERT_EQ(kTowerToAesLookupTable[0], 0);
  ASSERT_EQ(kTowerToAesLookupTable[1], 1);
  for (int a = 0; a < 256; ++a)
    for (int b = 0; b < 256; ++b)
      ASSERT_EQ(kTowerToAesLookupTable[towerMul(a, b)],
                aesMul(kTowerToAesLookupTable[a], kTowerToAesLookupTable[b]))
          << "homomorphism fails at a=" << a << " b=" << b;
}

// The GFNI affine matrices must realize the same map as the lookup table (and
// invert each other), under the vgf2p8affineqb bit convention.
TEST(BinaryFieldAesTables, MatricesMatchLookupUnderGfniConvention) {
  for (int x = 0; x < 256; ++x) {
    EXPECT_EQ(affine(kTowerToAesMatrix, static_cast<uint8_t>(x)),
              kTowerToAesLookupTable[x])
        << "tower->aes matrix disagrees with lookup at " << x;
    EXPECT_EQ(affine(kAesToTowerMatrix, kTowerToAesLookupTable[x]),
              static_cast<uint8_t>(x))
        << "aes->tower matrix is not the inverse at " << x;
  }
}

} // namespace
} // namespace mlir::prime_ir::field
