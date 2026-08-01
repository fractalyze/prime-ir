/* Copyright 2026 The PrimeIR Authors.

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

#ifndef PRIME_IR_DIALECT_FIELD_CONVERSIONS_BINARYFIELDTOARITH_TOWERFLATBASIS_H_
#define PRIME_IR_DIALECT_FIELD_CONVERSIONS_BINARYFIELDTOARITH_TOWERFLATBASIS_H_

#include <cstdint>

namespace mlir::prime_ir::field {

// GF(2)-linear basis-change constants between the Fan-Paar tower basis
// (BinaryFieldCodeGen.cpp: bit i of a bf<k> is the multilinear monomial
// prod_j X_{j+1}^{i_j}) and an isomorphic flat polynomial basis
// GF(2)[y]/(f_k), where a multiply is a single carry-less product plus a
// low-weight reduction. The f_k here DEFINE the `bf<4|5, flat>` types'
// semantics (FieldTypes.td isNarrowFlat) — every consumer of those types
// (the portable BinaryFieldToArith lowering, the NVPTX clmad specializer,
// and downstream emitters hoisting whole kernels into the flat basis) must
// agree on them through this header. Column i of kTowerToFlat* is the flat
// image of tower basis monomial i; kFlatToTower* is the inverse matrix.
// Values fit the field width; the tables are uint64_t so they feed the i64
// clmad domain without per-use widening. Generated and proven
// (irreducibility, homomorphism on all basis pairs, inverse) by
// tools/derive_tower_flat_basis.py.

// bf<4>: f_4(y) = y^16 + y^5 + y^3 + y + 1. Low part (f_4 minus the y^16
// term) drives the reduction fold.
inline constexpr uint64_t kFlatModLow16 = 0x2b;
inline constexpr uint64_t kTowerToFlat16[16] = {
    0x1,    0x732,  0xa785, 0x2f29, 0xf6bc, 0x2ee3, 0xb115, 0x46ff,
    0x3394, 0xd651, 0x644c, 0xcdd9, 0xc696, 0x88e0, 0xc682, 0xe708,
};
inline constexpr uint64_t kFlatToTower16[16] = {
    0x1,    0xd056, 0xdb5d, 0x8e62, 0x8b5d, 0x499c, 0x7533, 0xde9a,
    0xf5b5, 0x7d97, 0x9ab7, 0x6079, 0xb2a3, 0xb51b, 0xba0,  0xa24c,
};

// bf<5>: f_5(y) = y^32 + y^7 + y^3 + y^2 + 1.
inline constexpr uint64_t kFlatModLow32 = 0x8d;
inline constexpr uint64_t kTowerToFlat32[32] = {
    0x1,        0x54fd1264, 0xccec155c, 0xbaa88a9e, 0x843b36b,  0xcc269e97,
    0x664dc6fd, 0xc876bf36, 0xa733cc5a, 0x41bbf9ff, 0xaa45f221, 0xe618ffea,
    0x2e3bbbd1, 0xff2b2070, 0xb2b073d9, 0x94ebfc3f, 0x821f5c70, 0xdb81a57e,
    0x65def133, 0xb6c97b11, 0x9649d335, 0x419e9963, 0x2c04fc23, 0x6bd658c9,
    0x5364a11f, 0x7f52f2fc, 0x52adf73b, 0x50bb1f53, 0xffd615a8, 0x13385531,
    0xa3c08108, 0xe4df4d8c,
};
inline constexpr uint64_t kFlatToTower32[32] = {
    0x1,        0x844f21ff, 0x68356949, 0xe957cff,  0xfbc61024, 0x75024e3a,
    0x2ed00246, 0xd90bc8b7, 0x81c74306, 0x2768f200, 0x2bb29802, 0xdc866f4a,
    0x189cacd1, 0xd202507f, 0x27bb949b, 0xfdc28fb3, 0xd8b56f02, 0xa8612168,
    0xdafc5bda, 0xb1045c0a, 0x911c4d02, 0xb3bd3f19, 0x9f3b4d27, 0x7cb4f66e,
    0xa1f4fd7a, 0x6e3b232e, 0xf1eb9570, 0xaf14d03b, 0x67fc4d14, 0x92dd619b,
    0xe8670202, 0x8c0db4cc,
};

} // namespace mlir::prime_ir::field

// NOLINTNEXTLINE(whitespace/line_length)
#endif // PRIME_IR_DIALECT_FIELD_CONVERSIONS_BINARYFIELDTOARITH_TOWERFLATBASIS_H_
