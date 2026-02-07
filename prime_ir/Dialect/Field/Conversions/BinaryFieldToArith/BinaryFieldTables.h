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

#ifndef PRIME_IR_DIALECT_FIELD_CONVERSIONS_BINARYFIELDTOARITH_BINARYFIELDTABLES_H_
#define PRIME_IR_DIALECT_FIELD_CONVERSIONS_BINARYFIELDTOARITH_BINARYFIELDTABLES_H_

#include <cstdint>

namespace mlir::prime_ir::field {

//===----------------------------------------------------------------------===//
// Tower Alpha Constants
//===----------------------------------------------------------------------===//
//
// Tower alpha constants from zk_dtypes.
// These are the elements satisfying x² + x + α = 0 at each tower level.
// Level 0 (GF(2)) has no α.

constexpr uint64_t kTowerAlphas[8] = {
    0,                      // Level 0: GF(2), no extension
    1,                      // Level 1: x² + x + 1
    2,                      // Level 2: x² + x + 2
    8,                      // Level 3: x² + x + 8
    128,                    // Level 4: x² + x + 128
    32768,                  // Level 5: x² + x + 32768
    2147483648ULL,          // Level 6: x² + x + 2³¹
    9223372036854775808ULL, // Level 7: x² + x + 2⁶³
};

//===----------------------------------------------------------------------===//
// Tower <-> AES Field Isomorphism Tables
//===----------------------------------------------------------------------===//
//
// The AES field uses the irreducible polynomial x⁸ + x⁴ + x³ + x + 1 (0x11B)
// Our binary tower field at level 3 uses recursive extension polynomials:
//   Level 1: X² + X + 1 (α = 1)
//   Level 2: X² + X + 2 (α = 2)
//   Level 3: X² + X + 8 (α = 8)
//
// Source: Binius project
// - binius/crates/field/src/arch/x86_64/gfni/gfni_arithmetics.rs
// - binius/crates/field/src/arch/aarch64/simd_arithmetic.rs

// 8x8 binary transformation matrix: Tower field -> AES field
// Each byte represents a row of the matrix (LSB = column 0)
// For use with x86 vgf2p8affineqb instruction
constexpr uint8_t kTowerToAesMatrix[8] = {
    0x3E, // 0b00111110
    0x98, // 0b10011000
    0x4E, // 0b01001110
    0x96, // 0b10010110
    0xEA, // 0b11101010
    0x6A, // 0b01101010
    0x50, // 0b01010000
    0x31, // 0b00110001
};

// 8x8 binary transformation matrix: AES field -> Tower field
constexpr uint8_t kAesToTowerMatrix[8] = {
    0x0C, // 0b00001100
    0x70, // 0b01110000
    0xA2, // 0b10100010
    0x72, // 0b01110010
    0x3E, // 0b00111110
    0x86, // 0b10000110
    0xE8, // 0b11101000
    0xD1, // 0b11010001
};

// 256-entry lookup table for Tower -> AES field conversion
// For use on ARM where GFNI is not available
// clang-format off
constexpr uint8_t kTowerToAesLookupTable[256] = {
    0x00, 0x01, 0xBC, 0xBD, 0xB0, 0xB1, 0x0C, 0x0D,
    0xEC, 0xED, 0x50, 0x51, 0x5C, 0x5D, 0xE0, 0xE1,
    0xD3, 0xD2, 0x6F, 0x6E, 0x63, 0x62, 0xDF, 0xDE,
    0x3F, 0x3E, 0x83, 0x82, 0x8F, 0x8E, 0x33, 0x32,
    0x8D, 0x8C, 0x31, 0x30, 0x3D, 0x3C, 0x81, 0x80,
    0x61, 0x60, 0xDD, 0xDC, 0xD1, 0xD0, 0x6D, 0x6C,
    0x5E, 0x5F, 0xE2, 0xE3, 0xEE, 0xEF, 0x52, 0x53,
    0xB2, 0xB3, 0x0E, 0x0F, 0x02, 0x03, 0xBE, 0xBF,
    0x2E, 0x2F, 0x92, 0x93, 0x9E, 0x9F, 0x22, 0x23,
    0xC2, 0xC3, 0x7E, 0x7F, 0x72, 0x73, 0xCE, 0xCF,
    0xFD, 0xFC, 0x41, 0x40, 0x4D, 0x4C, 0xF1, 0xF0,
    0x11, 0x10, 0xAD, 0xAC, 0xA1, 0xA0, 0x1D, 0x1C,
    0xA3, 0xA2, 0x1F, 0x1E, 0x13, 0x12, 0xAF, 0xAE,
    0x4F, 0x4E, 0xF3, 0xF2, 0xFF, 0xFE, 0x43, 0x42,
    0x70, 0x71, 0xCC, 0xCD, 0xC0, 0xC1, 0x7C, 0x7D,
    0x9C, 0x9D, 0x20, 0x21, 0x2C, 0x2D, 0x90, 0x91,
    0x58, 0x59, 0xE4, 0xE5, 0xE8, 0xE9, 0x54, 0x55,
    0xB4, 0xB5, 0x08, 0x09, 0x04, 0x05, 0xB8, 0xB9,
    0x8B, 0x8A, 0x37, 0x36, 0x3B, 0x3A, 0x87, 0x86,
    0x67, 0x66, 0xDB, 0xDA, 0xD7, 0xD6, 0x6B, 0x6A,
    0xD5, 0xD4, 0x69, 0x68, 0x65, 0x64, 0xD9, 0xD8,
    0x39, 0x38, 0x85, 0x84, 0x89, 0x88, 0x35, 0x34,
    0x06, 0x07, 0xBA, 0xBB, 0xB6, 0xB7, 0x0A, 0x0B,
    0xEA, 0xEB, 0x56, 0x57, 0x5A, 0x5B, 0xE6, 0xE7,
    0x76, 0x77, 0xCA, 0xCB, 0xC6, 0xC7, 0x7A, 0x7B,
    0x9A, 0x9B, 0x26, 0x27, 0x2A, 0x2B, 0x96, 0x97,
    0xA5, 0xA4, 0x19, 0x18, 0x15, 0x14, 0xA9, 0xA8,
    0x49, 0x48, 0xF5, 0xF4, 0xF9, 0xF8, 0x45, 0x44,
    0xFB, 0xFA, 0x47, 0x46, 0x4B, 0x4A, 0xF7, 0xF6,
    0x17, 0x16, 0xAB, 0xAA, 0xA7, 0xA6, 0x1B, 0x1A,
    0x28, 0x29, 0x94, 0x95, 0x98, 0x99, 0x24, 0x25,
    0xC4, 0xC5, 0x78, 0x79, 0x74, 0x75, 0xC8, 0xC9,
};

// 256-entry lookup table for AES -> Tower field conversion
constexpr uint8_t kAesToTowerLookupTable[256] = {
    0x00, 0x01, 0x3C, 0x3D, 0x8C, 0x8D, 0xB0, 0xB1,
    0x8A, 0x8B, 0xB6, 0xB7, 0x06, 0x07, 0x3A, 0x3B,
    0x59, 0x58, 0x65, 0x64, 0xD5, 0xD4, 0xE9, 0xE8,
    0xD3, 0xD2, 0xEF, 0xEE, 0x5F, 0x5E, 0x63, 0x62,
    0x7A, 0x7B, 0x46, 0x47, 0xF6, 0xF7, 0xCA, 0xCB,
    0xF0, 0xF1, 0xCC, 0xCD, 0x7C, 0x7D, 0x40, 0x41,
    0x23, 0x22, 0x1F, 0x1E, 0xAF, 0xAE, 0x93, 0x92,
    0xA9, 0xA8, 0x95, 0x94, 0x25, 0x24, 0x19, 0x18,
    0x55, 0x54, 0x69, 0x68, 0xD9, 0xD8, 0xE5, 0xE4,
    0xDF, 0xDE, 0xE3, 0xE2, 0x53, 0x52, 0x6F, 0x6E,
    0x0C, 0x0D, 0x30, 0x31, 0x80, 0x81, 0xBC, 0xBD,
    0x86, 0x87, 0xBA, 0xBB, 0x0A, 0x0B, 0x36, 0x37,
    0x2F, 0x2E, 0x13, 0x12, 0xA3, 0xA2, 0x9F, 0x9E,
    0xA5, 0xA4, 0x99, 0x98, 0x29, 0x28, 0x15, 0x14,
    0x76, 0x77, 0x4A, 0x4B, 0xFA, 0xFB, 0xC6, 0xC7,
    0xFC, 0xFD, 0xC0, 0xC1, 0x70, 0x71, 0x4C, 0x4D,
    0x26, 0x27, 0x1A, 0x1B, 0xAA, 0xAB, 0x96, 0x97,
    0xAC, 0xAD, 0x90, 0x91, 0x20, 0x21, 0x1C, 0x1D,
    0x7F, 0x7E, 0x43, 0x42, 0xF3, 0xF2, 0xCF, 0xCE,
    0xF5, 0xF4, 0xC9, 0xC8, 0x79, 0x78, 0x45, 0x44,
    0x5C, 0x5D, 0x60, 0x61, 0xD0, 0xD1, 0xEC, 0xED,
    0xD6, 0xD7, 0xEA, 0xEB, 0x5A, 0x5B, 0x66, 0x67,
    0x05, 0x04, 0x39, 0x38, 0x89, 0x88, 0xB5, 0xB4,
    0x8F, 0x8E, 0xB3, 0xB2, 0x03, 0x02, 0x3F, 0x3E,
    0x73, 0x72, 0x4F, 0x4E, 0xFF, 0xFE, 0xC3, 0xC2,
    0xF9, 0xF8, 0xC5, 0xC4, 0x75, 0x74, 0x49, 0x48,
    0x2A, 0x2B, 0x16, 0x17, 0xA6, 0xA7, 0x9A, 0x9B,
    0xA0, 0xA1, 0x9C, 0x9D, 0x2C, 0x2D, 0x10, 0x11,
    0x09, 0x08, 0x35, 0x34, 0x85, 0x84, 0xB9, 0xB8,
    0x83, 0x82, 0xBF, 0xBE, 0x0F, 0x0E, 0x33, 0x32,
    0x50, 0x51, 0x6C, 0x6D, 0xDC, 0xDD, 0xE0, 0xE1,
    0xDA, 0xDB, 0xE6, 0xE7, 0x56, 0x57, 0x6A, 0x6B,
};
// clang-format on

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

// Pack 8-byte matrix into 64-bit value for vgf2p8affineqb instruction.
// The matrix is stored in column-major order for the affine instruction.
inline uint64_t packMatrixForAffine(const uint8_t matrix[8]) {
  uint64_t result = 0;
  for (int i = 0; i < 8; ++i) {
    result |= static_cast<uint64_t>(matrix[i]) << (i * 8);
  }
  return result;
}

// Get the packed Tower->AES transformation matrix for GFNI.
inline uint64_t getTowerToAesMatrixPacked() {
  return packMatrixForAffine(kTowerToAesMatrix);
}

// Get the packed AES->Tower transformation matrix for GFNI.
inline uint64_t getAesToTowerMatrixPacked() {
  return packMatrixForAffine(kAesToTowerMatrix);
}

//===----------------------------------------------------------------------===//
// Tower Polynomial Reduction Constants
//===----------------------------------------------------------------------===//
//
// Tower reduction polynomials for various tower levels:
// - Level 6 (64-bit):  x⁶⁴ + x⁴ + x³ + x + 1
// - Level 7 (128-bit): x¹²⁸ + x⁷ + x² + x + 1
//
// For reduction, we compute: high * (low_terms) where low_terms excludes xⁿ.
// E.g., for x¹²⁸ + x⁷ + x² + x + 1:
//   high * (x⁷ + x² + x + 1) = (high << 7) ^ (high << 2) ^ (high << 1) ^ high

// Reduction terms for tower level 6: x⁶⁴ + x⁴ + x³ + x + 1
// Returns shifts needed: {4, 3, 1, 0} (0 means just the value itself)
constexpr unsigned kTowerLevel6ReductionShifts[] = {4, 3, 1, 0};
constexpr unsigned kTowerLevel6ReductionShiftCount = 4;
constexpr unsigned kTowerLevel6OverflowShifts[] = {60, 61, 63}; // 64 - shift
constexpr unsigned kTowerLevel6OverflowShiftCount = 3;

// Reduction terms for tower level 7: x¹²⁸ + x⁷ + x² + x + 1
// Returns shifts needed: {7, 2, 1, 0}
constexpr unsigned kTowerLevel7ReductionShifts[] = {7, 2, 1, 0};
constexpr unsigned kTowerLevel7ReductionShiftCount = 4;
constexpr unsigned kTowerLevel7OverflowShifts[] = {57, 62, 63}; // 64 - shift
constexpr unsigned kTowerLevel7OverflowShiftCount = 3;

//===----------------------------------------------------------------------===//
// Polyval Field Constants
//===----------------------------------------------------------------------===//
//
// Polyval uses the irreducible polynomial: X¹²⁸ + X¹²⁷ + X¹²⁶ + X¹²¹ + 1
// This polynomial is specifically designed for efficient Montgomery reduction
// using PMULL/CLMUL instructions (only 2 polynomial multiplies needed).
//
// Reference: RFC 8452 (AES-GCM-SIV), binius project
//
// Key constants for Montgomery reduction on ARM:
// - Poly bits: 127, 126, 121 (high), 63, 62, 57 (low = mirrored at 64-bit
// boundary)
// - Combined as u128: (1 << 127) | (1 << 126) | (1 << 121) |
//                     (1 << 63)  | (1 << 62)  | (1 << 57)
// - As two u64: 0xE200000000000000 for both halves

// Polyval polynomial constant for Montgomery reduction (as u64 pair)
// Low and high 64-bits have the same pattern
constexpr uint64_t kPolyvalReductionConstant = 0xE200000000000000ULL;

// Polyval polynomial shift positions for the reduction
// High 64-bit: bits 63, 62, 57 (which become 127, 126, 121 in 128-bit)
// Low 64-bit: bits 63, 62, 57
constexpr unsigned kPolyvalHighBitShifts[] = {63, 62, 57};
constexpr unsigned kPolyvalHighBitShiftCount = 3;

//===----------------------------------------------------------------------===//
// Tower <-> Polyval Isomorphism Transformation
//===----------------------------------------------------------------------===//
//
// The Tower field (BinaryField128b) and Polyval field (BinaryField128bPolyval)
// are both GF(2^128) but use different irreducible polynomials:
// - Tower:   X¹²⁸ + X⁷ + X² + X + 1 (recursive tower construction)
// - Polyval: X¹²⁸ + X¹²⁷ + X¹²⁶ + X¹²¹ + 1 (sparse for fast reduction)
//
// These fields are isomorphic, and we can convert between them using
// 128x128 binary transformation matrices. Each transformation is applied by:
//   output = XOR of (basis[i] for each bit i set in input)
//
// Source: binius/crates/field/src/polyval.rs
//
// Usage:
//   To convert Tower -> Polyval:
//     polyval_val = 0;
//     for (int i = 0; i < 128; i++) {
//       if (tower_val & (1ULL << i)) {
//         polyval_val ^= kTowerToPolyvalBasis[i];
//       }
//     }
//
// Note: The values below are stored as {low_64, high_64} pairs.
// Values are in Montgomery form for Polyval field operations.

struct U128 {
  uint64_t lo;
  uint64_t hi;
};

// clang-format off
// Tower -> Polyval transformation basis (128 x 128-bit vectors)
// BINARY_TO_POLYVAL_TRANSFORMATION from binius
constexpr U128 kTowerToPolyvalBasis[128] = {
    {0x0000000000000001, 0xc200000000000000}, // [0]
    {0x3eb19c5f1a06b528, 0x21a09a4bf26aadcd}, // [1]
    {0x852cef0e61d7353d, 0xe62f1a804db43b94}, // [2]
    {0xa378ea68e992a5b6, 0xadcde131ca862a6b}, // [3]
    {0x72e9bdc82ec4fe6c, 0x5474611d07bdcd1f}, // [4]
    {0xaa3532aa6258c986, 0xf9a472d4a4965f4c}, // [5]
    {0x877681ed1a50b210, 0x10bd76c920260f81}, // [6]
    {0x6ef84934fdd225f2, 0xe7f3264523858ca3}, // [7]
    {0xedb8ddceb7f825d6, 0x586704bda927015f}, // [8]
    {0xb65f8aaec9cef096, 0x552dab8acfd831ae}, // [9]
    {0xde6792e475892fb3, 0xeccdac666a363def}, // [10]
    {0xe4a8327e33d95aa2, 0x4a621d01701247f6}, // [11]
    {0x9a11840f87149e2d, 0x8ed5002fed1f4b9a}, // [12]
    {0x302467db5a791e09, 0x3c65abbd41c759f0}, // [13]
    {0xaa643692e93caaab, 0xc2df68a5949a96b3}, // [14]
    {0x17daf9822eb57383, 0x4455027df88c1651}, // [15]
    {0x6dd1e116d55455fb, 0xc50e3a207f91d7cd}, // [16]
    {0x5fd08543d8caf5a2, 0xc89c3920b9b24b75}, // [17]
    {0xec180360b6548830, 0xfa583eb935de76a2}, // [18]
    {0x7800a5cd03690171, 0xc4d3d3b9938f3af7}, // [19]
    {0xbec91c0836143b44, 0xe1faff3b895be1e2}, // [20]
    {0x1c83552eeb1cd844, 0x256bd50f868b82cf}, // [21]
    {0x595cab38e9b59d79, 0x82fd35d590073ae9}, // [22]
    {0x2304a2533cdce9e6, 0x08dadd230bc90e19}, // [23]
    {0x502abeff6cead84c, 0xf4400f37acedc7d9}, // [24]
    {0x8cc88b7384deedfb, 0x5438d34e2b5b9032}, // [25]
    {0x447cd7d1d4a0385d, 0x7d798db71ef80a3e}, // [26]
    {0x8012303dc09cbf35, 0xa50d5ef4e33979db}, // [27]
    {0x0bb337efbc5b8115, 0x91c4b5e29de5759e}, // [28]
    {0x848f461ed0a4b110, 0xbbb0d4aaba0fab72}, // [29]
    {0x11cc078904076865, 0x3c9de86b9a306d6d}, // [30]
    {0x99db6d689ca1b370, 0xb5f43a166aa1f15f}, // [31]
    {0x243ecbd46378e59e, 0xa26153cb8c150af8}, // [32]
    {0xa876f81fe0c950ab, 0xccaa154bab1dd7ac}, // [33]
    {0x761a6139cdb07755, 0x4185b7e3ee1dddbc}, // [34]
    {0x653ed207337325f2, 0x2c9f95285b7aa574}, // [35]
    {0x42195c4c82d54dbb, 0xc8ba616ab131bfd2}, // [36]
    {0xaa36a28da1ab1c24, 0x2a9b07221a34865f}, // [37]
    {0x8b92900e0196dd39, 0x7e6e572804b548a8}, // [38]
    {0x9882a0015debd575, 0x4e9060deff44c9ef}, // [39]
    {0xc7ac9a5b424e1c65, 0x00a3a4d8c163c95a}, // [40]
    {0xf8f5eecba6033679, 0xf67c7eb5dde73d96}, // [41]
    {0x9b536094ba539fde, 0x54d78d187bbb57d1}, // [42]
    {0x033139975ab7f264, 0x76c553699edc5d4a}, // [43]
    {0xf3e41bbf5c6be650, 0x74ae8da43b2f587d}, // [44]
    {0xcd850aa6098e5fd2, 0x8a2941b59774c41a}, // [45]
    {0x0058165a063de84c, 0x9ddf65660a6f8f3c}, // [46]
    {0x1ff02ef96ee64cf3, 0xbb52da733635cc3d}, // [47]
    {0x7b7ed18bebf1c668, 0x564032a0d5d3773b}, // [48]
    {0x00222054ff0040ef, 0xef5c765e64b24b1b}, // [49]
    {0x3d484726e6249bee, 0xade661c18acba623}, // [50]
    {0xa29f2ef849c2d170, 0x9939ba35c969cdee}, // [51]
    {0xeb42d05b80174ce2, 0x2b100b39761d4f23}, // [52]
    {0xc765bd6229125d6c, 0xfbc25b179830f9ee}, // [53]
    {0x698e30184ab93141, 0xb58e089ebe7ad0b2}, // [54]
    {0xd12025afa876234c, 0x53874933a148be94}, // [55]
    {0x880f1d81fa580ffb, 0x41bbc7902188f4e9}, // [56]
    {0x25da1fe777b2dcbb, 0xea4199916a5d127d}, // [57]
    {0x7d9359ee0de0c287, 0xe7bc816547efbe98}, // [58]
    {0x5892155a7addd9da, 0x02e0f1f67e713983}, // [59]
    {0xe74955ca950af235, 0xdc6beb6eade9f875}, // [60]
    {0x6453a78d8f103230, 0x786d616edeadfa35}, // [61]
    {0xc8034da936737487, 0xe84e70191accadda}, // [62]
    {0x5363edfddd37fb3c, 0x012b8669ff3f451e}, // [63]
    {0x7833c194b9c943a0, 0x756209f0893e9687}, // [64]
    {0x9f63bd1e0d1439ac, 0xb2ac9efc9a189136}, // [65]
    {0xf650cc3994c3d2d8, 0x4de88e9a5bbb4c3d}, // [66]
    {0x9849e7c85e426b54, 0x8de7b5c85c07f335}, // [67]
    {0xf184e6761cf226d4, 0xcadd54ae6a7e72a4}, // [68]
    {0x55b5f3952f81bc30, 0xcdb182fb8d95496f}, // [69]
    {0x3a05bb2aca01a02e, 0x40013bc3c8172275}, // [70]
    {0x3e97351591adf18a, 0x704e7ce55e903388}, // [71]
    {0x988c3f36567d26f4, 0xf330cd9a74a5e884}, // [72]
    {0xc3bdf09d78cbde50, 0x18f4535304c0d74a}, // [73]
    {0x8885b838405c7e7e, 0xfe739c97fc26bed2}, // [74]
    {0xf980c3d74b3ec345, 0x492479260f2dcd8a}, // [75]
    {0x4ea2f744396691af, 0x96b6440a34de0aad}, // [76]
    {0x960a59aa564a7a26, 0x98355d1b4f7cfb03}, // [77]
    {0x8b1886b12ca37d64, 0x2703fda0532095ca}, // [78]
    {0x468c3c120f142822, 0x59c9dabe49bebf6b}, // [79]
    {0x1b14381a592e6cdd, 0xf8f3c35c671bac84}, // [80]
    {0xd80d2e9324894861, 0xd7b888791bd83b13}, // [81]
    {0x5aab9658137fa73f, 0x113ab0405354dd1c}, // [82]
    {0x461f797121b28ce6, 0xae56192d5e9c309e}, // [83]
    {0x811a6dac6b997783, 0xb7927ec7a84c2e04}, // [84]
    {0xba9b4189ce751cb4, 0x9e2f8d67fc600703}, // [85]
    {0xc8fc29729eb723ca, 0x574e95df2d8bb9e2}, // [86]
    {0xd9fa20f9a5088f26, 0x38bc6fc47739c06c}, // [87]
    {0xb3c38d8f95ce7a5f, 0x69d3b9b1d9483174}, // [88]
    {0x90e27e882f18640d, 0xd6e4bb147cc82b6e}, // [89]
    {0x85cd9fece12f7adc, 0x027338db641804d9}, // [90]
    {0xe76f523928c4364e, 0x523cb73968169ccc}, // [91]
    {0x8a11b0dcc941f2f6, 0xcdcf898117f92720}, // [92]
    {0x7f7892fec7a5b217, 0xc908287814c8cba6}, // [93]
    {0x104968d4cbbb285a, 0x92b99988bb26215d}, // [94]
    {0x4b95692534ef5068, 0x4dbca8fd835d00ea}, // [95]
    {0x167a2b851f32fd9c, 0xcd8b92c8a6e0e65e}, // [96]
    {0xc1e2d544628e7845, 0xc3473dfda9f97d6a}, // [97]
    {0xe0dc39a240365722, 0x0260e7badc64dbfd}, // [98]
    {0x9719c80e41953868, 0x3966125b40fe2bca}, // [99]
    {0x57b709a360d4a2c7, 0xac0211506eda3cba}, // [100]
    {0x5b337fefa219c52b, 0x0e4f0e47d02fedd1}, // [101]
    {0xace675511f754ee3, 0x1d5907ccdc659f7a}, // [102]
    {0x097284863b2a5b6e, 0x4ad5b368eaddc4bb}, // [103]
    {0xcef553a4a46cde5b, 0x2eae07273b8c4fc5}, // [104]
    {0x79d4a3b5d8dd9396, 0x096a310e7b1e3a31}, // [105]
    {0x1dde08d05018a353, 0x8c81362eeb1656a9}, // [106]
    {0xecf7f057b6fdba0b, 0x387e59e44cc0d53f}, // [107]
    {0xac82d91ca97561d6, 0x9d29670bbd0e8051}, // [108]
    {0x9714e48065be74a4, 0xaf1310d0f5cac4e8}, // [109]
    {0x411d14182a36fb6b, 0x9b684a3865c2b59c}, // [110]
    {0xca22b4e848340fbe, 0x3e7de163516ffdca}, // [111]
    {0xc2f5db315d5e7fda, 0x3c37dbe331de4b0d}, // [112]
    {0xe3d5a1c40c3769a0, 0x19e7f4b53ff86990}, // [113]
    {0xcc93fdb1b14a4775, 0x56469ab32b2b82e8}, // [114]
    {0x0d8ad49d260bb71b, 0x9c01cefde4781630}, // [115]
    {0x81366fec1e4e52c0, 0x6100101b8cebde73}, // [116]
    {0x32143fa65158ee4f, 0xa28d30c3cbd8b696}, // [117]
    {0x151c45f71eee6368, 0x3db7a902ec509e58}, // [118]
    {0x7107d37d79ebbaba, 0x42d5a505e8ab7009}, // [119]
    {0xc7d6d15c84cca8ce, 0xe47b83247cb2b162}, // [120]
    {0x3e4c87ff505737a5, 0x076caf0e23541c75}, // [121]
    {0x3980f5d1d3b84a89, 0x590a8d1cdbd17ae8}, // [122]
    {0xa53497edd34c4204, 0x77d649ff61a7cd0d}, // [123]
    {0xa4a8feed84fd3993, 0xefbe0c34eeab379e}, // [124]
    {0x51629cdde777f968, 0x90540cf7957a8a30}, // [125]
    {0x44c49c70aa92831f, 0x8749050496dd2882}, // [126]
    {0x370368d94947961a, 0x0fc80b1d600406b2}, // [127]
};

// Polyval -> Tower transformation basis (128 x 128-bit vectors)
// POLYVAL_TO_BINARY_TRANSFORMATION from binius
constexpr U128 kPolyvalToTowerBasis[128] = {
    {0xa8fc4d30a32dadcc, 0x66e1d645d7eb87dc}, // [0]
    {0xc5675d78c59c1901, 0x53ca87ba77172fd8}, // [1]
    {0xa15acb755a948567, 0x1a9cf63d31827dcd}, // [2]
    {0x474b0401a99f6c0a, 0xa8f28bdf6d29cee2}, // [3]
    {0x06b39ca9799c8d73, 0x4eefa9efe87ed19c}, // [4]
    {0x9885a6b2bc494f3e, 0x06ec578f505abf1e}, // [5]
    {0x9a96d3fb9cd3348a, 0x70ecdfe1f601f850}, // [6]
    {0xeb25f618fc3faf28, 0xcb0d16fc7f13733d}, // [7]
    {0xfcb578115fcbef3c, 0x4e9a97aa2c84139f}, // [8]
    {0x9a441bffe19219ad, 0xc6de6210afe8c6bd}, // [9]
    {0x1be5bf1e30c488d3, 0x73e3e8a7c5974860}, // [10]
    {0xb39e7f4bb37dce9c, 0x1f6d67e2e64bd6c4}, // [11]
    {0x5f5095b4c155f3b5, 0xc34135d567eada88}, // [12]
    {0x4790b8e2e37330e4, 0x23f165958d59a55e}, // [13]
    {0x05b88802add08d17, 0x4f2be978f16908e4}, // [14]
    {0x907936513c3a7d45, 0x6442b00f5bbf4009}, // [15]
    {0x5d61b9f18137026f, 0xac63f0397d911a7a}, // [16]
    {0xedf07cbc6698e144, 0x8e70543ae0e43313}, // [17]
    {0xaa5a07984066d026, 0xcb417a646d59f652}, // [18]
    {0x35bd8f76de7bb84e, 0xf028de8dd6163187}, // [19]
    {0xf15b4bcaa9bf186c, 0x2e03a12472d21599}, // [20]
    {0xa27d8e48d1b9ca76, 0x54a376cc03e5b2cf}, // [21]
    {0x201b87da07cb58ae, 0xd22894c253031b1b}, // [22]
    {0xf77d902dd5d2a563, 0x6bc1416afea6308f}, // [23]
    {0x50055f8ac3095121, 0x9958ecd28adbebf8}, // [24]
    {0xe6bb6f54c227fb91, 0x595a1b37062233d7}, // [25]
    {0xf671558ee315d809, 0x41ffcfcdda4583c4}, // [26]
    {0x63e982ec4b3e6ea2, 0x780c2490f3e5cb47}, // [27]
    {0x722a6b9037b6db34, 0xf7a450b35931fa76}, // [28]
    {0x28592772430ad07e, 0xe21991100e848213}, // [29]
    {0x60c65ec87d6f9277, 0x360d4079f62863cc}, // [30]
    {0xaca590e7a60dbe92, 0xd898bfa0b076cc4e}, // [31]
    {0x2e1647fc34b549bf, 0xcaacddd5e114fe5c}, // [32]
    {0x617776ddb2d3f888, 0x3042e34911c28e90}, // [33]
    {0xcfd8455b13cb9b14, 0x3728a3b0da53cdfe}, // [34]
    {0xa7c643bffbddc6b2, 0x2f2eb3d5bc7b2c48}, // [35]
    {0x501b04302706b908, 0x3b71a5c04010c0aa}, // [36]
    {0x9be54df766e48c51, 0x0701845b090e79bb}, // [37]
    {0xdb06fcfff7408f78, 0x1e9eac7bf45b14c8}, // [38]
    {0x0eb3bef69eee8b0b, 0x6b1b8e39a339423d}, // [39]
    {0x5d3a99cff1edcf0a, 0x8b06616385967df9}, // [40]
    {0x58e1dd1a51fe6a30, 0x5d921137890a3ded}, // [41]
    {0x628b705d38121acc, 0x828ed6fba42805b2}, // [42]
    {0xf70ecb6116cabd81, 0x9b7a95220e9d5b0f}, // [43]
    {0x047f136cab751c88, 0x0eb9055cb11711ed}, // [44]
    {0xca451290f7d5c78a, 0xd6f590777c17a6d0}, // [45]
    {0x91f910cb0893e71f, 0x401a922a6461fbe6}, // [46]
    {0xc927ebad9ed253f7, 0x15a549308bc53902}, // [47]
    {0x0f340a43f11a1b84, 0x45dccafc72a58448}, // [48]
    {0xe6d3e20451335d5b, 0x19d2a2c057d60656}, // [49]
    {0x99197c8b9a811454, 0x035af143a5827a0f}, // [50]
    {0x2191fd0e013f163a, 0x7ee35d174ad7cc69}, // [51]
    {0x9599fac8831effa9, 0xc4c0401d841f965c}, // [52]
    {0x4acfca3fc5630691, 0x63e809a843fc04f8}, // [53]
    {0x9fb7d78e2d6643c4, 0xdb2f3301594e3de4}, // [54]
    {0x3d709319cc130a7c, 0x1b31772535984ef9}, // [55]
    {0x918071b62a0593f3, 0x036dc9c884cd6d6c}, // [56]
    {0x132360b078027103, 0x4700cd0e81c88045}, // [57]
    {0xb0350e17ed2d625d, 0xdfa3f35eb236ea63}, // [58]
    {0xc28be91822978e15, 0xf0fd7c7760099f1a}, // [59]
    {0x5034e9eed1f21205, 0x852a1eba3ad160e9}, // [60]
    {0xca9efee1701763c3, 0x4a07dd461892df45}, // [61]
    {0x85fd61b42f707384, 0xadbbaa0add4c82fe}, // [62]
    {0xc231db13f0e15600, 0x5c63d0673f33c0f2}, // [63]
    {0x26e0e794dd4b3076, 0x24ddc15165011356}, // [64]
    {0x38afd02d201fb05b, 0xb60c601bbf72924e}, // [65]
    {0x84334bcf70649aeb, 0x2ef68918f416caca}, // [66]
    {0xd815534c707343f2, 0x0b72a3124c504bca}, // [67]
    {0x5d396f8523d80fe0, 0xcfd8b2076040c43d}, // [68]
    {0x504192bb27cc65e1, 0x098d9daf64154a63}, // [69]
    {0x283621f8fb6a6704, 0x3ae44070642e6720}, // [70]
    {0x6bfe2b373f47fd05, 0x19cd9b2843d0ff93}, // [71]
    {0xdb10450431d26122, 0x451e2e4159c78e65}, // [72]
    {0x423b36807c70f3ae, 0x797b753e29b9d0e9}, // [73]
    {0xea30600915664e22, 0xa8d0e8ba9bb634f6}, // [74]
    {0x9c504cb944475b0a, 0xdf8c74bbd66f8680}, // [75]
    {0x7a5a94d498128018, 0x32831a457ced3a41}, // [76]
    {0x47119b9b5f00350e, 0x1aca728985936a61}, // [77]
    {0x6b66764ed05bb1db, 0x6f436d64b4ee1a55}, // [78]
    {0x15e483cb21e5a1a2, 0x25930eaed3fd9829}, // [79]
    {0x06bf1d7e151780ab, 0x21735f5eb346e560}, // [80]
    {0x805eb16d7bd5345c, 0x55fc6f607f10e17f}, // [81]
    {0x4965292af4aeb57e, 0x4b4d289591f87811}, // [82]
    {0xf67998c1883c1cf3, 0x30608bc7444bcbaf}, // [83]
    {0x657c6e6395404343, 0xa12a72abe4152e4a}, // [84]
    {0xc73f9cd68fb0e2fb, 0x7579186d4e0959de}, // [85]
    {0x965c822892b7bfda, 0xb5560ce63f7894cc}, // [86]
    {0xba63d9fd645995d7, 0x6b06d7165072861e}, // [87]
    {0xde3c8ef8f9bf4e29, 0x359f439f5ec9107d}, // [88]
    {0x105821cd8b55b06b, 0xcbfe7985c6006a46}, // [89]
    {0x1129fb9076474061, 0x2110b3b51f5397ef}, // [90]
    {0x44c33b275c388c47, 0x1928478b6f3275c9}, // [91]
    {0x437111aa4652421a, 0x23f978e6a0a54802}, // [92]
    {0x1dd32dbedd310f5b, 0xe8c526bf924dc5cd}, // [93]
    {0xf43c73d22a05c8e4, 0xa0ac29f901f79ed5}, // [94]
    {0x47f4635b747145ea, 0x55e0871c6e97408f}, // [95]
    {0x7d3c2dfefd1ebcb3, 0x6c2114c3381f5366}, // [96]
    {0x863c3aceaaa3eef7, 0x42d23c18722fbd58}, // [97]
    {0x3838f8408a72fdf1, 0xbb0821ab38d5de13}, // [98]
    {0x31fa387773bb9153, 0x035d7239054762b1}, // [99]
    {0x9ab652e8979139e7, 0x8fa898aafe8b154f}, // [100]
    {0xc658193f16cb726c, 0x6a383e5cd4a16923}, // [101]
    {0x82022f32ae3f68b9, 0x9948caa8c6cefb01}, // [102]
    {0xdf7bac577ed73b44, 0x8d2a8decf9855bd4}, // [103]
    {0x59d548c5aa959879, 0x09c7b8300f0f9842}, // [104]
    {0xdca8b8e134047afc, 0x92e16d2d24e070ef}, // [105]
    {0xaf24877fb5031512, 0x47d8621457f4118a}, // [106]
    {0x19583a966a85667f, 0x25576941a55f0a0c}, // [107]
    {0x83fda3bc6285a8dc, 0xb113cad79cd35f2e}, // [108]
    {0xc3e6318431ffe580, 0xc76968eecb2748d0}, // [109]
    {0x39e6618395b68416, 0x7211122aa7e7f6fe}, // [110]
    {0xf450d00a45146d11, 0x88463599bf7d3e92}, // [111]
    {0x3bbb7f79a18ee123, 0x6e12b7d5adf95da3}, // [112]
    {0xeaca7e7b7280ff16, 0xe0a98ac4025bc568}, // [113]
    {0xf274057ac892ff77, 0xc13fc79f6c35048d}, // [114]
    {0xe39cae4de47eb505, 0x93c1a3145d4e47de}, // [115]
    {0xf1e5d7c53bdbd52b, 0x780064be3036df98}, // [116]
    {0x8b709172ecaff561, 0x48c467b5cec26562}, // [117]
    {0x7682094560524a7e, 0x5bbbab77ce5552ff}, // [118]
    {0xb128fec4e4a23a63, 0x551537ef6048831f}, // [119]
    {0x439317a13568b284, 0xe7ef397fcc095ead}, // [120]
    {0xf9d75d62d92c6332, 0xbc5d2927eac0a720}, // [121]
    {0xb2bc992b5b59e61e, 0x3bfeb420021f93e9}, // [122]
    {0x4af1b7307b574ed9, 0xc651dc438e2f1bc6}, // [123]
    {0x2a1ddb55413a4e43, 0xbfe0a17ee2b77754}, // [124]
    {0x7dfc01c05d732a32, 0xa062da2427df3d1a}, // [125]
    {0x93417ba0b085e1e8, 0x1e4889fd72b70ecf}, // [126]
    {0xc26a6bf2ca842f17, 0xc4f4769f4f9c2e33}, // [127]
};
// clang-format on

// Apply 128x128 binary transformation matrix to a 128-bit value.
// This is a simple reference implementation; optimized versions can use
// PMULL-based matrix multiplication for SIMD acceleration.
inline U128 applyTransformation(U128 input, const U128 basis[128]) {
  U128 result = {0, 0};
  for (int i = 0; i < 64; ++i) {
    if (input.lo & (1ULL << i)) {
      result.lo ^= basis[i].lo;
      result.hi ^= basis[i].hi;
    }
  }
  for (int i = 0; i < 64; ++i) {
    if (input.hi & (1ULL << i)) {
      result.lo ^= basis[64 + i].lo;
      result.hi ^= basis[64 + i].hi;
    }
  }
  return result;
}

// Convert Tower field element to Polyval field element.
inline U128 towerToPolyval(U128 tower) {
  return applyTransformation(tower, kTowerToPolyvalBasis);
}

// Convert Polyval field element to Tower field element.
inline U128 polyvalToTower(U128 polyval) {
  return applyTransformation(polyval, kPolyvalToTowerBasis);
}

//===----------------------------------------------------------------------===//
// Binary Field bf<8> Inverse Lookup Table
//===----------------------------------------------------------------------===//
//
// Direct inverse lookup table for GF(2^8) tower field (bf<3>).
// This provides O(1) inverse computation vs O(n) with Fermat's little theorem.
//
// For element a in GF(2^8), its inverse a^(-1) is:
//   kBinaryTower8bInverseTable[a]
//
// Note: inverse of 0 is defined as 0 for convenience (matches binius behavior).
//
// Source: binius/crates/field/src/arch/portable/pairwise_table_arithmetic.rs
//
// clang-format off
constexpr uint8_t kBinaryTower8bInverseTable[256] = {
    0x00, 0x01, 0x03, 0x02, 0x06, 0x0e, 0x04, 0x0f,
    0x0d, 0x0a, 0x09, 0x0c, 0x0b, 0x08, 0x05, 0x07,
    0x14, 0x67, 0x94, 0x7b, 0x10, 0x66, 0x9e, 0x7e,
    0xd2, 0x81, 0x27, 0x4b, 0xd1, 0x8f, 0x2f, 0x42,
    0x3c, 0xe6, 0xde, 0x7c, 0xb3, 0xc1, 0x4a, 0x1a,
    0x30, 0xe9, 0xdd, 0x79, 0xb1, 0xc6, 0x43, 0x1e,
    0x28, 0xe8, 0x9d, 0xb9, 0x63, 0x39, 0x8d, 0xc2,
    0x62, 0x35, 0x83, 0xc5, 0x20, 0xe7, 0x97, 0xbb,
    0x61, 0x48, 0x1f, 0x2e, 0xac, 0xc8, 0xbc, 0x56,
    0x41, 0x60, 0x26, 0x1b, 0xcf, 0xaa, 0x5b, 0xbe,
    0xef, 0x73, 0x6d, 0x5e, 0xf7, 0x86, 0x47, 0xbd,
    0x88, 0xfc, 0xbf, 0x4e, 0x76, 0xe0, 0x53, 0x6c,
    0x49, 0x40, 0x38, 0x34, 0xe4, 0xeb, 0x15, 0x11,
    0x8b, 0x85, 0xaf, 0xa9, 0x5f, 0x52, 0x98, 0x92,
    0xfb, 0xb5, 0xee, 0x51, 0xb7, 0xf0, 0x5c, 0xe1,
    0xdc, 0x2b, 0x95, 0x13, 0x23, 0xdf, 0x17, 0x9f,
    0xd3, 0x19, 0xc4, 0x3a, 0x8a, 0x69, 0x55, 0xf6,
    0x58, 0xfd, 0x84, 0x68, 0xc3, 0x36, 0xd0, 0x1d,
    0xa6, 0xf3, 0x6f, 0x99, 0x12, 0x7a, 0xba, 0x3e,
    0x6e, 0x93, 0xa0, 0xf8, 0xb8, 0x32, 0x16, 0x7f,
    0x9a, 0xf9, 0xe2, 0xdb, 0xed, 0xd8, 0x90, 0xf2,
    0xae, 0x6b, 0x4d, 0xce, 0x44, 0xc9, 0xa8, 0x6a,
    0xc7, 0x2c, 0xc0, 0x24, 0xfa, 0x71, 0xf1, 0x74,
    0x9c, 0x33, 0x96, 0x3f, 0x46, 0x57, 0x4f, 0x5a,
    0xb2, 0x25, 0x37, 0x8c, 0x82, 0x3b, 0x2d, 0xb0,
    0x45, 0xad, 0xd7, 0xff, 0xf4, 0xd4, 0xab, 0x4c,
    0x8e, 0x1c, 0x18, 0x80, 0xcd, 0xf5, 0xfe, 0xca,
    0xa5, 0xec, 0xe3, 0xa3, 0x78, 0x2a, 0x22, 0x7d,
    0x5d, 0x77, 0xa2, 0xda, 0x64, 0xea, 0x21, 0x3d,
    0x31, 0x29, 0xe5, 0x65, 0xd9, 0xa4, 0x72, 0x50,
    0x75, 0xb6, 0xa7, 0x91, 0xcc, 0xd5, 0x87, 0x54,
    0x9b, 0xa1, 0xb4, 0x70, 0x59, 0x89, 0xd6, 0xcb,
};
// clang-format on

} // namespace mlir::prime_ir::field

// NOLINTNEXTLINE(whitespace/line_length)
#endif // PRIME_IR_DIALECT_FIELD_CONVERSIONS_BINARYFIELDTOARITH_BINARYFIELDTABLES_H_
