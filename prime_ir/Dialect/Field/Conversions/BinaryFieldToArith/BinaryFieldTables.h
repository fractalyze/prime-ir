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

} // namespace mlir::prime_ir::field

// NOLINTNEXTLINE(whitespace/line_length)
#endif // PRIME_IR_DIALECT_FIELD_CONVERSIONS_BINARYFIELDTOARITH_BINARYFIELDTABLES_H_
