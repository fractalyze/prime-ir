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

// Binary field benchmark - ARM Polyval specialization
//
// Uses Polyval polynomial (X¹²⁸ + X¹²⁷ + X¹²⁶ + X¹²¹ + 1) with Montgomery reduction
// for faster bf<128> multiplication compared to Tower polynomial reduction.

!BF128 = !field.bf<7>  // GF(2¹²⁸) tower field

// BF128 multiplication using Polyval Montgomery reduction
func.func @bf128_mul_arm_polyval(%a: i128, %b: i128) -> i128 attributes { llvm.emit_c_interface } {
  %a_bf = builtin.unrealized_conversion_cast %a : i128 to !BF128
  %b_bf = builtin.unrealized_conversion_cast %b : i128 to !BF128
  %c_bf = field.mul %a_bf, %b_bf : !BF128
  %c = builtin.unrealized_conversion_cast %c_bf : !BF128 to i128
  return %c : i128
}
