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

// Binary field benchmark - ARM PMULL specialization
//
// Uses builtin.unrealized_conversion_cast to match what the specialization pass
// produces, allowing reconcile-unrealized-casts to clean up the cast chain.

!BF64 = !field.bf<6>  // GF(2⁶⁴) tower field
!BF128 = !field.bf<7>  // GF(2¹²⁸) tower field

// BF64 multiplication using PMULL
func.func @bf64_mul_arm(%a: i64, %b: i64) -> i64 attributes { llvm.emit_c_interface } {
  %a_bf = builtin.unrealized_conversion_cast %a : i64 to !BF64
  %b_bf = builtin.unrealized_conversion_cast %b : i64 to !BF64
  %c_bf = field.mul %a_bf, %b_bf : !BF64
  %c = builtin.unrealized_conversion_cast %c_bf : !BF64 to i64
  return %c : i64
}

// BF128 multiplication using PMULL
func.func @bf128_mul_arm(%a: i128, %b: i128) -> i128 attributes { llvm.emit_c_interface } {
  %a_bf = builtin.unrealized_conversion_cast %a : i128 to !BF128
  %b_bf = builtin.unrealized_conversion_cast %b : i128 to !BF128
  %c_bf = field.mul %a_bf, %b_bf : !BF128
  %c = builtin.unrealized_conversion_cast %c_bf : !BF128 to i128
  return %c : i128
}

// =============================================================================
// Square operations using PMULL
// =============================================================================

// BF64 square using PMULL
func.func @bf64_square_arm(%a: i64) -> i64 attributes { llvm.emit_c_interface } {
  %a_bf = builtin.unrealized_conversion_cast %a : i64 to !BF64
  %c_bf = field.square %a_bf : !BF64
  %c = builtin.unrealized_conversion_cast %c_bf : !BF64 to i64
  return %c : i64
}

// BF128 square using PMULL
func.func @bf128_square_arm(%a: i128) -> i128 attributes { llvm.emit_c_interface } {
  %a_bf = builtin.unrealized_conversion_cast %a : i128 to !BF128
  %c_bf = field.square %a_bf : !BF128
  %c = builtin.unrealized_conversion_cast %c_bf : !BF128 to i128
  return %c : i128
}
