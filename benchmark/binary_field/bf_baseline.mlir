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

// Binary field benchmark - Baseline (generic Karatsuba implementation)

!BF64 = !field.bf<6>  // GF(2⁶⁴) tower field
!BF128 = !field.bf<7>  // GF(2¹²⁸) tower field

// BF64 multiplication - After lowering: func(i64, i64) -> i64
func.func @bf64_mul_baseline(%a: !BF64, %b: !BF64) -> !BF64 attributes { llvm.emit_c_interface } {
  %c = field.mul %a, %b : !BF64
  return %c : !BF64
}

// BF128 multiplication - After lowering: func(i128, i128) -> i128
func.func @bf128_mul_baseline(%a: !BF128, %b: !BF128) -> !BF128 attributes { llvm.emit_c_interface } {
  %c = field.mul %a, %b : !BF128
  return %c : !BF128
}
