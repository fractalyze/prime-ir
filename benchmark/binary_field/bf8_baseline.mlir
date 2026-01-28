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

// Binary field benchmark - BF8 Baseline (generic Karatsuba implementation)

!BF8 = !field.bf<3>  // GF(2⁸) tower field

// BF8 scalar multiplication - After lowering: func(i8, i8) -> i8
func.func @bf8_mul_baseline(%a: !BF8, %b: !BF8) -> !BF8 attributes { llvm.emit_c_interface } {
  %c = field.mul %a, %b : !BF8
  return %c : !BF8
}
