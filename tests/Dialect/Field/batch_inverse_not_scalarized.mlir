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

// field.inverse is the one field arithmetic op deliberately left off
// ElementwiseMappable. That exclusion is load-bearing: it keeps a shaped
// inverse intact as a single op so FieldToModArith can lower it to Montgomery's
// batch inversion -- ONE inversion plus ~3(N-1) multiplies for N elements,
// instead of N independent inversions.
//
// Adding ElementwiseMappable to it (tempting, since it would let
// convert-elementwise-to-linalg scalarize a shaped inverse for the scalar-only
// binary-field lowering) silently costs orders of magnitude on the prime-field
// path: the result stays correct and every runner test still passes, so only a
// structural test catches it. This is that test.
//
// It pins BOTH field kinds, because the pass ordering in buildFieldToLLVM
// rests on the whole rule, not just the prime-field half.

// Nothing scalarizes a shaped inverse, of either field kind.
// RUN: prime-ir-opt --convert-elementwise-to-linalg --split-input-file %s \
// RUN:   | FileCheck %s --check-prefix=NOTELEMENTWISE
// A shaped prime-field inverse therefore reaches FieldToModArith whole and
// becomes one batch inversion; a binary-field one is not its business.
// RUN: prime-ir-opt --field-to-mod-arith --split-input-file %s \
// RUN:   | FileCheck %s --check-prefix=BATCH
// End to end: the prime-field case lowers, the binary-field case does not.
// RUN: prime-ir-opt --field-to-llvm --split-input-file \
// RUN:   --verify-diagnostics=only-expected %s

!PF = !field.pf<2013265921:i32>

// NOTELEMENTWISE-LABEL: @shaped_prime_inverse
// NOTELEMENTWISE-NOT: linalg.generic
// NOTELEMENTWISE: field.inverse

// 64 elements, ONE inversion.
// BATCH-LABEL: @shaped_prime_inverse
// BATCH-COUNT-1: mod_arith.inverse
// BATCH-NOT: mod_arith.inverse
func.func @shaped_prime_inverse(%a: tensor<64x!PF>) -> tensor<64x!PF> {
  %c = field.inverse %a : tensor<64x!PF>
  return %c : tensor<64x!PF>
}

// -----

// The binary-field half of the same rule. Binary fields have no batch
// inversion yet: FieldToModArith bails on BinaryFieldType, and
// BinaryFieldToArith's emitters are scalar-only, so a shaped binary-field
// inverse reaches that pass whole and cannot be legalized.
//
// This pins TODAY'S behaviour, not desired behaviour. Montgomery's trick is
// purely algebraic and applies to binary fields too -- measured, it wins
// ~11.7x for `ghash` and ~24x for `bf<6, flat>`, and loses for the `bf<7>`
// tower whose norm-descent inverse is only ~1.7x a multiply. Implementing it
// with a basis-dependent default is fractalyze/prime-ir#453; when that lands,
// this case starts lowering and the expected-error below must be replaced by
// a check for the batch form rather than deleted.

!BF128 = !field.bf<7>

// NOTELEMENTWISE-LABEL: @shaped_binary_inverse
// NOTELEMENTWISE-NOT: linalg.generic
// NOTELEMENTWISE: field.inverse

// FieldToModArith leaves binary fields alone, so the op survives it unchanged.
// BATCH-LABEL: @shaped_binary_inverse
// BATCH: field.inverse
func.func @shaped_binary_inverse(%a: tensor<4x!BF128>) -> tensor<4x!BF128> {
  %c = field.inverse %a : tensor<4x!BF128>
  // expected-error @+1 {{failed to legalize unresolved materialization}}
  return %c : tensor<4x!BF128>
}
