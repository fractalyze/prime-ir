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

// field.inverse is deliberately NOT ElementwiseMappable, unlike every other
// field arithmetic op. That exclusion is load-bearing: it keeps a shaped
// inverse intact as a single op so FieldToModArith can lower it to
// Montgomery's batch inversion -- ONE inversion plus ~3(N-1) multiplies for N
// elements, instead of N independent inversions.
//
// Adding ElementwiseMappable to it (tempting, since it would let
// convert-elementwise-to-linalg scalarize a shaped inverse for the
// scalar-only binary-field lowering) silently costs orders of magnitude on
// the prime-field path: the result stays correct and every runner test still
// passes, so only a structural test catches it. This is that test.

// Nothing scalarizes a shaped inverse.
// RUN: prime-ir-opt --convert-elementwise-to-linalg %s | FileCheck %s --check-prefix=NOTELEMENTWISE
// And it therefore reaches FieldToModArith whole, as one batch inversion.
// RUN: prime-ir-opt --field-to-mod-arith %s | FileCheck %s --check-prefix=BATCH

!PF = !field.pf<2013265921:i32>

// NOTELEMENTWISE-LABEL: @shaped_inverse
// NOTELEMENTWISE-NOT: linalg.generic
// NOTELEMENTWISE: field.inverse

// 64 elements, ONE inversion.
// BATCH-LABEL: @shaped_inverse
// BATCH-COUNT-1: mod_arith.inverse
// BATCH-NOT: mod_arith.inverse
func.func @shaped_inverse(%a: tensor<64x!PF>) -> tensor<64x!PF> {
  %c = field.inverse %a : tensor<64x!PF>
  return %c : tensor<64x!PF>
}
