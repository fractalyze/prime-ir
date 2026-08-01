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

// Shaped tower multiplies must NOT match the clmad specialization — the
// emitted trunci/shrui ladder is scalar-only, so a shaped match would build
// invalid IR. These pin the pattern's scalar dyn_cast guard against a
// getElementTypeOrSelf refactor. Kept out of
// specialize_binary_field_to_nvptx.mlir because its pipeline RUN also lowers
// through binary-field-to-arith, where shaped *tower* mul does not legalize
// yet (scalar materializations against the tensor type) — a pre-existing gap
// in that pass, unlike shaped ghash which the verifier rejects outright
// (ghash_shaped_unsupported.mlir).

// RUN: prime-ir-opt --specialize-binary-field-to-nvptx %s | FileCheck %s

!BF32 = !field.bf<5>

// CHECK-LABEL: @tensor_bf32_mul
// CHECK: field.mul
// CHECK-NOT: clmad
func.func @tensor_bf32_mul(%a: tensor<4x!BF32>, %b: tensor<4x!BF32>) -> tensor<4x!BF32> {
  %c = field.mul %a, %b : tensor<4x!BF32>
  return %c : tensor<4x!BF32>
}

// CHECK-LABEL: @vector_bf32_mul
// CHECK: field.mul
// CHECK-NOT: clmad
func.func @vector_bf32_mul(%a: vector<4x!BF32>, %b: vector<4x!BF32>) -> vector<4x!BF32> {
  %c = field.mul %a, %b : vector<4x!BF32>
  return %c : vector<4x!BF32>
}
