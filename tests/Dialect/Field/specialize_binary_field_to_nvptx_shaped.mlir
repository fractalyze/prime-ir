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

// Shaped `field.mul` cases that must NOT specialize to clmad. Shaped *flat*
// (ghash/aes) mul IS specialized element-wise — see
// specialize_binary_field_to_nvptx_shaped_clmad.mlir — but two shaped cases
// still stay portable and are pinned here:
//
//  * Shaped *tower* mul: the flat-basis conversion's trunci/shrui bit-matrix
//    ladder is scalar-only, so a shaped match would build invalid IR. This
//    also pins the tower pattern's scalar `dyn_cast` guard against the
//    getElementTypeOrSelf refactor that generalized only the flat patterns.
//  * Dynamic-shape flat mul: lane-by-lane unrolling needs a static element
//    count, so a dynamic tensor falls through to the portable software path.
//
// Kept out of specialize_binary_field_to_nvptx.mlir because these do not lower
// cleanly through a full binary-field-to-arith pipeline RUN — shaped tower mul
// leaves scalar materializations against the tensor type (a pre-existing gap
// in that pass), and dynamic flat mul is the same shaped-flat gap the verifier
// path documents (aes_shaped_unsupported.mlir / ghash_shaped_unsupported.mlir).

// RUN: prime-ir-opt --specialize-binary-field-to-nvptx %s | FileCheck %s

!BF32 = !field.bf<5>
!AES = !field.bf<3, aes>

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

// A dynamic-shape flat (aes) mul cannot be unrolled lane by lane, so it must
// stay portable rather than emit clmad.
// CHECK-LABEL: @dynamic_aes_mul
// CHECK: field.mul
// CHECK-NOT: clmad
func.func @dynamic_aes_mul(%a: tensor<?x!AES>, %b: tensor<?x!AES>) -> tensor<?x!AES> {
  %c = field.mul %a, %b : tensor<?x!AES>
  return %c : tensor<?x!AES>
}
