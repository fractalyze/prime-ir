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

// Shaped `field.mul` cases that must NOT specialize to clmad. Static shaped
// mul — flat and tower alike — IS specialized element-wise (see
// specialize_binary_field_to_nvptx_shaped_clmad.mlir); lane-by-lane
// unrolling needs a static element count, so only dynamic shapes stay
// portable, and those are pinned here.
//
// Kept out of specialize_binary_field_to_nvptx.mlir because a dynamic shaped
// mul does not lower cleanly through a full binary-field-to-arith pipeline
// RUN — the same shaped-flat gap the verifier path documents
// (aes_shaped_unsupported.mlir / ghash_shaped_unsupported.mlir).

// RUN: prime-ir-opt --specialize-binary-field-to-nvptx %s | FileCheck %s

!BF32 = !field.bf<5>
!AES = !field.bf<3, aes>
!F32 = !field.bf<5, flat>

// A dynamic-shape tower mul cannot be unrolled lane by lane either.
// CHECK-LABEL: @dynamic_bf32_mul
// CHECK: field.mul
// CHECK-NOT: clmad
func.func @dynamic_bf32_mul(%a: tensor<?x!BF32>, %b: tensor<?x!BF32>) -> tensor<?x!BF32> {
  %c = field.mul %a, %b : tensor<?x!BF32>
  return %c : tensor<?x!BF32>
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

// Same for the generic flat basis at another width.
// CHECK-LABEL: @dynamic_f32_mul
// CHECK: field.mul
// CHECK-NOT: clmad
func.func @dynamic_f32_mul(%a: tensor<?x!F32>, %b: tensor<?x!F32>) -> tensor<?x!F32> {
  %c = field.mul %a, %b : tensor<?x!F32>
  return %c : tensor<?x!F32>
}
