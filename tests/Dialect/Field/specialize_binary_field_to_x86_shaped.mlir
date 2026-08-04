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

// Shaped (tensor/vector) PCLMULQDQ specialization for the GHASH basis
// (`bf<7, ghash>`). vpclmulqdq is scalar inline asm, so a shaped `field.mul` is
// unrolled lane by lane: each element is extracted, run through the same
// Karatsuba ladder as the scalar path, and the results are reassembled with
// `tensor.from_elements`/`vector.from_elements`. This mirrors the NVPTX clmad
// path (specialize_binary_field_to_nvptx_shaped_clmad.mlir). The packed
// `bf<3, aes>` GFNI path is genuinely vectorized (one vgf2p8mulb over the whole
// vector) and needs no unrolling, so it is not repeated here.

// RUN: prime-ir-opt --specialize-binary-field-to-x86="use-gfni=false" %s | FileCheck %s --check-prefix=CHECK-PCLMULQDQ

// use-pclmulqdq=false disables specialization: the shaped mul stays portable.
// RUN: prime-ir-opt --specialize-binary-field-to-x86="use-pclmulqdq=false" %s | FileCheck %s --check-prefix=CHECK-OFF

// The emitted extract/from_elements scaffolding and the field<->iN casts must
// all be reconciled by the downstream binary-field-to-arith lowering -- no
// field ops or unrealized casts may survive.
// RUN: prime-ir-opt --specialize-binary-field-to-x86 --binary-field-to-arith %s | FileCheck %s --check-prefix=CHECK-PIPELINE
// CHECK-PIPELINE-NOT: field.mul
// CHECK-PIPELINE-NOT: unrealized_conversion_cast

!GHASH = !field.bf<7, ghash> // GF(2¹²⁸), flat GHASH polynomial basis

// Shaped GHASH (tensor): three vpclmulqdq (ll, hh, cross) build the 128×128
// Karatsuba product per lane, so 6 across the 2 lanes, wrapped by an extract
// per operand lane and a from_elements to rebuild the tensor.
// CHECK-PCLMULQDQ-LABEL: @tensor_ghash_mul
// CHECK-PCLMULQDQ: tensor.extract
// CHECK-PCLMULQDQ-COUNT-6: llvm.inline_asm{{.*}}vpclmulqdq
// CHECK-PCLMULQDQ: tensor.from_elements
// CHECK-OFF-LABEL: @tensor_ghash_mul
// CHECK-OFF: field.mul
// CHECK-OFF-NOT: vpclmulqdq
func.func @tensor_ghash_mul(%a: tensor<2x!GHASH>, %b: tensor<2x!GHASH>) -> tensor<2x!GHASH> {
  %c = field.mul %a, %b : tensor<2x!GHASH>
  return %c : tensor<2x!GHASH>
}

// Shaped GHASH (vector): same three vpclmulqdq per lane, reassembled with
// vector.extract / vector.from_elements.
// CHECK-PCLMULQDQ-LABEL: @vector_ghash_mul
// CHECK-PCLMULQDQ: vector.extract
// CHECK-PCLMULQDQ-COUNT-6: llvm.inline_asm{{.*}}vpclmulqdq
// CHECK-PCLMULQDQ: vector.from_elements
// CHECK-OFF-LABEL: @vector_ghash_mul
// CHECK-OFF: field.mul
// CHECK-OFF-NOT: vpclmulqdq
func.func @vector_ghash_mul(%a: vector<2x!GHASH>, %b: vector<2x!GHASH>) -> vector<2x!GHASH> {
  %c = field.mul %a, %b : vector<2x!GHASH>
  return %c : vector<2x!GHASH>
}

// The dynamic-shape guard (a dynamic ghash mul stays portable, since lane-by-lane
// unrolling needs a static element count) lives in the shared ShapedFieldMul
// helper and is covered once by @dynamic_aes_mul in
// specialize_binary_field_to_nvptx_shaped.mlir; a dynamic case here would also
// hit the pre-existing binary-field-to-arith gap on dynamic shaped field.mul,
// which the CHECK-PIPELINE run below would trip over.
