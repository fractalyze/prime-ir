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

// Shaped (tensor/vector) clmad specialization for the flat binary-field bases
// (`bf<7, ghash>`, `bf<3, aes>`). clmad is scalar inline asm, so a shaped
// `field.mul` is unrolled lane by lane: each element is extracted, run through
// the same scalar clmad ladder as the scalar path, and the results are
// reassembled with `tensor.from_elements`/`vector.from_elements`. This is the
// path the batched gf8 additive-NTT butterfly multiply (flock zerocheck) takes
// on GPU, so it must emit clmad instead of the software shift-XOR fallback.
//
// Tower multiplies stay scalar-only (see
// specialize_binary_field_to_nvptx_shaped.mlir): the flat bases are covered
// here because, unlike shaped tower mul, shaped flat mul fully legalizes
// through binary-field-to-arith once specialized.

// RUN: prime-ir-opt --specialize-binary-field-to-nvptx %s | FileCheck %s --check-prefix=CHECK-CLMAD

// use-clmad=false disables specialization: the shaped mul stays portable.
// RUN: prime-ir-opt --specialize-binary-field-to-nvptx="use-clmad=false" %s | FileCheck %s --check-prefix=CHECK-OFF

// The emitted extract/from_elements scaffolding and the field<->iN casts must
// all be reconciled by the downstream binary-field-to-arith lowering — no
// field ops or unrealized casts may survive.
// RUN: prime-ir-opt --specialize-binary-field-to-nvptx --binary-field-to-arith %s | FileCheck %s --check-prefix=CHECK-PIPELINE
// CHECK-PIPELINE-NOT: field.mul
// CHECK-PIPELINE-NOT: unrealized_conversion_cast

!AES = !field.bf<3, aes>     // GF(2⁸), flat AES polynomial basis
!GHASH = !field.bf<7, ghash> // GF(2¹²⁸), flat GHASH polynomial basis

// Shaped AES (tensor): each of the 2 lanes is one clmad.lo product + two
// reduction folds = three clmad.lo.u64, so 6 total, wrapped by an extract per
// operand lane and a from_elements to rebuild the tensor.
// CHECK-CLMAD-LABEL: @tensor_aes_mul
// CHECK-CLMAD: tensor.extract
// CHECK-CLMAD-COUNT-6: llvm.inline_asm{{.*}}clmad.lo{{.*}}u64
// CHECK-CLMAD: tensor.from_elements
// CHECK-OFF-LABEL: @tensor_aes_mul
// CHECK-OFF: field.mul
// CHECK-OFF-NOT: clmad
func.func @tensor_aes_mul(%a: tensor<2x!AES>, %b: tensor<2x!AES>) -> tensor<2x!AES> {
  %c = field.mul %a, %b : tensor<2x!AES>
  return %c : tensor<2x!AES>
}

// Shaped AES (vector): same three clmad.lo per lane, reassembled with
// vector.extract / vector.from_elements.
// CHECK-CLMAD-LABEL: @vector_aes_mul
// CHECK-CLMAD: vector.extract
// CHECK-CLMAD-COUNT-6: llvm.inline_asm{{.*}}clmad.lo{{.*}}u64
// CHECK-CLMAD: vector.from_elements
// CHECK-OFF-LABEL: @vector_aes_mul
// CHECK-OFF: field.mul
// CHECK-OFF-NOT: clmad
func.func @vector_aes_mul(%a: vector<2x!AES>, %b: vector<2x!AES>) -> vector<2x!AES> {
  %c = field.mul %a, %b : vector<2x!AES>
  return %c : vector<2x!AES>
}

// Shaped GHASH (tensor): eight clmad.{lo,hi}.u64 build the 128×128 product per
// lane, so 16 clmad across the 2 lanes.
// CHECK-CLMAD-LABEL: @tensor_ghash_mul
// CHECK-CLMAD: tensor.extract
// CHECK-CLMAD-COUNT-16: llvm.inline_asm{{.*}}clmad{{.*}}u64
// CHECK-CLMAD: tensor.from_elements
// CHECK-OFF-LABEL: @tensor_ghash_mul
// CHECK-OFF: field.mul
// CHECK-OFF-NOT: clmad
func.func @tensor_ghash_mul(%a: tensor<2x!GHASH>, %b: tensor<2x!GHASH>) -> tensor<2x!GHASH> {
  %c = field.mul %a, %b : tensor<2x!GHASH>
  return %c : tensor<2x!GHASH>
}

// A 2-D tensor exercises the row-major delinearization: 6 lanes × three
// clmad.lo = 18, and each extract carries two `index` operands.
// CHECK-CLMAD-LABEL: @tensor2d_aes_mul
// CHECK-CLMAD: tensor.extract %{{.*}}[%{{.*}}, %{{.*}}]
// CHECK-CLMAD-COUNT-18: llvm.inline_asm{{.*}}clmad.lo{{.*}}u64
// CHECK-CLMAD: tensor.from_elements
func.func @tensor2d_aes_mul(%a: tensor<2x3x!AES>, %b: tensor<2x3x!AES>) -> tensor<2x3x!AES> {
  %c = field.mul %a, %b : tensor<2x3x!AES>
  return %c : tensor<2x3x!AES>
}

// A static zero-element tensor unrolls to an empty `tensor.from_elements` (no
// lanes -> no clmad) rather than being rejected: rejecting it would leave an
// un-lowerable `field.mul` on the empty tensor, which the CHECK-PIPELINE run
// (--binary-field-to-arith) above would fail to legalize. Both flat bases.
// CHECK-CLMAD-LABEL: @tensor_aes_mul_empty
// CHECK-CLMAD: tensor.from_elements
// CHECK-CLMAD-NOT: clmad
// CHECK-OFF-LABEL: @tensor_aes_mul_empty
// CHECK-OFF: field.mul
func.func @tensor_aes_mul_empty(%a: tensor<0x!AES>, %b: tensor<0x!AES>) -> tensor<0x!AES> {
  %c = field.mul %a, %b : tensor<0x!AES>
  return %c : tensor<0x!AES>
}

// CHECK-CLMAD-LABEL: @tensor_ghash_mul_empty
// CHECK-CLMAD: tensor.from_elements
// CHECK-CLMAD-NOT: clmad
// CHECK-OFF-LABEL: @tensor_ghash_mul_empty
// CHECK-OFF: field.mul
func.func @tensor_ghash_mul_empty(%a: tensor<0x!GHASH>, %b: tensor<0x!GHASH>) -> tensor<0x!GHASH> {
  %c = field.mul %a, %b : tensor<0x!GHASH>
  return %c : tensor<0x!GHASH>
}
