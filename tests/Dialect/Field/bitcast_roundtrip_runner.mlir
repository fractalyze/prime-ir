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

// Tests that tensor reinterpret bitcasts in both directions (upward: PF → EF,
// downward: EF → PF) lower correctly through field-to-llvm when the bitcast
// crosses a function boundary.
//
// After bufferization, field::BitcastOp operates on memrefs. ConvertBitcast
// rebuilds the memref descriptor with correct sizes/strides/offset for the
// output element type.  Without this, the input descriptor would be forwarded
// as-is, leaving sizes in input-element units and causing memref.copy to
// compute the wrong byte count. Function boundaries need bufferization so that
// tensor<Nx!EF> doesn't leak into LLVM.
//
// Each bitcast is placed in a separate function to prevent the canonicalizer
// from folding bitcast(bitcast(x)) → x, which would hide lowering bugs.

// RUN: prime-ir-opt %s --field-to-llvm='bufferize-function-boundaries=true' \
// RUN:   | mlir-runner -e test_bitcast_roundtrips -entry-point-result=void \
// RUN:      --shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s < %t

!PF = !field.pf<7:i32>
!EF2 = !field.ef<2x!PF, 6:i32>
!EF6 = !field.ef<3x!EF2, 2:i32>  // Fp6 = (Fp2)³

func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

// --- Upward bitcast helpers (one per direction) ---

// PF → EF2 (degree ratio = 2)
func.func @bitcast_pf_to_ef2(%src: tensor<4x!PF>) -> tensor<2x!EF2> {
  %ef = field.bitcast %src : tensor<4x!PF> -> tensor<2x!EF2>
  return %ef : tensor<2x!EF2>
}

// PF → EF6 (degree ratio = 6)
func.func @bitcast_pf_to_ef6(%src: tensor<6x!PF>) -> tensor<1x!EF6> {
  %ef = field.bitcast %src : tensor<6x!PF> -> tensor<1x!EF6>
  return %ef : tensor<1x!EF6>
}

// EF2 → EF6 (tower-to-tower, degree ratio = 3)
func.func @bitcast_ef2_to_ef6(%src: tensor<3x!EF2>) -> tensor<1x!EF6> {
  %ef = field.bitcast %src : tensor<3x!EF2> -> tensor<1x!EF6>
  return %ef : tensor<1x!EF6>
}

// i32 → EF2 (integer → extension, degree ratio = 2)
func.func @bitcast_i32_to_ef2(%src: tensor<4xi32>) -> tensor<2x!EF2> {
  %ef = field.bitcast %src : tensor<4xi32> -> tensor<2x!EF2>
  return %ef : tensor<2x!EF2>
}

// i32 → EF6 (integer → tower extension, degree ratio = 6)
func.func @bitcast_i32_to_ef6(%src: tensor<6xi32>) -> tensor<1x!EF6> {
  %ef = field.bitcast %src : tensor<6xi32> -> tensor<1x!EF6>
  return %ef : tensor<1x!EF6>
}

// --- Downward bitcast helpers (one per direction) ---

// EF2 → PF (degree ratio = 2)
func.func @bitcast_ef2_to_pf(%src: tensor<2x!EF2>) -> tensor<4x!PF> {
  %pf = field.bitcast %src : tensor<2x!EF2> -> tensor<4x!PF>
  return %pf : tensor<4x!PF>
}

// EF6 → PF (degree ratio = 6)
func.func @bitcast_ef6_to_pf(%src: tensor<1x!EF6>) -> tensor<6x!PF> {
  %pf = field.bitcast %src : tensor<1x!EF6> -> tensor<6x!PF>
  return %pf : tensor<6x!PF>
}

// EF6 → EF2 (tower-to-tower, degree ratio = 3)
func.func @bitcast_ef6_to_ef2(%src: tensor<1x!EF6>) -> tensor<3x!EF2> {
  %ef = field.bitcast %src : tensor<1x!EF6> -> tensor<3x!EF2>
  return %ef : tensor<3x!EF2>
}

// EF2 → i32 (extension → integer, degree ratio = 2)
func.func @bitcast_ef2_to_i32(%src: tensor<2x!EF2>) -> tensor<4xi32> {
  %i = field.bitcast %src : tensor<2x!EF2> -> tensor<4xi32>
  return %i : tensor<4xi32>
}

// EF6 → i32 (tower extension → integer, degree ratio = 6)
func.func @bitcast_ef6_to_i32(%src: tensor<1x!EF6>) -> tensor<6xi32> {
  %i = field.bitcast %src : tensor<1x!EF6> -> tensor<6xi32>
  return %i : tensor<6xi32>
}

func.func @test_bitcast_roundtrips() {
  // Test 1: PF[4] → EF2[2] → i32[4]
  %pf4 = field.constant dense<[1, 2, 3, 4]> : tensor<4x!PF>
  %ef2_a = func.call @bitcast_pf_to_ef2(%pf4)
      : (tensor<4x!PF>) -> tensor<2x!EF2>
  %i32_a = field.bitcast %ef2_a : tensor<2x!EF2> -> tensor<4xi32>
  %m_a = bufferization.to_buffer %i32_a : tensor<4xi32> to memref<4xi32>
  %u_a = memref.cast %m_a : memref<4xi32> to memref<*xi32>
  func.call @printMemrefI32(%u_a) : (memref<*xi32>) -> ()
  // CHECK: [1, 2, 3, 4]

  // Test 2: PF[6] → EF6[1] → i32[6]
  %pf6 = field.constant dense<[1, 2, 3, 4, 5, 6]> : tensor<6x!PF>
  %ef6_b = func.call @bitcast_pf_to_ef6(%pf6)
      : (tensor<6x!PF>) -> tensor<1x!EF6>
  %i32_b = field.bitcast %ef6_b : tensor<1x!EF6> -> tensor<6xi32>
  %m_b = bufferization.to_buffer %i32_b : tensor<6xi32> to memref<6xi32>
  %u_b = memref.cast %m_b : memref<6xi32> to memref<*xi32>
  func.call @printMemrefI32(%u_b) : (memref<*xi32>) -> ()
  // CHECK: [1, 2, 3, 4, 5, 6]

  // Test 3: EF2[3] → EF6[1] → i32[6] (tower-to-tower)
  %ef2_3 = field.constant dense<[[1, 2], [3, 4], [5, 6]]> : tensor<3x!EF2>
  %ef6_c = func.call @bitcast_ef2_to_ef6(%ef2_3)
      : (tensor<3x!EF2>) -> tensor<1x!EF6>
  %i32_c = field.bitcast %ef6_c : tensor<1x!EF6> -> tensor<6xi32>
  %m_c = bufferization.to_buffer %i32_c : tensor<6xi32> to memref<6xi32>
  %u_c = memref.cast %m_c : memref<6xi32> to memref<*xi32>
  func.call @printMemrefI32(%u_c) : (memref<*xi32>) -> ()
  // CHECK: [1, 2, 3, 4, 5, 6]

  // Test 4: i32[4] → EF2[2] → i32[4]
  %i32_src4 = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
  %ef2_d = func.call @bitcast_i32_to_ef2(%i32_src4)
      : (tensor<4xi32>) -> tensor<2x!EF2>
  %i32_d = field.bitcast %ef2_d : tensor<2x!EF2> -> tensor<4xi32>
  %m_d = bufferization.to_buffer %i32_d : tensor<4xi32> to memref<4xi32>
  %u_d = memref.cast %m_d : memref<4xi32> to memref<*xi32>
  func.call @printMemrefI32(%u_d) : (memref<*xi32>) -> ()
  // CHECK: [1, 2, 3, 4]

  // Test 5: i32[6] → EF6[1] → i32[6]
  %i32_src6 = arith.constant dense<[1, 2, 3, 4, 5, 6]> : tensor<6xi32>
  %ef6_e = func.call @bitcast_i32_to_ef6(%i32_src6)
      : (tensor<6xi32>) -> tensor<1x!EF6>
  %i32_e = field.bitcast %ef6_e : tensor<1x!EF6> -> tensor<6xi32>
  %m_e = bufferization.to_buffer %i32_e : tensor<6xi32> to memref<6xi32>
  %u_e = memref.cast %m_e : memref<6xi32> to memref<*xi32>
  func.call @printMemrefI32(%u_e) : (memref<*xi32>) -> ()
  // CHECK: [1, 2, 3, 4, 5, 6]

  // --- Downward bitcasts (EF → PF/i32, EF6 → EF2) ---

  // Test 6: EF2[2] → PF[4] → i32[4]
  %ef2_f = field.constant dense<[[1, 2], [3, 4]]> : tensor<2x!EF2>
  %pf_f = func.call @bitcast_ef2_to_pf(%ef2_f)
      : (tensor<2x!EF2>) -> tensor<4x!PF>
  %i32_f = field.bitcast %pf_f : tensor<4x!PF> -> tensor<4xi32>
  %m_f = bufferization.to_buffer %i32_f : tensor<4xi32> to memref<4xi32>
  %u_f = memref.cast %m_f : memref<4xi32> to memref<*xi32>
  func.call @printMemrefI32(%u_f) : (memref<*xi32>) -> ()
  // CHECK: [1, 2, 3, 4]

  // Test 7: EF6[1] → PF[6] → i32[6]
  %ef6_g = field.constant dense<[[1, 2, 3, 4, 5, 6]]> : tensor<1x!EF6>
  %pf_g = func.call @bitcast_ef6_to_pf(%ef6_g)
      : (tensor<1x!EF6>) -> tensor<6x!PF>
  %i32_g = field.bitcast %pf_g : tensor<6x!PF> -> tensor<6xi32>
  %m_g = bufferization.to_buffer %i32_g : tensor<6xi32> to memref<6xi32>
  %u_g = memref.cast %m_g : memref<6xi32> to memref<*xi32>
  func.call @printMemrefI32(%u_g) : (memref<*xi32>) -> ()
  // CHECK: [1, 2, 3, 4, 5, 6]

  // Test 8: EF6[1] → EF2[3] → i32[6] (tower-to-tower downward)
  %ef6_h = field.constant dense<[[1, 2, 3, 4, 5, 6]]> : tensor<1x!EF6>
  %ef2_h = func.call @bitcast_ef6_to_ef2(%ef6_h)
      : (tensor<1x!EF6>) -> tensor<3x!EF2>
  %i32_h = field.bitcast %ef2_h : tensor<3x!EF2> -> tensor<6xi32>
  %m_h = bufferization.to_buffer %i32_h : tensor<6xi32> to memref<6xi32>
  %u_h = memref.cast %m_h : memref<6xi32> to memref<*xi32>
  func.call @printMemrefI32(%u_h) : (memref<*xi32>) -> ()
  // CHECK: [1, 2, 3, 4, 5, 6]

  // Test 9: EF2[2] → i32[4] (direct to integer)
  %ef2_i = field.constant dense<[[1, 2], [3, 4]]> : tensor<2x!EF2>
  %i32_i = func.call @bitcast_ef2_to_i32(%ef2_i)
      : (tensor<2x!EF2>) -> tensor<4xi32>
  %m_i = bufferization.to_buffer %i32_i : tensor<4xi32> to memref<4xi32>
  %u_i = memref.cast %m_i : memref<4xi32> to memref<*xi32>
  func.call @printMemrefI32(%u_i) : (memref<*xi32>) -> ()
  // CHECK: [1, 2, 3, 4]

  // Test 10: EF6[1] → i32[6] (tower direct to integer)
  %ef6_j = field.constant dense<[[1, 2, 3, 4, 5, 6]]> : tensor<1x!EF6>
  %i32_j = func.call @bitcast_ef6_to_i32(%ef6_j)
      : (tensor<1x!EF6>) -> tensor<6xi32>
  %m_j = bufferization.to_buffer %i32_j : tensor<6xi32> to memref<6xi32>
  %u_j = memref.cast %m_j : memref<6xi32> to memref<*xi32>
  func.call @printMemrefI32(%u_j) : (memref<*xi32>) -> ()
  // CHECK: [1, 2, 3, 4, 5, 6]

  return
}
