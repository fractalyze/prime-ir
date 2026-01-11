// Copyright 2025 The PrimeIR Authors.
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

// RUN: prime-ir-opt %s --field-to-llvm \
// RUN:   | mlir-runner -e test_cubic_ext_field -entry-point-result=void \
// RUN:      --shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s --check-prefix=CHECK_CUBIC < %t

!PF = !field.pf<7:i32>
!PFm = !field.pf<7:i32, true>
!CF = !field.f3<!PF, 2:i32>
!CFm = !field.f3<!PFm, 2:i32>

func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

func.func @test_cubic_ext_field() {
  // Create Fp3 elements: a = 1 + 2v + 3v², b = 2 + 3v + v²
  %a = field.constant [1, 2, 3] : !CF
  %b = field.constant [2, 3, 1] : !CF

  // Test 1: Addition - a + b = (3, 5, 4) in F₇
  %add_result = field.add %a, %b : !CF
  %add0, %add1, %add2 = field.ext_to_coeffs %add_result : (!CF) -> (!PF, !PF, !PF)
  %add0_i32 = field.bitcast %add0 : !PF -> i32
  %add1_i32 = field.bitcast %add1 : !PF -> i32
  %add2_i32 = field.bitcast %add2 : !PF -> i32
  %add_tensor = tensor.from_elements %add0_i32, %add1_i32, %add2_i32 : tensor<3xi32>
  %add_memref = bufferization.to_buffer %add_tensor : tensor<3xi32> to memref<3xi32>
  %add_unranked = memref.cast %add_memref : memref<3xi32> to memref<*xi32>
  func.call @printMemrefI32(%add_unranked) : (memref<*xi32>) -> ()

  // Test 2: Subtraction - a - b = (-1, -1, 2) = (6, 6, 2) in F₇
  %sub_result = field.sub %a, %b : !CF
  %sub0, %sub1, %sub2 = field.ext_to_coeffs %sub_result : (!CF) -> (!PF, !PF, !PF)
  %sub0_i32 = field.bitcast %sub0 : !PF -> i32
  %sub1_i32 = field.bitcast %sub1 : !PF -> i32
  %sub2_i32 = field.bitcast %sub2 : !PF -> i32
  %sub_tensor = tensor.from_elements %sub0_i32, %sub1_i32, %sub2_i32 : tensor<3xi32>
  %sub_memref = bufferization.to_buffer %sub_tensor : tensor<3xi32> to memref<3xi32>
  %sub_unranked = memref.cast %sub_memref : memref<3xi32> to memref<*xi32>
  func.call @printMemrefI32(%sub_unranked) : (memref<*xi32>) -> ()

  // Test 3: Multiplication - a * b (with ξ = 2, v³ = 2)
  %mul_result = field.mul %a, %b : !CF
  %mul0, %mul1, %mul2 = field.ext_to_coeffs %mul_result : (!CF) -> (!PF, !PF, !PF)
  %mul0_i32 = field.bitcast %mul0 : !PF -> i32
  %mul1_i32 = field.bitcast %mul1 : !PF -> i32
  %mul2_i32 = field.bitcast %mul2 : !PF -> i32
  %mul_tensor = tensor.from_elements %mul0_i32, %mul1_i32, %mul2_i32 : tensor<3xi32>
  %mul_memref = bufferization.to_buffer %mul_tensor : tensor<3xi32> to memref<3xi32>
  %mul_unranked = memref.cast %mul_memref : memref<3xi32> to memref<*xi32>
  func.call @printMemrefI32(%mul_unranked) : (memref<*xi32>) -> ()

  // Test 4: Square - a²
  %square_result = field.square %a : !CF
  %sq0, %sq1, %sq2 = field.ext_to_coeffs %square_result : (!CF) -> (!PF, !PF, !PF)
  %sq0_i32 = field.bitcast %sq0 : !PF -> i32
  %sq1_i32 = field.bitcast %sq1 : !PF -> i32
  %sq2_i32 = field.bitcast %sq2 : !PF -> i32
  %sq_tensor = tensor.from_elements %sq0_i32, %sq1_i32, %sq2_i32 : tensor<3xi32>
  %sq_memref = bufferization.to_buffer %sq_tensor : tensor<3xi32> to memref<3xi32>
  %sq_unranked = memref.cast %sq_memref : memref<3xi32> to memref<*xi32>
  func.call @printMemrefI32(%sq_unranked) : (memref<*xi32>) -> ()

  // Test 5: Inverse - a⁻¹
  %inv_result = field.inverse %a : !CF
  %inv0, %inv1, %inv2 = field.ext_to_coeffs %inv_result : (!CF) -> (!PF, !PF, !PF)
  %inv0_i32 = field.bitcast %inv0 : !PF -> i32
  %inv1_i32 = field.bitcast %inv1 : !PF -> i32
  %inv2_i32 = field.bitcast %inv2 : !PF -> i32
  %inv_tensor = tensor.from_elements %inv0_i32, %inv1_i32, %inv2_i32 : tensor<3xi32>
  %inv_memref = bufferization.to_buffer %inv_tensor : tensor<3xi32> to memref<3xi32>
  %inv_unranked = memref.cast %inv_memref : memref<3xi32> to memref<*xi32>
  func.call @printMemrefI32(%inv_unranked) : (memref<*xi32>) -> ()

  // Test 6: Montgomery form conversion
  %a_mont = field.to_mont %a : !CFm
  %a_back = field.from_mont %a_mont : !CF
  %back0, %back1, %back2 = field.ext_to_coeffs %a_back : (!CF) -> (!PF, !PF, !PF)
  %back0_i32 = field.bitcast %back0 : !PF -> i32
  %back1_i32 = field.bitcast %back1 : !PF -> i32
  %back2_i32 = field.bitcast %back2 : !PF -> i32
  %back_tensor = tensor.from_elements %back0_i32, %back1_i32, %back2_i32 : tensor<3xi32>
  %back_memref = bufferization.to_buffer %back_tensor : tensor<3xi32> to memref<3xi32>
  %back_unranked = memref.cast %back_memref : memref<3xi32> to memref<*xi32>
  func.call @printMemrefI32(%back_unranked) : (memref<*xi32>) -> ()

  return
}

// a + b = (1+2, 2+3, 3+1) = (3, 5, 4) mod 7
// CHECK_CUBIC: [3, 5, 4]

// a - b = (1-2, 2-3, 3-1) = (-1, -1, 2) = (6, 6, 2) mod 7
// CHECK_CUBIC: [6, 6, 2]

// a * b: Using v³ = ξ = 2
// c₀ = a₀b₀ + ξ(a₁b₂ + a₂b₁) = 1·2 + 2(2·1 + 3·3) = 2 + 2(2+9) = 2 + 2·11 = 2 + 2·4 = 2 + 8 = 10 = 3 mod 7
// c₁ = a₀b₁ + a₁b₀ + ξa₂b₂ = 1·3 + 2·2 + 2·3·1 = 3 + 4 + 6 = 13 = 6 mod 7
// c₂ = a₀b₂ + a₁b₁ + a₂b₀ = 1·1 + 2·3 + 3·2 = 1 + 6 + 6 = 13 = 6 mod 7
// CHECK_CUBIC: [3, 6, 6]

// a² = (1 + 2v + 3v²)²: Using v³ = ξ = 2
// c₀ = a₀² + 2ξa₁a₂ = 1 + 2·2·2·3 = 1 + 24 = 1 + 3 = 4 mod 7
// c₁ = 2a₀a₁ + ξa₂² = 2·1·2 + 2·9 = 4 + 18 = 4 + 4 = 8 = 1 mod 7
// c₂ = 2a₀a₂ + a₁² = 2·1·3 + 4 = 6 + 4 = 10 = 3 mod 7
// CHECK_CUBIC: [4, 1, 3]

// a⁻¹: det = 5, det⁻¹ = 3. t₀=3, t₁=2, t₂=1. c₀=3·3=2, c₁=2·3=6, c₂=1·3=3
// CHECK_CUBIC: [2, 6, 3]

// Montgomery form round-trip should return original: (1, 2, 3)
// CHECK_CUBIC: [1, 2, 3]
