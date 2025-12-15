// Copyright 2025 The ZKIR Authors.
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

// RUN: zkir-opt %s --field-to-llvm \
// RUN:   | mlir-runner -e test_quartic_ext_field -entry-point-result=void \
// RUN:      --shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s --check-prefix=CHECK_QUARTIC < %t

// p = 13, ξ = 2 (non-4th-power in F₁₃)
!PF = !field.pf<13:i32>
!PFm = !field.pf<13:i32, true>
!QF = !field.f4<!PF, 2:i32>
!QFm = !field.f4<!PFm, 2:i32>

func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

func.func @test_quartic_ext_field() {
  // Create Fp4 elements: a = 1 + 2v + 3v² + 4v³, b = 2 + 3v + v² + 2v³
  %a = field.constant [1, 2, 3, 4] : !QF
  %b = field.constant [2, 3, 1, 2] : !QF

  // Test 1: Addition - a + b = (3, 5, 4, 6) in F₁₃
  %add_result = field.add %a, %b : !QF
  %add0, %add1, %add2, %add3 = field.ext_to_coeffs %add_result : (!QF) -> (!PF, !PF, !PF, !PF)
  %add0_i32 = field.bitcast %add0 : !PF -> i32
  %add1_i32 = field.bitcast %add1 : !PF -> i32
  %add2_i32 = field.bitcast %add2 : !PF -> i32
  %add3_i32 = field.bitcast %add3 : !PF -> i32
  %add_tensor = tensor.from_elements %add0_i32, %add1_i32, %add2_i32, %add3_i32 : tensor<4xi32>
  %add_memref = bufferization.to_buffer %add_tensor : tensor<4xi32> to memref<4xi32>
  %add_unranked = memref.cast %add_memref : memref<4xi32> to memref<*xi32>
  func.call @printMemrefI32(%add_unranked) : (memref<*xi32>) -> ()

  // Test 2: Subtraction - a - b = (-1, -1, 2, 2) = (12, 12, 2, 2) in F₁₃
  %sub_result = field.sub %a, %b : !QF
  %sub0, %sub1, %sub2, %sub3 = field.ext_to_coeffs %sub_result : (!QF) -> (!PF, !PF, !PF, !PF)
  %sub0_i32 = field.bitcast %sub0 : !PF -> i32
  %sub1_i32 = field.bitcast %sub1 : !PF -> i32
  %sub2_i32 = field.bitcast %sub2 : !PF -> i32
  %sub3_i32 = field.bitcast %sub3 : !PF -> i32
  %sub_tensor = tensor.from_elements %sub0_i32, %sub1_i32, %sub2_i32, %sub3_i32 : tensor<4xi32>
  %sub_memref = bufferization.to_buffer %sub_tensor : tensor<4xi32> to memref<4xi32>
  %sub_unranked = memref.cast %sub_memref : memref<4xi32> to memref<*xi32>
  func.call @printMemrefI32(%sub_unranked) : (memref<*xi32>) -> ()

  // Test 3: Multiplication - a * b (with ξ = 2, v⁴ = 2)
  %mul_result = field.mul %a, %b : !QF
  %mul0, %mul1, %mul2, %mul3 = field.ext_to_coeffs %mul_result : (!QF) -> (!PF, !PF, !PF, !PF)
  %mul0_i32 = field.bitcast %mul0 : !PF -> i32
  %mul1_i32 = field.bitcast %mul1 : !PF -> i32
  %mul2_i32 = field.bitcast %mul2 : !PF -> i32
  %mul3_i32 = field.bitcast %mul3 : !PF -> i32
  %mul_tensor = tensor.from_elements %mul0_i32, %mul1_i32, %mul2_i32, %mul3_i32 : tensor<4xi32>
  %mul_memref = bufferization.to_buffer %mul_tensor : tensor<4xi32> to memref<4xi32>
  %mul_unranked = memref.cast %mul_memref : memref<4xi32> to memref<*xi32>
  func.call @printMemrefI32(%mul_unranked) : (memref<*xi32>) -> ()

  // Test 4: Square - a²
  %square_result = field.square %a : !QF
  %sq0, %sq1, %sq2, %sq3 = field.ext_to_coeffs %square_result : (!QF) -> (!PF, !PF, !PF, !PF)
  %sq0_i32 = field.bitcast %sq0 : !PF -> i32
  %sq1_i32 = field.bitcast %sq1 : !PF -> i32
  %sq2_i32 = field.bitcast %sq2 : !PF -> i32
  %sq3_i32 = field.bitcast %sq3 : !PF -> i32
  %sq_tensor = tensor.from_elements %sq0_i32, %sq1_i32, %sq2_i32, %sq3_i32 : tensor<4xi32>
  %sq_memref = bufferization.to_buffer %sq_tensor : tensor<4xi32> to memref<4xi32>
  %sq_unranked = memref.cast %sq_memref : memref<4xi32> to memref<*xi32>
  func.call @printMemrefI32(%sq_unranked) : (memref<*xi32>) -> ()

  // Test 5: Inverse - a⁻¹
  %inv_result = field.inverse %a : !QF
  %inv0, %inv1, %inv2, %inv3 = field.ext_to_coeffs %inv_result : (!QF) -> (!PF, !PF, !PF, !PF)
  %inv0_i32 = field.bitcast %inv0 : !PF -> i32
  %inv1_i32 = field.bitcast %inv1 : !PF -> i32
  %inv2_i32 = field.bitcast %inv2 : !PF -> i32
  %inv3_i32 = field.bitcast %inv3 : !PF -> i32
  %inv_tensor = tensor.from_elements %inv0_i32, %inv1_i32, %inv2_i32, %inv3_i32 : tensor<4xi32>
  %inv_memref = bufferization.to_buffer %inv_tensor : tensor<4xi32> to memref<4xi32>
  %inv_unranked = memref.cast %inv_memref : memref<4xi32> to memref<*xi32>
  func.call @printMemrefI32(%inv_unranked) : (memref<*xi32>) -> ()

  // Test 6: Montgomery form conversion (round-trip should return original)
  %a_mont = field.to_mont %a : !QFm
  %a_back = field.from_mont %a_mont : !QF
  %back0, %back1, %back2, %back3 = field.ext_to_coeffs %a_back : (!QF) -> (!PF, !PF, !PF, !PF)
  %back0_i32 = field.bitcast %back0 : !PF -> i32
  %back1_i32 = field.bitcast %back1 : !PF -> i32
  %back2_i32 = field.bitcast %back2 : !PF -> i32
  %back3_i32 = field.bitcast %back3 : !PF -> i32
  %back_tensor = tensor.from_elements %back0_i32, %back1_i32, %back2_i32, %back3_i32 : tensor<4xi32>
  %back_memref = bufferization.to_buffer %back_tensor : tensor<4xi32> to memref<4xi32>
  %back_unranked = memref.cast %back_memref : memref<4xi32> to memref<*xi32>
  func.call @printMemrefI32(%back_unranked) : (memref<*xi32>) -> ()

  return
}

// a + b = (1+2, 2+3, 3+1, 4+2) = (3, 5, 4, 6) mod 13
// CHECK_QUARTIC: [3, 5, 4, 6]

// a - b = (1-2, 2-3, 3-1, 4-2) = (-1, -1, 2, 2) = (12, 12, 2, 2) mod 13
// CHECK_QUARTIC: [12, 12, 2, 2]

// a * b: Using v⁴ = ξ = 2, p = 13
// a = 1 + 2v + 3v² + 4v³
// b = 2 + 3v + v² + 2v³
// Schoolbook multiplication:
// c₀ = a₀b₀ + ξ(a₁b₃ + a₂b₂ + a₃b₁)
//    = 1·2 + 2(2·2 + 3·1 + 4·3) = 2 + 2·19 = 40 = 1 mod 13
// c₁ = a₀b₁ + a₁b₀ + ξ(a₂b₃ + a₃b₂)
//    = 1·3 + 2·2 + 2(3·2 + 4·1) = 7 + 20 = 27 = 1 mod 13
// c₂ = a₀b₂ + a₁b₁ + a₂b₀ + ξ(a₃b₃)
//    = 1·1 + 2·3 + 3·2 + 2(4·2) = 13 + 16 = 29 = 3 mod 13
// c₃ = a₀b₃ + a₁b₂ + a₂b₁ + a₃b₀
//    = 1·2 + 2·1 + 3·3 + 4·2 = 21 = 8 mod 13
// CHECK_QUARTIC: [1, 1, 3, 8]

// a² = (1 + 2v + 3v² + 4v³)²: Using v⁴ = ξ = 2, p = 13
// c₀ = a₀² + ξ(2a₁a₃ + a₂²) = 1 + 2(16 + 9) = 51 = 12 mod 13
// c₁ = 2a₀a₁ + ξ(2a₂a₃) = 4 + 48 = 52 = 0 mod 13
// c₂ = 2a₀a₂ + a₁² + ξ(a₃²) = 6 + 4 + 32 = 42 = 3 mod 13
// c₃ = 2a₀a₃ + 2a₁a₂ = 8 + 12 = 20 = 7 mod 13
// CHECK_QUARTIC: [12, 0, 3, 7]

// Test 5: a⁻¹ (inverse of a = (1, 2, 3, 4) in Fp4)
// CHECK_QUARTIC: [9, 1, 8, 10]

// Test 6: Montgomery form round-trip should return original: (1, 2, 3, 4)
// CHECK_QUARTIC: [1, 2, 3, 4]
