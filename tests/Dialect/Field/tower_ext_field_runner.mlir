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

// RUN: prime-ir-opt %s --field-to-mod-arith --field-to-llvm \
// RUN:   | mlir-runner -e test_tower_ext_field -entry-point-result=void \
// RUN:      --shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s --check-prefix=CHECK_TOWER < %t

// Tower extension: Fp6 = (Fp2)^3 where Fp2 = Fp[v]/(v² - 6), Fp6 = Fp2[w]/(w³ - 2)
// An Fp6 element: (a₀ + a₁v) + (b₀ + b₁v)w + (c₀ + c₁v)w²
// Stored as 6 coefficients: [a₀, a₁, b₀, b₁, c₀, c₁]
// nonResidue for Fp6 over Fp2 is dense<[2, 0]> representing 2 + 0v = 2 in Fp2

!PF = !field.pf<7:i32>
!QF = !field.ef<2x!PF, 6:i32>
!TowerF6 = !field.ef<3x!QF, dense<[2, 0]> : tensor<2xi32>>

func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

// Helper to extract all 6 prime field coefficients from a tower element
func.func @extract_tower_coeffs(%x: !TowerF6) -> (i32, i32, i32, i32, i32, i32) {
  // First extract Fp2 coefficients
  %c0, %c1, %c2 = field.ext_to_coeffs %x : (!TowerF6) -> (!QF, !QF, !QF)

  // Then extract Fp coefficients from each Fp2
  %a0, %a1 = field.ext_to_coeffs %c0 : (!QF) -> (!PF, !PF)
  %b0, %b1 = field.ext_to_coeffs %c1 : (!QF) -> (!PF, !PF)
  %d0, %d1 = field.ext_to_coeffs %c2 : (!QF) -> (!PF, !PF)

  // Bitcast to i32
  %a0_i32 = field.bitcast %a0 : !PF -> i32
  %a1_i32 = field.bitcast %a1 : !PF -> i32
  %b0_i32 = field.bitcast %b0 : !PF -> i32
  %b1_i32 = field.bitcast %b1 : !PF -> i32
  %d0_i32 = field.bitcast %d0 : !PF -> i32
  %d1_i32 = field.bitcast %d1 : !PF -> i32

  return %a0_i32, %a1_i32, %b0_i32, %b1_i32, %d0_i32, %d1_i32 : i32, i32, i32, i32, i32, i32
}

func.func @print_tower(%x: !TowerF6) {
  %c0, %c1, %c2, %c3, %c4, %c5 = func.call @extract_tower_coeffs(%x)
    : (!TowerF6) -> (i32, i32, i32, i32, i32, i32)
  %tensor = tensor.from_elements %c0, %c1, %c2, %c3, %c4, %c5 : tensor<6xi32>
  %memref = bufferization.to_buffer %tensor : tensor<6xi32> to memref<6xi32>
  %unranked = memref.cast %memref : memref<6xi32> to memref<*xi32>
  func.call @printMemrefI32(%unranked) : (memref<*xi32>) -> ()
  return
}

func.func @test_tower_ext_field() {
  // Create Fp6 elements using nested coefficients
  // Element a: coeffs in Fp are [1, 2, 3, 4, 5, 6]
  // = (1 + 2v) + (3 + 4v)w + (5 + 6v)w²
  %a0 = field.constant [1, 2] : !QF  // 1 + 2v
  %a1 = field.constant [3, 4] : !QF  // 3 + 4v
  %a2 = field.constant [5, 6] : !QF  // 5 + 6v
  %a = field.ext_from_coeffs %a0, %a1, %a2 : (!QF, !QF, !QF) -> !TowerF6

  // Element b: coeffs in Fp are [1, 1, 1, 1, 1, 1]
  // = (1 + v) + (1 + v)w + (1 + v)w²
  %b0 = field.constant [1, 1] : !QF
  %b1 = field.constant [1, 1] : !QF
  %b2 = field.constant [1, 1] : !QF
  %b = field.ext_from_coeffs %b0, %b1, %b2 : (!QF, !QF, !QF) -> !TowerF6

  // Test 1: Addition - coefficient-wise add in Fp
  // a + b = [1+1, 2+1, 3+1, 4+1, 5+1, 6+1] = [2, 3, 4, 5, 6, 0] mod 7
  %add_result = field.add %a, %b : !TowerF6
  func.call @print_tower(%add_result) : (!TowerF6) -> ()

  // Test 2: Subtraction - coefficient-wise sub in Fp
  // a - b = [1-1, 2-1, 3-1, 4-1, 5-1, 6-1] = [0, 1, 2, 3, 4, 5] mod 7
  %sub_result = field.sub %a, %b : !TowerF6
  func.call @print_tower(%sub_result) : (!TowerF6) -> ()

  // Test 3: Negation
  // -a = [-1, -2, -3, -4, -5, -6] = [6, 5, 4, 3, 2, 1] mod 7
  %neg_result = field.negate %a : !TowerF6
  func.call @print_tower(%neg_result) : (!TowerF6) -> ()

  // Test 4: Double
  // 2*a = [2, 4, 6, 8, 10, 12] = [2, 4, 6, 1, 3, 5] mod 7
  %dbl_result = field.double %a : !TowerF6
  func.call @print_tower(%dbl_result) : (!TowerF6) -> ()

  // Test 5: Multiplication a * b
  // In Fp6 = Fp2[w]/(w³ - 2) with Fp2 = Fp[v]/(v² - 6):
  // a = (1+2v) + (3+4v)w + (5+6v)w², b = (1+v) + (1+v)w + (1+v)w²
  // c0 = a0*b0 + 2*(a1*b2 + a2*b1) = (6+3v) + 2*((6+0v)+(6+4v)) = (2+4v)
  // c1 = a0*b1 + a1*b0 + 2*a2*b2 = (6+3v) + (6+0v) + 2*(6+4v) = (3+4v)
  // c2 = a0*b2 + a1*b1 + a2*b0 = (6+3v) + (6+0v) + (6+4v) = (4+0v)
  // Result: [2, 4, 3, 4, 4, 0]
  %mul_result = field.mul %a, %b : !TowerF6
  func.call @print_tower(%mul_result) : (!TowerF6) -> ()

  // Test 6: Square a²
  // a0² = (1+2v)² = 1 + 4v + 24 = (4+4v)
  // a1² = (3+4v)² = 9 + 24v + 96 = (0+3v)
  // a2² = (5+6v)² = 25 + 60v + 216 = (3+4v)
  // a0*a1 = (1+2v)*(3+4v) = 3 + 10v + 48 = (2+3v)
  // a0*a2 = (1+2v)*(5+6v) = 5 + 16v + 72 = (0+2v)
  // a1*a2 = (3+4v)*(5+6v) = 15 + 38v + 144 = (5+3v)
  // c0 = a0² + 4*a1*a2 = (4+4v) + 4*(5+3v) = (3+2v)
  // c1 = 2*a0*a1 + 2*a2² = 2*(2+3v) + 2*(3+4v) = (3+0v)
  // c2 = 2*a0*a2 + a1² = 2*(0+2v) + (0+3v) = (0+0v)
  // Result: [3, 2, 3, 0, 0, 0]
  %square_result = field.square %a : !TowerF6
  func.call @print_tower(%square_result) : (!TowerF6) -> ()

  // Test 7: Inverse verification - a * inverse(a) should equal identity [1, 0, 0, 0, 0, 0]
  %inv_a = field.inverse %a : !TowerF6
  %inv_verify = field.mul %a, %inv_a : !TowerF6
  func.call @print_tower(%inv_verify) : (!TowerF6) -> ()

  return
}

// Test 1: a + b = [2, 3, 4, 5, 6, 0]
// CHECK_TOWER: [2, 3, 4, 5, 6, 0]

// Test 2: a - b = [0, 1, 2, 3, 4, 5]
// CHECK_TOWER: [0, 1, 2, 3, 4, 5]

// Test 3: -a = [6, 5, 4, 3, 2, 1]
// CHECK_TOWER: [6, 5, 4, 3, 2, 1]

// Test 4: 2*a = [2, 4, 6, 1, 3, 5]
// CHECK_TOWER: [2, 4, 6, 1, 3, 5]

// Test 5: a * b = [2, 4, 3, 4, 4, 0]
// CHECK_TOWER: [2, 4, 3, 4, 4, 0]

// Test 6: a² = [3, 2, 3, 0, 0, 0]
// CHECK_TOWER: [3, 2, 3, 0, 0, 0]

// Test 7: a * inverse(a) = identity = [1, 0, 0, 0, 0, 0]
// CHECK_TOWER: [1, 0, 0, 0, 0, 0]
