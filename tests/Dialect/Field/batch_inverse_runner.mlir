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

// RUN: prime-ir-opt %s --field-to-llvm \
// RUN:   | mlir-runner -e test_batch_inverse -entry-point-result=void \
// RUN:      --shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s --check-prefix=CHECK_BATCH < %t

// p = 149 ≡ 5 mod 8, ξ = 2 (non-4th-power in F₁₄₉)
!PF = !field.pf<149:i32>
!QF = !field.ef<4x!PF, 2:i32>

func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

// Helper: extract coefficients of an EF element and print as 4xi32.
func.func @print_ef(%val: !QF) {
  %c0, %c1, %c2, %c3 = field.ext_to_coeffs %val : (!QF) -> (!PF, !PF, !PF, !PF)
  %i0 = field.bitcast %c0 : !PF -> i32
  %i1 = field.bitcast %c1 : !PF -> i32
  %i2 = field.bitcast %c2 : !PF -> i32
  %i3 = field.bitcast %c3 : !PF -> i32
  %t = tensor.from_elements %i0, %i1, %i2, %i3 : tensor<4xi32>
  %m = bufferization.to_buffer %t : tensor<4xi32> to memref<4xi32>
  %u = memref.cast %m : memref<4xi32> to memref<*xi32>
  func.call @printMemrefI32(%u) : (memref<*xi32>) -> ()
  return
}

// Use function arguments to prevent constant folding of the batch inverse.
// InverseOp has a folder that would fold constant inputs at compile time.
func.func @run_batch_inverse(%a: !QF, %b: !QF) {
  // Build tensor<2x!QF> = [a, b].
  %t = tensor.from_elements %a, %b : tensor<2x!QF>

  // Batch inverse: [a⁻¹, b⁻¹].
  %inv = field.inverse %t : tensor<2x!QF>

  // Extract individual inverses.
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %inv_a = tensor.extract %inv[%c0] : tensor<2x!QF>
  %inv_b = tensor.extract %inv[%c1] : tensor<2x!QF>

  // Test 1: Print a⁻¹ — should match scalar inverse result.
  func.call @print_ef(%inv_a) : (!QF) -> ()

  // Test 2: Print b⁻¹.
  func.call @print_ef(%inv_b) : (!QF) -> ()

  // Test 3: a * a⁻¹ should be identity (1, 0, 0, 0).
  %check_a = field.mul %a, %inv_a : !QF
  func.call @print_ef(%check_a) : (!QF) -> ()

  // Test 4: b * b⁻¹ should be identity (1, 0, 0, 0).
  %check_b = field.mul %b, %inv_b : !QF
  func.call @print_ef(%check_b) : (!QF) -> ()

  return
}

func.func @test_batch_inverse() {
  // a = 1 + 2v + 3v² + 4v³, b = 2 + 3v + v² + 2v³
  %a = field.constant [1, 2, 3, 4] : !QF
  %b = field.constant [2, 3, 1, 2] : !QF
  func.call @run_batch_inverse(%a, %b) : (!QF, !QF) -> ()
  return
}

// Test 1: a⁻¹ matches scalar inverse from quartic_ext_field_runner.
// CHECK_BATCH: [114, 85, 92, 148]

// Test 2: b⁻¹ (verified by Test 4: b * b⁻¹ = 1).
// CHECK_BATCH: [22, 85, 42, 96]

// Test 3: a * a⁻¹ = identity
// CHECK_BATCH: [1, 0, 0, 0]

// Test 4: b * b⁻¹ = identity
// CHECK_BATCH: [1, 0, 0, 0]
