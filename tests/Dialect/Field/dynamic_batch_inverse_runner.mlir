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
// RUN:   | mlir-runner -e test_dynamic_batch_inverse -entry-point-result=void \
// RUN:      --shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s < %t

// p = 31 (small prime for readable test values)
!F = !field.pf<31:i32>

func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

func.func @print_pf(%val: !F) {
  %i = field.bitcast %val : !F -> i32
  %t = tensor.from_elements %i : tensor<1xi32>
  %m = bufferization.to_buffer %t : tensor<1xi32> to memref<1xi32>
  %u = memref.cast %m : memref<1xi32> to memref<*xi32>
  func.call @printMemrefI32(%u) : (memref<*xi32>) -> ()
  return
}

// Use function arguments to prevent constant folding.
func.func @run_dynamic_batch(%a: !F, %b: !F, %c: !F, %d: !F) {
  // Build static tensor, then cast to dynamic to force runtime dimension.
  %static = tensor.from_elements %a, %b, %c, %d : tensor<4x!F>
  %dynamic = tensor.cast %static : tensor<4x!F> to tensor<?x!F>

  // Batch inverse on dynamic tensor.
  %inv = field.inverse %dynamic : tensor<?x!F>

  // Extract and print inverses.
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index

  %inv0 = tensor.extract %inv[%c0] : tensor<?x!F>
  func.call @print_pf(%inv0) : (!F) -> ()
  %inv1 = tensor.extract %inv[%c1] : tensor<?x!F>
  func.call @print_pf(%inv1) : (!F) -> ()
  %inv2 = tensor.extract %inv[%c2] : tensor<?x!F>
  func.call @print_pf(%inv2) : (!F) -> ()
  %inv3 = tensor.extract %inv[%c3] : tensor<?x!F>
  func.call @print_pf(%inv3) : (!F) -> ()

  // Verify: a × a⁻¹ = 1 for each element.
  %check0 = field.mul %a, %inv0 : !F
  func.call @print_pf(%check0) : (!F) -> ()
  %check1 = field.mul %b, %inv1 : !F
  func.call @print_pf(%check1) : (!F) -> ()
  %check2 = field.mul %c, %inv2 : !F
  func.call @print_pf(%check2) : (!F) -> ()
  %check3 = field.mul %d, %inv3 : !F
  func.call @print_pf(%check3) : (!F) -> ()

  return
}

func.func @test_dynamic_batch_inverse() {
  // [2, 3, 5, 7] — all non-zero in F₃₁.
  %a = field.constant 2 : !F
  %b = field.constant 3 : !F
  %c = field.constant 5 : !F
  %d = field.constant 7 : !F
  func.call @run_dynamic_batch(%a, %b, %c, %d) : (!F, !F, !F, !F) -> ()
  return
}

// Inverses mod 31:
// 2⁻¹ = 16 (2 × 16 = 32 ≡ 1)
// CHECK: [16]
// 3⁻¹ = 21 (3 × 21 = 63 ≡ 1)
// CHECK: [21]
// 5⁻¹ = 25 (5 × 25 = 125 ≡ 1)
// CHECK: [25]
// 7⁻¹ = 9  (7 × 9  = 63 ≡ 1)
// CHECK: [9]
// Verification (all should be 1):
// CHECK: [1]
// CHECK: [1]
// CHECK: [1]
// CHECK: [1]
