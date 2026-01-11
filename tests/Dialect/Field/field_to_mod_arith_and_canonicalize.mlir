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

// RUN: prime-ir-opt -field-to-mod-arith -canonicalize %s | FileCheck %s -enable-var-scope

!PF = !field.pf<7:i32>
!QF = !field.f2<!PF, 6:i32>

// CHECK-LABEL: @test_mul
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]]) -> [[T]]
func.func @test_mul(%a: !QF, %b: !QF) -> !QF {
  // Karatsuba multiplication with canonicalization (β = 6 ≡ -1 mod 7):
  // c0 = v0 + β * v1 = v0 - v1, c1 = (a0 + a1)(b0 + b1) - v0 - v1
  // CHECK: %[[LHS:.*]]:2 = field.ext_to_coeffs %[[ARG0]] : ([[T]]) -> (!z7_i32, !z7_i32)
  // CHECK: %[[RHS:.*]]:2 = field.ext_to_coeffs %[[ARG1]] : ([[T]]) -> (!z7_i32, !z7_i32)
  // CHECK: %[[V0:.*]] = mod_arith.mul %[[LHS]]#0, %[[RHS]]#0 : !z7_i32
  // CHECK: %[[V1:.*]] = mod_arith.mul %[[LHS]]#1, %[[RHS]]#1 : !z7_i32
  // CHECK: %[[SUMLHS:.*]] = mod_arith.add %[[LHS]]#0, %[[LHS]]#1 : !z7_i32
  // CHECK: %[[SUMRHS:.*]] = mod_arith.add %[[RHS]]#0, %[[RHS]]#1 : !z7_i32
  // CHECK: %[[SUMPRODUCT:.*]] = mod_arith.mul %[[SUMLHS]], %[[SUMRHS]] : !z7_i32
  // CHECK: %[[TMP:.*]] = mod_arith.sub %[[SUMPRODUCT]], %[[V0]] : !z7_i32
  // CHECK: %[[C1:.*]] = mod_arith.sub %[[TMP]], %[[V1]] : !z7_i32
  // CHECK: %[[C0:.*]] = mod_arith.sub %[[V0]], %[[V1]] : !z7_i32
  // CHECK: %[[RESULT:.*]] = field.ext_from_coeffs %[[C0]], %[[C1]] : (!z7_i32, !z7_i32) -> [[T]]
  %mul = field.mul %a, %b : !QF
  // CHECK: return %[[RESULT]] : [[T]]
  return %mul: !QF
}

// CHECK-LABEL: @test_square
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_square(%a: !QF) -> !QF {
  // Custom square algorithm: c0 = (x0 - x1)(x0 + x1), c1 = 2 * x0 * x1
  // CHECK: %[[COEFFS:.*]]:2 = field.ext_to_coeffs %[[ARG0]] : ([[T]]) -> (!z7_i32, !z7_i32)
  // CHECK: %[[MUL:.*]] = mod_arith.mul %[[COEFFS]]#0, %[[COEFFS]]#1 : !z7_i32
  // CHECK: %[[DOUBLE:.*]] = mod_arith.double %[[MUL]] : !z7_i32
  // CHECK: %[[SUB:.*]] = mod_arith.sub %[[COEFFS]]#0, %[[COEFFS]]#1 : !z7_i32
  // CHECK: %[[ADD:.*]] = mod_arith.add %[[COEFFS]]#0, %[[COEFFS]]#1 : !z7_i32
  // CHECK: %[[MUL2:.*]] = mod_arith.mul %[[SUB]], %[[ADD]] : !z7_i32
  // CHECK: %[[RESULT:.*]] = field.ext_from_coeffs %[[MUL2]], %[[DOUBLE]] : (!z7_i32, !z7_i32) -> [[T]]
  %square = field.square %a : !QF
  // CHECK: return %[[RESULT]] : [[T]]
  return %square: !QF
}

// CHECK-LABEL: @test_inverse
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_inverse(%a: !QF) -> !QF {
  // CHECK: %[[COEFFS:.*]]:2 = field.ext_to_coeffs %[[ARG0]] : ([[T]]) -> (!z7_i32, !z7_i32)
  // CHECK: %[[SQUARE:.*]] = mod_arith.square %[[COEFFS]]#1 : !z7_i32
  // CHECK: %[[SQUARE2:.*]] = mod_arith.square %[[COEFFS]]#0 : !z7_i32
  // CHECK: %[[ADD:.*]] = mod_arith.add %[[SQUARE2]], %[[SQUARE]] : !z7_i32
  // CHECK: %[[INV:.*]] = mod_arith.inverse %[[ADD]] : !z7_i32
  // CHECK: %[[MUL:.*]] = mod_arith.mul %[[COEFFS]]#0, %[[INV]] : !z7_i32
  // CHECK: %[[NEG:.*]] = mod_arith.negate %[[COEFFS]]#1 : !z7_i32
  // CHECK: %[[MUL2:.*]] = mod_arith.mul %[[NEG]], %[[INV]] : !z7_i32
  // CHECK: %[[RESULT:.*]] = field.ext_from_coeffs %[[MUL]], %[[MUL2]] : (!z7_i32, !z7_i32) -> [[T]]
  %inverse = field.inverse %a : !QF
  // CHECK: return %[[RESULT]] : [[T]]
  return %inverse: !QF
}
