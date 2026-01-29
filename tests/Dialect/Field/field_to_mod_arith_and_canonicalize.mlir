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
!QF = !field.ef<2x!PF, 6:i32>

// CHECK-LABEL: @test_mul
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]]) -> [[T]]
func.func @test_mul(%a: !QF, %b: !QF) -> !QF {
  // Karatsuba multiplication with strength reduction (β = 6):
  // c0 = v0 + β * v1, c1 = (a0 + a1)(b0 + b1) - v0 - v1
  // β * v1 = 6 * v1 -> double(v1 + double(v1)) via strength reduction
  // CHECK: %[[LHS:.*]]:2 = field.ext_to_coeffs %[[ARG0]] : ([[T]]) -> (!z7_i32, !z7_i32)
  // CHECK: %[[RHS:.*]]:2 = field.ext_to_coeffs %[[ARG1]] : ([[T]]) -> (!z7_i32, !z7_i32)
  // CHECK: %[[V0:.*]] = mod_arith.mul %[[LHS]]#0, %[[RHS]]#0 : !z7_i32
  // CHECK: %[[V1:.*]] = mod_arith.mul %[[LHS]]#1, %[[RHS]]#1 : !z7_i32
  // CHECK: %[[SUMLHS:.*]] = mod_arith.add %[[LHS]]#0, %[[LHS]]#1 : !z7_i32
  // CHECK: %[[SUMRHS:.*]] = mod_arith.add %[[RHS]]#0, %[[RHS]]#1 : !z7_i32
  // CHECK: %[[SUMPRODUCT:.*]] = mod_arith.mul %[[SUMLHS]], %[[SUMRHS]] : !z7_i32
  // CHECK: %[[TMP:.*]] = mod_arith.sub %[[SUMPRODUCT]], %[[V0]] : !z7_i32
  // CHECK: %[[C1:.*]] = mod_arith.sub %[[TMP]], %[[V1]] : !z7_i32
  // 6 * v1 = double(v1 + double(v1))
  // CHECK: %[[D1:.*]] = mod_arith.double %[[V1]] : !z7_i32
  // CHECK: %[[T1:.*]] = mod_arith.add %[[V1]], %[[D1]] : !z7_i32
  // CHECK: %[[BETA_V1:.*]] = mod_arith.double %[[T1]] : !z7_i32
  // CHECK: %[[C0:.*]] = mod_arith.add %[[V0]], %[[BETA_V1]] : !z7_i32
  // CHECK: %[[RESULT:.*]] = field.ext_from_coeffs %[[C0]], %[[C1]] : (!z7_i32, !z7_i32) -> [[T]]
  %mul = field.mul %a, %b : !QF
  // CHECK: return %[[RESULT]] : [[T]]
  return %mul: !QF
}

// CHECK-LABEL: @test_square
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_square(%a: !QF) -> !QF {
  // Square algorithm with strength reduction:
  // c0 = x0² + β*x1², c1 = 2*x0*x1
  // β*x1² = 6*x1² -> double(x1² + double(x1²)) via strength reduction
  // CHECK: %[[COEFFS:.*]]:2 = field.ext_to_coeffs %[[ARG0]] : ([[T]]) -> (!z7_i32, !z7_i32)
  // CHECK: %[[SQ0:.*]] = mod_arith.square %[[COEFFS]]#0 : !z7_i32
  // CHECK: %[[SQ1:.*]] = mod_arith.square %[[COEFFS]]#1 : !z7_i32
  // CHECK: %[[MUL:.*]] = mod_arith.mul %[[COEFFS]]#0, %[[COEFFS]]#1 : !z7_i32
  // CHECK: %[[C1:.*]] = mod_arith.double %[[MUL]] : !z7_i32
  // 6 * x1² = double(x1² + double(x1²))
  // CHECK: %[[D1:.*]] = mod_arith.double %[[SQ1]] : !z7_i32
  // CHECK: %[[T1:.*]] = mod_arith.add %[[SQ1]], %[[D1]] : !z7_i32
  // CHECK: %[[BETA_SQ1:.*]] = mod_arith.double %[[T1]] : !z7_i32
  // CHECK: %[[C0:.*]] = mod_arith.add %[[SQ0]], %[[BETA_SQ1]] : !z7_i32
  // CHECK: %[[RESULT:.*]] = field.ext_from_coeffs %[[C0]], %[[C1]] : (!z7_i32, !z7_i32) -> [[T]]
  %square = field.square %a : !QF
  // CHECK: return %[[RESULT]] : [[T]]
  return %square: !QF
}

// CHECK-LABEL: @test_inverse
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_inverse(%a: !QF) -> !QF {
  // Inverse with strength reduction:
  // det = x0² - β*x1² = x0² - 6*x1²
  // β*x1² = 6*x1² -> double(x1² + double(x1²)) via strength reduction
  // CHECK: %[[COEFFS:.*]]:2 = field.ext_to_coeffs %[[ARG0]] : ([[T]]) -> (!z7_i32, !z7_i32)
  // CHECK: %[[SQ1:.*]] = mod_arith.square %[[COEFFS]]#1 : !z7_i32
  // CHECK: %[[SQ0:.*]] = mod_arith.square %[[COEFFS]]#0 : !z7_i32
  // 6 * x1² = double(x1² + double(x1²))
  // CHECK: %[[D1:.*]] = mod_arith.double %[[SQ1]] : !z7_i32
  // CHECK: %[[T1:.*]] = mod_arith.add %[[SQ1]], %[[D1]] : !z7_i32
  // CHECK: %[[BETA_SQ1:.*]] = mod_arith.double %[[T1]] : !z7_i32
  // CHECK: %[[DET:.*]] = mod_arith.sub %[[SQ0]], %[[BETA_SQ1]] : !z7_i32
  // CHECK: %[[INV:.*]] = mod_arith.inverse %[[DET]] : !z7_i32
  // CHECK: %[[C0:.*]] = mod_arith.mul %[[COEFFS]]#0, %[[INV]] : !z7_i32
  // CHECK: %[[NEG:.*]] = mod_arith.negate %[[COEFFS]]#1 : !z7_i32
  // CHECK: %[[C1:.*]] = mod_arith.mul %[[NEG]], %[[INV]] : !z7_i32
  // CHECK: %[[RESULT:.*]] = field.ext_from_coeffs %[[C0]], %[[C1]] : (!z7_i32, !z7_i32) -> [[T]]
  %inverse = field.inverse %a : !QF
  // CHECK: return %[[RESULT]] : [[T]]
  return %inverse: !QF
}
