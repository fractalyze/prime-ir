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

// RUN: prime-ir-opt -field-to-mod-arith %s -split-input-file | FileCheck %s -enable-var-scope

!PF = !field.pf<7:i32>
!QF = !field.ef<2x!PF, 6:i32>

//===----------------------------------------------------------------------===//
// Mixed-type add lowering
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_lower_add_ext_base
// CHECK-SAME: (%[[EXT:.*]]: [[EF:.*]], %[[BASE:.*]]: [[BF:.*]]) -> [[EF]] {
func.func @test_lower_add_ext_base(%ext: !QF, %base: !PF) -> !QF {
  // ext + base → ext_to_coeffs, add base to coeff₀, ext_from_coeffs
  // CHECK: %[[C:.*]]:2 = field.ext_to_coeffs %[[EXT]]
  // CHECK: %[[SUM:.*]] = mod_arith.add %[[C]]#0, %[[BASE]]
  // CHECK: %[[R:.*]] = field.ext_from_coeffs %[[SUM]], %[[C]]#1
  // CHECK: return %[[R]]
  %0 = field.add %ext, %base : !QF, !PF
  return %0 : !QF
}

// CHECK-LABEL: @test_lower_add_base_ext
// CHECK-SAME: (%[[BASE:.*]]: [[BF:.*]], %[[EXT:.*]]: [[EF:.*]]) -> [[EF]] {
func.func @test_lower_add_base_ext(%base: !PF, %ext: !QF) -> !QF {
  // base + ext → same result (commutative)
  // CHECK: %[[C:.*]]:2 = field.ext_to_coeffs %[[EXT]]
  // CHECK: %[[SUM:.*]] = mod_arith.add %[[C]]#0, %[[BASE]]
  // CHECK: %[[R:.*]] = field.ext_from_coeffs %[[SUM]], %[[C]]#1
  // CHECK: return %[[R]]
  %0 = field.add %base, %ext : !PF, !QF
  return %0 : !QF
}

// -----

!PF = !field.pf<7:i32>
!QF = !field.ef<2x!PF, 6:i32>

//===----------------------------------------------------------------------===//
// Mixed-type sub lowering
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_lower_sub_ext_base
// CHECK-SAME: (%[[EXT:.*]]: [[EF:.*]], %[[BASE:.*]]: [[BF:.*]]) -> [[EF]] {
func.func @test_lower_sub_ext_base(%ext: !QF, %base: !PF) -> !QF {
  // ext - base → ext_to_coeffs, sub base from coeff₀, ext_from_coeffs
  // CHECK: %[[C:.*]]:2 = field.ext_to_coeffs %[[EXT]]
  // CHECK: %[[DIFF:.*]] = mod_arith.sub %[[C]]#0, %[[BASE]]
  // CHECK: %[[R:.*]] = field.ext_from_coeffs %[[DIFF]], %[[C]]#1
  // CHECK: return %[[R]]
  %0 = field.sub %ext, %base : !QF, !PF
  return %0 : !QF
}

// CHECK-LABEL: @test_lower_sub_base_ext
func.func @test_lower_sub_base_ext(%base: !PF, %ext: !QF) -> !QF {
  // base - ext → -(ext - base): sub, then negate all coeffs
  // CHECK-NOT: field.sub
  // CHECK: mod_arith.sub
  // CHECK: mod_arith.negate
  // CHECK: mod_arith.negate
  %0 = field.sub %base, %ext : !PF, !QF
  return %0 : !QF
}

// -----

!PF = !field.pf<7:i32>
!QF = !field.ef<2x!PF, 6:i32>

//===----------------------------------------------------------------------===//
// Mixed-type mul lowering
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_lower_mul_ext_base
// CHECK-SAME: (%[[EXT:.*]]: [[EF:.*]], %[[BASE:.*]]: [[BF:.*]]) -> [[EF]] {
func.func @test_lower_mul_ext_base(%ext: !QF, %base: !PF) -> !QF {
  // ext * base → ext_to_coeffs, mul each coeff by base, ext_from_coeffs
  // CHECK: %[[C:.*]]:2 = field.ext_to_coeffs %[[EXT]]
  // CHECK: %[[P0:.*]] = mod_arith.mul %[[C]]#0, %[[BASE]]
  // CHECK: %[[P1:.*]] = mod_arith.mul %[[C]]#1, %[[BASE]]
  // CHECK: %[[R:.*]] = field.ext_from_coeffs %[[P0]], %[[P1]]
  // CHECK: return %[[R]]
  %0 = field.mul %ext, %base : !QF, !PF
  return %0 : !QF
}

// CHECK-LABEL: @test_lower_mul_base_ext
// CHECK-SAME: (%[[BASE:.*]]: [[BF:.*]], %[[EXT:.*]]: [[EF:.*]]) -> [[EF]] {
func.func @test_lower_mul_base_ext(%base: !PF, %ext: !QF) -> !QF {
  // base * ext → same as ext * base (commutative)
  // CHECK: %[[C:.*]]:2 = field.ext_to_coeffs %[[EXT]]
  // CHECK: %[[P0:.*]] = mod_arith.mul %[[C]]#0, %[[BASE]]
  // CHECK: %[[P1:.*]] = mod_arith.mul %[[C]]#1, %[[BASE]]
  // CHECK: %[[R:.*]] = field.ext_from_coeffs %[[P0]], %[[P1]]
  // CHECK: return %[[R]]
  %0 = field.mul %base, %ext : !PF, !QF
  return %0 : !QF
}

// -----

!PF = !field.pf<7:i32>
!CF = !field.ef<3x!PF, 5:i32>

//===----------------------------------------------------------------------===//
// Cubic extension field mixed-type lowering
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_lower_add_cubic_ext_base
// CHECK-SAME: (%[[EXT:.*]]: [[EF:.*]], %[[BASE:.*]]: [[BF:.*]]) -> [[EF]] {
func.func @test_lower_add_cubic_ext_base(%ext: !CF, %base: !PF) -> !CF {
  // CHECK: %[[C:.*]]:3 = field.ext_to_coeffs %[[EXT]]
  // CHECK: %[[SUM:.*]] = mod_arith.add %[[C]]#0, %[[BASE]]
  // CHECK: %[[R:.*]] = field.ext_from_coeffs %[[SUM]], %[[C]]#1, %[[C]]#2
  // CHECK: return %[[R]]
  %0 = field.add %ext, %base : !CF, !PF
  return %0 : !CF
}

// CHECK-LABEL: @test_lower_mul_cubic_ext_base
// CHECK-SAME: (%[[EXT:.*]]: [[EF:.*]], %[[BASE:.*]]: [[BF:.*]]) -> [[EF]] {
func.func @test_lower_mul_cubic_ext_base(%ext: !CF, %base: !PF) -> !CF {
  // CHECK: %[[C:.*]]:3 = field.ext_to_coeffs %[[EXT]]
  // CHECK: %[[P0:.*]] = mod_arith.mul %[[C]]#0, %[[BASE]]
  // CHECK: %[[P1:.*]] = mod_arith.mul %[[C]]#1, %[[BASE]]
  // CHECK: %[[P2:.*]] = mod_arith.mul %[[C]]#2, %[[BASE]]
  // CHECK: %[[R:.*]] = field.ext_from_coeffs %[[P0]], %[[P1]], %[[P2]]
  // CHECK: return %[[R]]
  %0 = field.mul %ext, %base : !CF, !PF
  return %0 : !CF
}
