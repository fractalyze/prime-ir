// RUN: zkir-opt -prime-field-to-mod-arith --split-input-file %s | FileCheck %s --enable-var-scope
!PF1 = !field.pf<3:i32>
!PFv = tensor<4x!PF1>
#elem = #field.pf_elem<31:i32> : !PF1

// CHECK-LABEL: @test_lower_constant
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_lower_constant() -> !PF1 {
  // CHECK: %[[RES:.*]] = mod_arith.constant 4 : [[T]]
  %res = field.pf.constant 4 : !PF1
  // CHECK: return %[[RES]] : [[T]]
  return %res : !PF1
}

// CHECK-LABEL: @test_lower_encapsulate
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]]) -> [[F:.*]] {
func.func @test_lower_encapsulate(%lhs : i32) -> !PF1 {
  // CHECK-NOT: field.pf.encapsulate
  // CHECK: %[[RES:.*]] = mod_arith.encapsulate %[[LHS]] : [[T]] -> [[F]]
  %res = field.pf.encapsulate %lhs : i32 -> !PF1
  // CHECK: return %[[RES]] : [[F]]
  return %res : !PF1
}

// CHECK-LABEL: @test_lower_encapsulate_vec
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]]) -> [[TF:.*]] {
func.func @test_lower_encapsulate_vec(%lhs : tensor<4xi32>) -> tensor<4x!PF1> {
  // CHECK-NOT: field.pf.encapsulate
  // CHECK: %[[RES:.*]] = mod_arith.encapsulate %[[LHS]] : [[T]] -> [[TF]]
  %res = field.pf.encapsulate %lhs : tensor<4xi32> -> tensor<4x!PF1>
  // CHECK: return %[[RES]] : [[TF]]
  return %res : tensor<4x!PF1>
}

// CHECK-LABEL: @test_lower_extract
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]]) -> [[F:.*]] {
func.func @test_lower_extract(%lhs : !PF1) -> i32 {
  // CHECK-NOT: field.pf.extract
  // CHECK: %[[RES:.*]] = mod_arith.extract %[[LHS]] : [[T]] -> [[F]]
  %res = field.pf.extract %lhs : !PF1 -> i32
  // CHECK: return %[[RES]] : [[F]]
  return %res : i32
}

// CHECK-LABEL: @test_lower_extract_vec
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]]) -> [[TF:.*]] {
func.func @test_lower_extract_vec(%lhs : tensor<4x!PF1>) -> tensor<4xi32> {
  // CHECK-NOT: field.pf.extract
  // CHECK: %[[RES:.*]] = mod_arith.extract %[[LHS]] : [[T]] -> [[TF]]
  %res = field.pf.extract %lhs : tensor<4x!PF1> -> tensor<4xi32>
  // CHECK: return %[[RES]] : [[TF]]
  return %res : tensor<4xi32>
}

// CHECK-LABEL: @test_lower_inverse
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_lower_inverse(%lhs : !PF1) -> !PF1 {
  // CHECK-NOT: field.pf.inverse
  // CHECK: %[[RES:.*]] = mod_arith.inverse %[[LHS]] : [[T]]
  %res = field.pf.inverse %lhs : !PF1
  return %res : !PF1
}

// CHECK-LABEL: @test_lower_inverse_vec
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_lower_inverse_vec(%lhs : !PFv) -> !PFv {
  // CHECK-NOT: field.pf.inverse
  // CHECK: %[[RES:.*]] = mod_arith.inverse %[[LHS]] : [[T]]
  %res = field.pf.inverse %lhs : !PFv
  return %res : !PFv
}

// CHECK-LABEL: @test_lower_add
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_lower_add() -> !PF1 {
  // CHECK: %[[C0:.*]] = mod_arith.constant 4 : [[T]]
  %c0 = field.pf.constant 4 : !PF1
  // CHECK: %[[RES:.*]] = mod_arith.add %[[C0]], %[[C0]] : [[T]]
  %res = field.pf.add %c0, %c0 : !PF1
  // CHECK: return %[[RES]] : [[T]]
  return %res : !PF1
}

// CHECK-LABEL: @test_lower_add_vec
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_add_vec(%lhs : !PFv, %rhs : !PFv) -> !PFv {
  // CHECK-NOT: field.pf.add
  // CHECK: %[[RES:.*]] = mod_arith.add %[[LHS]], %[[RHS]] : tensor<4x!Z3_i32_>
  %res = field.pf.add %lhs, %rhs : !PFv
  // CHECK: return %[[RES]] : [[T]]
  return %res : !PFv
}

// CHECK-LABEL: @test_lower_sub
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_lower_sub() -> !PF1 {
  // CHECK: %[[C0:.*]] = mod_arith.constant 4 : [[T]]
  %c0 = field.pf.constant 4 : !PF1
  // CHECK: %[[C1:.*]] = mod_arith.constant 5 : [[T]]
  %c1 = field.pf.constant 5 : !PF1
  // CHECK: %[[RES:.*]] = mod_arith.sub %[[C0]], %[[C1]] : [[T]]
  %res = field.pf.sub %c0, %c1 : !PF1
  // CHECK: return %[[RES]] : [[T]]
  return %res : !PF1
}

// CHECK-LABEL: @test_lower_sub_vec
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_sub_vec(%lhs : !PFv, %rhs : !PFv) -> !PFv {
  // CHECK-NOT: field.pf.sub
  // CHECK: %[[RES:.*]] = mod_arith.sub %[[LHS]], %[[RHS]] : [[T]]
  %res = field.pf.sub %lhs, %rhs : !PFv
  // CHECK: return %[[RES]] : [[T]]
  return %res : !PFv
}

// CHECK-LABEL: @test_lower_mul
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_lower_mul() -> !PF1 {
  // CHECK: %[[C0:.*]] = mod_arith.constant 4 : [[T]]
  %c0 = field.pf.constant 4 : !PF1
  // CHECK: %[[RES:.*]] = mod_arith.mul %[[C0]], %[[C0]] : [[T]]
  %res = field.pf.mul %c0, %c0 : !PF1
  // CHECK: return %[[RES]] : [[T]]
  return %res : !PF1
}

// CHECK-LABEL: @test_lower_mul_vec
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_mul_vec(%lhs : !PFv, %rhs : !PFv) -> !PFv {
  // CHECK-NOT: field.pf.mul
  // CHECK: %[[RES:.*]] = mod_arith.mul %[[LHS]], %[[RHS]] : [[T]]
  %res = field.pf.mul %lhs, %rhs : !PFv
  // CHECK: return %[[RES]] : [[T]]
  return %res : !PFv
}

// CHECK-LABEL: @test_lower_constant_tensor
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_lower_constant_tensor() -> !PFv {
  // CHECK-NOT: field.pf.constant
  // CHECK: %[[C0:.*]] = mod_arith.constant 5 : [[C:.*]]
  %c0 = field.pf.constant 5:  !PF1
  // CHECK: %[[RES:.*]] = tensor.from_elements %[[C0]], %[[C0]], %[[C0]], %[[C0]] : [[T]]
  %res = tensor.from_elements %c0, %c0, %c0, %c0 : !PFv
  // CHECK: return %[[RES]] : [[T]]
  return %res : !PFv
}
