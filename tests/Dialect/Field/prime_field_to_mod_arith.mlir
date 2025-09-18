// RUN: zkir-opt -field-to-mod-arith -split-input-file %s | FileCheck %s -enable-var-scope
!PF1 = !field.pf<97:i32>
!PF1m = !field.pf<97:i32, true>
!PFv = tensor<4x!PF1>
!PFmv = tensor<4x!PF1m>
#root_elem = #field.pf.elem<96:i32> : !PF1
#root = #field.root_of_unity<#root_elem, 2>

#mont = #mod_arith.montgomery<97:i32>

// CHECK-LABEL: @test_lower_constant
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_lower_constant() -> !PF1 {
  // CHECK: %[[RES:.*]] = mod_arith.constant 4 : [[T]]
  %res = field.constant 4 : !PF1
  // CHECK: return %[[RES]] : [[T]]
  return %res : !PF1
}

// CHECK-LABEL: @test_lower_bitcast
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]]) -> [[F:.*]] {
func.func @test_lower_bitcast(%lhs : !PF1) -> i32 {
  // CHECK-NOT: field.bitcast
  // CHECK: %[[RES:.*]] = mod_arith.bitcast %[[LHS]] : [[T]] -> [[F]]
  %res = field.bitcast %lhs : !PF1 -> i32
  // CHECK: return %[[RES]] : [[F]]
  return %res : i32
}

// CHECK-LABEL: @test_lower_bitcast_vec
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]]) -> [[F:.*]] {
func.func @test_lower_bitcast_vec(%lhs : !PFv) -> tensor<4xi32> {
  // CHECK-NOT: field.bitcast
  // CHECK: %[[RES:.*]] = mod_arith.bitcast %[[LHS]] : [[T]] -> [[F]]
  %res = field.bitcast %lhs : !PFv -> tensor<4xi32>
  // CHECK: return %[[RES]] : [[F]]
  return %res : tensor<4xi32>
}

// CHECK-LABEL: @test_lower_to_mont
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]]) -> [[Tm:.*]] {
func.func @test_lower_to_mont(%lhs : !PF1) -> !PF1m {
  // CHECK-NOT: field.to_mont
  // CHECK: %[[RES:.*]] = mod_arith.to_mont %[[LHS]] : [[Tm]]
  %res = field.to_mont %lhs : !PF1m
  // CHECK: return %[[RES]] : [[Tm]]
  return %res : !PF1m
}

// CHECK-LABEL: @test_lower_to_mont_vec
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]]) -> [[Tm:.*]] {
func.func @test_lower_to_mont_vec(%lhs : !PFv) -> !PFmv {
  // CHECK-NOT: field.to_mont
  // CHECK: %[[RES:.*]] = mod_arith.to_mont %[[LHS]] : [[Tm]]
  %res = field.to_mont %lhs : !PFmv
  // CHECK: return %[[RES]] : [[Tm]]
  return %res : !PFmv
}

// CHECK-LABEL: @test_lower_from_mont
// CHECK-SAME: (%[[LHS:.*]]: [[Tm:.*]]) -> [[T:.*]] {
func.func @test_lower_from_mont(%lhs : !PF1m) -> !PF1 {
  // CHECK-NOT: field.from_mont
  // CHECK: %[[RES:.*]] = mod_arith.from_mont %[[LHS]] : [[T]]
  %res = field.from_mont %lhs : !PF1
  // CHECK: return %[[RES]] : [[T]]
  return %res : !PF1
}

// CHECK-LABEL: @test_lower_from_mont_vec
// CHECK-SAME: (%[[LHS:.*]]: [[Tm:.*]]) -> [[T:.*]] {
func.func @test_lower_from_mont_vec(%lhs : !PFmv) -> !PFv {
  // CHECK-NOT: field.from_mont
  // CHECK: %[[RES:.*]] = mod_arith.from_mont %[[LHS]] : [[T]]
  %res = field.from_mont %lhs : !PFv
  // CHECK: return %[[RES]] : [[T]]
  return %res : !PFv
}

// CHECK-LABEL: @test_lower_inverse
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_lower_inverse(%lhs : !PF1) -> !PF1 {
  // CHECK-NOT: field.inverse
  // CHECK: %[[RES:.*]] = mod_arith.inverse %[[LHS]] : [[T]]
  %res = field.inverse %lhs : !PF1
  return %res : !PF1
}

// CHECK-LABEL: @test_lower_inverse_vec
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_lower_inverse_vec(%lhs : !PFv) -> !PFv {
  // CHECK-NOT: field.inverse
  // CHECK: %[[RES:.*]] = mod_arith.inverse %[[LHS]] : [[T]]
  %res = field.inverse %lhs : !PFv
  return %res : !PFv
}

// CHECK-LABEL: @test_lower_negate
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_lower_negate(%lhs : !PF1) -> !PF1 {
  // CHECK-NOT: field.negate
  // CHECK: %[[RES:.*]] = mod_arith.negate %[[LHS]] : [[T]]
  %res = field.negate %lhs : !PF1
  return %res : !PF1
}

// CHECK-LABEL: @test_lower_negate_vec
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_lower_negate_vec(%lhs : !PF1) -> !PF1 {
  // CHECK-NOT: field.negate
  // CHECK: %[[RES:.*]] = mod_arith.negate %[[LHS]] : [[T]]
  %res = field.negate %lhs : !PF1
  return %res : !PF1
}

// CHECK-LABEL: @test_lower_add
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_lower_add() -> !PF1 {
  // CHECK: %[[C0:.*]] = mod_arith.constant 4 : [[T]]
  %c0 = field.constant 4 : !PF1
  // CHECK: %[[RES:.*]] = mod_arith.add %[[C0]], %[[C0]] : [[T]]
  %res = field.add %c0, %c0 : !PF1
  // CHECK: return %[[RES]] : [[T]]
  return %res : !PF1
}

// CHECK-LABEL: @test_lower_add_vec
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_add_vec(%lhs : !PFv, %rhs : !PFv) -> !PFv {
  // CHECK-NOT: field.add
  // CHECK: %[[RES:.*]] = mod_arith.add %[[LHS]], %[[RHS]] : [[T]]
  %res = field.add %lhs, %rhs : !PFv
  // CHECK: return %[[RES]] : [[T]]
  return %res : !PFv
}

// CHECK-LABEL: @test_lower_double
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_lower_double() -> !PF1 {
  // CHECK: %[[C0:.*]] = mod_arith.constant 4 : [[T]]
  %c0 = field.constant 4 : !PF1
  // CHECK: %[[RES:.*]] = mod_arith.double %[[C0]] : [[T]]
  %res = field.double %c0 : !PF1
  // CHECK: return %[[RES]] : [[T]]
  return %res : !PF1
}

// CHECK-LABEL: @test_lower_double_vec
// CHECK-SAME: (%[[VAL:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_lower_double_vec(%val : !PFv) -> !PFv {
  // CHECK-NOT: field.double
  // CHECK: %[[RES:.*]] = mod_arith.double %[[VAL]] : [[T]]
  %res = field.double %val : !PFv
  // CHECK: return %[[RES]] : [[T]]
  return %res : !PFv
}

// CHECK-LABEL: @test_lower_sub
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_lower_sub() -> !PF1 {
  // CHECK: %[[C0:.*]] = mod_arith.constant 4 : [[T]]
  %c0 = field.constant 4 : !PF1
  // CHECK: %[[C1:.*]] = mod_arith.constant 5 : [[T]]
  %c1 = field.constant 5 : !PF1
  // CHECK: %[[RES:.*]] = mod_arith.sub %[[C0]], %[[C1]] : [[T]]
  %res = field.sub %c0, %c1 : !PF1
  // CHECK: return %[[RES]] : [[T]]
  return %res : !PF1
}

// CHECK-LABEL: @test_lower_sub_vec
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_sub_vec(%lhs : !PFv, %rhs : !PFv) -> !PFv {
  // CHECK-NOT: field.sub
  // CHECK: %[[RES:.*]] = mod_arith.sub %[[LHS]], %[[RHS]] : [[T]]
  %res = field.sub %lhs, %rhs : !PFv
  // CHECK: return %[[RES]] : [[T]]
  return %res : !PFv
}

// CHECK-LABEL: @test_lower_mul
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_lower_mul() -> !PF1 {
  // CHECK: %[[C0:.*]] = mod_arith.constant 4 : [[T]]
  %c0 = field.constant 4 : !PF1
  // CHECK: %[[RES:.*]] = mod_arith.mul %[[C0]], %[[C0]] : [[T]]
  %res = field.mul %c0, %c0 : !PF1
  // CHECK: return %[[RES]] : [[T]]
  return %res : !PF1
}

// CHECK-LABEL: @test_lower_mul_vec
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_mul_vec(%lhs : !PFv, %rhs : !PFv) -> !PFv {
  // CHECK-NOT: field.mul
  // CHECK: %[[RES:.*]] = mod_arith.mul %[[LHS]], %[[RHS]] : [[T]]
  %res = field.mul %lhs, %rhs : !PFv
  // CHECK: return %[[RES]] : [[T]]
  return %res : !PFv
}

// CHECK-LABEL: @test_lower_square
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_lower_square() -> !PF1 {
  // CHECK: %[[C0:.*]] = mod_arith.constant 4 : [[T]]
  %c0 = field.constant 4 : !PF1
  // CHECK: %[[RES:.*]] = mod_arith.square %[[C0]] : [[T]]
  %res = field.square %c0 : !PF1
  // CHECK: return %[[RES]] : [[T]]
  return %res : !PF1
}

// CHECK-LABEL: @test_lower_square_vec
// CHECK-SAME: (%[[VAL:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_lower_square_vec(%val : !PFv) -> !PFv {
  // CHECK-NOT: field.square
  // CHECK: %[[RES:.*]] = mod_arith.square %[[VAL]] : [[T]]
  %res = field.square %val : !PFv
  // CHECK: return %[[RES]] : [[T]]
  return %res : !PFv
}

// CHECK-LABEL: @test_lower_cmp
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]]) {
func.func @test_lower_cmp(%lhs : !PF1) {
  // CHECK: %[[RHS:.*]] = mod_arith.constant 5 : [[T]]
  %rhs = field.constant 5:  !PF1
  // CHECK-NOT: field.cmp
  // %[[EQUAL:.*]] = arith.cmpi [[eq:.*]], %[[LHS]], %[[RHS]] : [[i32]]
  // %[[NOTEQUAL:.*]] = arith.cmpi [[ne:.*]], %[[LHS]], %[[RHS]] : [[i32]]
  // %[[LESSTHAN:.*]] = arith.cmpi [[ult:.*]], %[[LHS]], %[[RHS]] : [[i32]]
  // %[[LESSTHANOREQUALS:.*]] = arith.cmpi [[ule:.*]], %[[LHS]], %[[RHS]] : [[i32]]
  // %[[GREATERTHAN:.*]] = arith.cmpi [[ugt:.*]], %[[LHS]], %[[RHS]] : [[i32]]
  // %[[GREATERTHANOREQUALS:.*]] = arith.cmpi [[uge:.*]], %[[LHS]], %[[RHS]] : [[i32]]
  %equal = field.cmp eq, %lhs, %rhs : !PF1
  %notEqual = field.cmp ne, %lhs, %rhs : !PF1
  %lessThan = field.cmp ult, %lhs, %rhs : !PF1
  %lessThanOrEquals = field.cmp ule, %lhs, %rhs : !PF1
  %greaterThan = field.cmp ugt, %lhs, %rhs : !PF1
  %greaterThanOrEquals = field.cmp uge, %lhs, %rhs : !PF1
  return
}
