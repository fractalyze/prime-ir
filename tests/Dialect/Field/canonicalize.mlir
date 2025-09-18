// RUN: zkir-opt -field-to-mod-arith -canonicalize %s | FileCheck %s -enable-var-scope

!PF = !field.pf<7:i32>

#beta = #field.pf.elem<6:i32> : !PF

!QF = !field.f2<!PF, #beta>

// CHECK-LABEL: @test_mul
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]]) -> [[T]]
func.func @test_mul(%a: !QF, %b: !QF) -> !QF {
  // CHECK: %[[LHS:.*]]:2 = field.ext_to_coeffs %[[ARG0]] : ([[T]]) -> (!z7_i32, !z7_i32)
  // CHECK: %[[RHS:.*]]:2 = field.ext_to_coeffs %[[ARG1]] : ([[T]]) -> (!z7_i32, !z7_i32)
  // CHECK: %[[MUL0:.*]] = mod_arith.mul %[[LHS]]#0, %[[RHS]]#0 : !z7_i32
  // CHECK: %[[MUL1:.*]] = mod_arith.mul %[[LHS]]#1, %[[RHS]]#1 : !z7_i32
  // CHECK: %[[SUB:.*]] = mod_arith.sub %[[MUL0]], %[[MUL1]] : !z7_i32
  // CHECK: %[[ADD0:.*]] = mod_arith.add %[[LHS]]#0, %[[LHS]]#1 : !z7_i32
  // CHECK: %[[ADD1:.*]] = mod_arith.add %[[RHS]]#0, %[[RHS]]#1 : !z7_i32
  // CHECK: %[[MUL2:.*]] = mod_arith.mul %[[ADD0]], %[[ADD1]] : !z7_i32
  // CHECK: %[[SUB2:.*]] = mod_arith.sub %[[MUL2]], %[[MUL0]] : !z7_i32
  // CHECK: %[[SUB3:.*]] = mod_arith.sub %[[SUB2]], %[[MUL1]] : !z7_i32
  // CHECK: %[[RESULT:.*]] = field.ext_from_coeffs %[[SUB]], %[[SUB3]] : (!z7_i32, !z7_i32) -> [[T]]
  %mul = field.mul %a, %b : !QF
  // CHECK: return %[[RESULT]] : [[T]]
  return %mul: !QF
}

// CHECK-LABEL: @test_square
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_square(%a: !QF) -> !QF {
  // CHECK: %[[COEFFS:.*]]:2 = field.ext_to_coeffs %[[ARG0]] : ([[T]]) -> (!z7_i32, !z7_i32)
  // CHECK: %[[SUB:.*]] = mod_arith.sub %[[COEFFS]]#0, %[[COEFFS]]#1 : !z7_i32
  // CHECK: %[[ADD:.*]] = mod_arith.add %[[COEFFS]]#0, %[[COEFFS]]#1 : !z7_i32
  // CHECK: %[[MUL:.*]] = mod_arith.mul %[[COEFFS]]#0, %[[COEFFS]]#1 : !z7_i32
  // CHECK: %[[MUL2:.*]] = mod_arith.mul %[[SUB]], %[[ADD]] : !z7_i32
  // CHECK: %[[DOUBLE:.*]] = mod_arith.double %[[MUL]] : !z7_i32
  // CHECK: %[[RESULT:.*]] = field.ext_from_coeffs %[[MUL2]], %[[DOUBLE]] : (!z7_i32, !z7_i32) -> [[T]]
  %square = field.square %a : !QF
  // CHECK: return %[[RESULT]] : [[T]]
  return %square: !QF
}

// CHECK-LABEL: @test_inverse
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_inverse(%a: !QF) -> !QF {
  // CHECK: %[[COEFFS:.*]]:2 = field.ext_to_coeffs %[[ARG0]] : ([[T]]) -> (!z7_i32, !z7_i32)
  // CHECK: %[[SQUARE0:.*]] = mod_arith.square %[[COEFFS]]#0 : !z7_i32
  // CHECK: %[[SQUARE1:.*]] = mod_arith.square %[[COEFFS]]#1 : !z7_i32
  // CHECK: %[[ADD:.*]] = mod_arith.add %[[SQUARE0]], %[[SQUARE1]] : !z7_i32
  // CHECK: %[[INV:.*]] = mod_arith.inverse %[[ADD]] : !z7_i32
  // CHECK: %[[MUL:.*]] = mod_arith.mul %[[COEFFS]]#0, %[[INV]] : !z7_i32
  // CHECK: %[[NEG:.*]] = mod_arith.negate %[[COEFFS]]#1 : !z7_i32
  // CHECK: %[[MUL2:.*]] = mod_arith.mul %[[NEG]], %[[INV]] : !z7_i32
  // CHECK: %[[RESULT:.*]] = field.ext_from_coeffs %[[MUL]], %[[MUL2]] : (!z7_i32, !z7_i32) -> [[T]]
  %inverse = field.inverse %a : !QF
  // CHECK: return %[[RESULT]] : [[T]]
  return %inverse: !QF
}
