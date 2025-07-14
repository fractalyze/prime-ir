// RUN: zkir-opt -field-to-mod-arith -canonicalize %s | FileCheck %s -enable-var-scope

!PF = !field.pf<7:i32>

#beta = #field.pf.elem<6:i32> : !PF

!QF = !field.f2<!PF, #beta>

// CHECK-LABEL: @test_mul
// CHECK-SAME: (%[[ARG0:.*]]: !z7_i32, %[[ARG1:.*]]: !z7_i32, %[[ARG2:.*]]: !z7_i32, %[[ARG3:.*]]: !z7_i32) -> (!z7_i32, !z7_i32)
func.func @test_mul(%a: !QF, %b: !QF) -> !QF {
  // CHECK: %[[MUL0:.*]] = mod_arith.mul %[[ARG0]], %[[ARG2]] : !z7_i32
  // CHECK: %[[MUL1:.*]] = mod_arith.mul %[[ARG1]], %[[ARG3]] : !z7_i32
  // CHECK: %[[SUB:.*]] = mod_arith.sub %[[MUL0]], %[[MUL1]] : !z7_i32
  // CHECK: %[[ADD0:.*]] = mod_arith.add %[[ARG0]], %[[ARG1]] : !z7_i32
  // CHECK: %[[ADD1:.*]] = mod_arith.add %[[ARG2]], %[[ARG3]] : !z7_i32
  // CHECK: %[[MUL2:.*]] = mod_arith.mul %[[ADD0]], %[[ADD1]] : !z7_i32
  // CHECK: %[[SUB2:.*]] = mod_arith.sub %[[MUL2]], %[[MUL0]] : !z7_i32
  // CHECK: %[[SUB3:.*]] = mod_arith.sub %[[SUB2]], %[[MUL1]] : !z7_i32
  %mul = field.mul %a, %b : !QF
  // CHECK: return %[[SUB]], %[[SUB3]] : !z7_i32, !z7_i32
  return %mul: !QF
}

// CHECK-LABEL: @test_square
// CHECK-SAME: (%[[ARG0:.*]]: !z7_i32, %[[ARG1:.*]]: !z7_i32) -> (!z7_i32, !z7_i32)
func.func @test_square(%a: !QF) -> !QF {
  // CHECK: %[[SUB:.*]] = mod_arith.sub %[[ARG0]], %[[ARG1]] : !z7_i32
  // CHECK: %[[ADD:.*]] = mod_arith.add %[[ARG0]], %[[ARG1]] : !z7_i32
  // CHECK: %[[MUL:.*]] = mod_arith.mul %[[ARG0]], %[[ARG1]] : !z7_i32
  // CHECK: %[[MUL2:.*]] = mod_arith.mul %[[SUB]], %[[ADD]] : !z7_i32
  // CHECK: %[[DOUBLE:.*]] = mod_arith.double %[[MUL]] : !z7_i32
  %square = field.square %a : !QF
  // CHECK: return %[[MUL2]], %[[DOUBLE]] : !z7_i32, !z7_i32
  return %square: !QF
}

// CHECK-LABEL: @test_inverse
// CHECK-SAME: (%[[ARG0:.*]]: !z7_i32, %[[ARG1:.*]]: !z7_i32) -> (!z7_i32, !z7_i32)
func.func @test_inverse(%a: !QF) -> !QF {
  // CHECK: %[[SQUARE0:.*]] = mod_arith.square %[[ARG0]] : !z7_i32
  // CHECK: %[[SQUARE1:.*]] = mod_arith.square %[[ARG1]] : !z7_i32
  // CHECK: %[[ADD:.*]] = mod_arith.add %[[SQUARE0]], %[[SQUARE1]] : !z7_i32
  // CHECK: %[[INV:.*]] = mod_arith.inverse %[[ADD]] : !z7_i32
  // CHECK: %[[MUL:.*]] = mod_arith.mul %[[ARG0]], %[[INV]] : !z7_i32
  // CHECK: %[[NEG:.*]] = mod_arith.negate %[[ARG1]] : !z7_i32
  // CHECK: %[[MUL2:.*]] = mod_arith.mul %[[NEG]], %[[INV]] : !z7_i32
  %inverse = field.inverse %a : !QF
  // CHECK: return %[[MUL]], %[[MUL2]] : !z7_i32, !z7_i32
  return %inverse: !QF
}
