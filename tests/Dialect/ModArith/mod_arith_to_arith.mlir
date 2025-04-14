// RUN: zkir-opt -mod-arith-to-arith --split-input-file %s | FileCheck %s --enable-var-scope

!Zp = !mod_arith.int<65537 : i32>
!Zpv = tensor<4x!Zp>

// CHECK-LABEL: @test_lower_constant
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_lower_constant() -> !mod_arith.int<3 : i5> {
  // CHECK-NOT: mod_arith.constant
  // CHECK: %[[CVAL:.*]] = arith.constant 2 : [[T]]
  // CHECK: return %[[CVAL]] : [[T]]
  %res = mod_arith.constant 5:  !mod_arith.int<3 : i5>
  return %res: !mod_arith.int<3 : i5>
}

// CHECK-LABEL: @test_lower_encapsulate
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_lower_encapsulate(%lhs : i32) -> !Zp {
  // CHECK-NOT: mod_arith.encapsulate
  // CHECK: return %[[LHS]] : [[T]]
  %res = mod_arith.encapsulate %lhs: i32 -> !Zp
  return %res : !Zp
}

// CHECK-LABEL: @test_lower_encapsulate_vec
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_lower_encapsulate_vec(%lhs : tensor<4xi32>) -> !Zpv {
  // CHECK-NOT: mod_arith.encapsulate
  // CHECK: return %[[LHS]] : [[T]]
  %res = mod_arith.encapsulate %lhs: tensor<4xi32> -> !Zpv
  return %res : !Zpv
}

// CHECK-LABEL: @test_lower_extract
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_lower_extract(%lhs : !Zp) -> i32 {
  // CHECK-NOT: mod_arith.extract
  // CHECK: return %[[LHS]] : [[T]]
  %res = mod_arith.extract %lhs: !Zp -> i32
  return %res : i32
}

// CHECK-LABEL: @test_lower_extract_vec
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_lower_extract_vec(%lhs : !Zpv) -> tensor<4xi32> {
  // CHECK-NOT: mod_arith.extract
  // CHECK: return %[[LHS]] : [[T]]
  %res = mod_arith.extract %lhs: !Zpv -> tensor<4xi32>
  return %res : tensor<4xi32>
}

// CHECK-LABEL: @test_lower_reduce
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_lower_reduce(%lhs : !Zp) -> !Zp {
  // CHECK-NOT: mod_arith.reduce
  // CHECK: %[[CMOD:.*]] = arith.constant 65537 : [[T]]
  // CHECK: %[[REMS:.*]] = arith.remsi %[[LHS]], %[[CMOD]] : [[T]]
  // CHECK: %[[ADD:.*]] = arith.addi %[[REMS]], %[[CMOD]] : [[T]]
  // CHECK: %[[REM:.*]] = arith.remui %[[ADD]], %[[CMOD]] : [[T]]
  // CHECK: return %[[REM]] : [[T]]
  %res = mod_arith.reduce %lhs: !Zp
  return %res : !Zp
}

// CHECK-LABEL: @test_lower_reduce_vec
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_lower_reduce_vec(%lhs : !Zpv) -> !Zpv {
  // CHECK-NOT: mod_arith.reduce
  // CHECK: %[[CMOD:.*]] = arith.constant dense<65537> : [[T]]
  // CHECK: %[[REMS:.*]] = arith.remsi %[[LHS]], %[[CMOD]] : [[T]]
  // CHECK: %[[ADD:.*]] = arith.addi %[[REMS]], %[[CMOD]] : [[T]]
  // CHECK: %[[REM:.*]] = arith.remui %[[ADD]], %[[CMOD]] : [[T]]
  // CHECK: return %[[REM]] : [[T]]
  %res = mod_arith.reduce %lhs: !Zpv
  return %res : !Zpv
}

// CHECK-LABEL: @test_lower_inverse
// CHECK-SAME: (%[[INPUT:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_lower_inverse(%lhs : !Zp) -> !Zp {
  // CHECK-NOT: mod_arith.inverse
  // CHECK: %[[C0:.*]] = arith.constant 0 : [[T]]
  // CHECK: %[[C1:.*]] = arith.constant 1 : [[T]]
  // CHECK: %[[CMOD:.*]] = arith.constant 65537 : [[T]]
  // CHECK:      %[[WHILE_RES:.*]]:4 = scf.while (%[[ARG1:.*]] = %[[CMOD]], %[[ARG2:.*]] = %[[INPUT]], %[[ARG3:.*]] = %[[C0]], %[[ARG4:.*]] = %[[C1]]) : ([[T]], [[T]], [[T]], [[T]]) -> ([[T]], [[T]], [[T]], [[T]]) {
  // CHECK-NEXT:   %[[COND:.*]] = arith.cmpi ne, %[[ARG2]], %[[C0]] : [[T]]
  // CHECK-NEXT:   scf.condition(%[[COND]]) %[[ARG1]], %[[ARG2]], %[[ARG3]], %[[ARG4]] : [[T]], [[T]], [[T]], [[T]]
  // CHECK-NEXT: } do {
  // CHECK-NEXT: ^bb0(%[[ARG1_LOOP:.*]]: [[T]], %[[ARG2_LOOP:.*]]: [[T]], %[[ARG3_LOOP:.*]]: [[T]], %[[ARG4_LOOP:.*]]: [[T]]):
  // CHECK-NEXT:   %[[DIV:.*]] = arith.divui %[[ARG1_LOOP]], %[[ARG2_LOOP]] : [[T]]
  // CHECK-NEXT:   %[[REM:.*]] = arith.remui %[[ARG1_LOOP]], %[[ARG2_LOOP]] : [[T]]
  // CHECK-NEXT:   %[[MUL:.*]] = arith.muli %[[ARG4_LOOP]], %[[DIV]] : [[T]]
  // CHECK-NEXT:   %[[SUB:.*]] = arith.subi %[[ARG3_LOOP]], %[[MUL]] : [[T]]
  // CHECK-NEXT:   scf.yield %[[ARG2_LOOP]], %[[REM]], %[[ARG4_LOOP]], %[[SUB]] : [[T]], [[T]], [[T]], [[T]]
  // CHECK-NEXT: }
  // CHECK:      %[[EQ_COND:.*]] = arith.cmpi eq, %[[WHILE_RES]]#0, %[[C1]] : [[T]]
  // CHECK-NEXT: %[[IF_RES:.*]] = scf.if %[[EQ_COND]] -> ([[T]]) {
  // CHECK:      %[[SLT_COND:.*]] = arith.cmpi slt, %[[WHILE_RES]]#2, %[[C0]] : [[T]]
  // CHECK-NEXT: %[[INNER_IF_RES:.*]] = scf.if %[[SLT_COND]] -> ([[T]]) {
  // CHECK:      %[[C65537:.*]] = arith.constant 65537 : [[T]]
  // CHECK-NEXT: %[[ADD:.*]] = arith.addi %[[WHILE_RES]]#2, %[[C65537]] : [[T]]
  // CHECK-NEXT: scf.yield %[[ADD]] : [[T]]
  // CHECK:      scf.yield %[[WHILE_RES]]#2 : [[T]]
  // CHECK:      scf.yield %[[INNER_IF_RES]] : [[T]]
  // CHECK:      scf.yield %[[C0]] : [[T]]
  // CHECK:      return %[[IF_RES]] : [[T]]
  %res = mod_arith.inverse %lhs : !Zp
  return %res : !Zp
}

// CHECK-LABEL: @test_lower_add
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_add(%lhs : !Zp, %rhs : !Zp) -> !Zp {
  // CHECK-NOT: mod_arith.add
  // CHECK: %[[CMOD:.*]] = arith.constant 65537 : [[T]]
  // CHECK: %[[ADD:.*]] = arith.addi %[[LHS]], %[[RHS]] : [[T]]
  // CHECK: %[[IFGE:.*]] = arith.cmpi uge, %[[ADD]], %[[CMOD]] : [[T]]
  // CHECK: %[[SUB:.*]] = arith.subi %[[ADD]], %[[CMOD]] : [[T]]
  // CHECK: %[[REM:.*]] = arith.select %[[IFGE]], %[[SUB]], %[[ADD]] : [[T]]
  // CHECK: return %[[REM]] : [[T]]
  %res = mod_arith.add %lhs, %rhs : !Zp
  return %res : !Zp
}

// CHECK-LABEL: @test_lower_add_vec
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_add_vec(%lhs : !Zpv, %rhs : !Zpv) -> !Zpv {
  // CHECK-NOT: mod_arith.add
  // CHECK: %[[CMOD:.*]] = arith.constant dense<65537> : [[T]]
  // CHECK: %[[ADD:.*]] = arith.addi %[[LHS]], %[[RHS]] : [[T]]
  // CHECK: %[[IFGE:.*]] = arith.cmpi uge, %[[ADD]], %[[CMOD]] : [[T]]
  // CHECK: %[[SUB:.*]] = arith.subi %[[ADD]], %[[CMOD]] : [[T]]
  // CHECK: %[[REM:.*]] = arith.select %[[IFGE]], %[[SUB]], %[[ADD]] : tensor<4xi1>, [[T]]
  // CHECK: return %[[REM]] : [[T]]
  %res = mod_arith.add %lhs, %rhs : !Zpv
  return %res : !Zpv
}

// CHECK-LABEL: @test_lower_sub
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_sub(%lhs : !Zp, %rhs : !Zp) -> !Zp {
  // CHECK-NOT: mod_arith.sub
  // CHECK: %[[CMOD:.*]] = arith.constant 65537 : [[T]]
  // CHECK: %[[SUB:.*]] = arith.subi %[[LHS]], %[[RHS]] : [[T]]
  // CHECK: %[[ADD:.*]] = arith.addi %[[SUB]], %[[CMOD]] : [[T]]
  // CHECK: %[[IFGE:.*]] = arith.cmpi uge, %[[LHS]], %[[RHS]] : [[T]]
  // CHECK: %[[SELECT:.*]] = arith.select %[[IFGE]], %[[SUB]], %[[ADD]] : [[T]]
  // CHECK: return %[[SELECT]] : [[T]]
  %res = mod_arith.sub %lhs, %rhs : !Zp
  return %res : !Zp
}

// CHECK-LABEL: @test_lower_sub_vec
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_sub_vec(%lhs : !Zpv, %rhs : !Zpv) -> !Zpv {
  // CHECK-NOT: mod_arith.sub
  // CHECK: %[[CMOD:.*]] = arith.constant dense<65537> : [[T]]
  // CHECK: %[[SUB:.*]] = arith.subi %[[LHS]], %[[RHS]] : [[T]]
  // CHECK: %[[ADD:.*]] = arith.addi %[[SUB]], %[[CMOD]] : [[T]]
  // CHECK: %[[IFGE:.*]] = arith.cmpi uge, %[[LHS]], %[[RHS]] : [[T]]
  // CHECK: %[[SELECT:.*]] = arith.select %[[IFGE]], %[[SUB]], %[[ADD]] : tensor<4xi1>, [[T]]
  // CHECK: return %[[SELECT]] : [[T]]
  %res = mod_arith.sub %lhs, %rhs : !Zpv
  return %res : !Zpv
}

// CHECK-LABEL: @test_lower_mul
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_mul(%lhs : !Zp, %rhs : !Zp) -> !Zp {
  // CHECK-NOT: mod_arith.mul
  // CHECK: %[[CMOD:.*]] = arith.constant 65537 : [[TEXT:.*]]
  // CHECK: %[[EXT0:.*]] = arith.extui %[[LHS]] : [[T]] to [[TEXT]]
  // CHECK: %[[EXT1:.*]] = arith.extui %[[RHS]] : [[T]] to [[TEXT]]
  // CHECK: %[[MUL:.*]] = arith.muli %[[EXT0]], %[[EXT1]] : [[TEXT]]
  // CHECK: %[[REM:.*]] = arith.remui %[[MUL]], %[[CMOD]] : [[TEXT]]
  // CHECK: %[[TRUNC:.*]] = arith.trunci %[[REM]] : [[TEXT]] to [[T]]
  // CHECK: return %[[TRUNC]] : [[T]]
  %res = mod_arith.mul %lhs, %rhs : !Zp
  return %res : !Zp
}

// CHECK-LABEL: @test_lower_mul_vec
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_mul_vec(%lhs : !Zpv, %rhs : !Zpv) -> !Zpv {
  // CHECK-NOT: mod_arith.mul
  // CHECK: %[[CMOD:.*]] = arith.constant dense<65537> : [[TEXT:.*]]
  // CHECK: %[[EXT0:.*]] = arith.extui %[[LHS]] : [[T]] to [[TEXT]]
  // CHECK: %[[EXT1:.*]] = arith.extui %[[RHS]] : [[T]] to [[TEXT]]
  // CHECK: %[[MUL:.*]] = arith.muli %[[EXT0]], %[[EXT1]] : [[TEXT]]
  // CHECK: %[[REM:.*]] = arith.remui %[[MUL]], %[[CMOD]] : [[TEXT]]
  // CHECK: %[[TRUNC:.*]] = arith.trunci %[[REM]] : [[TEXT]] to [[T]]
  // CHECK: return %[[TRUNC]] : [[T]]
  %res = mod_arith.mul %lhs, %rhs : !Zpv
  return %res : !Zpv
}

// CHECK-LABEL: @test_lower_constant_tensor
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_lower_constant_tensor() -> !Zpv {
  // CHECK-NOT: mod_arith.constant
  // CHECK: %[[C0:.*]] = arith.constant 5 : [[INT:.*]]
  %c0 = mod_arith.constant 5:  !Zp
  // CHECK: %[[RES:.*]] = tensor.from_elements %[[C0]], %[[C0]], %[[C0]], %[[C0]] : [[T]]
  %res = tensor.from_elements %c0, %c0, %c0, %c0 : !Zpv
  // CHECK: return %[[RES]] : [[T]]
  return %res : !Zpv
}

// CHECK-LABEL: @test_lower_mac
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]], %[[ACC:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_mac(%lhs : !Zp, %rhs : !Zp, %acc : !Zp) -> !Zp {
  // CHECK-NOT: mod_arith.mac
  // CHECK: %[[CMOD:.*]] = arith.constant 65537 : [[TEXT:.*]]
  // CHECK: %[[EXT0:.*]] = arith.extui %[[LHS]] : [[T]] to [[TEXT]]
  // CHECK: %[[EXT1:.*]] = arith.extui %[[RHS]] : [[T]] to [[TEXT]]
  // CHECK: %[[EXT2:.*]] = arith.extui %[[ACC]] : [[T]] to [[TEXT]]
  // CHECK: %[[MUL:.*]] = arith.muli %[[EXT0]], %[[EXT1]] : [[TEXT]]
  // CHECK: %[[ADD:.*]] = arith.addi %[[MUL]], %[[EXT2]] : [[TEXT]]
  // CHECK: %[[REM:.*]] = arith.remui %[[ADD]], %[[CMOD]] : [[TEXT]]
  // CHECK: %[[TRUNC:.*]] = arith.trunci %[[REM]] : [[TEXT]] to [[T]]
  // CHECK: return %[[TRUNC]] : [[T]]
  %res = mod_arith.mac %lhs, %rhs, %acc : !Zp
  return %res : !Zp
}

// CHECK-LABEL: @test_lower_mac_vec
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]], %[[ACC:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_mac_vec(%lhs : !Zpv, %rhs : !Zpv, %acc : !Zpv) -> !Zpv {
  // CHECK-NOT: mod_arith.mac
  // CHECK: %[[CMOD:.*]] = arith.constant dense<65537> : [[TEXT:.*]]
  // CHECK: %[[EXT0:.*]] = arith.extui %[[LHS]] : [[T]] to [[TEXT]]
  // CHECK: %[[EXT1:.*]] = arith.extui %[[RHS]] : [[T]] to [[TEXT]]
  // CHECK: %[[EXT2:.*]] = arith.extui %[[ACC]] : [[T]] to [[TEXT]]
  // CHECK: %[[MUL:.*]] = arith.muli %[[EXT0]], %[[EXT1]] : [[TEXT]]
  // CHECK: %[[ADD:.*]] = arith.addi %[[MUL]], %[[EXT2]] : [[TEXT]]
  // CHECK: %[[REM:.*]] = arith.remui %[[ADD]], %[[CMOD]] : [[TEXT]]
  // CHECK: %[[TRUNC:.*]] = arith.trunci %[[REM]] : [[TEXT]] to [[T]]
  // CHECK: return %[[TRUNC]] : [[T]]
  %res = mod_arith.mac %lhs, %rhs, %acc : !Zpv
  return %res : !Zpv
}

// -----

// CHECK-LABEL: @test_lower_subifge
// CHECK-SAME: (%[[LHS:.*]]: [[TENSOR_TYPE:.*]], %[[RHS:.*]]: [[TENSOR_TYPE]]) -> [[TENSOR_TYPE]] {
func.func @test_lower_subifge(%lhs : tensor<4xi8>, %rhs : tensor<4xi8>) -> tensor<4xi8> {

  // CHECK: %[[SUB:.*]] = arith.subi %[[LHS]], %[[RHS]] : [[TENSOR_TYPE]]
  // CHECK: %[[CMP:.*]] = arith.cmpi uge, %[[LHS]], %[[RHS]] : [[TENSOR_TYPE]]
  // CHECK: %[[RES:.*]] = arith.select %[[CMP]], %[[SUB]], %[[LHS]] : tensor<4xi1>, [[TENSOR_TYPE]]
  %res = mod_arith.subifge %lhs, %rhs: tensor<4xi8>
  // CHECK: return %[[RES]] : [[TENSOR_TYPE]]
  return %res : tensor<4xi8>
}

// -----

// CHECK-LABEL: @test_lower_subifge_int
// CHECK-SAME: (%[[LHS:.*]]: [[INT_TYPE:.*]], %[[RHS:.*]]: [[INT_TYPE]]) -> [[INT_TYPE]] {
func.func @test_lower_subifge_int(%lhs : i8, %rhs : i8) -> i8 {

  // CHECK: %[[SUB:.*]] = arith.subi %[[LHS]], %[[RHS]] : [[INT_TYPE]]
  // CHECK: %[[CMP:.*]] = arith.cmpi uge, %[[LHS]], %[[RHS]] : [[INT_TYPE]]
  // CHECK: %[[RES:.*]] = arith.select %[[CMP]], %[[SUB]], %[[LHS]] : [[INT_TYPE]]
  %res = mod_arith.subifge %lhs, %rhs: i8
  // CHECK: return %[[RES]] : [[INT_TYPE]]
  return %res : i8
}
