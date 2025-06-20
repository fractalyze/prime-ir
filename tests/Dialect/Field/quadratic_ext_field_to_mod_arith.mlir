// RUN: zkir-opt -field-to-mod-arith -split-input-file %s | FileCheck %s -enable-var-scope
!mod = !mod_arith.int<7:i32>
#mont = #mod_arith.montgomery<!mod>
!PF = !field.pf<7:i32>
!PFm = !field.pf<7:i32, true>

#beta = #field.pf.elem<6:i32> : !PF
#beta_mont = #field.pf.elem<3:i32> : !PFm
!QF = !field.f2<!PF, #beta>
!QFm = !field.f2<!PFm, #beta_mont>
#ef = #field.f2.elem<#beta, #beta> : !QF

// CHECK-LABEL: @test_lower_inverse
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]]) -> ([[T]], [[T]]) {
func.func @test_lower_inverse(%arg0: !QF) -> !QF {
    // CHECK-NOT: field.inverse
    %0 = field.inverse %arg0 : !QF
    return %0 : !QF
}

// CHECK-LABEL: @test_lower_double
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]]) -> ([[T]], [[T]]) {
func.func @test_lower_double(%arg0: !QF) -> !QF {
    // CHECK-NOT: field.double
    %0 = field.double %arg0 : !QF
    return %0 : !QF
}

// CHECK-LABEL: @test_lower_square
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]]) -> ([[T]], [[T]]) {
func.func @test_lower_square(%arg0: !QF) -> !QF {
    // CHECK-NOT: field.square
    %0 = field.square %arg0 : !QF
    return %0 : !QF
}

// CHECK-LABEL: @test_lower_add
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]], %[[ARG2:.*]]: [[T]], %[[ARG3:.*]]: [[T]]) -> ([[T]], [[T]]) {
func.func @test_lower_add(%arg0: !QF, %arg1: !QF) -> !QF {
    // CHECK: %[[C0:.*]] = mod_arith.add %[[ARG0]], %[[ARG2]] : [[T]]
    // CHECK: %[[C1:.*]] = mod_arith.add %[[ARG1]], %[[ARG3]] : [[T]]
    // CHECK: return %[[C0]], %[[C1]] : [[T]], [[T]]
    %0 = field.add %arg0, %arg1 : !QF
    return %0 : !QF
}

// CHECK-LABEL: @test_lower_mul
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]], %[[ARG2:.*]]: [[T]], %[[ARG3:.*]]: [[T]]) -> ([[T]], [[T]]) {
func.func @test_lower_mul(%arg0: !QF, %arg1: !QF) -> !QF {
    // CHECK: %[[BETA:.*]] = mod_arith.constant 6 : [[T]]
    // CHECK: %[[V0:.*]] = mod_arith.mul %[[ARG0]], %[[ARG2]] : [[T]]
    // CHECK: %[[V1:.*]] = mod_arith.mul %[[ARG1]], %[[ARG3]] : [[T]]
    // CHECK: %[[BETATIMESV1:.*]] = mod_arith.mul %[[BETA]], %[[V1]] : [[T]]
    // CHECK: %[[C0:.*]] = mod_arith.add %[[V0]], %[[BETATIMESV1]] : [[T]]
    // CHECK: %[[SUMLHS:.*]] = mod_arith.add %[[ARG0]], %[[ARG1]] : [[T]]
    // CHECK: %[[SUMRHS:.*]] = mod_arith.add %[[ARG2]], %[[ARG3]] : [[T]]
    // CHECK: %[[SUMPRODUCT:.*]] = mod_arith.mul %[[SUMLHS]], %[[SUMRHS]] : [[T]]
    // CHECK: %[[TMP:.*]] = mod_arith.sub %[[SUMPRODUCT]], %[[V0]] : [[T]]
    // CHECK: %[[C1:.*]] = mod_arith.sub %[[TMP]], %[[V1]] : [[T]]
    // CHECK: return %[[C0]], %[[C1]] : [[T]], [[T]]
    %0 = field.mul %arg0, %arg1 : !QF
    return %0 : !QF
}

// CHECK-LABEL: @test_lower_from_elements
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]]) -> tensor<2x2x[[T]]> {
func.func @test_lower_from_elements(%arg0: !PF, %arg1: !PF) -> tensor<2x!QF> {
    %0 = field.f2.constant %arg0, %arg1 : !QF
    %1 = field.f2.constant %arg1, %arg0 : !QF
    tensor.from_elements %arg0, %arg1 : tensor<2x!PF>
    // CHECK: %[[TENSOR:.*]] = tensor.from_elements %[[ARG0]], %[[ARG1]], %[[ARG1]], %[[ARG0]] : tensor<2x2x[[T]]>
    %2 = tensor.from_elements %0, %1 : tensor<2x!QF>
    // CHECK: return %[[TENSOR]] : tensor<2x2x[[T]]>
    return %2 : tensor<2x!QF>
}

// CHECK-LABEL: @test_lower_from_mont
// CHECK-SAME: (%[[ARG0:.*]]: [[Tm:.*]], %[[ARG1:.*]]: [[Tm]]) -> ([[T:.*]], [[T:.*]]) {
func.func @test_lower_from_mont(%arg0: !QFm) -> !QF {
    %0 = field.from_mont %arg0 : !QF
    return %0 : !QF
}

// CHECK-LABEL: @test_lower_to_mont
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]]) -> ([[Tm:.*]], [[Tm]]) {
func.func @test_lower_to_mont(%arg0: !QF) -> !QFm {
    %0 = field.to_mont %arg0 : !QFm
    return %0 : !QFm
}

// CHECK-LABEL: @test_lower_tensor_extract
// CHECK-SAME: (%[[ARG0:.*]]: tensor<3x2x2x[[T:.*]]>) -> ([[T]], [[T]]) {
func.func @test_lower_tensor_extract(%arg0: tensor<3x2x!QF>) -> !QF {
    // CHECK: %[[I1:.*]] = arith.constant 1 : index
    %i1 = arith.constant 1 : index

    // CHECK: %[[C0:.*]] = arith.constant 0 : index
    // CHECK: %[[VALUE0:.*]] = tensor.extract %[[ARG0]][%[[I1]], %[[I1]], %[[C0]]] : tensor<3x2x2x[[T]]>
    // CHECK: %[[C1:.*]] = arith.constant 1 : index
    // CHECK: %[[VALUE1:.*]] = tensor.extract %[[ARG0]][%[[I1]], %[[I1]], %[[C1]]] : tensor<3x2x2x[[T]]>
    %1 = tensor.extract %arg0[%i1, %i1] : tensor<3x2x!QF>
    // CHECK: return %[[VALUE0]], %[[VALUE1]] : [[T]], [[T]]
    return %1 : !QF
}

// CHECK-LABEL: @test_lower_memref
// CHECK-SAME: (%[[ARG0:.*]]: memref<3x2x2x[[T:.*]]>) -> ([[T]], [[T]]) {
func.func @test_lower_memref(%arg0: memref<3x2x!QF>) -> !QF {
    %t = bufferization.to_tensor %arg0 : memref<3x2x!QF> to tensor<3x2x!QF>
    %i1 = arith.constant 1 : index
    %1 = tensor.extract %t[%i1, %i1] : tensor<3x2x!QF>
    return %1 : !QF
}
