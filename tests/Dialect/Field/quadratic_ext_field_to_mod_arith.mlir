// RUN: zkir-opt -field-to-mod-arith --split-input-file %s | FileCheck %s --enable-var-scope
!PF = !field.pf<7:i32>
#beta = #field.pf_elem<6:i32> : !PF
!QF = !field.f2<!PF, #beta>

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
