// RUN: zkir-opt -elliptic-curve-to-field %s | FileCheck %s -enable-var-scope

!PF = !field.pf<97:i32>

#1 = #field.pf.elem<1:i32> : !PF
#2 = #field.pf.elem<2:i32> : !PF
#3 = #field.pf.elem<3:i32> : !PF
#4 = #field.pf.elem<4:i32> : !PF

#curve = #elliptic_curve.sw<#1, #2, (#3, #4)>
!affine = !elliptic_curve.affine<#curve>

// CHECK-LABEL: @test_tensor_extract
// CHECK-SAME: (%[[INPUT:.*]]: [[INPUT_TYPE:.*]]) -> ([[T:.*]], [[T:.*]]) {
func.func @test_tensor_extract(%input : tensor<4x!affine>) -> !affine {
  %c0 = arith.constant 0 : index
  // CHECK: %[[RESULT0:.*]] = tensor.extract %[[INPUT]][%[[C0:.*]], %[[C0_0:.*]]] : [[INPUT_TYPE]]
  // CHECK: %[[RESULT1:.*]] = tensor.extract %[[INPUT]][%[[C0:.*]], %[[C1:.*]]] : [[INPUT_TYPE]]
  %result = tensor.extract %input[%c0] : tensor<4x!affine>
  // CHECK: return %[[RESULT0]], %[[RESULT1]] : [[T]], [[T]]
  return %result : !affine
}

// CHECK-LABEL: @test_tensor_from_elements
// CHECK-SAME: (%[[ELEM0:.*]]: [[ELEM_TYPE:.*]], %[[ELEM1:.*]]: [[ELEM_TYPE:.*]]) -> [[T:.*]] {
func.func @test_tensor_from_elements(%point : !affine) -> tensor<2x!affine> {
  // CHECK: %[[TENSOR:.*]] = tensor.from_elements %[[ELEM0]], %[[ELEM1]], %[[ELEM0]], %[[ELEM1]] : [[T]]
  %tensor = tensor.from_elements %point, %point : tensor<2x!affine>
  // CHECK: return %[[TENSOR]] : [[T]]
  return %tensor : tensor<2x!affine>
}
