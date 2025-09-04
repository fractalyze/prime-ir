// RUN: cat %S/../../bn254_field_defs.mlir %S/../../bn254_ec_defs.mlir %s \
// RUN:   | zkir-opt -elliptic-curve-to-field \
// RUN:   | FileCheck %s -enable-var-scope

// CHECK-LABEL: @test_tensor_dim
// CHECK-SAME: (%[[INPUT:.*]]: [[INPUT_TYPE:.*]]) -> [[T:.*]] {
func.func @test_tensor_dim(%input : tensor<?x!affine>) -> index {
  %c0 = arith.constant 0 : index
  // CHECK: %[[DIM:.*]] = tensor.dim %[[INPUT]], %[[C0:.*]] : [[INPUT_TYPE]]
  %dim = tensor.dim %input, %c0 : tensor<?x!affine>
  // CHECK: return %[[DIM]] : [[T]]
  return %dim : index
}

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

// CHECK-LABEL: @test_tensor_extract_slice
// CHECK-SAME: (%[[INPUT:.*]]: [[INPUT_TYPE:.*]]) -> [[T:.*]] {
func.func @test_tensor_extract_slice(%input : tensor<4x!affine>) -> tensor<?x!affine> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  // CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[INPUT]][%[[C0:.*]]] [%[[C2:.*]]] [%[[C1:.*]]] : [[INPUT_TYPE]] to [[T]]
  %slice = tensor.extract_slice %input[%c0] [%c2] [%c1] : tensor<4x!affine> to tensor<?x!affine>
  // CHECK: return %[[SLICE]] : [[T]]
  return %slice : tensor<?x!affine>
}
