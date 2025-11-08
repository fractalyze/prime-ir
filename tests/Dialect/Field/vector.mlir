// RUN: zkir-opt -field-to-mod-arith %s | FileCheck %s -enable-var-scope

!PF = !field.pf<7:i32, true>

// CHECK-LABEL: @test_vector_splat
// CHECK-SAME: (%[[INPUT:.*]]: [[INPUT_TYPE:.*]]) -> [[VEC:.*]] {
func.func @test_vector_splat(%input : !PF) -> vector<4x!PF> {
  // CHECK: %[[SPLAT:.*]] = vector.splat %[[INPUT]] : [[VEC]]
  %splat = vector.splat %input : vector<4x!PF>
  // CHECK: return %[[SPLAT]] : [[VEC]]
  return %splat : vector<4x!PF>
}
