// RUN: zkir-opt -mod-arith-to-arith %s | FileCheck %s -enable-var-scope

!mod = !mod_arith.int<7:i32, true>

// CHECK-LABEL: @test_vector_splat
// CHECK-SAME: (%[[INPUT:.*]]: [[INPUT_TYPE:.*]]) -> [[VEC:.*]] {
func.func @test_vector_splat(%input : !mod) -> vector<4x!mod> {
  // CHECK: %[[SPLAT:.*]] = vector.splat %[[INPUT]] : [[VEC]]
  %splat = vector.splat %input : vector<4x!mod>
  // CHECK: return %[[SPLAT]] : [[VEC]]
  return %splat : vector<4x!mod>
}
