// RUN: emitters_opt %s -zkx-flatten-tensors | FileCheck %s

// This test reproduces a convergence failure in FlattenTensorsPass.
// When flattening vector<1x1xT> types in scf.for loops, the linearization
// map (d0, d1) -> (d0 + d1) with d0, d1 in [0, 0] causes a cycle between
// ApplyIndexingOp canonicalization patterns (SimplifyIndexingMap,
// FoldApplyIndexingResults, and others), exceeding applyPatternsGreedily's
// maxIterations=10.
//
// Related: //zkx/backends/gpu/codegen/emitters/tests:scatter/sorted_indices.hlo.test

// CHECK-LABEL: func.func @vector_1x1_for_loop
// CHECK-SAME:    vector<1xi32>
// CHECK-NOT:     vector<1x1xi32>
func.func @vector_1x1_for_loop(%v: vector<1x1xi32>, %val: i32)
    -> vector<1x1xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %result = scf.for %i = %c0 to %c1 step %c1
      iter_args(%acc = %v) -> (vector<1x1xi32>) {
    %inner = scf.for %j = %c0 to %c1 step %c1
        iter_args(%acc2 = %acc) -> (vector<1x1xi32>) {
      %elem = vector.extract %acc2[%i, %j] : i32 from vector<1x1xi32>
      %new = arith.addi %elem, %val : i32
      %inserted = vector.insert %new, %acc2 [%i, %j] : i32 into vector<1x1xi32>
      scf.yield %inserted : vector<1x1xi32>
    }
    scf.yield %inner : vector<1x1xi32>
  }
  return %result : vector<1x1xi32>
}
