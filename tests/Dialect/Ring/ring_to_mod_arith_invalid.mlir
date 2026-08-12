// A ring already occupies two tensor dimensions, so a rank-k module over it
// would land on a rank-(k+2) residue tensor that no pattern here builds. The
// type converter refuses the container rather than passing it through with the
// ring type intact and the pass reporting success.

// RUN: not prime-ir-opt %s -ring-to-mod-arith 2>&1 | FileCheck %s

// CHECK: failed to legalize
func.func @module_over_the_ring(%m: tensor<2x!ring.rq<[12289], 8 : i32>>)
    -> tensor<2x!ring.rq<[12289], 8 : i32>> {
  return %m : tensor<2x!ring.rq<[12289], 8 : i32>>
}
