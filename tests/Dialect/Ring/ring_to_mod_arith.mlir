// RUN: prime-ir-opt %s -ring-to-mod-arith \
// RUN:   | FileCheck %s --implicit-check-not=arith.remui --implicit-check-not=arith.muli

// base_convert from RNS basis [3, 5] (Q = 15) to [7], degree N = 2.
// Hand-computed CRT fast-basis-extension constants:
//   yHatInv = [ (15/3)^-1 mod 3 = 2, (15/5)^-1 mod 5 = 2 ]
//   table   = [ 15/3 mod 7 = 5,  15/5 mod 7 = 3 ]
// The lowering must route every reduction through mod_arith (Barrett), so the
// implicit-check-not above forbids arith.remui / arith.muli anywhere.

// CHECK-LABEL: func.func @bconv
// CHECK-SAME: (%{{.*}}: tensor<2x2xi64>) -> tensor<1x2xi64>
func.func @bconv(%x: !ring.rq<[3, 5], 2 : i32>) -> !ring.rq<[7], 2 : i32> {
  // Residues become mod_arith.int; multiplies/adds are mod_arith (Barrett).
  // CHECK-DAG: mod_arith.constant
  // CHECK-DAG: mod_arith.bitcast
  // CHECK-DAG: mod_arith.mul
  // CHECK-DAG: mod_arith.add
  // CHECK-DAG: tensor.extract_slice
  // CHECK-DAG: tensor.insert_slice
  // CHECK: return %{{.*}} : tensor<1x2xi64>
  %y = ring.base_convert %x : !ring.rq<[3, 5], 2 : i32> to !ring.rq<[7], 2 : i32>
  return %y : !ring.rq<[7], 2 : i32>
}
