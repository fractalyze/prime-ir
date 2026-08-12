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

// -----

// The permutation is static and the same in every limb, so the whole [L, N]
// tensor lowers to one constant index table and one gather -- not one op per
// coefficient, and not one nest per limb. At CKKS degrees (N = 2^15) an
// unrolled permute would emit hundreds of thousands of ops.
// CHECK-LABEL: func.func @automorphism_gathers_once
// CHECK: arith.constant dense<{{.*}}> : tensor<256xi32>
// CHECK: linalg.generic
// CHECK-NOT: linalg.generic
// CHECK-NOT: tensor.insert %
func.func @automorphism_gathers_once(
    %x: !ring.rq<[12289, 40961], 256 : i32, eval>)
    -> !ring.rq<[12289, 40961], 256 : i32, eval> {
  %y = ring.automorphism %x {exponent = 3 : i64}
      : !ring.rq<[12289, 40961], 256 : i32, eval>
  return %y : !ring.rq<[12289, 40961], 256 : i32, eval>
}
