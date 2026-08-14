// RUN: prime-ir-opt %s | prime-ir-opt | FileCheck %s

// base_convert extends a 2-limb RNS basis to 3 limbs (ModUp shape), same N.
// CHECK-LABEL: func.func @base_convert_extend
func.func @base_convert_extend(
    %x: !ring.rq<[1152921504606830593, 1152921504598720513], 4096 : i32>)
    -> !ring.rq<[1152921504606830593, 1152921504598720513, 1152921504597016577], 4096 : i32> {
  // Op printer strips the same-dialect `!ring.rq` prefix on operand/result types.
  // CHECK: ring.base_convert %{{.*}} : <[{{.*}}], 4096 : i32> to <[{{.*}}], 4096 : i32>
  %y = ring.base_convert %x
     : !ring.rq<[1152921504606830593, 1152921504598720513], 4096 : i32>
    to !ring.rq<[1152921504606830593, 1152921504598720513, 1152921504597016577], 4096 : i32>
  return %y : !ring.rq<[1152921504606830593, 1152921504598720513, 1152921504597016577], 4096 : i32>
}

// base_convert also models the rescale shape (drop a limb), same N.
// CHECK-LABEL: func.func @base_convert_drop
func.func @base_convert_drop(
    %x: !ring.rq<[12289, 40961], 8 : i32>) -> !ring.rq<[12289], 8 : i32> {
  // CHECK: ring.base_convert
  %y = ring.base_convert %x : !ring.rq<[12289, 40961], 8 : i32> to !ring.rq<[12289], 8 : i32>
  return %y : !ring.rq<[12289], 8 : i32>
}
