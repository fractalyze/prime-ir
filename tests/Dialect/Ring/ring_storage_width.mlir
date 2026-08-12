// RUN: prime-ir-opt %s --split-input-file | FileCheck %s
// RUN: prime-ir-opt %s --split-input-file -ring-to-mod-arith | FileCheck %s --check-prefix=LOWER

// RNS moduli are NTT-word primes that usually fit a 32-bit word, and a limb
// arrives typed by its own field dtype. The residue storage therefore has to
// be sayable on the ring; i64 is only the default.

// CHECK-LABEL: func.func @rq_i32_storage
// CHECK-SAME: !ring.rq<[12289, 40961], 8 : i32, i32, eval>
func.func @rq_i32_storage(%x: !ring.rq<[12289, 40961], 8 : i32, i32, eval>)
    -> !ring.rq<[12289, 40961], 8 : i32, i32, eval> {
  %y = ring.mul %x, %x : !ring.rq<[12289, 40961], 8 : i32, i32, eval>
  return %y : !ring.rq<[12289, 40961], 8 : i32, i32, eval>
}

// -----

// Omitting the storage keeps the i64 spelling, so nothing that predates this
// parameter has to say it.
// CHECK-LABEL: func.func @rq_default_storage
// CHECK-SAME: !ring.rq<[12289], 8 : i32>
func.func @rq_default_storage(%x: !ring.rq<[12289], 8 : i32>)
    -> !ring.rq<[12289], 8 : i32> {
  return %x : !ring.rq<[12289], 8 : i32>
}

// -----

// The limb bridge is where the widths meet: a 32-bit field tensor reaches the
// ring's residue rows without a widening copy.
// LOWER-LABEL: func.func @limbs_i32
// LOWER: tensor<2x8xi32>
func.func @limbs_i32(%a: tensor<8x!field.pf<12289:i32>>,
                     %b: tensor<8x!field.pf<40961:i32>>)
    -> !ring.rq<[12289, 40961], 8 : i32, i32, eval> {
  %r = ring.from_limbs %a, %b : tensor<8x!field.pf<12289:i32>>,
                                tensor<8x!field.pf<40961:i32>>
      to !ring.rq<[12289, 40961], 8 : i32, i32, eval>
  return %r : !ring.rq<[12289, 40961], 8 : i32, i32, eval>
}
