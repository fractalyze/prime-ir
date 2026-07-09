// RUN: prime-ir-opt %s -ring-to-mod-arith | FileCheck %s

// gadget_product accumulates the per-term negacyclic products in the NTT
// (evaluation) domain: each digit and key is forward-transformed once, the
// pointwise products are summed, and a SINGLE inverse NTT reconstructs the
// coefficient result -- not one inverse per term. For levels = 2 that is 4
// forward transforms and exactly 1 inverse (vs. 2 inverses if unfused). This
// pins the fused shape so a regression back to per-term inverse NTTs is caught.

// CHECK-LABEL: func.func @gp
// CHECK-COUNT-1: poly.ntt {{.*}}inverse = true
// CHECK-NOT: inverse = true
func.func @gp(%x: !ring.rq<[17], 4 : i32>, %k0: !ring.rq<[17], 4 : i32>,
              %k1: !ring.rq<[17], 4 : i32>) -> !ring.rq<[17], 4 : i32> {
  %r = ring.gadget_product %x, %k0, %k1 {baseBits = 2 : i64}
      : !ring.rq<[17], 4 : i32>, !ring.rq<[17], 4 : i32>, !ring.rq<[17], 4 : i32>
      -> !ring.rq<[17], 4 : i32>
  return %r : !ring.rq<[17], 4 : i32>
}
