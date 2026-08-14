// RUN: prime-ir-opt %s -ring-to-mod-arith | FileCheck %s

// The limb bridge is where a modulus stops being a type parameter of a field
// tensor and becomes one entry of the ring's modulus list. Each limb arrives
// carrying its own q_i in `!field.pf`, so the verifier checks the moduli
// against the ring type rather than trusting an attribute to have been kept in
// sync.

!PF17 = !field.pf<17:i64>
!PF41 = !field.pf<41:i64>
!Rq = !ring.rq<[17, 41], 4 : i32, eval>

// The lowered form is the [L, N] residue tensor, so the bridge is a bitcast per
// limb to its storage integers followed by a row insert.
// CHECK-LABEL: func.func @from_limbs_assembles_rows
// CHECK: field.bitcast
// CHECK: tensor.insert_slice
// CHECK: field.bitcast
// CHECK: tensor.insert_slice
func.func @from_limbs_assembles_rows(%l0: tensor<4x!PF17>, %l1: tensor<4x!PF41>)
    -> !Rq {
  %r = ring.from_limbs %l0, %l1
      : tensor<4x!PF17>, tensor<4x!PF41> to !Rq
  return %r : !Rq
}

// CHECK-LABEL: func.func @to_limbs_splits_rows
// CHECK: tensor.extract_slice
// CHECK: field.bitcast
// CHECK: tensor.extract_slice
// CHECK: field.bitcast
func.func @to_limbs_splits_rows(%r: !Rq)
    -> (tensor<4x!PF17>, tensor<4x!PF41>) {
  %l0, %l1 = ring.to_limbs %r
      : !Rq to tensor<4x!PF17>, tensor<4x!PF41>
  return %l0, %l1 : tensor<4x!PF17>, tensor<4x!PF41>
}

// A single-modulus ring is the lattice-PCS case: the limb tensor and the ring
// value describe the same N residues, so nothing is stacked.
!PF12289 = !field.pf<12289:i64>
!Rq1 = !ring.rq<[12289], 8 : i32, eval>

// CHECK-LABEL: func.func @round_trips_through_one_limb
func.func @round_trips_through_one_limb(%l: tensor<8x!PF12289>)
    -> tensor<8x!PF12289> {
  %r = ring.from_limbs %l : tensor<8x!PF12289> to !Rq1
  %o = ring.to_limbs %r : !Rq1 to tensor<8x!PF12289>
  return %o : tensor<8x!PF12289>
}
