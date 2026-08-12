// RUN: prime-ir-opt %s --split-input-file --verify-diagnostics

// A residue has to fit its own storage word. Residues are unsigned, so the
// bound is on q - 1: an i16 ring tops out at q = 65536.
// expected-error @+1 {{modulus 65537 does not fit in i16 storage}}
func.func @rq_rejects_narrow_storage(%x: !ring.rq<[65537], 8 : i32, i16>) {
  return
}

// -----

// The limb's storage is part of its type, and the bridge is a reinterpret --
// a width mismatch here would be a silent widening copy.
func.func @from_limbs_rejects_storage_mismatch(%a: tensor<8x!field.pf<12289:i64>>)
    -> !ring.rq<[12289], 8 : i32, i32, eval> {
  // expected-error @+1 {{limb 0 is stored in i64, but the ring's residues are i32}}
  %r = ring.from_limbs %a : tensor<8x!field.pf<12289:i64>>
      to !ring.rq<[12289], 8 : i32, i32, eval>
  return %r : !ring.rq<[12289], 8 : i32, i32, eval>
}
