// RUN: prime-ir-opt --split-input-file --verify-diagnostics %s

// base_convert changes the RNS basis but must preserve the ring degree N.
func.func @base_convert_bad_degree(
    %x: !ring.rq<[12289], 8 : i32>) -> !ring.rq<[12289], 16 : i32> {
  // expected-error @+1 {{input and output rings must share the degree N}}
  %y = ring.base_convert %x : !ring.rq<[12289], 8 : i32> to !ring.rq<[12289], 16 : i32>
  return %y : !ring.rq<[12289], 16 : i32>
}

// -----

// X^N+1 splits only where 2N | q-1. At N = 8 the prime 41 admits an N-th root
// (8 | 40) but no 2N-th root (16 does not divide 40), so this ring has no
// evaluation basis for XLA's transform to produce.
// expected-error @+1 {{2N = 16 does not divide 41 - 1}}
func.func @no_eval_basis_without_a_2n_th_root(%x: !ring.rq<[41], 8 : i32, eval>) {
  return
}

// -----

// The Galois exponent must be odd to be coprime to 2N.
func.func @automorphism_rejects_even_exponent(
    %x: !ring.rq<[12289], 8 : i32>) -> !ring.rq<[12289], 8 : i32> {
  // expected-error @+1 {{exponent must be a positive odd integer}}
  %y = ring.automorphism %x {exponent = 4 : i64} : !ring.rq<[12289], 8 : i32>
  return %y : !ring.rq<[12289], 8 : i32>
}

// -----

// Digit extraction does not commute with the CRT map.
func.func @gadget_decompose_rejects_eval(
    %x: !ring.rq<[12289], 8 : i32, eval>) -> !ring.rq<[12289], 8 : i32, eval> {
  // expected-error @+1 {{input must be in the coeff basis}}
  %d:2 = ring.gadget_decompose %x {baseBits = 4 : i64, levels = 2 : i64}
      : !ring.rq<[12289], 8 : i32, eval> -> !ring.rq<[12289], 8 : i32, eval>, !ring.rq<[12289], 8 : i32, eval>
  return %d#0 : !ring.rq<[12289], 8 : i32, eval>
}

// -----

// The coefficient-basis product is a negacyclic convolution, not anything this
// dialect can spell.
func.func @mul_rejects_coeff(
    %x: !ring.rq<[12289], 8 : i32>) -> !ring.rq<[12289], 8 : i32> {
  // expected-error @+1 {{operands must be in the eval basis}}
  %y = ring.mul %x, %x : !ring.rq<[12289], 8 : i32>
  return %y : !ring.rq<[12289], 8 : i32>
}

// -----

// rescale drops a modulus; it does not change basis.
func.func @rescale_preserves_basis(
    %x: !ring.rq<[12289, 40961], 8 : i32, eval>) -> !ring.rq<[12289], 8 : i32> {
  // expected-error @+1 {{rescale does not change basis}}
  %y = ring.rescale %x : !ring.rq<[12289, 40961], 8 : i32, eval> to !ring.rq<[12289], 8 : i32>
  return %y : !ring.rq<[12289], 8 : i32>
}
