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

// The coefficient-basis product is a negacyclic convolution, not anything this
// dialect can spell.
func.func @mul_rejects_coeff(
    %x: !ring.rq<[12289], 8 : i32>) -> !ring.rq<[12289], 8 : i32> {
  // expected-error @+1 {{operands must be in the eval basis}}
  %y = ring.mul %x, %x : !ring.rq<[12289], 8 : i32>
  return %y : !ring.rq<[12289], 8 : i32>
}

// -----

// Position k of the dropped limb only stands for the same coefficient as
// position k of the surviving ones in the coefficient basis; in the evaluation
// basis each limb evaluates at its own root of unity.
func.func @rescale_rejects_eval_basis(
    %x: !ring.rq<[12289, 40961], 8 : i32, eval>)
    -> !ring.rq<[12289], 8 : i32, eval> {
  // expected-error @+1 {{only the coefficient basis relates}}
  %y = ring.rescale %x
      : !ring.rq<[12289, 40961], 8 : i32, eval> to !ring.rq<[12289], 8 : i32, eval>
  return %y : !ring.rq<[12289], 8 : i32, eval>
}

// -----

// Same for the other cross-limb op.
func.func @base_convert_rejects_eval_basis(
    %x: !ring.rq<[12289], 8 : i32, eval>) -> !ring.rq<[40961], 8 : i32, eval> {
  // expected-error @+1 {{only the coefficient basis relates}}
  %y = ring.base_convert %x
      : !ring.rq<[12289], 8 : i32, eval> to !ring.rq<[40961], 8 : i32, eval>
  return %y : !ring.rq<[40961], 8 : i32, eval>
}

// -----

// The result is written with the input's residue word, so a ring that stores
// its residues differently is not a rescale target.
func.func @rescale_rejects_storage_mismatch(
    %x: !ring.rq<[12289, 40961], 8 : i32, i32>) -> !ring.rq<[12289], 8 : i32> {
  // expected-error @+1 {{must share the residue storage type}}
  %y = ring.rescale %x
      : !ring.rq<[12289, 40961], 8 : i32, i32> to !ring.rq<[12289], 8 : i32>
  return %y : !ring.rq<[12289], 8 : i32>
}

// -----

// A Montgomery limb carries a factor of R that the ring's canonical residues
// do not, and nothing downstream would take it back out.
func.func @from_limbs_rejects_montgomery(
    %l: tensor<8x!field.pf<12289 : i64, true>>) -> !ring.rq<[12289], 8 : i32> {
  // expected-error @+1 {{is in Montgomery form}}
  %r = ring.from_limbs %l
      : tensor<8x!field.pf<12289 : i64, true>> to !ring.rq<[12289], 8 : i32>
  return %r : !ring.rq<[12289], 8 : i32>
}

// -----

// A limb modulus is an arbitrary-width integer; comparing it must not depend
// on it fitting a word.
func.func @from_limbs_rejects_wide_modulus(
    %l: tensor<8x!field.pf<21888242871839275222246405745257275088548364400416034343698204186575808495617:i256>>)
    -> !ring.rq<[12289], 8 : i32> {
  // expected-error @+1 {{but the ring's is 12289}}
  %r = ring.from_limbs %l
      : tensor<8x!field.pf<21888242871839275222246405745257275088548364400416034343698204186575808495617:i256>>
      to !ring.rq<[12289], 8 : i32>
  return %r : !ring.rq<[12289], 8 : i32>
}

// -----

// The tensor IS the residue layout, so it has to be [L, N] in the ring's word.
func.func @from_tensor_rejects_wrong_shape(%t: tensor<3x2xi64>)
    -> !ring.rq<[7, 11, 13], 4 : i32> {
  // expected-error @+1 {{residue tensor must be 3x4}}
  %r = ring.from_tensor %t : tensor<3x2xi64> to !ring.rq<[7, 11, 13], 4 : i32>
  return %r : !ring.rq<[7, 11, 13], 4 : i32>
}

// -----

func.func @to_tensor_rejects_wrong_element(%x: !ring.rq<[12289], 8 : i32, i32>)
    -> tensor<1x8xi64> {
  // expected-error @+1 {{must be the ring's storage type i32}}
  %t = ring.to_tensor %x : !ring.rq<[12289], 8 : i32, i32> to tensor<1x8xi64>
  return %t : tensor<1x8xi64>
}

// -----

// One limb per modulus: the operand count is what fixes which q_i each limb
// answers to, so a mismatch has no reading to fall back on.
func.func @from_limbs_rejects_wrong_limb_count(
    %l0: tensor<8x!field.pf<12289:i64>>, %l1: tensor<8x!field.pf<40961:i64>>)
    -> !ring.rq<[12289], 8 : i32> {
  // expected-error @+1 {{expects 1 limb, but got 2}}
  %r = ring.from_limbs %l0, %l1
      : tensor<8x!field.pf<12289:i64>>, tensor<8x!field.pf<40961:i64>>
      to !ring.rq<[12289], 8 : i32>
  return %r : !ring.rq<[12289], 8 : i32>
}

// -----

// The limb carries its modulus in its own type, which is the whole point of
// the bridge: the check is against the type, not against a restated attribute.
func.func @from_limbs_rejects_modulus_mismatch(
    %l0: tensor<8x!field.pf<40961:i64>>) -> !ring.rq<[12289], 8 : i32> {
  // expected-error @+1 {{limb 0 has modulus 40961, but the ring's is 12289}}
  %r = ring.from_limbs %l0
      : tensor<8x!field.pf<40961:i64>> to !ring.rq<[12289], 8 : i32>
  return %r : !ring.rq<[12289], 8 : i32>
}

// -----

// A limb holds the ring's N residues; a shorter one is not a truncated ring
// element, it is a different object.
func.func @from_limbs_rejects_wrong_degree(
    %l0: tensor<4x!field.pf<12289:i64>>) -> !ring.rq<[12289], 8 : i32> {
  // expected-error @+1 {{limb 0 must have 8 elements, but got 4}}
  %r = ring.from_limbs %l0
      : tensor<4x!field.pf<12289:i64>> to !ring.rq<[12289], 8 : i32>
  return %r : !ring.rq<[12289], 8 : i32>
}

// -----

// to_limbs answers the same constraints from the result side.
func.func @to_limbs_rejects_modulus_mismatch(
    %r: !ring.rq<[12289], 8 : i32>) -> tensor<8x!field.pf<40961:i64>> {
  // expected-error @+1 {{limb 0 has modulus 40961, but the ring's is 12289}}
  %l = ring.to_limbs %r
      : !ring.rq<[12289], 8 : i32> to tensor<8x!field.pf<40961:i64>>
  return %l : tensor<8x!field.pf<40961:i64>>
}

// -----

// CRT is an isomorphism only for a pairwise coprime basis. Distinctness is not
// enough: a shared factor makes the residues redundant, so base_convert and
// rescale would compute against a modulus product that is not Q.
// expected-error @+1 {{moduli must be pairwise coprime, but 9 and 21 share the factor 3}}
func.func @rq_rejects_non_coprime_moduli(%x: !ring.rq<[9, 21], 2 : i32>) {
  return
}

// -----

// The bridges are reinterprets, so a value that just came out of one carries
// the basis of the ring it came from. Naming a different basis on the way back
// in is a relabel with no transform between, and the transform is not this
// dialect's to perform.
func.func @from_tensor_cannot_relabel_the_basis(
    %x: !ring.rq<[12289], 8 : i32>) -> !ring.rq<[12289], 8 : i32, eval> {
  %t = ring.to_tensor %x : !ring.rq<[12289], 8 : i32> to tensor<1x8xi64>
  // expected-error @+1 {{cannot relabel the basis}}
  %r = ring.from_tensor %t : tensor<1x8xi64> to !ring.rq<[12289], 8 : i32, eval>
  return %r : !ring.rq<[12289], 8 : i32, eval>
}

// -----

func.func @from_limbs_cannot_relabel_the_basis(
    %x: !ring.rq<[12289], 8 : i32>) -> !ring.rq<[12289], 8 : i32, eval> {
  %l = ring.to_limbs %x
      : !ring.rq<[12289], 8 : i32> to tensor<8x!field.pf<12289:i64>>
  // expected-error @+1 {{cannot relabel the basis}}
  %r = ring.from_limbs %l
      : tensor<8x!field.pf<12289:i64>> to !ring.rq<[12289], 8 : i32, eval>
  return %r : !ring.rq<[12289], 8 : i32, eval>
}
