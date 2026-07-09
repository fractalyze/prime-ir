// RUN: prime-ir-opt %s | prime-ir-opt | FileCheck %s

// Parse + print round-trip of the ring.rq type.

// An RNS ring: two ~60-bit coprime limbs, N = 4096 (Q = q0 * q1).
// CHECK-LABEL: func.func @rq_rns_roundtrip
// CHECK-SAME: !ring.rq<[1152921504606830593, 1152921504598720513], 4096 : i32>
func.func @rq_rns_roundtrip(
    %arg0: !ring.rq<[1152921504606830593, 1152921504598720513], 4096 : i32>)
    -> !ring.rq<[1152921504606830593, 1152921504598720513], 4096 : i32> {
  // CHECK: return %arg0 : !ring.rq<[1152921504606830593, 1152921504598720513], 4096 : i32>
  return %arg0 : !ring.rq<[1152921504606830593, 1152921504598720513], 4096 : i32>
}

// A single-prime ring (length-1 modulus list) at N = 8 also round-trips.
// CHECK-LABEL: func.func @rq_single_prime
// CHECK-SAME: !ring.rq<[12289], 8 : i32>
func.func @rq_single_prime(
    %arg0: !ring.rq<[12289], 8 : i32>) -> !ring.rq<[12289], 8 : i32> {
  return %arg0 : !ring.rq<[12289], 8 : i32>
}

// A ciphertext is the free module R_Q^2 = a tensor of two ring elements.
// CHECK-LABEL: func.func @ciphertext_is_tensor_of_ring
// CHECK-SAME: tensor<2x!ring.rq<[12289], 8 : i32>>
func.func @ciphertext_is_tensor_of_ring(
    %arg0: tensor<2x!ring.rq<[12289], 8 : i32>>)
    -> tensor<2x!ring.rq<[12289], 8 : i32>> {
  return %arg0 : tensor<2x!ring.rq<[12289], 8 : i32>>
}
