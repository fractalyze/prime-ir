// Copyright 2025 The PrimeIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

// RUN: prime-ir-opt %s -split-input-file -verify-diagnostics

// A negacyclic transform over `n` coefficients evaluates at odd powers of a
// `2n`-th root, so an `n`-th root is the wrong one. Caught at the op rather than
// producing a transform over the wrong ring.

!coeff_ty = !field.pf<7681:i32, true>
// 3383 has order 4 — correct for the cyclic transform, half of what negacyclic
// over 4 coefficients needs.
#omega = #field.root_of_unity<3383:i32, 4:i32> : !coeff_ty

func.func @negacyclic_rejects_an_nth_root(%t : tensor<4x!coeff_ty>) -> tensor<4x!coeff_ty> {
  // expected-error @+1 {{needs `root^4 == -1`}}
  %r = poly.ntt %t into %t {root=#omega} negacyclic=true : tensor<4x!coeff_ty>
  return %r : tensor<4x!coeff_ty>
}

// -----

// The same order-4 root *relabelled* as degree 8. `RootOfUnityAttr` verifies only
// `root^degree == 1`, which is divisibility and not order, so 3383^8 = 1 passes
// it — a check on the stated degree would wave this through and then drive the
// core with a root of half the needed order. Rejecting on `psi^n == -1` is what
// makes the claim a checked value.
!coeff_ty = !field.pf<7681:i32, true>
#mislabelled = #field.root_of_unity<3383:i32, 8:i32> : !coeff_ty

func.func @negacyclic_rejects_a_mislabelled_root(%t : tensor<4x!coeff_ty>) -> tensor<4x!coeff_ty> {
  // expected-error @+1 {{needs `root^4 == -1`}}
  %r = poly.ntt %t into %t {root=#mislabelled} negacyclic=true : tensor<4x!coeff_ty>
  return %r : tensor<4x!coeff_ty>
}

// -----

!coeff_ty = !field.pf<7681:i32, true>

func.func @negacyclic_requires_a_root(%t : tensor<4x!coeff_ty>) -> tensor<4x!coeff_ty> {
  // expected-error @+1 {{negacyclic requires `root`}}
  %r = poly.ntt %t into %t negacyclic=true : tensor<4x!coeff_ty>
  return %r : tensor<4x!coeff_ty>
}

// -----

// `twiddles` cannot encode the per-coefficient twist, so the pair is rejected
// rather than silently lowering to a cyclic transform.
!coeff_ty = !field.pf<7681:i32, true>
#psi = #field.root_of_unity<1925:i32, 8:i32> : !coeff_ty

func.func @negacyclic_rejects_twiddles(%t : tensor<4x!coeff_ty>, %tw : tensor<4x!coeff_ty>) -> tensor<4x!coeff_ty> {
  // expected-error @+1 {{negacyclic is not supported with `twiddles`}}
  %r = poly.ntt %t into %t with %tw {root=#psi} negacyclic=true : tensor<4x!coeff_ty>
  return %r : tensor<4x!coeff_ty>
}

// -----

// The twist rides the natural index, which only `bit_reverse` establishes.
!coeff_ty = !field.pf<7681:i32, true>
#psi = #field.root_of_unity<1925:i32, 8:i32> : !coeff_ty

func.func @negacyclic_requires_bit_reverse(%t : tensor<4x!coeff_ty>) -> tensor<4x!coeff_ty> {
  // expected-error @+1 {{negacyclic requires `bit_reverse`}}
  %r = poly.ntt %t into %t {root=#psi} bit_reverse=false negacyclic=true : tensor<4x!coeff_ty>
  return %r : tensor<4x!coeff_ty>
}

// -----

// `psi^n == -1` does not imply a power-of-two length: 3383 has order 4, so it
// satisfies the equation at n = 6 as well. Unchecked, this reaches the
// lowering's power-of-two assert — a crash rather than a diagnostic.
!coeff_ty = !field.pf<7681:i32, true>
#omega = #field.root_of_unity<3383:i32, 4:i32> : !coeff_ty

func.func @negacyclic_rejects_a_non_power_of_two(%t : tensor<6x!coeff_ty>) -> tensor<6x!coeff_ty> {
  // expected-error @+1 {{negacyclic requires a power-of-two length, got 6}}
  %r = poly.ntt %t into %t {root=#omega} negacyclic=true : tensor<6x!coeff_ty>
  return %r : tensor<6x!coeff_ty>
}

// -----

!coeff_ty = !field.pf<7681:i32, true>
#psi = #field.root_of_unity<1925:i32, 8:i32> : !coeff_ty

func.func @negacyclic_rejects_rank_zero(%t : tensor<!coeff_ty>) -> tensor<!coeff_ty> {
  // expected-error @+1 {{negacyclic requires a rank-1 tensor}}
  %r = poly.ntt %t into %t {root=#psi} negacyclic=true : tensor<!coeff_ty>
  return %r : tensor<!coeff_ty>
}

// -----

// The positive control: a genuine 2n-th root is accepted, so the rejections above
// are about the property and not about `negacyclic` itself.

!coeff_ty = !field.pf<7681:i32, true>
#psi = #field.root_of_unity<1925:i32, 8:i32> : !coeff_ty

func.func @negacyclic_accepts_a_2nth_root(%t : tensor<4x!coeff_ty>) -> tensor<4x!coeff_ty> {
  %r = poly.ntt %t into %t {root=#psi} negacyclic=true : tensor<4x!coeff_ty>
  return %r : tensor<4x!coeff_ty>
}
