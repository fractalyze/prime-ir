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

// The shape contract `poly.ntt` shares between the cyclic and negacyclic paths:
// the transform runs on the minor dimension and every leading dimension is a
// batch. Each rejection below stands between a caller and the lowering's
// power-of-two `assert`, which is a crash rather than a diagnostic.

!coeff_ty = !field.pf<7681:i32, true>
#omega = #field.root_of_unity<3383:i32, 4:i32> : !coeff_ty

func.func @ntt_rejects_rank_zero(%t : tensor<!coeff_ty>) -> tensor<!coeff_ty> {
  // expected-error @+1 {{requires a tensor of rank 1 or higher}}
  %r = poly.ntt %t into %t {root=#omega} : tensor<!coeff_ty>
  return %r : tensor<!coeff_ty>
}

// -----

!coeff_ty = !field.pf<7681:i32, true>
#omega = #field.root_of_unity<3383:i32, 4:i32> : !coeff_ty

func.func @ntt_rejects_a_non_power_of_two(%t : tensor<6x!coeff_ty>) -> tensor<6x!coeff_ty> {
  // expected-error @+1 {{requires a power-of-two transform length, got 6}}
  %r = poly.ntt %t into %t {root=#omega} : tensor<6x!coeff_ty>
  return %r : tensor<6x!coeff_ty>
}

// -----

// The length is read off the *minor* dimension, so a batch axis is free to be
// any size while the transform axis is not. The pair below and the positive
// control that follows are what pin which axis is which — a reversed reading
// would accept this one and reject that one.
!coeff_ty = !field.pf<7681:i32, true>
#omega = #field.root_of_unity<3383:i32, 4:i32> : !coeff_ty

func.func @ntt_rejects_a_non_power_of_two_minor_dim(%t : tensor<4x6x!coeff_ty>) -> tensor<4x6x!coeff_ty> {
  // expected-error @+1 {{requires a power-of-two transform length, got 6}}
  %r = poly.ntt %t into %t {root=#omega} : tensor<4x6x!coeff_ty>
  return %r : tensor<4x6x!coeff_ty>
}

// -----

!coeff_ty = !field.pf<7681:i32, true>
#omega = #field.root_of_unity<3383:i32, 4:i32> : !coeff_ty

func.func @ntt_accepts_a_non_power_of_two_batch(%t : tensor<6x4x!coeff_ty>) -> tensor<6x4x!coeff_ty> {
  %r = poly.ntt %t into %t {root=#omega} : tensor<6x4x!coeff_ty>
  return %r : tensor<6x4x!coeff_ty>
}

// -----

// One table serves the whole batch, so it stays rank-1 as the transformed
// tensor grows leading dimensions. The assembly format derives its type from
// `type($output)` rather than spelling it, so a batched table is not a verifier
// rejection but an unwritable one — the parser resolves `%tw` to the rank-1
// type and the mismatch lands on the prior use.
!coeff_ty = !field.pf<7681:i32, true>

// expected-note @+1 {{prior use here}}
func.func @ntt_rejects_batched_twiddles(%t : tensor<2x4x!coeff_ty>, %tw : tensor<2x4x!coeff_ty>) -> tensor<2x4x!coeff_ty> {
  // expected-error @+1 {{expects different type than prior uses: 'tensor<4x!field.pf<7681 : i32, true>>' vs 'tensor<2x4x!field.pf<7681 : i32, true>>'}}
  %r = poly.ntt %t into %t with %tw : tensor<2x4x!coeff_ty>
  return %r : tensor<2x4x!coeff_ty>
}

// -----

!coeff_ty = !field.pf<7681:i32, true>

// expected-note @+1 {{prior use here}}
func.func @ntt_rejects_a_mismatched_twiddle_length(%t : tensor<2x4x!coeff_ty>, %tw : tensor<8x!coeff_ty>) -> tensor<2x4x!coeff_ty> {
  // expected-error @+1 {{expects different type than prior uses: 'tensor<4x!field.pf<7681 : i32, true>>' vs 'tensor<8x!field.pf<7681 : i32, true>>'}}
  %r = poly.ntt %t into %t with %tw : tensor<2x4x!coeff_ty>
  return %r : tensor<2x4x!coeff_ty>
}
