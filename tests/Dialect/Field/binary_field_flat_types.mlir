// Copyright 2026 The PrimeIR Authors.
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

// Round-trip, canonical normalization, and rejection for the binary-field
// flat-basis syntax. `flat` is the canonical modulus of any level 1..7;
// `poly<f>` gives an explicit modulus (full polynomial bitmask) and uniques
// to the canonical spelling when it matches; custom moduli are limited to
// levels 1..6 with 2*deg(f - y^n) <= n and must be irreducible.

// RUN: prime-ir-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// CHECK-LABEL: @roundtrip_canonical
// CHECK-SAME: !field.bf<1, flat>
// CHECK-SAME: !field.bf<2, flat>
// CHECK-SAME: !field.bf<4, flat>
// CHECK-SAME: !field.bf<5, flat>
// CHECK-SAME: !field.bf<6, flat>
func.func @roundtrip_canonical(%a: !field.bf<1, flat>, %b: !field.bf<2, flat>,
                               %c: !field.bf<4, flat>, %d: !field.bf<5, flat>,
                               %e: !field.bf<6, flat>) {
  return
}

// -----

// A canonical modulus spelled via poly<> or flat uniques to the named basis:
// levels 3/7 print as aes/ghash.
// CHECK-LABEL: @canonical_normalization
// CHECK-SAME: !field.bf<3, aes>
// CHECK-SAME: !field.bf<7, ghash>
// CHECK-SAME: !field.bf<5, flat>
func.func @canonical_normalization(
    %a: !field.bf<3, poly<0x11b>>, %b: !field.bf<7, flat>,
    %c: !field.bf<5, poly<0x10000008d>>) {
  return
}

// -----

// A custom modulus is its own type and prints in full.
// CHECK-LABEL: @custom_modulus
// CHECK-SAME: !field.bf<3, poly<0x11D>>
func.func @custom_modulus(%a: !field.bf<3, poly<0x11d>>) {
  return
}

// -----

// expected-error @+1 {{GF(2) has a single basis}}
func.func @flat_level0(%a: !field.bf<0, flat>) {
  return
}

// -----

// 0x11c = x^8+x^4+x^3+x^2 has no constant term (divisible by x).
// expected-error @below {{flat modulus must have a constant term}}
func.func @reducible_modulus(%a: !field.bf<3, poly<0x11c>>) {
  return
}

// -----

// x^8 + x^2 + x + 1 factors over GF(2) — f(1) = 0 — despite having a
// constant term and a low-degree tail.
// expected-error @below {{is reducible over GF(2)}}
func.func @reducible_modulus2(%a: !field.bf<3, poly<0x107>>) {
  return
}

// -----

// Degree of the low part exceeds n/2: the two-fold reduction would not
// converge.
// expected-error @below {{2*deg <= 8}}
func.func @degree_violation(%a: !field.bf<3, poly<0x1e1>>) {
  return
}

// -----

// Level 7 admits only the canonical GHASH modulus.
// expected-error @+2 {{level-7 flat modulus must be the canonical GHASH}}
func.func @level7_custom(
    %a: !field.bf<7, poly<0x100000000000000000000000000000425>>) {
  return
}
