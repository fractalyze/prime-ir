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

// `inverse-algorithm` selects the prime-field scalar-inverse lowering. Fermat
// (x^(p-2)) emits a fully unrolled, branch-free square-and-multiply chain
// (mod_arith.mul / mod_arith.square, no mod_arith.inverse and no scf.while);
// Bernstein-Yang emits mod_arith.inverse (a safegcd loop). `auto` picks Fermat
// for prime fields whose modulus is <= 64 bits, Bernstein-Yang otherwise. The
// default is bernstein-yang; `auto` is opt-in (see FieldToModArith.td).

// RUN: prime-ir-opt -field-to-mod-arith="use-elementwise-inverse=true inverse-algorithm=auto" -split-input-file %s | FileCheck %s --check-prefix=AUTO
// RUN: prime-ir-opt -field-to-mod-arith="use-elementwise-inverse=true inverse-algorithm=bernstein-yang" -split-input-file %s | FileCheck %s --check-prefix=FORCEBY
// RUN: prime-ir-opt -field-to-mod-arith="use-elementwise-inverse=true inverse-algorithm=fermat" -split-input-file %s | FileCheck %s --check-prefix=FORCEFERMAT

// Small prime field (modulus 97, 7 bits): auto -> Fermat.
!PF = !field.pf<97:i32, true>

// AUTO-LABEL: @small_field
// AUTO: linalg.generic
// AUTO: mod_arith.mul
// AUTO-NOT: mod_arith.inverse
// AUTO-NOT: scf.while

// Forcing bernstein-yang overrides auto's Fermat choice.
// FORCEBY-LABEL: @small_field
// FORCEBY: mod_arith.inverse

// Forcing fermat matches auto here (small field).
// FORCEFERMAT-LABEL: @small_field
// FORCEFERMAT: mod_arith.mul
// FORCEFERMAT-NOT: mod_arith.inverse
func.func @small_field(%arg0: tensor<4x!PF>) -> tensor<4x!PF> {
  %inv = field.inverse %arg0 : tensor<4x!PF>
  return %inv : tensor<4x!PF>
}

// -----

// Large prime field (bn254 scalar, 254 bits): auto -> Bernstein-Yang, because
// the unrolled Fermat chain (~p_bits wide multiplications) is more expensive
// than safegcd for wide fields.
!PF = !field.pf<21888242871839275222246405745257275088548364400416034343698204186575808495617:i256, true>

// AUTO-LABEL: @large_field
// AUTO: mod_arith.inverse

// FORCEBY-LABEL: @large_field
// FORCEBY: mod_arith.inverse

// Forcing fermat overrides auto's width gate: even the 254-bit field lowers to
// the unrolled square-and-multiply chain, never mod_arith.inverse.
// FORCEFERMAT-LABEL: @large_field
// FORCEFERMAT: mod_arith.mul
// FORCEFERMAT-NOT: mod_arith.inverse
func.func @large_field(%arg0: tensor<4x!PF>) -> tensor<4x!PF> {
  %inv = field.inverse %arg0 : tensor<4x!PF>
  return %inv : tensor<4x!PF>
}
