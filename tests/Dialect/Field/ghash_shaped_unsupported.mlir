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

// The portable GHASH multiply (`emitGhashMul`) is scalar-only: it hard-codes
// i64/i128 truncs, shifts, and constants. Shaped (tensor/vector) `bf<7, ghash>`
// mul/square is not lowered yet, so the conversion must fail cleanly instead of
// building invalid IR (e.g. an i128->i64 trunc of a tensor). Scalar GHASH is
// covered end-to-end by ghash_runner.mlir.

// RUN: prime-ir-opt %s --binary-field-to-arith --split-input-file --verify-diagnostics

!G = !field.bf<7, ghash>
func.func @tensor_ghash_mul(%a: tensor<4x!G>, %b: tensor<4x!G>) -> tensor<4x!G> {
  // expected-error @+1 {{operand #0 must be field-like}}
  %c = field.mul %a, %b : tensor<4x!G>
  return %c : tensor<4x!G>
}

// -----

!G = !field.bf<7, ghash>
func.func @vector_ghash_square(%a: vector<4x!G>) -> vector<4x!G> {
  // expected-error @+1 {{operand #0 must be field-like}}
  %c = field.square %a : vector<4x!G>
  return %c : vector<4x!G>
}

// -----

!G = !field.bf<7, ghash>
func.func @tensor_ghash_inverse(%a: tensor<4x!G>) -> tensor<4x!G> {
  // expected-error @+1 {{operand #0 must be field-like}}
  %c = field.inverse %a : tensor<4x!G>
  return %c : tensor<4x!G>
}
