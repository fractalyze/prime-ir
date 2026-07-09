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

// The portable AES multiply (`emitAesMul`) is scalar-only, like the GHASH one:
// it hard-codes i8/i16 truncs, shifts, and constants. Shaped (tensor/vector)
// `bf<3, aes>` mul/square is not lowered yet, so the conversion must fail
// cleanly instead of building invalid IR. Scalar AES is covered end-to-end by
// aes_runner.mlir.

// RUN: prime-ir-opt %s --binary-field-to-arith --split-input-file --verify-diagnostics

!A = !field.bf<3, aes>
func.func @tensor_aes_mul(%a: tensor<4x!A>, %b: tensor<4x!A>) -> tensor<4x!A> {
  // expected-error @+1 {{operand #0 must be field-like}}
  %c = field.mul %a, %b : tensor<4x!A>
  return %c : tensor<4x!A>
}

// -----

!A = !field.bf<3, aes>
func.func @vector_aes_square(%a: vector<4x!A>) -> vector<4x!A> {
  // expected-error @+1 {{operand #0 must be field-like}}
  %c = field.square %a : vector<4x!A>
  return %c : vector<4x!A>
}
