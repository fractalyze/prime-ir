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

// Tower outlining (fractalyze/prime-ir#390) must be a no-op on SHAPED operands.
//
// The outlined helpers are declared over the scalar element type i(2^k), so
// binding a tensor/vector operand to one would build a call that does not
// verify. Shaped tower ops therefore keep expanding inline.
//
// Lowering a shaped tower mul/square is separately broken today — the
// recursion truncates to a scalar half-width type, so the result can no longer
// be materialized back to the shaped type (fractalyze/prime-ir#452). These
// cases pin that failure as it stands and, by running the SAME expectations
// with outlining on and off, prove outlining neither causes it nor changes it.
//
// When #452 is fixed these expected-errors will start failing, which is the
// point: whoever fixes shaped lowering must also decide what outlining does
// with a shaped operand rather than silently emitting an ill-typed call.

// RUN: prime-ir-opt %s --binary-field-to-arith \
// RUN:   --split-input-file --verify-diagnostics=only-expected
// RUN: prime-ir-opt %s \
// RUN:     --binary-field-to-arith="outline-tower-ops=true outline-min-tower-level=3" \
// RUN:   --split-input-file --verify-diagnostics=only-expected

!BF128 = !field.bf<7>

func.func @tensor_tower_mul(%a: tensor<4x!BF128>, %b: tensor<4x!BF128>)
    -> tensor<4x!BF128> {
  %c = field.mul %a, %b : tensor<4x!BF128>
  // expected-error @+1 {{failed to legalize unresolved materialization}}
  return %c : tensor<4x!BF128>
}

// -----

!BF128 = !field.bf<7>

func.func @tensor_tower_square(%a: tensor<4x!BF128>) -> tensor<4x!BF128> {
  %c = field.square %a : tensor<4x!BF128>
  // expected-error @+1 {{failed to legalize unresolved materialization}}
  return %c : tensor<4x!BF128>
}

// -----

// A narrower tower shows the same collapse, so this is the recursion and not
// something specific to level 7.
!BF32 = !field.bf<5>

func.func @tensor_tower_mul_narrow(%a: tensor<4x!BF32>, %b: tensor<4x!BF32>)
    -> tensor<4x!BF32> {
  %c = field.mul %a, %b : tensor<4x!BF32>
  // expected-error @+1 {{failed to legalize unresolved materialization}}
  return %c : tensor<4x!BF32>
}
