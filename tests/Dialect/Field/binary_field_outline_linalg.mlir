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

// Tower outlining (fractalyze/prime-ir#390) through the SHIPPED pipeline on a
// shaped operand.
//
// `binary-field-to-arith` is scalar-only: its emitters truncate to a scalar
// half-width type, so a shaped operand cannot be lowered. `buildFieldToLLVM`
// therefore runs `convert-elementwise-to-linalg` ahead of it, leaving only the
// scalar body of a `linalg.generic` for the lowering to see. This runs the
// real `--field-to-llvm` rather than composing an order by hand, so if that
// ordering is ever inverted again this test fails instead of silently
// exercising a path the pipeline does not take.

// RUN: prime-ir-opt %s --field-to-llvm="outline-tower-ops=true" | FileCheck %s
// The default (outlining off) must lower the same input just as well: the
// scalarization above is what makes shaped tower ops legal at all, and that is
// independent of outlining. Without this run, a regression that only lowers
// when outlining happens to be on would go unnoticed.
// RUN: prime-ir-opt %s --field-to-llvm | FileCheck %s --check-prefix=DEFAULT

!BF128 = !field.bf<7>   // GF(2¹²⁸)

// The whole 3⁷ Karatsuba tower reduces to one call per element. The entry
// function keeps its tensor signature (function boundaries are not bufferized
// by default), so only its body is in the LLVM dialect.
// CHECK-LABEL: func.func @tensor_tower_mul
// CHECK: llvm.call @__prime_ir_bf_mul_l7

func.func @tensor_tower_mul(%a: tensor<4x!BF128>, %b: tensor<4x!BF128>)
    -> tensor<4x!BF128> {
  %c = field.mul %a, %b : tensor<4x!BF128>
  return %c : tensor<4x!BF128>
}

// The same input lowers in the default mode too, inline and with no helpers.
// DEFAULT-LABEL: func.func @tensor_tower_mul
// DEFAULT-NOT: __prime_ir_bf_
