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

// Tower outlining (fractalyze/prime-ir#390) on the shape a real consumer feeds
// it. `binary-field-to-arith` is a scalar-only tower path by contract:
// elementwise field ops are converted to `linalg.generic` first, so the
// lowering only ever sees the scalar body. This pins that this is where the
// outlining lands — one shared helper called from inside the generic body,
// rather than a tower expanded per loop nest.

// RUN: prime-ir-opt %s --convert-elementwise-to-linalg \
// RUN:     --binary-field-to-arith="outline-tower-ops=true" \
// RUN:   | FileCheck %s

!BF128 = !field.bf<7>   // GF(2¹²⁸)

// CHECK-LABEL: func.func @tensor_tower_mul
// CHECK: linalg.generic
// The whole 3⁷ Karatsuba tower reduces to a single call in the loop body.
// CHECK: ^bb0(%[[LHS:.*]]: i128, %[[RHS:.*]]: i128, %{{.*}}: i128):
// CHECK-NEXT: %[[R:.*]] = func.call @__prime_ir_bf_mul_l7(%[[LHS]], %[[RHS]])
// CHECK-NEXT: linalg.yield %[[R]]
func.func @tensor_tower_mul(%a: tensor<4x!BF128>, %b: tensor<4x!BF128>)
    -> tensor<4x!BF128> {
  %c = field.mul %a, %b : tensor<4x!BF128>
  return %c : tensor<4x!BF128>
}

// The helper is emitted once at module scope and shared by every generic that
// needs it, which is the point of outlining on a per-element loop body.
// CHECK: func.func private @__prime_ir_bf_mul_l7
// CHECK-SAME: llvm.no_inline
