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

// Tests that tensor extension field operations are properly lowered through
// convert-elementwise-to-linalg and field-to-mod-arith passes.
//
// The full pipeline to LLVM requires additional bufferization passes, which
// are tested separately in e2e tests.

// RUN: prime-ir-opt -convert-elementwise-to-linalg -field-to-mod-arith %s \
// RUN:   | FileCheck %s -enable-var-scope

!PF = !field.pf<7:i32>
!EF2 = !field.ef<2x!PF, 6:i32>
!EF6 = !field.ef<3x!EF2, 2:i32>

// CHECK-LABEL: @test_tensor_ext_field_add
// CHECK: linalg.generic
// CHECK: mod_arith.add
// CHECK-NOT: field.add
func.func @test_tensor_ext_field_add(%a: tensor<10x!EF6>, %b: tensor<10x!EF6>) -> tensor<10x!EF6> {
  %c = field.add %a, %b : tensor<10x!EF6>
  return %c : tensor<10x!EF6>
}

// CHECK-LABEL: @test_tensor_ext_field_mul
// CHECK: linalg.generic
// CHECK: mod_arith.mul
// CHECK-NOT: field.mul
func.func @test_tensor_ext_field_mul(%a: tensor<4x!EF2>, %b: tensor<4x!EF2>) -> tensor<4x!EF2> {
  %c = field.mul %a, %b : tensor<4x!EF2>
  return %c : tensor<4x!EF2>
}

// CHECK-LABEL: @test_tensor_ext_field_sub
// CHECK: linalg.generic
// CHECK: mod_arith.sub
// CHECK-NOT: field.sub
func.func @test_tensor_ext_field_sub(%a: tensor<8x!EF2>, %b: tensor<8x!EF2>) -> tensor<8x!EF2> {
  %c = field.sub %a, %b : tensor<8x!EF2>
  return %c : tensor<8x!EF2>
}
