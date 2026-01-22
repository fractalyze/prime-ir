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

// RUN: prime-ir-opt %s -split-input-file -verify-diagnostics

!PF = !field.pf<7:i32>
!PF2 = !field.pf<11:i32>
!EF3 = !field.ef<3x!PF, 2:i32>

// Test that different base prime field types are rejected
func.func @test_invalid_base_field(%arg0: tensor<3x!PF2>) -> tensor<1x!EF3> {
  // expected-error @+1 {{are cast incompatible}}
  %0 = field.bitcast %arg0 : tensor<3x!PF2> -> tensor<1x!EF3>
  return %0 : tensor<1x!EF3>
}

// -----

!PF = !field.pf<7:i32>
!EF3 = !field.ef<3x!PF, 2:i32>

// Test that mismatched element count is rejected
func.func @test_invalid_count(%arg0: tensor<4x!PF>) -> tensor<1x!EF3> {
  // expected-error @+1 {{are cast incompatible}}
  %0 = field.bitcast %arg0 : tensor<4x!PF> -> tensor<1x!EF3>
  return %0 : tensor<1x!EF3>
}

// -----

!PF = !field.pf<7:i32>

// Test that prime field to prime field is rejected (need at least one extension field)
func.func @test_pf_to_pf(%arg0: tensor<3x!PF>) -> tensor<3x!PF> {
  // expected-error @+1 {{are cast incompatible}}
  %0 = field.bitcast %arg0 : tensor<3x!PF> -> tensor<3x!PF>
  return %0 : tensor<3x!PF>
}
