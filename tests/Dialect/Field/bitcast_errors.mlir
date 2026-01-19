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

// RUN: prime-ir-opt %s -verify-diagnostics

!PF = !field.pf<7:i32>
!PF_alt = !field.pf<13:i32>
!EF3 = !field.ef<3x!PF, 2:i32>
!EF3_alt = !field.ef<3x!PF_alt, 2:i32>
!EF4 = !field.ef<4x!PF, 2:i32>

func.func @test_bitcast_tensor_ef_to_tensor_pf_shape_mismatch(%ef: tensor<4x!EF3>) -> tensor<2x3x!PF> {
  // expected-error@+1 {{are cast incompatible}}
  %pf = field.bitcast %ef : tensor<4x!EF3> -> tensor<2x3x!PF>
  return %pf : tensor<2x3x!PF>
}


func.func @test_bitcast_tensor_pf_to_tensor_ef_shape_mismatch(%pf: tensor<2x4x!PF>) -> tensor<2x!EF3> {
  // expected-error@+1 {{are cast incompatible}}
  %ef = field.bitcast %pf : tensor<2x4x!PF> -> tensor<2x!EF3>
  return %ef : tensor<2x!EF3>
}

func.func @test_bitcast_ef_degree_dim_mismatch(%ef: !EF4) -> tensor<3x!PF> {
  // expected-error@+1 {{are cast incompatible}}
  %tensor = field.bitcast %ef : !EF4 -> tensor<3x!PF>
  return %tensor : tensor<3x!PF>
}

func.func @test_bitcast_tensor_to_ef_degree_dim_mismatch(%tensor: tensor<4x!PF>) -> !EF3 {
  // expected-error@+1 {{are cast incompatible}}
  %ef = field.bitcast %tensor : tensor<4x!PF> -> !EF3
  return %ef : !EF3
}

func.func @test_bitcast_ef_base_field_type_mismatch(%ef: !EF3_alt) -> tensor<3x!PF> {
  // expected-error@+1 {{are cast incompatible}}
  %tensor = field.bitcast %ef : !EF3_alt -> tensor<3x!PF>
  return %tensor : tensor<3x!PF>
}

func.func @test_bitcast_tensor_base_field_type_mismatch(%tensor: tensor<3x!PF>) -> !EF3_alt {
  // expected-error@+1 {{are cast incompatible}}
  %ef = field.bitcast %tensor : tensor<3x!PF> -> !EF3_alt
  return %ef : !EF3_alt
}
