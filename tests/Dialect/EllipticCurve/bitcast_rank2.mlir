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

// A point's K coordinates sit contiguously, so point/field reinterpret relates
// element counts rather than one distinguished dimension: batching points into
// rows describes the same bytes as laying them out flat, and the type rule is
// the same at every rank.

// RUN: prime-ir-opt --split-input-file --verify-diagnostics %s | FileCheck %s

!PF = !field.pf<7:i256>
#curve = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !PF
!jacobian = !elliptic_curve.jacobian<#curve>

// Jacobian over a prime field: K = 3 coordinates, so 4 points are 12 field
// elements however the two tensors distribute them across dimensions.
// CHECK-LABEL: func.func @rank2_points_to_rank2_fields
func.func @rank2_points_to_rank2_fields(
    %pts: tensor<2x2x!jacobian>) -> tensor<2x6x!PF> {
  // CHECK: elliptic_curve.bitcast
  %f = elliptic_curve.bitcast %pts : tensor<2x2x!jacobian> -> tensor<2x6x!PF>
  return %f : tensor<2x6x!PF>
}

// -----

!PF = !field.pf<7:i256>
#curve = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !PF
!jacobian = !elliptic_curve.jacobian<#curve>

// The ranks need not agree either; only the element counts do.
// CHECK-LABEL: func.func @rank2_points_to_flat_fields
func.func @rank2_points_to_flat_fields(
    %pts: tensor<2x2x!jacobian>) -> tensor<12x!PF> {
  // CHECK: elliptic_curve.bitcast
  %f = elliptic_curve.bitcast %pts : tensor<2x2x!jacobian> -> tensor<12x!PF>
  return %f : tensor<12x!PF>
}

// -----

!PF = !field.pf<7:i256>
#curve = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !PF
!jacobian = !elliptic_curve.jacobian<#curve>

// Rank freedom is not count freedom: 4 points need 12 field elements, not 10.
func.func @rank2_element_count_mismatch(
    %pts: tensor<2x2x!jacobian>) -> tensor<2x5x!PF> {
  // expected-error @+1 {{are cast incompatible}}
  %f = elliptic_curve.bitcast %pts : tensor<2x2x!jacobian> -> tensor<2x5x!PF>
  return %f : tensor<2x5x!PF>
}
