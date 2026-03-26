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

// Verify that EF constant print→parse round-trips are idempotent:
// canonicalize once to fold constants, then canonicalize again to confirm
// the printed output parses back to the same IR.
// RUN: prime-ir-opt -canonicalize %s | prime-ir-opt -canonicalize | FileCheck %s

!PF = !field.pf<7:i32>
!PFm = !field.pf<7:i32, true>
!QF = !field.ef<2x!PF, 6:i32>
!QFm = !field.ef<2x!PFm, 6:i32>
!Fp6 = !field.ef<3x!QF, 2:i32>

//===----------------------------------------------------------------------===//
// Scalar EF constants (list syntax [c₀, c₁, ...])
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @roundtrip_scalar_ef
func.func @roundtrip_scalar_ef() -> !QF {
  // CHECK: field.constant [3, 5]
  %0 = field.constant [3, 5] : !QF
  return %0 : !QF
}

// CHECK-LABEL: @roundtrip_scalar_ef_zero
func.func @roundtrip_scalar_ef_zero() -> !QF {
  // CHECK: field.constant [0, 0]
  %0 = field.constant [0, 0] : !QF
  return %0 : !QF
}

// CHECK-LABEL: @roundtrip_scalar_ef_mont
func.func @roundtrip_scalar_ef_mont() -> !QFm {
  // CHECK: field.constant [2, 4]
  %0 = field.constant [2, 4] : !QFm
  return %0 : !QFm
}

//===----------------------------------------------------------------------===//
// Tower extension constants
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @roundtrip_scalar_tower
func.func @roundtrip_scalar_tower() -> !Fp6 {
  // CHECK: field.constant [1, 2, 3, 4, 5, 6]
  %0 = field.constant [1, 2, 3, 4, 5, 6] : !Fp6
  return %0 : !Fp6
}

//===----------------------------------------------------------------------===//
// Tensor EF constants (dense syntax)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @roundtrip_tensor_ef
func.func @roundtrip_tensor_ef() -> tensor<2x!QF> {
  // CHECK: field.constant dense
  // CHECK-SAME: [1, 2], [3, 4]
  %0 = field.constant dense<[[1, 2], [3, 4]]> : tensor<2x!QF>
  return %0 : tensor<2x!QF>
}

// CHECK-LABEL: @roundtrip_tensor_ef_mont
func.func @roundtrip_tensor_ef_mont() -> tensor<2x!QFm> {
  // CHECK: field.constant dense
  // CHECK-SAME: [1, 2], [3, 4]
  %0 = field.constant dense<[[1, 2], [3, 4]]> : tensor<2x!QFm>
  return %0 : tensor<2x!QFm>
}

// CHECK-LABEL: @roundtrip_tensor_tower
func.func @roundtrip_tensor_tower() -> tensor<2x!Fp6> {
  // CHECK: field.constant dense
  // CHECK-SAME: [1, 2], [3, 4], [5, 6]
  %0 = field.constant dense<[[[1, 2], [3, 4], [5, 6]], [[6, 5], [4, 3], [2, 1]]]> : tensor<2x!Fp6>
  return %0 : tensor<2x!Fp6>
}

//===----------------------------------------------------------------------===//
// Splat tensor EF constants
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @roundtrip_tensor_ef_splat
func.func @roundtrip_tensor_ef_splat() -> tensor<2x!QF> {
  // CHECK: field.constant dense<1>
  %0 = field.constant dense<1> : tensor<2x!QF>
  return %0 : tensor<2x!QF>
}

//===----------------------------------------------------------------------===//
// Folded constants (canonicalize produces new constants that must round-trip)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @roundtrip_folded_negate
func.func @roundtrip_folded_negate() -> !QF {
  // CHECK: field.constant [4, 5]
  %0 = field.constant [3, 2] : !QF
  %1 = field.negate %0 : !QF
  return %1 : !QF
}

// CHECK-LABEL: @roundtrip_folded_tensor_add
func.func @roundtrip_folded_tensor_add() -> tensor<2x!QF> {
  // [1, 2] + [3, 0] = [4, 2]; [3, 4] + [1, 0] = [4, 4]
  // CHECK: field.constant dense
  // CHECK-SAME: [4, 2], [4, 4]
  %0 = field.constant dense<[[1, 2], [3, 4]]> : tensor<2x!QF>
  %1 = field.constant dense<[[3, 0], [1, 0]]> : tensor<2x!QF>
  %2 = field.add %0, %1 : tensor<2x!QF>
  return %2 : tensor<2x!QF>
}
