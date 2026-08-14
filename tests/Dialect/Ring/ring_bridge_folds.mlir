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

// Both bridges are reinterprets, so a round trip through either one is the
// identity and leaves nothing behind.

// RUN: prime-ir-opt %s --canonicalize --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @tensor_round_trip
// CHECK-SAME:    (%[[X:.*]]: !ring.rq
// CHECK-NEXT:    return %[[X]]
func.func @tensor_round_trip(%x: !ring.rq<[12289], 8 : i32>)
    -> !ring.rq<[12289], 8 : i32> {
  %t = ring.to_tensor %x : !ring.rq<[12289], 8 : i32> to tensor<1x8xi64>
  %r = ring.from_tensor %t : tensor<1x8xi64> to !ring.rq<[12289], 8 : i32>
  return %r : !ring.rq<[12289], 8 : i32>
}

// -----

// CHECK-LABEL: func.func @tensor_round_trip_the_other_way
// CHECK-SAME:    (%[[T:.*]]: tensor<1x8xi64>
// CHECK-NEXT:    return %[[T]]
func.func @tensor_round_trip_the_other_way(%t: tensor<1x8xi64>)
    -> tensor<1x8xi64> {
  %r = ring.from_tensor %t : tensor<1x8xi64> to !ring.rq<[12289], 8 : i32>
  %out = ring.to_tensor %r : !ring.rq<[12289], 8 : i32> to tensor<1x8xi64>
  return %out : tensor<1x8xi64>
}

// -----

// CHECK-LABEL: func.func @limb_round_trip
// CHECK-SAME:    (%[[X:.*]]: !ring.rq
// CHECK-NEXT:    return %[[X]]
func.func @limb_round_trip(%x: !ring.rq<[12289, 40961], 8 : i32>)
    -> !ring.rq<[12289, 40961], 8 : i32> {
  %l0, %l1 = ring.to_limbs %x : !ring.rq<[12289, 40961], 8 : i32>
      to tensor<8x!field.pf<12289:i64>>, tensor<8x!field.pf<40961:i64>>
  %r = ring.from_limbs %l0, %l1
      : tensor<8x!field.pf<12289:i64>>, tensor<8x!field.pf<40961:i64>>
      to !ring.rq<[12289, 40961], 8 : i32>
  return %r : !ring.rq<[12289, 40961], 8 : i32>
}

// -----

// CHECK-LABEL: func.func @limb_round_trip_the_other_way
// CHECK-SAME:    (%[[L:.*]]: tensor<8x!{{.*}}>
// CHECK-NEXT:    return %[[L]]
func.func @limb_round_trip_the_other_way(%l: tensor<8x!field.pf<12289:i64>>)
    -> tensor<8x!field.pf<12289:i64>> {
  %r = ring.from_limbs %l
      : tensor<8x!field.pf<12289:i64>> to !ring.rq<[12289], 8 : i32>
  %out = ring.to_limbs %r
      : !ring.rq<[12289], 8 : i32> to tensor<8x!field.pf<12289:i64>>
  return %out : tensor<8x!field.pf<12289:i64>>
}
