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

// RUN: prime-ir-opt -linalg-fuse-elementwise-ops %s | FileCheck %s -enable-var-scope

!pf_babybear_mont = !field.pf<2013265921 : i32, true>
#map = affine_map<() -> ()>

func.func @fusion(%arg0: tensor<!pf_babybear_mont>, %arg1: tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont> {
  // CHECK-LABEL: @fusion
  // CHECK-SAME: (%[[ARG0:.*]]: tensor<!pf_babybear_mont>, %[[ARG1:.*]]: tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
  // CHECK: %[[CST:.*]] = field.constant dense<536870908> : tensor<!pf_babybear_mont>
  // CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<!pf_babybear_mont>
  // CHECK: %[[RESULT:.*]] = linalg.generic
  // The constant is passed as an input and used via block argument
  // CHECK: ^bb0(%[[IN:.*]]: !pf_babybear_mont, %[[IN_0:.*]]: !pf_babybear_mont, %[[IN_1:.*]]: !pf_babybear_mont, %[[OUT:.*]]: !pf_babybear_mont):
  // CHECK:   %[[ADD:.*]] = field.add %[[IN]], %[[IN_0]] : !pf_babybear_mont
  // CHECK:   %[[MUL1:.*]] = field.mul %[[ADD]], %[[IN]] : !pf_babybear_mont
  // CHECK:   %[[MUL2:.*]] = field.mul %[[MUL1]], %[[IN_1]] : !pf_babybear_mont
  // CHECK:   linalg.yield %[[MUL2]] : !pf_babybear_mont
  // CHECK: return %[[RESULT]] : tensor<!pf_babybear_mont>
  %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%arg0, %arg1 : tensor<!pf_babybear_mont>, tensor<!pf_babybear_mont>) outs(%arg0 : tensor<!pf_babybear_mont>) {
  ^bb0(%in: !pf_babybear_mont, %in_0: !pf_babybear_mont, %out: !pf_babybear_mont):
    %ret = field.add %in, %in_0 : !pf_babybear_mont
    linalg.yield %ret : !pf_babybear_mont
  } -> tensor<!pf_babybear_mont>
  %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%0, %arg0 : tensor<!pf_babybear_mont>, tensor<!pf_babybear_mont>) outs(%0 : tensor<!pf_babybear_mont>) {
  ^bb0(%in: !pf_babybear_mont, %in_0: !pf_babybear_mont, %out: !pf_babybear_mont):
    %ret = field.mul %in, %in_0 : !pf_babybear_mont
    linalg.yield %ret : !pf_babybear_mont
  } -> tensor<!pf_babybear_mont>
  %2 = field.constant dense<536870908> : tensor<!pf_babybear_mont>
  %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%1, %2 : tensor<!pf_babybear_mont>, tensor<!pf_babybear_mont>) outs(%1 : tensor<!pf_babybear_mont>) {
  ^bb0(%in: !pf_babybear_mont, %in_0: !pf_babybear_mont, %out: !pf_babybear_mont):
    %ret = field.mul %in, %in_0 : !pf_babybear_mont
    linalg.yield %ret : !pf_babybear_mont
  } -> tensor<!pf_babybear_mont>
  return %3 : tensor<!pf_babybear_mont>
}
