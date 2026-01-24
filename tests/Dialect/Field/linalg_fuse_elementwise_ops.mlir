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

!pf_babybear = !field.pf<2013265921 : i32, true>
#map = affine_map<() -> ()>

func.func @fusion(%arg0: tensor<!pf_babybear>, %arg1: tensor<!pf_babybear>) -> tensor<!pf_babybear> {
  // CHECK-LABEL: @fusion
  // CHECK-SAME: (%[[ARG0:.*]]: tensor<!pf_babybear>, %[[ARG1:.*]]: tensor<!pf_babybear>) -> tensor<!pf_babybear>
  // CHECK: %[[CST:.*]] = field.constant dense<536870908> : tensor<!pf_babybear>
  // CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<!pf_babybear>
  // CHECK: %[[RESULT:.*]] = linalg.generic
  // The constant is passed as an input and used via block argument
  // CHECK: ^bb0(%[[IN:.*]]: !pf_babybear, %[[IN_0:.*]]: !pf_babybear, %[[IN_1:.*]]: !pf_babybear, %[[OUT:.*]]: !pf_babybear):
  // CHECK:   %[[ADD:.*]] = field.add %[[IN]], %[[IN_0]] : !pf_babybear
  // CHECK:   %[[MUL1:.*]] = field.mul %[[ADD]], %[[IN]] : !pf_babybear
  // CHECK:   %[[MUL2:.*]] = field.mul %[[MUL1]], %[[IN_1]] : !pf_babybear
  // CHECK:   linalg.yield %[[MUL2]] : !pf_babybear
  // CHECK: return %[[RESULT]] : tensor<!pf_babybear>
  %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%arg0, %arg1 : tensor<!pf_babybear>, tensor<!pf_babybear>) outs(%arg0 : tensor<!pf_babybear>) {
  ^bb0(%in: !pf_babybear, %in_0: !pf_babybear, %out: !pf_babybear):
    %ret = field.add %in, %in_0 : !pf_babybear
    linalg.yield %ret : !pf_babybear
  } -> tensor<!pf_babybear>
  %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%0, %arg0 : tensor<!pf_babybear>, tensor<!pf_babybear>) outs(%0 : tensor<!pf_babybear>) {
  ^bb0(%in: !pf_babybear, %in_0: !pf_babybear, %out: !pf_babybear):
    %ret = field.mul %in, %in_0 : !pf_babybear
    linalg.yield %ret : !pf_babybear
  } -> tensor<!pf_babybear>
  %2 = field.constant dense<536870908> : tensor<!pf_babybear>
  %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%1, %2 : tensor<!pf_babybear>, tensor<!pf_babybear>) outs(%1 : tensor<!pf_babybear>) {
  ^bb0(%in: !pf_babybear, %in_0: !pf_babybear, %out: !pf_babybear):
    %ret = field.mul %in, %in_0 : !pf_babybear
    linalg.yield %ret : !pf_babybear
  } -> tensor<!pf_babybear>
  return %3 : tensor<!pf_babybear>
}
