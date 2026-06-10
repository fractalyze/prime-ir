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

// RUN: prime-ir-opt -reassociate-field-add %s | FileCheck %s -enable-var-scope

!PF = !field.pf<17:i32>
!QF = !field.ef<2x!PF, 6:i32>

//===----------------------------------------------------------------------===//
// Serial chain -> balanced tree
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @serial_chain_to_balanced_tree
func.func @serial_chain_to_balanced_tree(
    %a: !PF, %b: !PF, %c: !PF, %d: !PF,
    %e: !PF, %f: !PF, %g: !PF, %h: !PF) -> !PF {
  // CHECK: %[[S0:.*]] = field.add %arg0, %arg1
  // CHECK: %[[S1:.*]] = field.add %arg2, %arg3
  // CHECK: %[[S2:.*]] = field.add %arg4, %arg5
  // CHECK: %[[S3:.*]] = field.add %arg6, %arg7
  // CHECK: %[[S4:.*]] = field.add %[[S0]], %[[S1]]
  // CHECK: %[[S5:.*]] = field.add %[[S2]], %[[S3]]
  // CHECK: %[[S6:.*]] = field.add %[[S4]], %[[S5]]
  // CHECK: return %[[S6]]
  %0 = field.add %a, %b : !PF
  %1 = field.add %0, %c : !PF
  %2 = field.add %1, %d : !PF
  %3 = field.add %2, %e : !PF
  %4 = field.add %3, %f : !PF
  %5 = field.add %4, %g : !PF
  %6 = field.add %5, %h : !PF
  return %6 : !PF
}

//===----------------------------------------------------------------------===//
// Rank ordering: the operand at the end of a multiply chain is the deepest,
// so it must fold into the tree last (rightmost operand of the final add),
// keeping it off the critical path of the other adds.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @rank_orders_deep_operand_last
func.func @rank_orders_deep_operand_last(
    %a: !PF, %b: !PF, %c: !PF, %d: !PF, %x: !PF) -> !PF {
  // CHECK: %[[M1:.*]] = field.mul %arg4, %arg4
  // CHECK: %[[M2:.*]] = field.mul %[[M1]], %arg4
  // CHECK: %[[M3:.*]] = field.mul %[[M2]], %arg4
  // CHECK: %[[S0:.*]] = field.add %arg0, %arg1
  // CHECK: %[[S1:.*]] = field.add %arg2, %arg3
  // CHECK: %[[S2:.*]] = field.add %[[S0]], %[[S1]]
  // CHECK: %[[S3:.*]] = field.add %[[S2]], %[[M3]]
  // CHECK: return %[[S3]]
  %m1 = field.mul %x, %x : !PF
  %m2 = field.mul %m1, %x : !PF
  %m3 = field.mul %m2, %x : !PF
  %0 = field.add %m3, %a : !PF
  %1 = field.add %0, %b : !PF
  %2 = field.add %1, %c : !PF
  %3 = field.add %2, %d : !PF
  return %3 : !PF
}

//===----------------------------------------------------------------------===//
// Extension-field chains rebalance the same way.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @ext_field_chain
func.func @ext_field_chain(%a: !QF, %b: !QF, %c: !QF, %d: !QF) -> !QF {
  // CHECK: %[[S0:.*]] = field.add %arg0, %arg1
  // CHECK: %[[S1:.*]] = field.add %arg2, %arg3
  // CHECK: %[[S2:.*]] = field.add %[[S0]], %[[S1]]
  // CHECK: return %[[S2]]
  %0 = field.add %a, %b : !QF
  %1 = field.add %0, %c : !QF
  %2 = field.add %1, %d : !QF
  return %2 : !QF
}

//===----------------------------------------------------------------------===//
// Tensor (elementwise) chains are field-like containers and rebalance too.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @tensor_elementwise_chain
func.func @tensor_elementwise_chain(
    %a: tensor<2x!PF>, %b: tensor<2x!PF>,
    %c: tensor<2x!PF>, %d: tensor<2x!PF>) -> tensor<2x!PF> {
  // CHECK: %[[S0:.*]] = field.add %arg0, %arg1
  // CHECK: %[[S1:.*]] = field.add %arg2, %arg3
  // CHECK: %[[S2:.*]] = field.add %[[S0]], %[[S1]]
  // CHECK: return %[[S2]]
  %0 = field.add %a, %b : tensor<2x!PF>
  %1 = field.add %0, %c : tensor<2x!PF>
  %2 = field.add %1, %d : tensor<2x!PF>
  return %2 : tensor<2x!PF>
}

//===----------------------------------------------------------------------===//
// A single add is not a chain; it must be left untouched.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @single_add_untouched
func.func @single_add_untouched(%a: !PF, %b: !PF) -> !PF {
  // CHECK: %[[S:.*]] = field.add %arg0, %arg1
  // CHECK: return %[[S]]
  %0 = field.add %a, %b : !PF
  return %0 : !PF
}

//===----------------------------------------------------------------------===//
// A multi-use intermediate is a chain boundary: collapsing it into the
// outer sum would duplicate work, so both adds stay as written.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @multi_use_intermediate_untouched
func.func @multi_use_intermediate_untouched(
    %a: !PF, %b: !PF, %c: !PF) -> (!PF, !PF) {
  // CHECK: %[[S0:.*]] = field.add %arg0, %arg1
  // CHECK: %[[S1:.*]] = field.add %[[S0]], %arg2
  // CHECK: return %[[S0]], %[[S1]]
  %0 = field.add %a, %b : !PF
  %1 = field.add %0, %c : !PF
  return %0, %1 : !PF, !PF
}

//===----------------------------------------------------------------------===//
// A mixed-type add (extension field + base field) is a chain boundary, but
// the uniform sub-chain feeding it still rebalances.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @mixed_type_add_is_boundary
func.func @mixed_type_add_is_boundary(
    %a: !PF, %b: !PF, %c: !PF, %d: !PF, %e: !QF) -> !QF {
  // CHECK: %[[S0:.*]] = field.add %arg0, %arg1
  // CHECK: %[[S1:.*]] = field.add %arg2, %arg3
  // CHECK: %[[S2:.*]] = field.add %[[S0]], %[[S1]]
  // CHECK: %[[R:.*]] = field.add %arg4, %[[S2]] : !field.ef<{{.*}}>, !pf
  // CHECK: return %[[R]]
  %0 = field.add %a, %b : !PF
  %1 = field.add %0, %c : !PF
  %2 = field.add %1, %d : !PF
  %3 = field.add %e, %2 : !QF, !PF
  return %3 : !QF
}
