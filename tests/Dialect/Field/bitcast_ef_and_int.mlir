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

// Test that field bitcasts lower to noops through the full field-to-llvm
// pipeline. Scalar EF <-> integer uses alloca + store + load (LLVM's SROA
// eliminates the alloca at -O1+). PF <-> integer folds away entirely. Tensor
// bitcasts consumed inline compile to a direct GEP + load with no copies.

// RUN: prime-ir-opt %s \
// RUN:   --field-to-llvm='bufferize-function-boundaries=true' \
// RUN:   | FileCheck %s

!PF = !field.pf<2013265921:i32, true>
!EF4 = !field.ef<4x!PF, 11:i32>

// --- Scalar EF <-> integer: alloca + store + load (noop after SROA) ---

// CHECK-LABEL: llvm.func @bitcast_ef_to_i128
// CHECK:         llvm.alloca
// CHECK-NEXT:    llvm.store
// CHECK-NEXT:    llvm.load
// CHECK-NOT:     llvm.extractvalue
// CHECK-NOT:     llvm.shl
// CHECK-NOT:     llvm.or
// CHECK:         llvm.return
func.func @bitcast_ef_to_i128(%arg0: !EF4) -> i128 {
  %0 = field.bitcast %arg0 : !EF4 -> i128
  return %0 : i128
}

// CHECK-LABEL: llvm.func @bitcast_i128_to_ef
// CHECK:         llvm.alloca
// CHECK-NEXT:    llvm.store
// CHECK-NEXT:    llvm.load
// CHECK-NOT:     llvm.lshr
// CHECK-NOT:     llvm.insertvalue
// CHECK:         llvm.return
func.func @bitcast_i128_to_ef(%arg0: i128) -> !EF4 {
  %0 = field.bitcast %arg0 : i128 -> !EF4
  return %0 : !EF4
}

// --- Scalar PF <-> integer: true noop (type converter maps PF to i32) ---

// CHECK-LABEL: llvm.func @bitcast_pf_to_i32
// CHECK-NEXT:    llvm.return %arg0
func.func @bitcast_pf_to_i32(%arg0: !PF) -> i32 {
  %0 = field.bitcast %arg0 : !PF -> i32
  return %0 : i32
}

// CHECK-LABEL: llvm.func @bitcast_i32_to_pf
// CHECK-NEXT:    llvm.return %arg0
func.func @bitcast_i32_to_pf(%arg0: i32) -> !PF {
  %0 = field.bitcast %arg0 : i32 -> !PF
  return %0 : !PF
}

// --- Tensor bitcasts consumed inline: true noops (GEP + load, no copies) ---

// EF → PF reinterpret + extract: just a GEP into the underlying PF buffer.
//
// CHECK-LABEL: llvm.func @tensor_ef_to_pf_extract
// CHECK:         llvm.getelementptr
// CHECK-NEXT:    llvm.load
// CHECK-NEXT:    llvm.return
func.func @tensor_ef_to_pf_extract(%arg0: tensor<2x!EF4>) -> i32 {
  %pf = field.bitcast %arg0 : tensor<2x!EF4> -> tensor<8x!PF>
  %idx = arith.constant 2 : index
  %elem = tensor.extract %pf[%idx] : tensor<8x!PF>
  %i = field.bitcast %elem : !PF -> i32
  return %i : i32
}

// Same-shape EF → i128 + extract: just a GEP with i128 element type.
//
// CHECK-LABEL: llvm.func @tensor_ef_to_i128_extract
// CHECK:         llvm.getelementptr
// CHECK-NEXT:    llvm.load
// CHECK-NEXT:    llvm.return
func.func @tensor_ef_to_i128_extract(%arg0: tensor<4x!EF4>) -> i128 {
  %ints = field.bitcast %arg0 : tensor<4x!EF4> -> tensor<4xi128>
  %idx = arith.constant 2 : index
  %elem = tensor.extract %ints[%idx] : tensor<4xi128>
  return %elem : i128
}
