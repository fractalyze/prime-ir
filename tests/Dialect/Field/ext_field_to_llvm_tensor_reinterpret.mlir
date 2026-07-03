// Copyright 2025 The PrimeIR Authors.
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

// RUN: prime-ir-opt --convert-ext-field-to-llvm -reconcile-unrealized-casts \
// RUN:   -split-input-file %s | FileCheck %s

// Value-path (no bufferization) tensor reinterpret bitcast — regression guard
// for xla#163.  The GPU MLIR emitter lowers tensors to raw !llvm.ptr loads
// (LowerTensors) with no bufferization stage, so a count-changing base<->EF
// reinterpret survives as `field.bitcast` on a pointer-backed tensor rather than
// as a memref bitcast.  ConvertBitcast forwards it as an unrealized cast to the
// converted element type, and reconcile-unrealized-casts collapses the
//   ptr -> tensor<in> -> tensor<out> -> ptr
// chain back to the original pointer — the reinterpret is a pure buffer
// identity (input and output hold the same total storage integers).

!PF  = !field.pf<7:i32>
!EF2 = !field.ef<2x!PF, 6:i32>

// int -> EF: the packed EF constant-pool direction that fails on GPU without
// the fix (tensor<4xi32> is 2 EF elements of degree 2).
// CHECK-LABEL: @bitcast_tensor_i32_to_ef2
//  CHECK-SAME: (%[[PTR:.+]]: !llvm.ptr)
//   CHECK-NOT:   field.bitcast
//       CHECK:   return %[[PTR]] : !llvm.ptr
func.func @bitcast_tensor_i32_to_ef2(%ptr: !llvm.ptr) -> !llvm.ptr {
  %t = builtin.unrealized_conversion_cast %ptr : !llvm.ptr to tensor<4xi32>
  %b = field.bitcast %t : tensor<4xi32> -> tensor<2x!EF2>
  %r = builtin.unrealized_conversion_cast %b : tensor<2x!EF2> to !llvm.ptr
  return %r : !llvm.ptr
}

// -----

!PF  = !field.pf<7:i32>
!EF2 = !field.ef<2x!PF, 6:i32>

// EF -> int: the reverse direction.
// CHECK-LABEL: @bitcast_tensor_ef2_to_i32
//  CHECK-SAME: (%[[PTR:.+]]: !llvm.ptr)
//   CHECK-NOT:   field.bitcast
//       CHECK:   return %[[PTR]] : !llvm.ptr
func.func @bitcast_tensor_ef2_to_i32(%ptr: !llvm.ptr) -> !llvm.ptr {
  %t = builtin.unrealized_conversion_cast %ptr : !llvm.ptr to tensor<2x!EF2>
  %b = field.bitcast %t : tensor<2x!EF2> -> tensor<4xi32>
  %r = builtin.unrealized_conversion_cast %b : tensor<4xi32> to !llvm.ptr
  return %r : !llvm.ptr
}
