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

// RUN: prime-ir-opt -canonicalize -mod-arith-to-arith %s | FileCheck %s --check-prefix=CHECK-LOWERING

// RUN: prime-ir-opt %s -canonicalize -mod-arith-to-arith -convert-to-llvm \
// RUN:   | mlir-runner -e main -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s < %t

// Goldilocks leaves no headroom bit in its i64 carrier, so for an odd operand
// above 2^64 - modulus an add-modulus-then-shift halve wraps and the shifted
// result loses exactly 2^63. Pins the carry-free shift-then-add lowering on
// operands that exercised the wrap (values with bit 63 set).

func.func private @printMemrefI64(memref<*xi64>) attributes { llvm.emit_c_interface }

!mod = !mod_arith.int<18446744069414584321 : i64>

// CHECK-LOWERING-LABEL: func.func @halve(
// CHECK-LOWERING: %[[SHIFTED:.*]] = arith.shrui %{{.*}}, %{{.*}} : i64
// CHECK-LOWERING: %[[ODD_SUM:.*]] = arith.addi %[[SHIFTED]], %{{.*}} : i64
// CHECK-LOWERING: arith.select %{{.*}}, %[[ODD_SUM]], %[[SHIFTED]] : i64
func.func @halve(%arg0 : !mod) -> !mod {
  %c2 = mod_arith.constant 2 : !mod
  %inv_two = mod_arith.inverse %c2 : !mod
  %res = mod_arith.mul %arg0, %inv_two : !mod
  return %res : !mod
}

func.func @halve_negated(%arg0 : !mod) -> !mod {
  %c2 = mod_arith.constant 2 : !mod
  %inv_two = mod_arith.inverse %c2 : !mod
  %neg_inv_two = mod_arith.negate %inv_two : !mod
  %res = mod_arith.mul %arg0, %neg_inv_two : !mod
  return %res : !mod
}

func.func @main() {
  %mem = memref.alloca() : memref<1xi64>
  %mem_cast = memref.cast %mem : memref<1xi64> to memref<*xi64>
  %idx = arith.constant 0 : index

  // 2^63 + 2^62 + 12345: odd, and odd + modulus wraps the i64 carrier.
  %odd = mod_arith.constant 13835058055282176057 : !mod
  %r0 = func.call @halve(%odd) : (!mod) -> !mod
  %r0_int = mod_arith.bitcast %r0 : !mod -> i64
  memref.store %r0_int, %mem[%idx] : memref<1xi64>
  func.call @printMemrefI64(%mem_cast) : (memref<*xi64>) -> ()

  // Its even neighbor halves by the plain shift on both lowerings.
  %even = mod_arith.constant 13835058055282176058 : !mod
  %r1 = func.call @halve(%even) : (!mod) -> !mod
  %r1_int = mod_arith.bitcast %r1 : !mod -> i64
  memref.store %r1_int, %mem[%idx] : memref<1xi64>
  func.call @printMemrefI64(%mem_cast) : (memref<*xi64>) -> ()

  %r2 = func.call @halve_negated(%odd) : (!mod) -> !mod
  %r2_int = mod_arith.bitcast %r2 : !mod -> i64
  memref.store %r2_int, %mem[%idx] : memref<1xi64>
  func.call @printMemrefI64(%mem_cast) : (memref<*xi64>) -> ()
  return
}

// (odd + modulus)/2 = 16140901062348380189; printMemrefI64 renders i64
// signed, so the bit-63-set result prints negative.
// CHECK: [-2305843011361171427]
// CHECK: [6917529027641088029]
// CHECK: [2305843007066204132]
