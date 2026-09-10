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

// RUN: prime-ir-opt '-mod-arith-to-arith=lazy-reduction=true' %s | FileCheck %s --check-prefix=CHECK-LOWERING

// RUN: prime-ir-opt %s '-mod-arith-to-arith=lazy-reduction=true' -convert-to-llvm \
// RUN:   | mlir-runner -e main -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s < %t

// A lazy Goldilocks add hands the square a value in [0, 2^64), which the
// square pre-reduces to minui(a - p, a). Reading a - p as a signed i64 is exact
// only when p < 2^63; for Goldilocks it wraps to a + (2^64 mod p), one
// Montgomery unit, so a signed square would compute (a + 1)^2.

func.func private @printMemrefI64(memref<*xi64>) attributes { llvm.emit_c_interface }

!Gl = !mod_arith.int<18446744069414584321 : i64>
!Glm = !mod_arith.int<18446744069414584321 : i64, true>

// CHECK-LOWERING-LABEL: func.func @square_of_sum(
// CHECK-LOWERING-NOT: arith.mulsi_extended
// CHECK-LOWERING: arith.mului_extended
func.func @square_of_sum(%a : !Glm, %b : !Glm) -> !Glm {
  %s = mod_arith.add %a, %b : !Glm
  %r = mod_arith.square %s : !Glm
  return %r : !Glm
}

func.func @main() {
  %a = mod_arith.constant 3 : !Gl
  %b = mod_arith.constant 11 : !Gl
  %a_mont = mod_arith.to_mont %a : !Glm
  %b_mont = mod_arith.to_mont %b : !Glm
  %r_mont = func.call @square_of_sum(%a_mont, %b_mont) : (!Glm, !Glm) -> !Glm
  %r = mod_arith.from_mont %r_mont : !Gl
  %v = mod_arith.bitcast %r : !Gl -> i64
  %mem = memref.alloca() : memref<1xi64>
  %c0 = arith.constant 0 : index
  memref.store %v, %mem[%c0] : memref<1xi64>
  %u = memref.cast %mem : memref<1xi64> to memref<*xi64>
  func.call @printMemrefI64(%u) : (memref<*xi64>) -> ()
  return
}

// (3 + 11)^2 = 196; the signed square prints 225 = 15^2.
// CHECK: [196]
