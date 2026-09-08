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

// End-to-end check that outlining the tower expansion (fractalyze/prime-ir#390)
// is value-preserving. The same expectations are checked twice: once with the
// tower expanded inline, once with it outlined into shared helpers. Outlining
// only changes WHERE the arithmetic is emitted, never what it computes, so any
// divergence between the two runs is a bug in the outliner.
//
// The operands are full-width values rather than small-subfield constants
// (2, 3, 5 stay inside GF(4)/GF(16) and never exercise the upper levels), so
// every Karatsuba level and the whole norm descent is on the path.

// Inline: the default lowering.
// RUN: prime-ir-opt %s --field-to-llvm \
// RUN:   | mlir-runner -e main -entry-point-result=void \
// RUN:      --shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t.inline
// RUN: FileCheck %s < %t.inline

// Outlined at the default threshold, enabled through the PIPELINE option --
// this is the path a consumer (xla) actually takes, so it is the one worth
// executing rather than a standalone pass run.
// RUN: prime-ir-opt %s --field-to-llvm="outline-tower-ops=true" \
// RUN:   | mlir-runner -e main -entry-point-result=void \
// RUN:      --shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t.outlined
// RUN: FileCheck %s < %t.outlined

// Outlined at the lowest useful threshold — the deepest call chain, so the
// most recursion in the outliner itself.
// RUN: prime-ir-opt %s \
// RUN:     --field-to-llvm="outline-tower-ops=true outline-min-tower-level=3" \
// RUN:   | mlir-runner -e main -entry-point-result=void \
// RUN:      --shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t.outlined3
// RUN: FileCheck %s < %t.outlined3

// The runs must agree value-for-value, not merely each satisfy the checks
// above. mlir-runner prints a heap address per memref, which differs between
// processes, so compare only the printed data lines.
// RUN: grep -E "^ *\[[0-9]+\]$" %t.inline > %t.inline.vals
// RUN: grep -E "^ *\[[0-9]+\]$" %t.outlined > %t.outlined.vals
// RUN: grep -E "^ *\[[0-9]+\]$" %t.outlined3 > %t.outlined3.vals
// RUN: diff %t.inline.vals %t.outlined.vals
// RUN: diff %t.inline.vals %t.outlined3.vals

!BF64 = !field.bf<6>    // GF(2⁶⁴)
!BF128 = !field.bf<7>   // GF(2¹²⁸)

func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

func.func @emit(%v: i32) {
  %t = tensor.from_elements %v : tensor<1xi32>
  %b = bufferization.to_buffer %t : tensor<1xi32> to memref<1xi32>
  %c = memref.cast %b : memref<1xi32> to memref<*xi32>
  func.call @printMemrefI32(%c) : (memref<*xi32>) -> ()
  return
}

func.func @main() {
  %a = field.constant 123456789012345678901234567890 : !BF128
  %b = field.constant 98765432109876543210987654321 : !BF128

  // Level-7 multiply, low 32 bits of the product.
  %m = field.mul %a, %b : !BF128
  %mi = field.bitcast %m : !BF128 -> i128
  %m32 = arith.trunci %mi : i128 to i32
  func.call @emit(%m32) : (i32) -> ()
  // CHECK: {{^}}[160412733]

  // Level-7 square.
  %s = field.square %a : !BF128
  %si = field.bitcast %s : !BF128 -> i128
  %s32 = arith.trunci %si : i128 to i32
  func.call @emit(%s32) : (i32) -> ()
  // CHECK: {{^}}[471905104]

  // The inverse itself, low 32 bits. Pinning the value (not just the
  // roundtrip) catches an inverse that is self-consistently wrong.
  %inv = field.inverse %a : !BF128
  %ivi = field.bitcast %inv : !BF128 -> i128
  %iv32 = arith.trunci %ivi : i128 to i32
  func.call @emit(%iv32) : (i32) -> ()
  // CHECK: {{^}}[317449641]

  // a · a⁻¹ = 1 exercises the norm descent and the multiply together: the
  // descent is built from level-6 muls and squares, all outlined.
  %one = field.mul %a, %inv : !BF128
  %oi = field.bitcast %one : !BF128 -> i128
  %o32 = arith.trunci %oi : i128 to i32
  func.call @emit(%o32) : (i32) -> ()
  // CHECK: {{^}}[1]

  // Same roundtrip one level down, so a level-6-only regression cannot hide
  // behind the level-7 path.
  %d = field.constant 12345678901234567890 : !BF64
  %dinv = field.inverse %d : !BF64
  %done = field.mul %d, %dinv : !BF64
  %di = field.bitcast %done : !BF64 -> i64
  %d32 = arith.trunci %di : i64 to i32
  func.call @emit(%d32) : (i32) -> ()
  // CHECK: {{^}}[1]
  return
}
