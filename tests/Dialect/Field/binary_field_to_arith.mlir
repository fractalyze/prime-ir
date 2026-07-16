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

// RUN: prime-ir-opt --binary-field-to-arith %s | FileCheck %s

// Test binary field to arith lowering

!BF8 = !field.bf<3>  // GF(2^8)
!BF64 = !field.bf<6>  // GF(2^64) tower
!BF2E = !field.bf<1>  // GF(4), 2-bit logical, i8 carrier
!GHASH = !field.bf<7, ghash>  // GF(2^128), flat GHASH basis

// CHECK-LABEL: @test_bf_constant
// CHECK-SAME: () -> i8
func.func @test_bf_constant() -> !BF8 {
  // CHECK: arith.constant 42 : i8
  %c = field.constant 42 : !BF8
  return %c : !BF8
}

// CHECK-LABEL: @test_bf_add
// CHECK-SAME: (%arg0: i8, %arg1: i8) -> i8
func.func @test_bf_add(%a: !BF8, %b: !BF8) -> !BF8 {
  // CHECK: arith.xori
  %c = field.add %a, %b : !BF8
  return %c : !BF8
}

// CHECK-LABEL: @test_bf_sub
// CHECK-SAME: (%arg0: i8, %arg1: i8) -> i8
func.func @test_bf_sub(%a: !BF8, %b: !BF8) -> !BF8 {
  // CHECK: arith.xori
  %c = field.sub %a, %b : !BF8
  return %c : !BF8
}

// CHECK-LABEL: @test_bf_negate
// CHECK-SAME: (%arg0: i8) -> i8
func.func @test_bf_negate(%a: !BF8) -> !BF8 {
  // CHECK: return %arg0 : i8
  %c = field.negate %a : !BF8
  return %c : !BF8
}

// CHECK-LABEL: @test_bf_double
// CHECK-SAME: (%arg0: i8) -> i8
func.func @test_bf_double(%a: !BF8) -> !BF8 {
  // CHECK: arith.constant 0 : i8
  %c = field.double %a : !BF8
  return %c : !BF8
}

// CHECK-LABEL: @test_bf_mul
// CHECK-SAME: (%arg0: i8, %arg1: i8) -> i8
func.func @test_bf_mul(%a: !BF8, %b: !BF8) -> !BF8 {
  // CHECK: arith.trunci
  // CHECK: arith.shrui
  // CHECK: arith.xori
  // CHECK: arith.ori
  %c = field.mul %a, %b : !BF8
  return %c : !BF8
}

// CHECK-LABEL: @test_bf_square
// CHECK-SAME: (%arg0: i8) -> i8
func.func @test_bf_square(%a: !BF8) -> !BF8 {
  // CHECK: arith.trunci
  // CHECK: arith.shrui
  // CHECK: arith.xori
  // CHECK: arith.ori
  %c = field.square %a : !BF8
  return %c : !BF8
}

// arith.select can carry binary-field types (e.g. from a fused HLO select);
// it must be retyped onto the storage int.
// CHECK-LABEL: @test_bf_select
// CHECK-SAME: (%arg0: i1, %arg1: i8, %arg2: i8) -> i8
func.func @test_bf_select(%c: i1, %a: !BF8, %b: !BF8) -> !BF8 {
  // CHECK: arith.select %arg0, %arg1, %arg2 : i8
  %s = arith.select %c, %a, %b : !BF8
  return %s : !BF8
}

// Same, on the flat GHASH basis — i128 storage.
// CHECK-LABEL: @test_bf_ghash_select
// CHECK-SAME: (%arg0: i1, %arg1: i128, %arg2: i128) -> i128
func.func @test_bf_ghash_select(%c: i1, %a: !GHASH, %b: !GHASH) -> !GHASH {
  // CHECK: arith.select %arg0, %arg1, %arg2 : i128
  %s = arith.select %c, %a, %b : !GHASH
  return %s : !GHASH
}

// Scalar GHASH inverse lowers to the Fermat chain (linearized squares +
// carryless multiplies) on the i128 storage — no field ops survive.
// CHECK-LABEL: @test_bf_ghash_inverse
// CHECK-SAME: (%arg0: i128) -> i128
func.func @test_bf_ghash_inverse(%a: !GHASH) -> !GHASH {
  // CHECK: arith.trunci
  // CHECK: arith.shrui
  // CHECK: arith.xori
  // CHECK: arith.ori
  // CHECK-NOT: field.inverse
  %c = field.inverse %a : !GHASH
  return %c : !GHASH
}

// Wide tower inverse lowers via recursive descent: the presence of the
// level-3 lookup's scf.index_switch proves the descent reached its base
// instead of unrolling a Fermat chain (which never emits a switch and is
// ~0.5M ops at this width).
// CHECK-LABEL: @test_bf64_inverse_descent
// CHECK-SAME: (%arg0: i64) -> i64
func.func @test_bf64_inverse_descent(%a: !BF64) -> !BF64 {
  // CHECK: scf.index_switch
  // CHECK-NOT: field.inverse
  %c = field.inverse %a : !BF64
  return %c : !BF64
}

// -----

// Sub-byte binary fields lower onto a byte-rounded i8 carrier (XLA stores
// t0-t2 byte-per-element; raw i2/i4 tensor types would trigger downstream
// sub-byte packing against unpacked buffers). Math still runs on the logical
// width: trunc on entry, extui back to the carrier on exit.

// CHECK-LABEL: @test_bf_subbyte_add
// CHECK-SAME: (%arg0: i8, %arg1: i8) -> i8
func.func @test_bf_subbyte_add(%a: !BF2E, %b: !BF2E) -> !BF2E {
  // CHECK: %[[A:.*]] = arith.trunci %arg0 : i8 to i2
  // CHECK: %[[B:.*]] = arith.trunci %arg1 : i8 to i2
  // CHECK: %[[X:.*]] = arith.xori %[[A]], %[[B]] : i2
  // CHECK: %[[R:.*]] = arith.extui %[[X]] : i2 to i8
  // CHECK: return %[[R]] : i8
  %c = field.add %a, %b : !BF2E
  return %c : !BF2E
}

// Sub-byte constants re-type their i2 attribute onto the i8 carrier.
// CHECK-LABEL: @test_bf_subbyte_constant
// CHECK-SAME: () -> i8
func.func @test_bf_subbyte_constant() -> !BF2E {
  // CHECK: arith.constant 3 : i8
  %c = field.constant 3 : !BF2E
  return %c : !BF2E
}

// A bitcast whose integer side is the logical width (emitter-inserted
// truncations produce these) widens back to the carrier instead of leaving
// an unresolved i2->i8 materialization.
// CHECK-LABEL: @test_bf_subbyte_bitcast_narrow_int
// CHECK-SAME: (%arg0: i2) -> i8
func.func @test_bf_subbyte_bitcast_narrow_int(%a: i2) -> !BF2E {
  // CHECK: %[[R:.*]] = arith.extui %arg0 : i2 to i8
  // CHECK: return %[[R]] : i8
  %c = field.bitcast %a : i2 -> !BF2E
  return %c : !BF2E
}

// The reverse direction: a bf value on the i8 carrier bitcast to its exact
// logical integer width truncates back down.
// CHECK-LABEL: @test_bf_subbyte_bitcast_to_narrow_int
// CHECK-SAME: (%arg0: i8) -> i2
func.func @test_bf_subbyte_bitcast_to_narrow_int(%a: !BF2E) -> i2 {
  // CHECK: %[[R:.*]] = arith.trunci %arg0 : i8 to i2
  // CHECK: return %[[R]] : i2
  %c = field.bitcast %a : !BF2E -> i2
  return %c : i2
}
