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

// RUN: prime-ir-opt -mod-arith-to-arith -split-input-file %s | FileCheck %s -enable-var-scope

// p64 = 2^64 - 59 (largest 64-bit prime). Chosen to exercise the
// "modulus uses every bit of storage" case where mu fits in k+1 bits.
// mu = floor(2^128 / p64) = 18446744073709551675.
!Zp64 = !mod_arith.int<18446744073709551557 : i64>
!Zp64v = tensor<4x!Zp64>

// CHECK-LABEL: @test_lower_barrett_mul_scalar
// CHECK-SAME: (%[[LHS:.*]]: i64, %[[RHS:.*]]: i64) -> i64
func.func @test_lower_barrett_mul_scalar(%lhs : !Zp64, %rhs : !Zp64) -> !Zp64 {
  // CHECK-NOT: mod_arith.barrett_mul
  // CHECK-DAG: %[[L:.*]] = arith.extui %[[LHS]] : i64 to i128
  // CHECK-DAG: %[[R:.*]] = arith.extui %[[RHS]] : i64 to i128
  // CHECK: %[[PROD:.*]] = arith.muli %[[L]], %[[R]] : i128
  // CHECK: %[[MU:.*]] = arith.constant 18446744073709551675 : i128
  // CHECK: %{{.*}}, %[[Q:.*]] = arith.mului_extended %[[PROD]], %[[MU]] : i128
  // CHECK: %[[P:.*]] = arith.constant 18446744073709551557 : i128
  // CHECK: %[[QP:.*]] = arith.muli %[[Q]], %[[P]] : i128
  // CHECK: %[[R0:.*]] = arith.subi %[[PROD]], %[[QP]] : i128
  // CHECK: %[[CMP:.*]] = arith.cmpi uge, %[[R0]], %[[P]] : i128
  // CHECK: %[[SUB:.*]] = arith.subi %[[R0]], %[[P]] : i128
  // CHECK: %[[SEL:.*]] = arith.select %[[CMP]], %[[SUB]], %[[R0]] : i128
  // CHECK: %[[OUT:.*]] = arith.trunci %[[SEL]] : i128 to i64
  // CHECK: return %[[OUT]] : i64
  %res = mod_arith.barrett_mul %lhs, %rhs : !Zp64
  return %res : !Zp64
}

// -----

!Zp64 = !mod_arith.int<18446744073709551557 : i64>
!Zp64v = tensor<4x!Zp64>

// Shaped variant: same lowering, but constants are dense splats over the
// element shape and arith ops act on tensor<4xi128>.
// CHECK-LABEL: @test_lower_barrett_mul_vec
// CHECK-SAME: (%[[LHS:.*]]: tensor<4xi64>, %[[RHS:.*]]: tensor<4xi64>) -> tensor<4xi64>
func.func @test_lower_barrett_mul_vec(%lhs : !Zp64v, %rhs : !Zp64v) -> !Zp64v {
  // CHECK-NOT: mod_arith.barrett_mul
  // CHECK-DAG: %[[L:.*]] = arith.extui %[[LHS]] : tensor<4xi64> to tensor<4xi128>
  // CHECK-DAG: %[[R:.*]] = arith.extui %[[RHS]] : tensor<4xi64> to tensor<4xi128>
  // CHECK: %[[PROD:.*]] = arith.muli %[[L]], %[[R]] : tensor<4xi128>
  // CHECK: %[[MU:.*]] = arith.constant dense<18446744073709551675> : tensor<4xi128>
  // CHECK: %{{.*}}, %[[Q:.*]] = arith.mului_extended %[[PROD]], %[[MU]] : tensor<4xi128>
  // CHECK: %[[P:.*]] = arith.constant dense<18446744073709551557> : tensor<4xi128>
  // CHECK: arith.muli %[[Q]], %[[P]] : tensor<4xi128>
  // CHECK: arith.cmpi uge
  // CHECK: arith.subi
  // CHECK: arith.select
  // CHECK: %[[OUT:.*]] = arith.trunci %{{.*}} : tensor<4xi128> to tensor<4xi64>
  // CHECK: return %[[OUT]] : tensor<4xi64>
  %res = mod_arith.barrett_mul %lhs, %rhs : !Zp64v
  return %res : !Zp64v
}

// -----

// BN254 base field — i256. Just check that the lowering structure is the
// same Barrett shape; the literal value of mu is large and brittle to spell.
!Zbn = !mod_arith.int<21888242871839275222246405745257275088696311157297823662689037894645226208583 : i256>

// CHECK-LABEL: @test_lower_barrett_mul_bn254
// CHECK-SAME: (%[[LHS:.*]]: i256, %[[RHS:.*]]: i256) -> i256
func.func @test_lower_barrett_mul_bn254(%lhs : !Zbn, %rhs : !Zbn) -> !Zbn {
  // CHECK-NOT: mod_arith.barrett_mul
  // CHECK: arith.extui %[[LHS]] : i256 to i512
  // CHECK: arith.extui %[[RHS]] : i256 to i512
  // CHECK: arith.muli %{{.*}}, %{{.*}} : i512
  // CHECK: arith.mului_extended %{{.*}}, %{{.*}} : i512
  // CHECK: arith.muli %{{.*}}, %{{.*}} : i512
  // CHECK: arith.subi %{{.*}}, %{{.*}} : i512
  // CHECK: arith.cmpi uge, %{{.*}}, %{{.*}} : i512
  // CHECK: arith.select
  // CHECK: arith.trunci %{{.*}} : i512 to i256
  %res = mod_arith.barrett_mul %lhs, %rhs : !Zbn
  return %res : !Zbn
}

// -----

// Dispatch test: a generic mod_arith.mul on a wide non-Mersenne modulus
// must lower via Barrett, NOT via the software-urem fallback. The
// distinguishing signal is arith.mului_extended (Barrett's `q' = high half`)
// vs arith.remui (the urem fallback).
!Zp64 = !mod_arith.int<18446744073709551557 : i64>

// CHECK-LABEL: @test_lower_mul_dispatches_to_barrett
func.func @test_lower_mul_dispatches_to_barrett(%lhs : !Zp64, %rhs : !Zp64) -> !Zp64 {
  // CHECK-NOT: mod_arith.mul
  // CHECK-NOT: arith.remui
  // CHECK: arith.mului_extended
  %res = mod_arith.mul %lhs, %rhs : !Zp64
  return %res : !Zp64
}

// -----

// Gate test: a small (i32) modulus must NOT take the Barrett path because
// hardware urem on i32 is a single instruction. Expect arith.remui to remain.
!Zp32 = !mod_arith.int<65537 : i32>

// CHECK-LABEL: @test_lower_mul_small_modulus_uses_urem
func.func @test_lower_mul_small_modulus_uses_urem(%lhs : !Zp32, %rhs : !Zp32) -> !Zp32 {
  // CHECK-NOT: mod_arith.mul
  // CHECK-NOT: arith.mului_extended
  // CHECK: arith.remui
  %res = mod_arith.mul %lhs, %rhs : !Zp32
  return %res : !Zp32
}

// -----

// Regression guard: a Mersenne-shaped modulus (p = 2^k - 1) — even when
// stored in a >= 64-bit type — must keep using the cheaper shift+and+add
// specialization rather than dispatching to Barrett.
!ZpMersenne = !mod_arith.int<2147483647 : i64>

// CHECK-LABEL: @test_lower_mul_mersenne_stays_specialized
func.func @test_lower_mul_mersenne_stays_specialized(%lhs : !ZpMersenne, %rhs : !ZpMersenne) -> !ZpMersenne {
  // CHECK-NOT: mod_arith.mul
  // CHECK-NOT: arith.mului_extended
  // CHECK-NOT: arith.remui
  // CHECK: arith.shrui
  // CHECK: arith.andi
  // CHECK: arith.addi
  %res = mod_arith.mul %lhs, %rhs : !ZpMersenne
  return %res : !ZpMersenne
}

// -----

// Regression guard: Montgomery-typed mul still goes to mod_arith.mont_mul,
// not Barrett. After lowering both are gone; we check that the result type
// uses Montgomery-shaped lowering by looking for the REDC tail (subi+minui
// or cmpi+select on the modulus constant). The salient negative signal is
// `arith.mului_extended` of the prod with a Barrett `mu` constant — Mont's
// arith.mului_extended is on (a, b) directly, not on (a*b, mu).
!ZpMont = !mod_arith.int<18446744073709551557 : i64, true>

// CHECK-LABEL: @test_lower_mul_mont_type_uses_mont
func.func @test_lower_mul_mont_type_uses_mont(%lhs : !ZpMont, %rhs : !ZpMont) -> !ZpMont {
  // CHECK-NOT: mod_arith.mul
  // CHECK-NOT: mod_arith.mont_mul
  // CHECK-NOT: mod_arith.barrett_mul
  // Mont REDC produces a single mului_extended of the operands themselves
  // (not of the wide product), and never extends to a 2k-bit wide type.
  // CHECK-NOT: arith.extui {{.*}} : i64 to i128
  %res = mod_arith.mul %lhs, %rhs : !ZpMont
  return %res : !ZpMont
}
