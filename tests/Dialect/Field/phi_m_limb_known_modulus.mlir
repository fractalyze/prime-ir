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

// Verifies the six Goldilocks-class Phi_m ciphertext-limb primes
// (goldilocks2..goldilocks7) registered upstream in zk_dtypes resolve through
// getKnownModulusAlias. When a mod_arith type carries one of these moduli the
// ModArith OpAsmDialectInterface emits the `!z_goldilocksN` type alias; this is
// only possible if registerKnownModulusAliases picked the field up via
// ZK_DTYPES_ALL_PRIME_FIELD_TYPE_LIST. The aliases below are the gate.

// RUN: prime-ir-opt -field-to-mod-arith %s | FileCheck %s

// CHECK-DAG: !z_goldilocks2 = !mod_arith.int<18446744056529682433 : i64>
// CHECK-DAG: !z_goldilocks3 = !mod_arith.int<18446743880436023297 : i64>
// CHECK-DAG: !z_goldilocks4 = !mod_arith.int<18446743841781317633 : i64>
// CHECK-DAG: !z_goldilocks5 = !mod_arith.int<18446743824601448449 : i64>
// CHECK-DAG: !z_goldilocks6 = !mod_arith.int<18446743751587004417 : i64>
// CHECK-DAG: !z_goldilocks7 = !mod_arith.int<18446743738702102529 : i64>

!PF2 = !field.pf<18446744056529682433:i64>
!PF3 = !field.pf<18446743880436023297:i64>
!PF4 = !field.pf<18446743841781317633:i64>
!PF5 = !field.pf<18446743824601448449:i64>
!PF6 = !field.pf<18446743751587004417:i64>
!PF7 = !field.pf<18446743738702102529:i64>

// CHECK-LABEL: @goldilocks2_add
func.func @goldilocks2_add(%a: !PF2, %b: !PF2) -> !PF2 {
  %0 = field.add %a, %b : !PF2
  return %0 : !PF2
}

// CHECK-LABEL: @goldilocks3_add
func.func @goldilocks3_add(%a: !PF3, %b: !PF3) -> !PF3 {
  %0 = field.add %a, %b : !PF3
  return %0 : !PF3
}

// CHECK-LABEL: @goldilocks4_add
func.func @goldilocks4_add(%a: !PF4, %b: !PF4) -> !PF4 {
  %0 = field.add %a, %b : !PF4
  return %0 : !PF4
}

// CHECK-LABEL: @goldilocks5_add
func.func @goldilocks5_add(%a: !PF5, %b: !PF5) -> !PF5 {
  %0 = field.add %a, %b : !PF5
  return %0 : !PF5
}

// CHECK-LABEL: @goldilocks6_add
func.func @goldilocks6_add(%a: !PF6, %b: !PF6) -> !PF6 {
  %0 = field.add %a, %b : !PF6
  return %0 : !PF6
}

// CHECK-LABEL: @goldilocks7_add
func.func @goldilocks7_add(%a: !PF7, %b: !PF7) -> !PF7 {
  %0 = field.add %a, %b : !PF7
  return %0 : !PF7
}
