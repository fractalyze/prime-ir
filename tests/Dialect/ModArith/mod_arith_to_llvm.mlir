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

// Tests that ModArith ops lower through the ModArith → Arith → LLVM chain
// via the unified convert-to-llvm pass.

// RUN: prime-ir-opt -convert-to-llvm %s | FileCheck %s

!Zp = !mod_arith.int<65537 : i32>

// CHECK-LABEL: @test_mod_arith_constant
// CHECK-NOT: mod_arith.constant
func.func @test_mod_arith_constant() -> !Zp {
  %c = mod_arith.constant 5 : !Zp
  return %c : !Zp
}

// CHECK-LABEL: @test_mod_arith_add
// CHECK-NOT: mod_arith.add
func.func @test_mod_arith_add(%a: !Zp, %b: !Zp) -> !Zp {
  %c = mod_arith.add %a, %b : !Zp
  return %c : !Zp
}

// CHECK-LABEL: @test_mod_arith_mul
// CHECK-NOT: mod_arith.mul
func.func @test_mod_arith_mul(%a: !Zp, %b: !Zp) -> !Zp {
  %c = mod_arith.mul %a, %b : !Zp
  return %c : !Zp
}

// CHECK-LABEL: @test_mod_arith_sub
// CHECK-NOT: mod_arith.sub
func.func @test_mod_arith_sub(%a: !Zp, %b: !Zp) -> !Zp {
  %c = mod_arith.sub %a, %b : !Zp
  return %c : !Zp
}

// CHECK-LABEL: @test_mod_arith_negate
// CHECK-NOT: mod_arith.negate
func.func @test_mod_arith_negate(%a: !Zp) -> !Zp {
  %c = mod_arith.negate %a : !Zp
  return %c : !Zp
}
