// Copyright 2025 The ZKIR Authors.
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

// RUN: cat %S/../../bn254_field_defs.mlir %s \
// RUN:   | zkir-opt -convert-to-llvm -split-input-file \
// RUN:   | FileCheck %s -enable-var-scope

// CHECK-LABEL: @test_ext_from_coeffs
func.func @test_ext_from_coeffs(%var1: i256, %var2: i256) -> !QFm {
  // CHECK-NOT: field.ext_from_coeffs
  %ext = field.ext_from_coeffs %var1, %var2 : (i256, i256) -> !QFm
  return %ext : !QFm
}

// CHECK-LABEL: @test_ext_to_coeffs
func.func @test_ext_to_coeffs(%ext: !QFm) -> i256 {
  // CHECK-NOT: field.ext_to_coeffs
  %coeffs:2 = field.ext_to_coeffs %ext : (!QFm) -> (i256, i256)
  return %coeffs#1 : i256
}
