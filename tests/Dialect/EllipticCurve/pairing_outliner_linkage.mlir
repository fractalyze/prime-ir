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

// Pairing-check lowering outlines Fp12 helpers (CyclotomicSquare, Fp12Mul,
// MulBy034, MulBy014) into private func.func ops. The host LLVM pipeline
// requires those to lower with Internal linkage; setPrivate() alone only
// updates symbol visibility, so FunctionOutlinerBase must also attach
// `llvm.linkage = #llvm.linkage<internal>` to each outlined function.

// RUN: cat %S/../../bn254_defs.mlir %s \
// RUN:   | prime-ir-opt -elliptic-curve-to-field \
// RUN:   | FileCheck %s -enable-var-scope

func.func @drive_pairing_check(
    %g1_pts: tensor<2x!affinem>,
    %g2_pts: tensor<2x!g2affinem>) -> i1 {
  %r = elliptic_curve.pairing_check %g1_pts, %g2_pts
      : tensor<2x!affinem>, tensor<2x!g2affinem> -> i1
  return %r : i1
}

// Each outlined Fp12 helper carries Internal linkage.
// CHECK-DAG: func.func private @__prime_ir_cyclotomic_square_
// CHECK-DAG-SAME: attributes {llvm.linkage = #llvm.linkage<internal>}
// CHECK-DAG: func.func private @__prime_ir_fp12_mul_
// CHECK-DAG-SAME: attributes {llvm.linkage = #llvm.linkage<internal>}
// CHECK-DAG: func.func private @__prime_ir_mul_by_034_
// CHECK-DAG-SAME: attributes {llvm.linkage = #llvm.linkage<internal>}

// No outlined helper should be missing the linkage attribute (and
// therefore default to external).
// CHECK-NOT: func.func private @__prime_ir_{{.*}}
// CHECK-NOT:   #llvm.linkage<external>
