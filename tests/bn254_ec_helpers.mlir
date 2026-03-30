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

// G1 standard-form print helpers and utility functions.
// Requires bn254_defs.mlir to be concatenated before this file.

func.func private @printMemrefBn254G1Affine(memref<*x!affine>) attributes { llvm.emit_c_interface }

func.func private @printMemrefG1Affine(%affine: memref<*x!affine>) {
  func.call @printMemrefBn254G1Affine(%affine) : (memref<*x!affine>) -> ()
  return
}

func.func private @printMemrefBn254G1Jacobian(memref<*x!jacobian>) attributes { llvm.emit_c_interface }

func.func private @printMemrefG1Jacobian(%jacobian: memref<*x!jacobian>) {
  func.call @printMemrefBn254G1Jacobian(%jacobian) : (memref<*x!jacobian>) -> ()
  return
}

func.func private @printMemrefBn254G1Xyzz(memref<*x!xyzz>) attributes { llvm.emit_c_interface }

func.func private @printMemrefG1Xyzz(%xyzz: memref<*x!xyzz>) {
  func.call @printMemrefBn254G1Xyzz(%xyzz) : (memref<*x!xyzz>) -> ()
  return
}

// assumes standard form scalar input and outputs affine with standard form coordinates
func.func @getG1GeneratorMultiple(%k: !SF) -> !affine {
  %one = field.constant 1 : !PF
  %two = field.constant 2 : !PF
  %g = elliptic_curve.from_coords %one, %two : (!PF, !PF) -> !affine
  %g_multiple = elliptic_curve.scalar_mul %k, %g : !SF, !affine -> !jacobian
  %g_multiple_affine = elliptic_curve.convert_point_type %g_multiple : !jacobian -> !affine
  return %g_multiple_affine : !affine
}
