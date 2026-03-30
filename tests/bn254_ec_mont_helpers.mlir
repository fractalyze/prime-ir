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

// G1 Montgomery-form print helpers and utility functions.
// Requires bn254_defs.mlir to be concatenated before this file.

func.func private @printMemrefBn254G1AffineMont(memref<*x!affinem>) attributes { llvm.emit_c_interface }

func.func private @printMemrefG1AffineMont(%affine: memref<*x!affinem>) {
  func.call @printMemrefBn254G1AffineMont(%affine) : (memref<*x!affinem>) -> ()
  return
}

func.func private @printMemrefBn254G1JacobianMont(memref<*x!jacobianm>) attributes { llvm.emit_c_interface }

func.func private @printMemrefG1JacobianMont(%jacobian: memref<*x!jacobianm>) {
  func.call @printMemrefBn254G1JacobianMont(%jacobian) : (memref<*x!jacobianm>) -> ()
  return
}

func.func private @printMemrefBn254G1XyzzMont(memref<*x!xyzzm>) attributes { llvm.emit_c_interface }

func.func private @printMemrefG1XyzzMont(%xyzz: memref<*x!xyzzm>) {
  func.call @printMemrefBn254G1XyzzMont(%xyzz) : (memref<*x!xyzzm>) -> ()
  return
}

func.func @getG1GeneratorMultipleMont(%k: !SF) -> !affinem {
  %one = field.constant 1 : !PFm
  %two = field.constant 2 : !PFm
  %g = elliptic_curve.from_coords %one, %two : (!PFm, !PFm) -> !affinem
  %g_multiple = elliptic_curve.scalar_mul %k, %g : !SF, !affinem -> !jacobianm
  %g_multiple_affine = elliptic_curve.convert_point_type %g_multiple : !jacobianm -> !affinem
  return %g_multiple_affine : !affinem
}
