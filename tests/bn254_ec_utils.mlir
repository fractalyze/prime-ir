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

func.func @printG1AffineMont(%affine: !affinem) {
  %c0 = arith.constant 0 : index
  %affine_memref = memref.alloca() : memref<1x!affinem>
  memref.store %affine, %affine_memref[%c0] : memref<1x!affinem>
  %affine_memref_cast = memref.cast %affine_memref : memref<1x!affinem> to memref<*x!affinem>
  func.call @printMemrefG1AffineMont(%affine_memref_cast) : (memref<*x!affinem>) -> ()
  return
}

func.func @printG1JacobianMont(%jacobian: !jacobianm) {
  %c0 = arith.constant 0 : index
  %jacobian_memref = memref.alloca() : memref<1x!jacobianm>
  memref.store %jacobian, %jacobian_memref[%c0] : memref<1x!jacobianm>
  %jacobian_memref_cast = memref.cast %jacobian_memref : memref<1x!jacobianm> to memref<*x!jacobianm>
  func.call @printMemrefG1JacobianMont(%jacobian_memref_cast) : (memref<*x!jacobianm>) -> ()
  return
}

func.func @printG1XyzzMont(%xyzz: !xyzzm) {
  %c0 = arith.constant 0 : index
  %xyzz_memref = memref.alloca() : memref<1x!xyzzm>
  memref.store %xyzz, %xyzz_memref[%c0] : memref<1x!xyzzm>
  %xyzz_memref_cast = memref.cast %xyzz_memref : memref<1x!xyzzm> to memref<*x!xyzzm>
  func.call @printMemrefG1XyzzMont(%xyzz_memref_cast) : (memref<*x!xyzzm>) -> ()
  return
}

func.func @printG1AffineFromJacobianMont(%jacobian: !jacobianm) {
  %affine = elliptic_curve.convert_point_type %jacobian : !jacobianm -> !affinem
  func.call @printG1AffineMont(%affine) : (!affinem) -> ()
  return
}

func.func @printG1AffineFromXyzzMont(%xyzz: !xyzzm) {
  %affine = elliptic_curve.convert_point_type %xyzz : !xyzzm -> !affinem
  func.call @printG1AffineMont(%affine) : (!affinem) -> ()
  return
}
