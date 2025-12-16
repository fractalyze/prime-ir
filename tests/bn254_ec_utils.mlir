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

func.func @printG1Affine(%affine: !affine) {
  %c0 = arith.constant 0 : index
  %affine_memref = memref.alloca() : memref<1x!affine>
  memref.store %affine, %affine_memref[%c0] : memref<1x!affine>
  %affine_memref_cast = memref.cast %affine_memref : memref<1x!affine> to memref<*x!affine>
  func.call @printMemrefG1Affine(%affine_memref_cast) : (memref<*x!affine>) -> ()
  return
}

func.func @printG1Jacobian(%jacobian: !jacobian) {
  %c0 = arith.constant 0 : index
  %jacobian_memref = memref.alloca() : memref<1x!jacobian>
  memref.store %jacobian, %jacobian_memref[%c0] : memref<1x!jacobian>
  %jacobian_memref_cast = memref.cast %jacobian_memref : memref<1x!jacobian> to memref<*x!jacobian>
  func.call @printMemrefG1Jacobian(%jacobian_memref_cast) : (memref<*x!jacobian>) -> ()
  return
}

func.func @printG1Xyzz(%xyzz: !xyzz) {
  %c0 = arith.constant 0 : index
  %xyzz_memref = memref.alloca() : memref<1x!xyzz>
  memref.store %xyzz, %xyzz_memref[%c0] : memref<1x!xyzz>
  %xyzz_memref_cast = memref.cast %xyzz_memref : memref<1x!xyzz> to memref<*x!xyzz>
  func.call @printMemrefG1Xyzz(%xyzz_memref_cast) : (memref<*x!xyzz>) -> ()
  return
}

func.func @printG1AffineFromJacobian(%jacobian: !jacobian) {
  %affine = elliptic_curve.convert_point_type %jacobian : !jacobian -> !affine
  func.call @printG1Affine(%affine) : (!affine) -> ()
  return
}

func.func @printG1AffineFromXyzz(%xyzz: !xyzz) {
  %affine = elliptic_curve.convert_point_type %xyzz : !xyzz -> !affine
  func.call @printG1Affine(%affine) : (!affine) -> ()
  return
}
