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

!coeff_ty = !field.pf<21888242871839275222246405745257275088548364400416034343698204186575808495617 : i256>
!coefft_ty = tensor<1048576x!coeff_ty>
!memref_ty = memref<1048576x!coeff_ty>

!coeff_ty_mont = !field.pf<21888242871839275222246405745257275088548364400416034343698204186575808495617 : i256, true>
!coefft_ty_mont = tensor<1048576x!coeff_ty_mont>
!memref_ty_mont = memref<1048576x!coeff_ty_mont>

#root_of_unity = #field.root_of_unity<17220337697351015657950521176323262483320249231368149235373741788599650842711:i256, 1048576:i256> : !coeff_ty

func.func @ntt(%arg0 : !memref_ty) attributes { llvm.emit_c_interface } {
  %t = bufferization.to_tensor %arg0 restrict writable : !memref_ty to !coefft_ty
  %res = poly.ntt %t into %t {root=#root_of_unity} : !coefft_ty
  bufferization.materialize_in_destination %res in writable %arg0 : (!coefft_ty, !memref_ty) -> ()
  return
}

func.func @intt(%arg0 : !memref_ty) attributes { llvm.emit_c_interface } {
  %t = bufferization.to_tensor %arg0 restrict writable : !memref_ty to !coefft_ty
  %res = poly.ntt %t into %t {root=#root_of_unity} inverse=true : !coefft_ty
  bufferization.materialize_in_destination %res in writable %arg0 : (!coefft_ty, !memref_ty) -> ()
  return
}

func.func @ntt_mont(%arg0 : !memref_ty_mont) attributes { llvm.emit_c_interface } {
  %t = bufferization.to_tensor %arg0 restrict writable : !memref_ty_mont to !coefft_ty_mont
  %res = poly.ntt %t into %t {root=#root_of_unity} : !coefft_ty_mont
  bufferization.materialize_in_destination %res in writable %arg0 : (!coefft_ty_mont, !memref_ty_mont) -> ()
  return
}

func.func @intt_mont(%arg0 : !memref_ty_mont) attributes { llvm.emit_c_interface } {
  %t = bufferization.to_tensor %arg0 restrict writable : !memref_ty_mont to !coefft_ty_mont
  %res = poly.ntt %t into %t {root=#root_of_unity} inverse=true : !coefft_ty_mont
  bufferization.materialize_in_destination %res in writable %arg0 : (!coefft_ty_mont, !memref_ty_mont) -> ()
  return
}
