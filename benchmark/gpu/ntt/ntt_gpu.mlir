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

!PF = !field.pf<21888242871839275222246405745257275088548364400416034343698204186575808495617 : i256>
!PFm = !field.pf<21888242871839275222246405745257275088548364400416034343698204186575808495617 : i256, true>

#root_elem = #field.pf.elem<17220337697351015657950521176323262483320249231368149235373741788599650842711:i256> : !PF
#root_of_unity = #field.root_of_unity<#root_elem, 1048576:i256>

func.func @ntt_gpu(%arg0 : memref<1048576x!PFm>) attributes { llvm.emit_c_interface } {
  %t = bufferization.to_tensor %arg0 restrict writable : memref<1048576x!PFm> to tensor<1048576x!PFm>
  %res = poly.ntt %t into %t {
    root=#root_of_unity,
    tileX=256,
    ntt_gpu_mapping = [
      #gpu.loop_dim_map<processor = block_x,  map = (d0) -> (d0), bound = (d0) -> (d0)>,
      #gpu.loop_dim_map<processor = block_y,  map = (d0) -> (d0), bound = (d0) -> (d0)>,
      #gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>,
      #gpu.loop_dim_map<processor = thread_y, map = (d0) -> (d0), bound = (d0) -> (d0)>
    ]
  }: tensor<1048576x!PFm>
  bufferization.materialize_in_destination %res in writable %arg0 : (tensor<1048576x!PFm>, memref<1048576x!PFm>) -> ()
  return
}

func.func @intt_gpu(%arg0 : memref<1048576x!PFm>) attributes { llvm.emit_c_interface } {
  %t = bufferization.to_tensor %arg0 restrict writable : memref<1048576x!PFm> to tensor<1048576x!PFm>
  %res = poly.ntt %t into %t {
    root=#root_of_unity,
    tileX=256,
    ntt_gpu_mapping = [
      #gpu.loop_dim_map<processor = block_x,  map = (d0) -> (d0), bound = (d0) -> (d0)>,
      #gpu.loop_dim_map<processor = block_y,  map = (d0) -> (d0), bound = (d0) -> (d0)>,
      #gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>,
      #gpu.loop_dim_map<processor = thread_y, map = (d0) -> (d0), bound = (d0) -> (d0)>
    ]
  } inverse=true: tensor<1048576x!PFm>
  bufferization.materialize_in_destination %res in writable %arg0 : (tensor<1048576x!PFm>, memref<1048576x!PFm>) -> ()
  return
}
