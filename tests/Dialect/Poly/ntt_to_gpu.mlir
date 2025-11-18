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

// RUN: zkir-opt %s -poly-to-field -field-to-gpu="bufferize-function-boundaries parallelize-affine target-format=llvm nvvm-use-bare-ptr-call-conv" | FileCheck %s

!PF = !field.pf<7681:i32>
#elem = #field.pf.elem<3383:i32>  : !PF
#root_of_unity = #field.root_of_unity<#elem, 4:i32>

// CHECK-LABEL: @ntt_in_place
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @ntt_in_place(%arg0: tensor<4x!PF>) -> tensor<4x!PF> {
  // CHECK-NOT: poly.ntt
  // CHECK: gpu.launch_func @ntt_in_place_kernel
  // CHECK: gpu.launch_func @ntt_in_place_kernel
  %1 = poly.ntt %arg0 into %arg0 {
    root=#root_of_unity,
    tileX=256,
    ntt_gpu_mapping = [
      #gpu.loop_dim_map<processor = block_x,  map = (d0) -> (d0), bound = (d0) -> (d0)>,
      #gpu.loop_dim_map<processor = block_y,  map = (d0) -> (d0), bound = (d0) -> (d0)>,
      #gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>,
      #gpu.loop_dim_map<processor = thread_y, map = (d0) -> (d0), bound = (d0) -> (d0)>
    ],
    bit_reverse_gpu_mapping = [#gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]
  } : tensor<4x!PF>
  return %1 : tensor<4x!PF>
}

// CHECK-LABEL: @intt_in_place
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @intt_in_place(%arg0: tensor<4x!PF>) -> tensor<4x!PF> {
  // CHECK-NOT: poly.ntt
  // CHECK: gpu.launch_func @intt_in_place_kernel
  // CHECK: gpu.launch_func @intt_in_place_kernel
  %1 = poly.ntt %arg0 into %arg0 {
    root=#root_of_unity,
    tileX=256,
    ntt_gpu_mapping = [
      #gpu.loop_dim_map<processor = block_x,  map = (d0) -> (d0), bound = (d0) -> (d0)>,
      #gpu.loop_dim_map<processor = block_y,  map = (d0) -> (d0), bound = (d0) -> (d0)>,
      #gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>,
      #gpu.loop_dim_map<processor = thread_y, map = (d0) -> (d0), bound = (d0) -> (d0)>
    ],
    bit_reverse_gpu_mapping = [#gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]
    } inverse=true : tensor<4x!PF>
  return %1 : tensor<4x!PF>
}

// CHECK-LABEL: @ntt_out_of_place
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @ntt_out_of_place(%arg0: tensor<4x!PF> {bufferization.writable = false}) -> tensor<4x!PF> {
  // CHECK-NOT: poly.ntt
  // CHECK: gpu.launch_func @ntt_out_of_place_kernel
  // CHECK: gpu.launch_func @ntt_out_of_place_kernel
  %tmp = bufferization.alloc_tensor() : tensor<4x!PF>
  %1 = poly.ntt %arg0 into %tmp {
    root=#root_of_unity,
    ntt_gpu_mapping = [
      #gpu.loop_dim_map<processor = block_x,  map = (d0) -> (d0), bound = (d0) -> (d0)>,
      #gpu.loop_dim_map<processor = block_y,  map = (d0) -> (d0), bound = (d0) -> (d0)>,
      #gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>,
      #gpu.loop_dim_map<processor = thread_y, map = (d0) -> (d0), bound = (d0) -> (d0)>
    ],
    bit_reverse_gpu_mapping = [#gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]
    } : tensor<4x!PF>
  return %1 : tensor<4x!PF>
}

// CHECK-LABEL: @intt_out_of_place
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @intt_out_of_place(%arg0: tensor<4x!PF>) -> tensor<4x!PF> {
  // CHECK-NOT: poly.ntt
  // CHECK: gpu.launch_func @intt_out_of_place_kernel
  // CHECK: gpu.launch_func @intt_out_of_place_kernel
  %tmp = bufferization.alloc_tensor() : tensor<4x!PF>
  %1 = poly.ntt %arg0 into %tmp {
    root=#root_of_unity,
    ntt_gpu_mapping = [
      #gpu.loop_dim_map<processor = block_x,  map = (d0) -> (d0), bound = (d0) -> (d0)>,
      #gpu.loop_dim_map<processor = block_y,  map = (d0) -> (d0), bound = (d0) -> (d0)>,
      #gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>,
      #gpu.loop_dim_map<processor = thread_y, map = (d0) -> (d0), bound = (d0) -> (d0)>
    ],
    bit_reverse_gpu_mapping = [#gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]
    } inverse=true : tensor<4x!PF>
  return %1 : tensor<4x!PF>
}
