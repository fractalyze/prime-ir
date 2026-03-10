/* Copyright 2026 The ZKX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "zkx/backends/gpu/runtime/msm/msm_runner.h"

#include <cuda_runtime.h>

// ICICLE headers
#include "icicle/curves/params/bn254.h"
#include "icicle/msm.h"
#include "third_party/icicle/cuda_msm.cuh"

namespace zkx::gpu {

using scalar_t = bn254::scalar_t;
using fq_t = bn254::point_field_t;
using affine_t = bn254::affine_t;
using projective_t = bn254::projective_t;

// Convert ICICLE's projective (Jacobian) result to affine on GPU.
// ICICLE internally converts inputs from Montgomery form and computes in
// standard form, so the output is in standard form. This kernel converts back
// to Montgomery form via to_montgomery() after the affine conversion.
__global__ void projective_to_affine_mont_kernel(const projective_t* proj,
                                                 affine_t* aff, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    affine_t a = projective_t::to_affine(proj[tid]);
    aff[tid] = {fq_t::to_montgomery(a.x), fq_t::to_montgomery(a.y)};
  }
}

// Convert projective to affine without Montgomery conversion (standard form).
__global__ void projective_to_affine_kernel(const projective_t* proj,
                                            affine_t* aff, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    aff[tid] = projective_t::to_affine(proj[tid]);
  }
}

int RunBn254G1Msm(const void* d_scalars, const void* d_bases, int msm_size,
                  void* d_result_affine, void* stream, int window_bits,
                  bool scalars_montgomery, bool points_montgomery,
                  bool result_montgomery, const char** error_msg) {
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

  // Configure ICICLE MSM.
  auto config = icicle::default_msm_config();
  config.stream = stream;
  config.are_scalars_on_device = true;
  config.are_scalars_montgomery_form = scalars_montgomery;
  config.are_points_on_device = true;
  config.are_points_montgomery_form = points_montgomery;
  config.are_results_on_device = true;
  config.is_async = true;
  if (window_bits > 0) {
    config.c = window_bits;
  }

  // ICICLE outputs projective (Jacobian) points (3 × Fq, 96 bytes each).
  // The output buffer expects affine points (2 × Fq, 64 bytes each).
  // Allocate a temporary buffer for projective results, then convert.
  projective_t* d_proj = nullptr;
  cudaError_t err = cudaMallocAsync(&d_proj, sizeof(projective_t), cuda_stream);
  if (err != cudaSuccess) {
    if (error_msg) *error_msg = cudaGetErrorString(err);
    return static_cast<int>(err);
  }

  err = msm::msm_cuda<scalar_t, affine_t, projective_t>(
      reinterpret_cast<const scalar_t*>(d_scalars),
      reinterpret_cast<const affine_t*>(d_bases), msm_size, config, d_proj);
  if (err != cudaSuccess) {
    cudaFreeAsync(d_proj, cuda_stream);
    if (error_msg) *error_msg = cudaGetErrorString(err);
    return static_cast<int>(err);
  }

  // ICICLE with is_async=true spawns a detached cleanup thread that frees
  // scratch buffers only after stream synchronization. We must sync here to
  // ensure scratch is released before any subsequent MSM call allocates,
  // preventing OOM from accumulated scratch buffers.
  err = cudaStreamSynchronize(cuda_stream);
  if (err != cudaSuccess) {
    cudaFreeAsync(d_proj, cuda_stream);
    if (error_msg) *error_msg = cudaGetErrorString(err);
    return static_cast<int>(err);
  }

  // Convert projective → affine on GPU.
  if (result_montgomery) {
    projective_to_affine_mont_kernel<<<1, 1, 0, cuda_stream>>>(
        d_proj, reinterpret_cast<affine_t*>(d_result_affine), 1);
  } else {
    projective_to_affine_kernel<<<1, 1, 0, cuda_stream>>>(
        d_proj, reinterpret_cast<affine_t*>(d_result_affine), 1);
  }
  err = cudaGetLastError();
  cudaFreeAsync(d_proj, cuda_stream);

  if (err != cudaSuccess) {
    if (error_msg) *error_msg = cudaGetErrorString(err);
    return static_cast<int>(err);
  }

  return 0;
}

// G2 types
using g2_fq_t = bn254::g2_point_field_t;
using g2_affine_t = bn254::g2_affine_t;
using g2_projective_t = bn254::g2_projective_t;

// Convert ICICLE's G2 projective result to affine Montgomery form on GPU.
__global__ void g2_projective_to_affine_mont_kernel(const g2_projective_t* proj,
                                                    g2_affine_t* aff, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    g2_affine_t a = g2_projective_t::to_affine(proj[tid]);
    aff[tid] = {g2_fq_t::to_montgomery(a.x), g2_fq_t::to_montgomery(a.y)};
  }
}

// Convert G2 projective to affine without Montgomery conversion (standard
// form).
__global__ void g2_projective_to_affine_kernel(const g2_projective_t* proj,
                                               g2_affine_t* aff, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    aff[tid] = g2_projective_t::to_affine(proj[tid]);
  }
}

int RunBn254G2Msm(const void* d_scalars, const void* d_bases, int msm_size,
                  void* d_result_affine, void* stream, int window_bits,
                  bool scalars_montgomery, bool points_montgomery,
                  bool result_montgomery, const char** error_msg) {
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

  // Configure ICICLE MSM.
  auto config = icicle::default_msm_config();
  config.stream = stream;
  config.are_scalars_on_device = true;
  config.are_scalars_montgomery_form = scalars_montgomery;
  config.are_points_on_device = true;
  config.are_points_montgomery_form = points_montgomery;
  config.are_results_on_device = true;
  config.is_async = true;
  if (window_bits > 0) {
    config.c = window_bits;
  }

  // Allocate temporary buffer for projective result.
  g2_projective_t* d_proj = nullptr;
  cudaError_t err =
      cudaMallocAsync(&d_proj, sizeof(g2_projective_t), cuda_stream);
  if (err != cudaSuccess) {
    if (error_msg) *error_msg = cudaGetErrorString(err);
    return static_cast<int>(err);
  }

  err = msm::msm_cuda<scalar_t, g2_affine_t, g2_projective_t>(
      reinterpret_cast<const scalar_t*>(d_scalars),
      reinterpret_cast<const g2_affine_t*>(d_bases), msm_size, config, d_proj);
  if (err != cudaSuccess) {
    cudaFreeAsync(d_proj, cuda_stream);
    if (error_msg) *error_msg = cudaGetErrorString(err);
    return static_cast<int>(err);
  }

  // Sync to let ICICLE's async cleanup thread free scratch buffers.
  err = cudaStreamSynchronize(cuda_stream);
  if (err != cudaSuccess) {
    cudaFreeAsync(d_proj, cuda_stream);
    if (error_msg) *error_msg = cudaGetErrorString(err);
    return static_cast<int>(err);
  }

  // Convert projective → affine on GPU.
  if (result_montgomery) {
    g2_projective_to_affine_mont_kernel<<<1, 1, 0, cuda_stream>>>(
        d_proj, reinterpret_cast<g2_affine_t*>(d_result_affine), 1);
  } else {
    g2_projective_to_affine_kernel<<<1, 1, 0, cuda_stream>>>(
        d_proj, reinterpret_cast<g2_affine_t*>(d_result_affine), 1);
  }
  err = cudaGetLastError();
  cudaFreeAsync(d_proj, cuda_stream);

  if (err != cudaSuccess) {
    if (error_msg) *error_msg = cudaGetErrorString(err);
    return static_cast<int>(err);
  }

  return 0;
}

}  // namespace zkx::gpu
