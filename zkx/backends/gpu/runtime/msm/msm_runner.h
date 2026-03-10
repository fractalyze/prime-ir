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

#ifndef ZKX_BACKENDS_GPU_RUNTIME_MSM_MSM_RUNNER_H_
#define ZKX_BACKENDS_GPU_RUNTIME_MSM_MSM_RUNNER_H_

#include <cstdint>

namespace zkx::gpu {

// Runs BN254 G1 MSM using ICICLE's bucket-method CUDA implementation.
//
// All pointers must point to device memory. Scalars are Fr elements (32 bytes
// each), bases are affine G1 points (64 bytes each: x,y in Fq), and result is
// a single affine G1 point (64 bytes: x,y in Fq).
//
// The Montgomery flags control whether ICICLE performs Montgomery ↔ standard
// conversions on inputs/outputs. Passing standard-form inputs with the
// corresponding flag set to false skips the internal conversion (faster).
//
// Returns 0 on success, non-zero on error.
// The error_msg pointer (if non-null) is set to a static error string.
int RunBn254G1Msm(const void* d_scalars, const void* d_bases, int msm_size,
                  void* d_result_affine, void* stream, int window_bits = 0,
                  bool scalars_montgomery = true, bool points_montgomery = true,
                  bool result_montgomery = true,
                  const char** error_msg = nullptr);

// Runs BN254 G2 MSM using ICICLE's bucket-method CUDA implementation.
//
// All pointers must point to device memory. Scalars are Fr elements (32 bytes
// each), bases are affine G2 points (128 bytes each: x,y in Fq2), and result
// is a single affine G2 point (128 bytes: x,y in Fq2).
//
// Montgomery flag semantics are the same as RunBn254G1Msm.
// Returns 0 on success, non-zero on error.
int RunBn254G2Msm(const void* d_scalars, const void* d_bases, int msm_size,
                  void* d_result_affine, void* stream, int window_bits = 0,
                  bool scalars_montgomery = true, bool points_montgomery = true,
                  bool result_montgomery = true,
                  const char** error_msg = nullptr);

}  // namespace zkx::gpu

#endif  // ZKX_BACKENDS_GPU_RUNTIME_MSM_MSM_RUNNER_H_
