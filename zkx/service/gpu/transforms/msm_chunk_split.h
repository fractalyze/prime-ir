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

#ifndef ZKX_SERVICE_GPU_TRANSFORMS_MSM_CHUNK_SPLIT_H_
#define ZKX_SERVICE_GPU_TRANSFORMS_MSM_CHUNK_SPLIT_H_

#include <cstdint>
#include <string_view>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"

#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/hlo/pass/hlo_pass_interface.h"

namespace zkx::gpu {

// Splits large MSM instructions into smaller chunks that fit in GPU memory.
//
// Estimates the GPU memory required for an MSM of size N (input buffers +
// ICICLE internal working memory) and only splits when it would exceed the
// available GPU memory budget. If the MSM fits, it is left untouched.
//
// When splitting is needed:
//
//   MSM(s[0..N), b[0..N)) =
//       MSM(s[0..K), b[0..K)) + MSM(s[K..2K), b[K..2K)) + ...
//
// This runs BEFORE MsmBatchFusion so that the resulting smaller MSMs can
// potentially be re-batched.
class MsmChunkSplit : public HloModulePass {
 public:
  // device_memory_bytes: total GPU memory in bytes (e.g., 24 GB for RTX 4090).
  //   Used to compute the maximum MSM size that fits in memory.
  // memory_fraction: fraction of GPU memory to budget for MSM (default 0.8).
  //   Leaves headroom for other allocations (HLO buffers, framework overhead).
  explicit MsmChunkSplit(int64_t device_memory_bytes,
                         double memory_fraction = 0.8)
      : device_memory_bytes_(device_memory_bytes),
        memory_fraction_(memory_fraction) {}

  std::string_view name() const override { return "msm-chunk-split"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<std::string_view>& execution_threads) override;

 private:
  int64_t device_memory_bytes_;
  double memory_fraction_;

  // Estimates the GPU memory required for an MSM of size N.
  //
  // Mirrors ICICLE's compute_required_memory() from cuda_msm.cuh:
  //   scalars_mem  = scalar_bytes × N
  //   indices_mem  = 7 × 4 × N × nof_bms       (sorting + temp storage)
  //   points_mem   = base_bytes × N × precompute_factor
  //   buckets_mem  = 4 × projective_bytes × 2^c × nof_bms_after_precomp
  //
  // where c = min(max(ceil(log₂(N)) - 4, 1), 20)   [ICICLE's get_optimal_c]
  //       nof_bms = ceil(bitsize / c)
  //       nof_bms_after_precomp = ceil(nof_bms / precompute_factor)
  static int64_t EstimateMsmMemoryBytes(int64_t msm_size, int64_t scalar_bytes,
                                        int64_t base_bytes,
                                        int64_t projective_bytes,
                                        int32_t precompute_factor);

  // Computes the maximum MSM size that fits within the memory budget
  // by binary-searching EstimateMsmMemoryBytes.
  int64_t MaxMsmSize(int64_t scalar_bytes, int64_t base_bytes,
                     int64_t projective_bytes, int32_t precompute_factor) const;

  // ICICLE's get_optimal_c: c = min(max(ceil(log₂(N)) - 4, 1), 20).
  static int GetOptimalC(int64_t msm_size);

  // BN254 scalar field bitsize.
  static constexpr int kBitsize = 254;

  // Hard cap on MSM chunk size. ICICLE's CUDA bucket-method kernels produce
  // illegal memory accesses above ~5M elements (observed on RTX 5090).
  // This is a kernel bug, not a memory issue — it crashes regardless of
  // available GPU memory. 5M is a conservative safe limit for all GPUs.
  // TODO(ryank): Remove this cap once ICICLE fixes the bucket-method kernel
  // bug for large element counts. Track upstream:
  // https://github.com/ingonyama-zk/icicle/issues
  static constexpr int64_t kMaxChunkSize = 5'000'000;
};

}  // namespace zkx::gpu

#endif  // ZKX_SERVICE_GPU_TRANSFORMS_MSM_CHUNK_SPLIT_H_
