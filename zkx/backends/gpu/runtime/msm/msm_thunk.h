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

#ifndef ZKX_BACKENDS_GPU_RUNTIME_MSM_MSM_THUNK_H_
#define ZKX_BACKENDS_GPU_RUNTIME_MSM_MSM_THUNK_H_

#include <cstdint>

#include "zkx/backends/gpu/runtime/thunk.h"
#include "zkx/service/buffer_assignment.h"
#include "zkx/zkx_data.pb.h"

namespace zkx::gpu {

// Thunk that runs BN254 G1/G2 MSM using ICICLE's CUDA implementation.
//
// Operands:
//   - scalars: [N] array of Fr elements (32 bytes each)
//   - bases:   [N] array of affine G1/G2 points
// Result:
//   - A single affine G1/G2 point
//
// Montgomery/standard form is derived from the element types at thunk creation
// time. Passing standard-form inputs skips ICICLE's internal Montgomery
// conversion (faster).
class MsmThunk : public Thunk {
 public:
  MsmThunk(ThunkInfo thunk_info, BufferAllocation::Slice scalars,
           BufferAllocation::Slice bases, BufferAllocation::Slice result,
           int32_t msm_size, int32_t window_bits,
           PrimitiveType scalars_element_type, PrimitiveType bases_element_type,
           PrimitiveType result_element_type);

  MsmThunk(const MsmThunk&) = delete;
  MsmThunk& operator=(const MsmThunk&) = delete;

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

  std::string ToString(int indent) const override;

 private:
  BufferAllocation::Slice scalars_;
  BufferAllocation::Slice bases_;
  BufferAllocation::Slice result_;
  int32_t msm_size_;
  int32_t window_bits_;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-private-field"
  PrimitiveType scalars_element_type_;
  PrimitiveType bases_element_type_;
  PrimitiveType result_element_type_;
#pragma GCC diagnostic pop
};

}  // namespace zkx::gpu

#endif  // ZKX_BACKENDS_GPU_RUNTIME_MSM_MSM_THUNK_H_
