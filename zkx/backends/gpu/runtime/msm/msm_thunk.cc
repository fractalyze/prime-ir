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

#include "zkx/backends/gpu/runtime/msm/msm_thunk.h"

#include "absl/log/log.h"
#include "absl/strings/str_format.h"

#include "zkx/primitive_util.h"

#if GOOGLE_CUDA
#include "zkx/backends/gpu/runtime/msm/msm_runner.h"
#include "zkx/stream_executor/device_memory.h"
#include "zkx/stream_executor/gpu/gpu_stream.h"
#endif  // GOOGLE_CUDA

namespace zkx::gpu {

namespace {

#if GOOGLE_CUDA
bool IsG2BasesType(PrimitiveType type) {
  return type == BN254_G2_AFFINE || type == BN254_G2_AFFINE_MONT;
}
#endif  // GOOGLE_CUDA

}  // namespace

MsmThunk::MsmThunk(ThunkInfo thunk_info, BufferAllocation::Slice scalars,
                   BufferAllocation::Slice bases,
                   BufferAllocation::Slice result, int32_t msm_size,
                   int32_t window_bits, PrimitiveType scalars_element_type,
                   PrimitiveType bases_element_type,
                   PrimitiveType result_element_type)
    : Thunk(Kind::kMsm, std::move(thunk_info)),
      scalars_(scalars),
      bases_(bases),
      result_(result),
      msm_size_(msm_size),
      window_bits_(window_bits),
      scalars_element_type_(scalars_element_type),
      bases_element_type_(bases_element_type),
      result_element_type_(result_element_type) {}

std::string MsmThunk::ToString(int) const {
  return absl::StrFormat("msm_size = %d, window_bits = %d", msm_size_,
                         window_bits_);
}

absl::Status MsmThunk::ExecuteOnStream(const ExecuteParams& params) {
#if GOOGLE_CUDA
  TF_ASSIGN_OR_RETURN(
      se::Stream * stream,
      GetStreamForExecution(Thunk::execution_stream_id(), params));

  se::DeviceMemoryBase scalars_buf =
      params.buffer_allocations->GetDeviceAddress(scalars_);
  se::DeviceMemoryBase bases_buf =
      params.buffer_allocations->GetDeviceAddress(bases_);
  se::DeviceMemoryBase result_buf =
      params.buffer_allocations->GetDeviceAddress(result_);

  bool scalars_mont = primitive_util::IsMontgomeryForm(scalars_element_type_);
  bool points_mont = primitive_util::IsMontgomeryForm(bases_element_type_);
  bool result_mont = primitive_util::IsMontgomeryForm(result_element_type_);
  bool is_g2 = IsG2BasesType(bases_element_type_);

  VLOG(3) << "Running BN254 " << (is_g2 ? "G2" : "G1")
          << " MSM: size=" << msm_size_ << " window_bits=" << window_bits_
          << " scalars_mont=" << scalars_mont << " points_mont=" << points_mont
          << " result_mont=" << result_mont;

  void* cuda_stream =
      reinterpret_cast<void*>(se::gpu::AsGpuStreamValue(stream));

  const char* error_msg = nullptr;
  int rc =
      is_g2 ? RunBn254G2Msm(scalars_buf.opaque(), bases_buf.opaque(), msm_size_,
                            result_buf.opaque(), cuda_stream, window_bits_,
                            scalars_mont, points_mont, result_mont, &error_msg)
            : RunBn254G1Msm(scalars_buf.opaque(), bases_buf.opaque(), msm_size_,
                            result_buf.opaque(), cuda_stream, window_bits_,
                            scalars_mont, points_mont, result_mont, &error_msg);
  if (rc != 0) {
    return absl::InternalError(absl::StrFormat(
        "BN254 %s MSM failed (code %d): %s", is_g2 ? "G2" : "G1", rc,
        error_msg ? error_msg : "unknown error"));
  }
  return absl::OkStatus();
#else
  return absl::UnimplementedError("MSM requires CUDA");
#endif  // GOOGLE_CUDA
}

}  // namespace zkx::gpu
