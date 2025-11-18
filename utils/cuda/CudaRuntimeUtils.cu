/* Copyright 2025 The ZKIR Authors.

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

// clang-format off
#include <cub/device/device_radix_sort.cuh>
// clang-format on

#include <cstddef>

#include "mlir/ExecutionEngine/RunnerUtils.h"
#include "utils/cuda/CudaUtils.h"

template <typename KeyType, typename ValueType>
void sortPairs(const KeyType *keysIn, KeyType *keysOut, const ValueType *valsIn,
               ValueType *valsOut, int64_t numItems, CUstream inputStream) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(inputStream);
  zkir::utils::CudaAsyncUniquePtr<std::byte> dTempStorage = nullptr;
  size_t tempStorageBytes = 0;

  // First call to DeviceRadixSort::SortPairs determines the required size for
  // temporary storage.
  CHECK_CUDA_ERROR(cub::DeviceRadixSort::SortPairs(
      dTempStorage.get(), tempStorageBytes, keysIn, keysOut, valsIn, valsOut,
      numItems, 0, sizeof(KeyType) * 8, stream));

  dTempStorage =
      zkir::utils::makeCudaAsyncUnique<std::byte>(tempStorageBytes, stream);

  // Second call to DeviceRadixSort::SortPairs performs the actual sorting.
  CHECK_CUDA_ERROR(cub::DeviceRadixSort::SortPairs(
      dTempStorage.get(), tempStorageBytes, keysIn, keysOut, valsIn, valsOut,
      numItems, 0, sizeof(KeyType) * 8, stream));
}

extern "C" MLIR_RUNNERUTILS_EXPORT void
sortPairsI64I64(const int64_t *keysIn, int64_t *keysOut, const int64_t *valsIn,
                int64_t *valsOut, int64_t numItems, CUstream inputStream) {
  sortPairs<int64_t, int64_t>(keysIn, keysOut, valsIn, valsOut, numItems,
                              inputStream);
}
