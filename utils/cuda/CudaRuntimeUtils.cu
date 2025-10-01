// clang-format off
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
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

template <typename Type>
void Encode(const Type *in, Type *uniqueOut, Type *countsOut, Type *numRunsOut,
            int64_t numRuns, CUstream inputStream) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(inputStream);
  zkir::utils::CudaAsyncUniquePtr<std::byte> dTempStorage = nullptr;
  size_t tempStorageBytes = 0;

  // First call to DeviceRunLengthEncode::Encode determines the required size
  // for temporary storage.
  CHECK_CUDA_ERROR(cub::DeviceRunLengthEncode::Encode(
      dTempStorage.get(), tempStorageBytes, in, uniqueOut, countsOut,
      numRunsOut, numRuns, stream));

  dTempStorage =
      zkir::utils::makeCudaAsyncUnique<std::byte>(tempStorageBytes, stream);

  // Second call to DeviceRunLengthEncode::Encode performs the actual encoding.
  CHECK_CUDA_ERROR(cub::DeviceRunLengthEncode::Encode(
      dTempStorage.get(), tempStorageBytes, in, uniqueOut, countsOut,
      numRunsOut, numRuns, stream));
}

extern "C" MLIR_RUNNERUTILS_EXPORT void
EncodeI64(const int64_t *in, int64_t *uniqueOut, int64_t *countsOut,
          int64_t *numRunsOut, int64_t numRuns, CUstream inputStream) {
  Encode<int64_t>(in, uniqueOut, countsOut, numRunsOut, numRuns, inputStream);
}
