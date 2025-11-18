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

#ifndef UTILS_CUDA_CUDAUTILS_H_
#define UTILS_CUDA_CUDAUTILS_H_

#include <cstdio>
#include <cstdlib>
#include <memory>

#include "cuda_runtime_api.h" // NOLINT(build/include_subdir)
#include "llvm/Support/raw_ostream.h"

namespace zkir::utils {

#define CHECK_CUDA_ERROR(call)                                                 \
  do {                                                                         \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error at %s:%d - %s failed: %s (%d)\n",       \
                   __FILE__, __LINE__, #call, cudaGetErrorString(error),       \
                   error);                                                     \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

// CUDA device memory deleter for use with unique_ptr
struct CudaDeleter {
  void operator()(void *ptr) const noexcept {
    if (ptr) {
      // In destructor context, we can't abort() as it would prevent proper
      // cleanup of other objects. Handle errors gracefully instead.
      cudaError_t error = cudaFree(ptr);
      if (error != cudaSuccess) {
        llvm::errs() << "CUDA error in destructor - cudaAsync failed: "
                     << cudaGetErrorString(error) << " (" << error << ")\n";
      }
    }
  }
};

// Convenience alias for CUDA device memory unique_ptr
template <typename T>
using CudaUniquePtr = std::unique_ptr<T, CudaDeleter>;

// Helper function to allocate CUDA device memory with RAII
template <typename T>
CudaUniquePtr<T> makeCudaUnique(size_t count) {
  T *ptr = nullptr;
  CHECK_CUDA_ERROR(
      cudaMalloc(reinterpret_cast<void **>(&ptr), count * sizeof(T)));
  return CudaUniquePtr<T>(ptr);
}

// CUDA device memory async deleter for use with unique_ptr
struct CudaAsyncDeleter {
  cudaStream_t stream{nullptr}; // store stream inside the deleter

  void operator()(void *ptr) const noexcept {
    if (ptr) {
      // In destructor context, we can't abort() as it would prevent proper
      // cleanup of other objects. Handle errors gracefully instead.
      cudaError_t error = cudaFreeAsync(ptr, stream);
      if (error != cudaSuccess) {
        llvm::errs() << "CUDA error in destructor - cudaFreeAsync failed: "
                     << cudaGetErrorString(error) << " (" << error << ")\n";
      }
    }
  }
};

template <typename T>
using CudaAsyncUniquePtr = std::unique_ptr<T, CudaAsyncDeleter>;

template <typename T>
CudaAsyncUniquePtr<T> makeCudaAsyncUnique(size_t count, cudaStream_t stream) {
  T *ptr = nullptr;
  CHECK_CUDA_ERROR(cudaMallocAsync(reinterpret_cast<void **>(&ptr),
                                   count * sizeof(T), stream));
  return CudaAsyncUniquePtr<T>(ptr, CudaAsyncDeleter{stream});
}

} // namespace zkir::utils

#endif // UTILS_CUDA_CUDAUTILS_H_
