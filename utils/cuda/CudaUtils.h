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

} // namespace zkir::utils

#endif // UTILS_CUDA_CUDAUTILS_H_
