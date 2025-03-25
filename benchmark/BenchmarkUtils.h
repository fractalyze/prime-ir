#ifndef BENCHMARK_BENCHMARKUTILS_H_
#define BENCHMARK_BENCHMARKUTILS_H_

#include <cstdint>
#include <cstdlib>

namespace zkir {
namespace benchmark {

// For reference, see
// https://mlir.llvm.org/docs/TargetLLVMIR/#c-compatible-wrapper-emission
template <typename T>
class Memref {
 public:
  Memref(size_t h, size_t w) {
    allocatedPtr = reinterpret_cast<T*>(malloc(sizeof(T) * w * h));
    alignedPtr = allocatedPtr;

    offset = 0;
    sizes[0] = h;
    sizes[1] = w;
    strides[0] = w;
    strides[1] = 1;
  }

  T* pget(size_t i, size_t j) const {
    return &alignedPtr[offset + i * strides[0] + j * strides[1]];
  }

  T get(size_t i, size_t j) const { return *pget(i, j); }

 private:
  T* allocatedPtr;
  T* alignedPtr;
  size_t offset;
  size_t sizes[2];
  size_t strides[2];
};

}  // namespace benchmark
}  // namespace zkir

#endif  // BENCHMARK_BENCHMARKUTILS_H_
