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

#include <climits>
#include <cstring>

#include "benchmark/benchmark.h"
#include "cuda_runtime_api.h" // NOLINT(build/include_subdir)
#include "gtest/gtest.h"
#include "mlir/ExecutionEngine/MemRefUtils.h"
#include "mlir/Support/LLVM.h"
#include "utils/cuda/CudaUtils.h"
#include "zk_dtypes/include/field/goldilocks/goldilocks.h"

#define NUM_COEFFS (1 << 20)

namespace mlir::zkir::benchmark {
namespace {

using F = zk_dtypes::Goldilocks;

extern "C" void _mlir_ciface_matvec_cpu(StridedMemRefType<F, 2> *mat,
                                        StridedMemRefType<F, 1> *vec,
                                        StridedMemRefType<F, 1> *out);
extern "C" void _mlir_ciface_matvec_gpu(StridedMemRefType<F, 2> *mat,
                                        StridedMemRefType<F, 1> *vec,
                                        StridedMemRefType<F, 1> *out);

void fillWithRandom(F &elem, ArrayRef<int64_t> coords) { elem = F::Random(); }

template <bool kIsGPU>
void BM_matvec_benchmark(::benchmark::State &state) {
  OwningMemRef<F, 2> hMat({NUM_COEFFS, 100}, {}, fillWithRandom);
  OwningMemRef<F, 1> hVec({100}, {}, fillWithRandom);
  OwningMemRef<F, 1> hOut({NUM_COEFFS}, {}, {});

  const size_t bytesMat = NUM_COEFFS * 100 * sizeof(F);
  const size_t bytesVec = 100 * sizeof(F);
  const size_t bytesOut = NUM_COEFFS * sizeof(F);

  if constexpr (kIsGPU) {
    auto dMatBuf = ::zkir::utils::makeCudaUnique<F>(NUM_COEFFS * 100);
    auto dVecBuf = ::zkir::utils::makeCudaUnique<F>(100);
    auto dOutBuf = ::zkir::utils::makeCudaUnique<F>(NUM_COEFFS);

    CHECK_CUDA_ERROR(cudaMemcpy(dMatBuf.get(), hMat->data, bytesMat,
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(dVecBuf.get(), hVec->data, bytesVec,
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    StridedMemRefType<F, 2> dMatRef{/*basePtr=*/dMatBuf.get(),
                                    /*data=*/dMatBuf.get(),
                                    /*offset=*/0,
                                    /*sizes=*/{NUM_COEFFS, 100},
                                    /*strides=*/{100, 1}};
    StridedMemRefType<F, 1> dVecRef{/*basePtr=*/dVecBuf.get(),
                                    /*data=*/dVecBuf.get(),
                                    /*offset=*/0,
                                    /*sizes=*/{100},
                                    /*strides=*/{1}};
    StridedMemRefType<F, 1> dOutRef{/*basePtr=*/dOutBuf.get(),
                                    /*data=*/dOutBuf.get(),
                                    /*offset=*/0,
                                    /*sizes=*/{NUM_COEFFS},
                                    /*strides=*/{1}};

    for (auto _ : state) {
      state.PauseTiming();
      // Reset the output buffer to 0 since the reduction happens on top of
      // the output buffer.
      CHECK_CUDA_ERROR(cudaMemset(dOutBuf.get(), 0, NUM_COEFFS * sizeof(F)));
      CHECK_CUDA_ERROR(cudaDeviceSynchronize());
      state.ResumeTiming();

      _mlir_ciface_matvec_gpu(&dMatRef, &dVecRef, &dOutRef);
      CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    // Copy back to host for a correctness check
    OwningMemRef<F, 1> hOutGpu({NUM_COEFFS}, {}, {});
    CHECK_CUDA_ERROR(cudaMemcpy(hOutGpu->data, dOutBuf.get(), bytesOut,
                                cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    _mlir_ciface_matvec_cpu(&*hMat, &*hVec, &*hOut);
    for (int i = 0; i < NUM_COEFFS; i++) {
      EXPECT_EQ((*hOut)[i], (*hOutGpu)[i]);
    }
  } else {
    for (auto _ : state) {
      _mlir_ciface_matvec_cpu(&*hMat, &*hVec, &*hOut);
    }
  }
}

BENCHMARK_TEMPLATE(BM_matvec_benchmark, /*kIsGPU=*/false)
    ->Unit(::benchmark::kMillisecond)
    ->Name("matvec_cpu");
BENCHMARK_TEMPLATE(BM_matvec_benchmark, /*kIsGPU=*/true)
    ->Unit(::benchmark::kMillisecond)
    ->Name("matvec_gpu");

} // namespace
} // namespace mlir::zkir::benchmark

// clang-format off
// NOLINTBEGIN(whitespace/line_length)
//
// 2025-08-22T10:00:03+00:00
// Run on AMD Ryzen 9 9950X3D (32 X 5550.91 MHz CPU s)
// CPU Caches:
//   L1 Data 48 KiB (x16)
//   L1 Instruction 32 KiB (x16)
//   L2 Unified 1024 KiB (x16)
//   L3 Unified 98304 KiB (x2)
// Load Average: 0.44, 0.59, 0.73
// -----------------------------------------------------
// Benchmark           Time             CPU   Iterations
// -----------------------------------------------------
// matvec_cpu       25.0 ms         24.8 ms           32
// matvec_gpu        189 ms          189 ms            4
// NOLINTEND()
