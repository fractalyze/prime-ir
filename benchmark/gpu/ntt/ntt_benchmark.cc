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
#include "zk_dtypes/include/elliptic_curve/bn/bn254/fr.h"

#define NUM_COEFFS (1 << 20)

namespace mlir::zkir::benchmark {
namespace {

using F = zk_dtypes::bn254::Fr;

extern "C" void _mlir_ciface_ntt_cpu(StridedMemRefType<F, 1> *input);
extern "C" void _mlir_ciface_intt_cpu(StridedMemRefType<F, 1> *input);
extern "C" void _mlir_ciface_ntt_gpu(StridedMemRefType<F, 1> *input);
extern "C" void _mlir_ciface_intt_gpu(StridedMemRefType<F, 1> *input);

void fillWithRandom(F &elem, ArrayRef<int64_t> coords) { elem = F::Random(); }

template <bool kIsGPU>
void BM_ntt_benchmark(::benchmark::State &state) {
  OwningMemRef<F, 1> hInput(/*shape=*/{NUM_COEFFS}, /*shapeAlloc=*/{},
                            /*init=*/fillWithRandom);
  OwningMemRef<F, 1> hTemp({NUM_COEFFS}, {});

  const size_t bytes = sizeof(F) * NUM_COEFFS;

  if constexpr (kIsGPU) {
    auto dInputBuf = ::zkir::utils::makeCudaUnique<F>(NUM_COEFFS);
    auto dTmpBuf = ::zkir::utils::makeCudaUnique<F>(NUM_COEFFS);

    CHECK_CUDA_ERROR(cudaMemcpy(dInputBuf.get(), hInput->data, bytes,
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    StridedMemRefType<F, 1> dTmpRef{/*basePtr=*/dTmpBuf.get(),
                                    /*data=*/dTmpBuf.get(),
                                    /*offset=*/0,
                                    /*sizes=*/{NUM_COEFFS},
                                    /*strides=*/{1}};

    for (auto _ : state) {
      state.PauseTiming();
      CHECK_CUDA_ERROR(cudaMemcpy(dTmpBuf.get(), dInputBuf.get(), bytes,
                                  cudaMemcpyDeviceToDevice));
      CHECK_CUDA_ERROR(cudaDeviceSynchronize());
      state.ResumeTiming();

      _mlir_ciface_ntt_gpu(&dTmpRef);
    }

    _mlir_ciface_intt_gpu(&dTmpRef);

    // Copy back to host for a correctness check
    CHECK_CUDA_ERROR(
        cudaMemcpy(hTemp->data, dTmpBuf.get(), bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
  } else {
    for (auto _ : state) {
      state.PauseTiming();
      std::memcpy(hTemp->data, hInput->data, bytes);
      state.ResumeTiming();

      _mlir_ciface_ntt_cpu(&*hTemp);
    }

    _mlir_ciface_intt_cpu(&*hTemp);
  }

  for (int i = 0; i < NUM_COEFFS; i++) {
    EXPECT_EQ((*hTemp)[i], (*hInput)[i]);
  }
}

BENCHMARK_TEMPLATE(BM_ntt_benchmark, /*kIsGPU=*/false)
    ->Unit(::benchmark::kMillisecond)
    ->Name("ntt_cpu");
BENCHMARK_TEMPLATE(BM_ntt_benchmark, /*kIsGPU=*/true)
    ->Unit(::benchmark::kMillisecond)
    ->Name("ntt_gpu");

} // namespace
} // namespace mlir::zkir::benchmark

// clang-format off
// NOLINTBEGIN(whitespace/line_length)
//
// 2025-08-21T10:20:25+00:00
// Run on AMD Ryzen 9 9950X3D (32 X 624 MHz CPU s)
// CPU Caches:
//   L1 Data 48 KiB (x16)
//   L1 Instruction 32 KiB (x16)
//   L2 Unified 1024 KiB (x16)
//   L3 Unified 98304 KiB (x2)
// Load Average: 4.43, 3.88, 4.71
// -----------------------------------------------------
// Benchmark           Time             CPU   Iterations
// -----------------------------------------------------
// ntt_cpu          49.5 ms         43.0 ms           18
// ntt_gpu          6.02 ms         6.02 ms          116
// NOLINTEND()
