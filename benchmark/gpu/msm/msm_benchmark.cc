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
#include <random>

#include "benchmark/BenchmarkUtils.h"
#include "benchmark/benchmark.h"
#include "cuda_runtime_api.h" // NOLINT(build/include_subdir)
#include "mlir/ExecutionEngine/MemRefUtils.h"
#include "mlir/Support/LLVM.h"
#include "utils/cuda/CudaUtils.h"

#define NUM_SCALARMULS (1 << 20)

namespace mlir::zkir::benchmark {
namespace {

using i256 = BigInt<4>;

// `kPrime` =
// 21888242871839275222246405745257275088548364400416034343698204186575808495617
const i256 kPrimeBase = i256::fromHexString(
    "0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47");

const i256 kPrimeScalar = i256::fromHexString(
    "0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001");

// Set up the random number generator.
std::mt19937_64 rng(std::random_device{}()); // NOLINT(whitespace/braces)
std::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);

template <bool kIsScalar>
void fillWithRandom(i256 &elem, ArrayRef<int64_t> coords) {
  if constexpr (kIsScalar) {
    elem = i256::randomLT(kPrimeScalar, rng, dist);
  } else {
    elem = i256::randomLT(kPrimeBase, rng, dist);
  }
}

extern "C" void _mlir_ciface_msm_cpu(StridedMemRefType<i256, 1> *scalars,
                                     StridedMemRefType<i256, 2> *points);
extern "C" void _mlir_ciface_msm_gpu(StridedMemRefType<i256, 1> *scalars,
                                     StridedMemRefType<i256, 2> *points);

template <bool kIsGPU>
void BM_msm_benchmark(::benchmark::State &state) {
  OwningMemRef<i256, 1> hScalars(/*shape=*/{NUM_SCALARMULS}, /*shapeAlloc=*/{},
                                 /*init=*/fillWithRandom<true>);
  OwningMemRef<i256, 2> hPoints(/*shape=*/{NUM_SCALARMULS, 2},
                                /*shapeAlloc=*/{},
                                /*init=*/fillWithRandom<false>);

  const size_t scalarBytes = sizeof(i256) * NUM_SCALARMULS;
  const size_t pointBytes = sizeof(i256) * NUM_SCALARMULS * 2;

  if constexpr (kIsGPU) {
    auto dScalarBuf = ::zkir::utils::makeCudaUnique<i256>(NUM_SCALARMULS);
    auto dPointBuf = ::zkir::utils::makeCudaUnique<i256>(NUM_SCALARMULS * 2);

    // Stage host input to device once
    CHECK_CUDA_ERROR(cudaMemcpy(dScalarBuf.get(), hScalars->data, scalarBytes,
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(dPointBuf.get(), hPoints->data, pointBytes,
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Build device-backed memrefs (same shape/stride as host)
    StridedMemRefType<i256, 1> dScalarRef{/*basePtr=*/dScalarBuf.get(),
                                          /*data=*/dScalarBuf.get(),
                                          /*offset=*/0,
                                          /*sizes=*/{NUM_SCALARMULS},
                                          /*strides=*/{1}};
    StridedMemRefType<i256, 2> dPointRef{/*basePtr=*/dPointBuf.get(),
                                         /*data=*/dPointBuf.get(),
                                         /*offset=*/0,
                                         /*sizes=*/{NUM_SCALARMULS, 2},
                                         /*strides=*/{2, 1}};

    for (auto _ : state) {
      _mlir_ciface_msm_gpu(&dScalarRef, &dPointRef);
    }
  } else {
    for (auto _ : state) {
      _mlir_ciface_msm_cpu(&*hScalars, &*hPoints);
    }
  }
}

BENCHMARK_TEMPLATE(BM_msm_benchmark, /*kIsGPU=*/false)
    ->Iterations(5)
    ->Unit(::benchmark::kMillisecond)
    ->Name("msm_cpu");

BENCHMARK_TEMPLATE(BM_msm_benchmark, /*kIsGPU=*/true)
    ->Iterations(5)
    ->Unit(::benchmark::kMillisecond)
    ->Name("msm_gpu");

} // namespace
} // namespace mlir::zkir::benchmark

// clang-format off
// NOLINTBEGIN(whitespace/line_length)
//
// 2025-08-20T04:37:37+00:00
// Run on AMD Ryzen 9 9950X3D (32 X 624 MHz CPU s)
// CPU Caches:
//   L1 Data 48 KiB (x16)
//   L1 Instruction 32 KiB (x16)
//   L2 Unified 1024 KiB (x16)
//   L3 Unified 98304 KiB (x2)
// Load Average: 0.95, 2.65, 1.79
// ---------------------------------------------------------------
// Benchmark                     Time             CPU   Iterations
// ---------------------------------------------------------------
// msm_cpu/iterations:5        316 ms          316 ms            5
// msm_gpu/iterations:5       5369 ms         5368 ms            5
// NOLINTEND()
