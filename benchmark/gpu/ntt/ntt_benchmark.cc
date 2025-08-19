#include <climits>
#include <cstring>
#include <random>

#include "benchmark/BenchmarkUtils.h"
#include "benchmark/CudaUtils.h"
#include "benchmark/benchmark.h"
#include "cuda_runtime_api.h" // NOLINT(build/include_subdir)
#include "gtest/gtest.h"
#include "mlir/ExecutionEngine/MemRefUtils.h"
#include "mlir/Support/LLVM.h"

#define NUM_COEFFS (1 << 20)

namespace mlir::zkir::benchmark {
namespace {

using i256 = BigInt<4>;

extern "C" void _mlir_ciface_ntt_cpu(StridedMemRefType<i256, 1> *input);
extern "C" void _mlir_ciface_intt_cpu(StridedMemRefType<i256, 1> *input);
extern "C" void _mlir_ciface_ntt_gpu(StridedMemRefType<i256, 1> *input);
extern "C" void _mlir_ciface_intt_gpu(StridedMemRefType<i256, 1> *input);

// `kPrime` =
// 21888242871839275222246405745257275088548364400416034343698204186575808495617
const i256 kPrime = i256::fromHexString(
    "0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001");

// Set up the random number generator.
std::mt19937_64 rng(std::random_device{}()); // NOLINT(whitespace/braces)
std::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);

// Set the element to random number in [0, kPrime).
void fillWithRandom(i256 &elem, ArrayRef<int64_t> coords) {
  elem = i256::randomLT(kPrime, rng, dist);
}

template <bool kIsGPU>
void BM_ntt_benchmark(::benchmark::State &state) {
  OwningMemRef<i256, 1> hInput(/*shape=*/{NUM_COEFFS}, /*shapeAlloc=*/{},
                               /*init=*/fillWithRandom);
  OwningMemRef<i256, 1> hTemp({NUM_COEFFS}, {});

  const size_t bytes = sizeof(i256) * NUM_COEFFS;

  if constexpr (kIsGPU) {
    auto dInputBuf = makeCudaUnique<i256>(NUM_COEFFS);
    auto dTmpBuf = makeCudaUnique<i256>(NUM_COEFFS);

    CHECK_CUDA_ERROR(cudaMemcpy(dInputBuf.get(), hInput->data, bytes,
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    StridedMemRefType<i256, 1> dTmpRef{/*basePtr=*/dTmpBuf.get(),
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
