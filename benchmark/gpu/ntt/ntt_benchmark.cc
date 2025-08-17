#include <climits>
#include <cstring>
#include <random>

#include "benchmark/BenchmarkUtils.h"
#include "benchmark/benchmark.h"
#include "gtest/gtest.h"
#include "mlir/ExecutionEngine/MemRefUtils.h"
#include "mlir/Support/LLVM.h"

#define NUM_COEFFS (1 << 20)

namespace mlir::zkir::benchmark {
namespace {

using i64 = BigInt<1>;

extern "C" void _mlir_ciface_ntt_cpu(StridedMemRefType<i64, 1> *input);
extern "C" void _mlir_ciface_intt_cpu(StridedMemRefType<i64, 1> *input);
extern "C" void _mlir_ciface_ntt_gpu(StridedMemRefType<i64, 1> *input);
extern "C" void _mlir_ciface_intt_gpu(StridedMemRefType<i64, 1> *input);

// `kPrime` = 9223372036836950017
const i64 kPrime = i64({9223372036836950017});

// Set up the random number generator.
std::mt19937_64 rng(std::random_device{}()); // NOLINT(whitespace/braces)
std::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);

// Set the element to random number in [0, kPrime).
void fillWithRandom(i64 &elem, ArrayRef<int64_t> coords) {
  elem = i64::randomLT(kPrime, rng, dist);
}

template <bool kIsGPU>
void BM_ntt_benchmark(::benchmark::State &state) {
  OwningMemRef<i64, 1> input(/*shape=*/{NUM_COEFFS}, /*shapeAlloc=*/{},
                             /*init=*/fillWithRandom);

  OwningMemRef<i64, 1> tmp({NUM_COEFFS}, {});
  for (auto _ : state) {
    state.PauseTiming();
    memcpy((*tmp).data, (*input).data, sizeof(i64) * NUM_COEFFS);
    state.ResumeTiming();
    if constexpr (kIsGPU) {
      _mlir_ciface_ntt_gpu(&*tmp);
    } else {
      _mlir_ciface_ntt_cpu(&*tmp);
    }
  }

  if constexpr (kIsGPU) {
    _mlir_ciface_intt_gpu(&*tmp);
  } else {
    _mlir_ciface_intt_cpu(&*tmp);
  }

  // FIXME(batzor): The NTT benchmark is not working on GPU because the
  // `cuLaunchKernel` fails with `CUDA_ERROR_INVALID_VALUE`.
  if constexpr (!kIsGPU) {
    for (int i = 0; i < NUM_COEFFS; i++) {
      EXPECT_EQ((*tmp)[i], (*input)[i]);
    }
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
// 2025-08-07T08:20:47+00:00
// Run on AMD Ryzen 9 9950X3D (32 X 624 MHz CPU s)
// CPU Caches:
//   L1 Data 48 KiB (x16)
//   L1 Instruction 32 KiB (x16)
//   L2 Unified 1024 KiB (x16)
//   L3 Unified 98304 KiB (x2)
// Load Average: 24.83, 28.23, 17.45
// -----------------------------------------------------
// Benchmark           Time             CPU   Iterations
// -----------------------------------------------------
// ntt_cpu          5.42 ms         5.34 ms          157
// ntt_gpu          ---- ms         ---- ms          ---
// NOLINTEND()
