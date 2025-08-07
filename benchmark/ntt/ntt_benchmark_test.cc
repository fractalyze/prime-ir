#include <climits>
#include <random>

#include "benchmark/BenchmarkUtils.h"
#include "benchmark/benchmark.h"
#include "gtest/gtest.h"

#define NUM_COEFFS (1 << 20)

namespace mlir::zkir::benchmark {
namespace {

using i256 = BigInt<4>;

// `kPrime` =
// 21888242871839275222246405745257275088548364400416034343698204186575808495617
const i256 kPrime = i256::fromHexString(
    "0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001");

// Fill the input with random numbers in [0, prime).
void fillWithRandom(Memref<i256> *input, const i256 &kPrime) {
  // Set up the random number generator.
  std::mt19937_64 rng(std::random_device{}());  // NOLINT(whitespace/braces)
  std::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);
  for (int i = 0; i < NUM_COEFFS; i++) {
    *input->pget(i, 0) = i256::randomLT(kPrime, rng, dist);
  }
}

extern "C" void _mlir_ciface_ntt(Memref<i256> *buffer);
extern "C" void _mlir_ciface_intt(Memref<i256> *buffer);

extern "C" void _mlir_ciface_ntt_mont(Memref<i256> *buffer);
extern "C" void _mlir_ciface_intt_mont(Memref<i256> *buffer);

template <bool kIsMont>
void BM_ntt_benchmark(::benchmark::State &state) {
  Memref<i256> input(NUM_COEFFS, 1);
  fillWithRandom(&input, kPrime);

  Memref<i256> ntt(NUM_COEFFS, 1);
  for (auto _ : state) {
    state.PauseTiming();
    memcpy(ntt.pget(0, 0), input.pget(0, 0), sizeof(i256) * NUM_COEFFS);
    state.ResumeTiming();
    if constexpr (kIsMont) {
      _mlir_ciface_ntt_mont(&ntt);
    } else {
      _mlir_ciface_ntt(&ntt);
    }
  }

  if constexpr (kIsMont) {
    _mlir_ciface_intt_mont(&ntt);
  } else {
    _mlir_ciface_intt(&ntt);
  }

  for (int i = 0; i < NUM_COEFFS; i++) {
    for (int j = 0; j < 4; j++) {
      EXPECT_EQ(ntt.pget(i, 0)->limbs[j], input.pget(i, 0)->limbs[j]);
    }
  }
}

template <bool kIsMont>
void BM_intt_benchmark(::benchmark::State &state) {
  Memref<i256> input(NUM_COEFFS, 1);
  fillWithRandom(&input, kPrime);

  Memref<i256> ntt(NUM_COEFFS, 1);
  memcpy(ntt.pget(0, 0), input.pget(0, 0), sizeof(i256) * NUM_COEFFS);
  if constexpr (kIsMont) {
    _mlir_ciface_ntt_mont(&ntt);
  } else {
    _mlir_ciface_ntt(&ntt);
  }

  Memref<i256> intt(NUM_COEFFS, 1);
  for (auto _ : state) {
    state.PauseTiming();
    memcpy(intt.pget(0, 0), ntt.pget(0, 0), sizeof(i256) * NUM_COEFFS);
    state.ResumeTiming();
    if constexpr (kIsMont) {
      _mlir_ciface_intt_mont(&intt);
    } else {
      _mlir_ciface_intt(&intt);
    }
  }

  for (int i = 0; i < NUM_COEFFS; i++) {
    for (int j = 0; j < 4; j++) {
      EXPECT_EQ(intt.pget(i, 0)->limbs[j], input.pget(i, 0)->limbs[j]);
    }
  }
}

BENCHMARK_TEMPLATE(BM_ntt_benchmark, /*kIsMont=*/false)
    ->Unit(::benchmark::kMillisecond)
    ->Name("ntt");
BENCHMARK_TEMPLATE(BM_intt_benchmark, /*kIsMont=*/false)
    ->Unit(::benchmark::kMillisecond)
    ->Name("intt");
BENCHMARK_TEMPLATE(BM_ntt_benchmark, /*kIsMont=*/true)
    ->Unit(::benchmark::kMillisecond)
    ->Name("ntt_mont");
BENCHMARK_TEMPLATE(BM_intt_benchmark, /*kIsMont=*/true)
    ->Unit(::benchmark::kMillisecond)
    ->Name("intt_mont");

}  // namespace
}  // namespace mlir::zkir::benchmark

// clang-format off
// NOLINTBEGIN(whitespace/line_length)
// 2025-08-07T01:17:56+00:00
// Run on AMD Ryzen 9 9950X3D (32 X 5515.87 MHz CPU s)
// CPU Caches:
//   L1 Data 48 KiB (x16)
//   L1 Instruction 32 KiB (x16)
//   L2 Unified 1024 KiB (x16)
//   L3 Unified 98304 KiB (x2)
// Load Average: 1.53, 2.65, 5.49
// -----------------------------------------------------
// Benchmark           Time             CPU   Iterations
// -----------------------------------------------------
// ntt              1287 ms         1286 ms            1
// intt             1358 ms         1348 ms            1
// ntt_mont         12.9 ms         12.9 ms           54
// intt_mont        13.6 ms         13.6 ms           46
// NOLINTEND()
