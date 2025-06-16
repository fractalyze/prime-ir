#include <climits>
#include <random>

#include "benchmark/BenchmarkUtils.h"
#include "benchmark/benchmark.h"
#include "gtest/gtest.h"

#define NUM_SCALARMULS (1 << 20)

namespace zkir {
namespace {

using benchmark::Memref;

using i256 = benchmark::BigInt<4>;

// `kPrime` =
// 21888242871839275222246405745257275088548364400416034343698204186575808495617
const i256 kPrimeBase = i256::fromHexString(
    "0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47");

const i256 kPrimeScalar = i256::fromHexString(
    "0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001");

// Fill the input with random numbers in [0, prime).
static void fillWithRandom(Memref<i256> *input, const i256 &kPrime) {
  // Set up the random number generator.
  std::mt19937_64 rng(std::random_device{}());  // NOLINT(whitespace/braces)
  std::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);
  for (int i = 0; i < NUM_SCALARMULS; i++) {
    *input->pget(i, 0) = i256::randomLT(kPrime, rng, dist);
  }
}

// Fill the input with random numbers in [0, prime).
static void fillWithRandomPoints(Memref<i256> *input, const i256 &kPrime) {
  // Set up the random number generator.
  std::mt19937_64 rng(std::random_device{}());  // NOLINT(whitespace/braces)
  std::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);
  for (int i = 0; i < NUM_SCALARMULS; i++) {
    *input->pget(i, 0) = i256::randomLT(kPrime, rng, dist);
    *input->pget(i, 1) = i256::randomLT(kPrime, rng, dist);
  }
}

extern "C" void _mlir_ciface_msm(Memref<i256> *scalars, Memref<i256> *points);

void BM_msm_benchmark(::benchmark::State &state) {
  Memref<i256> scalars(NUM_SCALARMULS, 1);
  fillWithRandom(&scalars, kPrimeScalar);
  Memref<i256> points(NUM_SCALARMULS, 2);
  fillWithRandomPoints(&points, kPrimeBase);

  for (auto _ : state) {
    _mlir_ciface_msm(&scalars, &points);
  }
}

BENCHMARK(BM_msm_benchmark)->Iterations(20)->Unit(::benchmark::kMillisecond);

}  // namespace
}  // namespace zkir

// clang-format off
// NOLINTBEGIN(whitespace/line_length)

// $bazel run //benchmark/msm:msm_benchmark_test_serial
// 2025-06-16
//
// Run on (8 X 24 MHz CPU s)
// CPU Caches:
//   L1 Data 64 KiB
//   L1 Instruction 128 KiB
//   L2 Unified 4096 KiB (x8)
// Load Average: 1.70, 2.47, 3.13
// -------------------------------------------------------------------------
// Benchmark                               Time             CPU   Iterations
// -------------------------------------------------------------------------
// BM_msm_benchmark/iterations:20       4263 ms         4254 ms           20

// $bazel run //benchmark/msm:msm_benchmark_test_parallel
// 2025-06-16
//
// Run on (8 X 24 MHz CPU s)
// CPU Caches:
//   L1 Data 64 KiB
//   L1 Instruction 128 KiB
//   L2 Unified 4096 KiB (x8)
// Load Average: 7.68, 4.68, 3.73
// -------------------------------------------------------------------------
// Benchmark                               Time             CPU   Iterations
// -------------------------------------------------------------------------
// BM_msm_benchmark/iterations:20        946 ms          916 ms           20
// NOLINTEND()
