#include "benchmark/BenchmarkUtils.h"
#include "benchmark/benchmark.h"
#include "gtest/gtest.h"

namespace zkir {
namespace {

using benchmark::Memref;

using i256 = benchmark::BigInt<4>;

// `kPrime` =
// 21888242871839275222246405745257275088548364400416034343698204186575808495617
const i256 kPrime = i256::fromHexString(
    "0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001");

extern "C" void _mlir_ciface_mul(Memref<i256> *output, Memref<i256> *input);
extern "C" void _mlir_ciface_mont_mul(Memref<i256> *output,
                                      Memref<i256> *input);

void BM_mul_benchmark(::benchmark::State &state) {
  Memref<i256> input(1, 1);

  std::mt19937_64 rng(std::random_device{}());  // NOLINT(whitespace/braces)
  std::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);

  Memref<i256> output(1, 1);
  for (auto _ : state) {
    state.PauseTiming();
    *input.pget(0, 0) = i256::randomLT(kPrime, rng, dist);
    state.ResumeTiming();
    _mlir_ciface_mul(&output, &input);
  }
}

BENCHMARK(BM_mul_benchmark);

void BM_mont_mul_benchmark(::benchmark::State &state) {
  Memref<i256> input(1, 1);

  std::mt19937_64 rng(std::random_device{}());  // NOLINT(whitespace/braces)
  std::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);

  Memref<i256> mont_output(1, 1);
  for (auto _ : state) {
    state.PauseTiming();
    *input.pget(0, 0) = i256::randomLT(kPrime, rng, dist);
    state.ResumeTiming();
    _mlir_ciface_mont_mul(&mont_output, &input);
  }
}

BENCHMARK(BM_mont_mul_benchmark);

}  // namespace
}  // namespace zkir

// Run on (14 X 24 MHz CPU s)
// CPU Caches:
//   L1 Data 64 KiB
//   L1 Instruction 128 KiB
//   L2 Unified 4096 KiB (x14)
// Load Average: 2.27, 2.16, 2.53
// ----------------------------------------------------------------
// Benchmark                      Time             CPU   Iterations
// ----------------------------------------------------------------
// BM_mul_benchmark            2219 ns         2221 ns       318374
// BM_mont_mul_benchmark        411 ns          413 ns      1678053
