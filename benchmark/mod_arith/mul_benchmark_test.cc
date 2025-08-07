#include "benchmark/BenchmarkUtils.h"
#include "benchmark/benchmark.h"
#include "gtest/gtest.h"

namespace mlir::zkir::benchmark {
namespace {

using i256 = BigInt<4>;

// `kPrime` =
// 21888242871839275222246405745257275088548364400416034343698204186575808495617
const i256 kPrime = i256::fromHexString(
    "0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001");

extern "C" void _mlir_ciface_mul(Memref<i256> *output, Memref<i256> *input);
extern "C" void _mlir_ciface_mont_mul(Memref<i256> *output,
                                      Memref<i256> *input);

template <bool kIsMont>
void BM_mul_benchmark(::benchmark::State &state) {
  Memref<i256> input(1, 1);

  std::mt19937_64 rng(std::random_device{}());  // NOLINT(whitespace/braces)
  std::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);

  Memref<i256> output(1, 1);
  for (auto _ : state) {
    state.PauseTiming();
    *input.pget(0, 0) = i256::randomLT(kPrime, rng, dist);
    state.ResumeTiming();
    if constexpr (kIsMont) {
      _mlir_ciface_mont_mul(&output, &input);
    } else {
      _mlir_ciface_mul(&output, &input);
    }
  }
}

BENCHMARK_TEMPLATE(BM_mul_benchmark, /*kIsMont=*/false)->Name("mul");
BENCHMARK_TEMPLATE(BM_mul_benchmark, /*kIsMont=*/true)->Name("mont_mul");

}  // namespace
}  // namespace mlir::zkir::benchmark

// clang-format off
// NOLINTBEGIN(whitespace/line_length)
//
// 2025-08-07T01:12:10+00:00
// Run on AMD Ryzen 9 9950X3D (32 X 5529.37 MHz CPU s)
// CPU Caches:
//   L1 Data 48 KiB (x16)
//   L1 Instruction 32 KiB (x16)
//   L2 Unified 1024 KiB (x16)
//   L3 Unified 98304 KiB (x2)
// Load Average: 0.49, 8.35, 8.82
// -----------------------------------------------------
// Benchmark           Time             CPU   Iterations
// -----------------------------------------------------
// mul               124 ns          124 ns      5624619
// mont_mul          128 ns          128 ns      5472464
// NOLINTEND()
