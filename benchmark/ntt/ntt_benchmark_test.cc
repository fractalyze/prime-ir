#include <climits>
#include <random>

#include "benchmark/BenchmarkUtils.h"
#include "benchmark/benchmark.h"
#include "gtest/gtest.h"

#define NUM_COEFFS (1 << 20)

namespace zkir {
namespace {

using ::zkir::benchmark::Memref;

using i256 = benchmark::BigInt<4>;

// `kPrime` =
// 21888242871839275222246405745257275088548364400416034343698204186575808495617
const i256 kPrime = i256::fromHexString(
    "0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001");

// Fill the input with random numbers in [0, prime).
static void fillWithRandom(Memref<i256> *input, const i256 &kPrime) {
  // Set up the random number generator.
  std::mt19937_64 rng(std::random_device{}());  // NOLINT(whitespace/braces)
  std::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);
  for (int i = 0; i < NUM_COEFFS; i++) {
    *input->pget(i, 0) = i256::randomLT(kPrime, rng, dist);
  }
}

extern "C" void _mlir_ciface_ntt(Memref<i256> *output, Memref<i256> *input);
extern "C" void _mlir_ciface_intt(Memref<i256> *output, Memref<i256> *input);

extern "C" void _mlir_ciface_ntt_mont(Memref<i256> *output,
                                      Memref<i256> *input);
extern "C" void _mlir_ciface_intt_mont(Memref<i256> *output,
                                       Memref<i256> *input);

void BM_ntt_benchmark(::benchmark::State &state) {
  Memref<i256> input(NUM_COEFFS, 1);
  fillWithRandom(&input, kPrime);

  Memref<i256> ntt(NUM_COEFFS, 1);
  for (auto _ : state) {
    _mlir_ciface_ntt(&ntt, &input);
  }

  Memref<i256> intt(NUM_COEFFS, 1);
  _mlir_ciface_intt(&intt, &ntt);

  for (int i = 0; i < NUM_COEFFS; i++) {
    for (int j = 0; j < 4; j++) {
      EXPECT_EQ(intt.pget(i, 0)->limbs[j], input.pget(i, 0)->limbs[j]);
    }
  }
}

BENCHMARK(BM_ntt_benchmark)->Unit(::benchmark::kMillisecond);

void BM_intt_benchmark(::benchmark::State &state) {
  Memref<i256> input(NUM_COEFFS, 1);
  fillWithRandom(&input, kPrime);

  Memref<i256> ntt(NUM_COEFFS, 1);
  _mlir_ciface_ntt(&ntt, &input);

  Memref<i256> intt(NUM_COEFFS, 1);
  for (auto _ : state) {
    _mlir_ciface_intt(&intt, &ntt);
  }

  for (int i = 0; i < NUM_COEFFS; i++) {
    for (int j = 0; j < 4; j++) {
      EXPECT_EQ(intt.pget(i, 0)->limbs[j], input.pget(i, 0)->limbs[j]);
    }
  }
}

BENCHMARK(BM_intt_benchmark)->Unit(::benchmark::kMillisecond);

void BM_ntt_mont_benchmark(::benchmark::State &state) {
  Memref<i256> input(NUM_COEFFS, 1);
  fillWithRandom(&input, kPrime);

  Memref<i256> ntt(NUM_COEFFS, 1);
  for (auto _ : state) {
    _mlir_ciface_ntt_mont(&ntt, &input);
  }

  Memref<i256> intt(NUM_COEFFS, 1);
  _mlir_ciface_intt_mont(&intt, &ntt);

  for (int i = 0; i < NUM_COEFFS; i++) {
    for (int j = 0; j < 4; j++) {
      EXPECT_EQ(intt.pget(i, 0)->limbs[j], input.pget(i, 0)->limbs[j]);
    }
  }
}

BENCHMARK(BM_ntt_mont_benchmark)->Unit(::benchmark::kMillisecond);

void BM_intt_mont_benchmark(::benchmark::State &state) {
  Memref<i256> input(NUM_COEFFS, 1);
  fillWithRandom(&input, kPrime);

  Memref<i256> ntt(NUM_COEFFS, 1);
  _mlir_ciface_ntt_mont(&ntt, &input);

  Memref<i256> intt(NUM_COEFFS, 1);
  for (auto _ : state) {
    _mlir_ciface_intt_mont(&intt, &ntt);
  }

  for (int i = 0; i < NUM_COEFFS; i++) {
    for (int j = 0; j < 4; j++) {
      EXPECT_EQ(intt.pget(i, 0)->limbs[j], input.pget(i, 0)->limbs[j]);
    }
  }
}

BENCHMARK(BM_intt_mont_benchmark)->Unit(::benchmark::kMillisecond);

}  // namespace
}  // namespace zkir

// clang-format off
// NOLINTBEGIN(whitespace/line_length)
// Run on (14 X 24 MHz CPU s)
// CPU Caches:
//   L1 Data 64 KiB
//   L1 Instruction 128 KiB
//   L2 Unified 4096 KiB (x14)
// Load Average: 6.49, 5.64, 5.49
// -------------------------------------------------------------------------
// Benchmark                               Time             CPU   Iterations
// -------------------------------------------------------------------------
// BM_ntt_benchmark                     1656 ms         1050 ms            1
// BM_intt_benchmark/iterations:1       1791 ms         1090 ms            1
// BM_ntt_mont_benchmark                38.6 ms         18.6 ms           40
// BM_intt_mont_benchmark               99.4 ms         56.4 ms           11
// NOLINTEND()
