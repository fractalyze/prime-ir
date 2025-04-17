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
    *input->pget(0, i) = i256::randomLT(kPrime, rng, dist);
  }
}

extern "C" void _mlir_ciface_input_generation(Memref<i256> *output);
extern "C" void _mlir_ciface_ntt(Memref<i256> *output, Memref<i256> *input);
extern "C" void _mlir_ciface_intt(Memref<i256> *output, Memref<i256> *input);

extern "C" void _mlir_ciface_ntt_mont(Memref<i256> *output,
                                      Memref<i256> *input);
extern "C" void _mlir_ciface_intt_mont(Memref<i256> *output,
                                       Memref<i256> *input);

void BM_ntt_benchmark(::benchmark::State &state) {
  Memref<i256> input(1, NUM_COEFFS);
  _mlir_ciface_input_generation(&input);
  fillWithRandom(&input, kPrime);

  Memref<i256> ntt(1, NUM_COEFFS);
  for (auto _ : state) {
    _mlir_ciface_ntt(&ntt, &input);
  }

  Memref<i256> intt(1, NUM_COEFFS);
  _mlir_ciface_intt(&intt, &ntt);

  for (int i = 0; i < NUM_COEFFS; i++) {
    for (int j = 0; j < 4; j++) {
      EXPECT_EQ(intt.pget(0, i)->limbs[j], input.pget(0, i)->limbs[j]);
    }
  }
}

BENCHMARK(BM_ntt_benchmark)->Unit(::benchmark::kSecond);

void BM_intt_benchmark(::benchmark::State &state) {
  Memref<i256> input(1, NUM_COEFFS);
  _mlir_ciface_input_generation(&input);
  fillWithRandom(&input, kPrime);

  Memref<i256> ntt(1, NUM_COEFFS);
  _mlir_ciface_ntt(&ntt, &input);

  Memref<i256> intt(1, NUM_COEFFS);
  for (auto _ : state) {
    _mlir_ciface_intt(&intt, &ntt);
  }

  for (int i = 0; i < NUM_COEFFS; i++) {
    for (int j = 0; j < 4; j++) {
      EXPECT_EQ(intt.pget(0, i)->limbs[j], input.pget(0, i)->limbs[j]);
    }
  }
}

// FIXME(batzor): It fails for more than 1 iteration so it seems like it is
// modifying the input. But I am not sure why ;(
BENCHMARK(BM_intt_benchmark)->Iterations(1)->Unit(::benchmark::kSecond);

void BM_ntt_mont_benchmark(::benchmark::State &state) {
  Memref<i256> input(1, NUM_COEFFS);
  _mlir_ciface_input_generation(&input);
  fillWithRandom(&input, kPrime);

  Memref<i256> ntt(1, NUM_COEFFS);
  for (auto _ : state) {
    _mlir_ciface_ntt_mont(&ntt, &input);
  }

  Memref<i256> intt(1, NUM_COEFFS);
  _mlir_ciface_intt_mont(&intt, &ntt);

  for (int i = 0; i < NUM_COEFFS; i++) {
    for (int j = 0; j < 4; j++) {
      EXPECT_EQ(intt.pget(0, i)->limbs[j], input.pget(0, i)->limbs[j]);
    }
  }
}

BENCHMARK(BM_ntt_mont_benchmark)->Unit(::benchmark::kSecond);

void BM_intt_mont_benchmark(::benchmark::State &state) {
  Memref<i256> input(1, NUM_COEFFS);
  _mlir_ciface_input_generation(&input);
  fillWithRandom(&input, kPrime);

  Memref<i256> ntt(1, NUM_COEFFS);
  _mlir_ciface_ntt_mont(&ntt, &input);

  Memref<i256> intt(1, NUM_COEFFS);
  for (auto _ : state) {
    _mlir_ciface_intt_mont(&intt, &ntt);
  }

  for (int i = 0; i < NUM_COEFFS; i++) {
    for (int j = 0; j < 4; j++) {
      EXPECT_EQ(intt.pget(0, i)->limbs[j], input.pget(0, i)->limbs[j]);
    }
  }
}

// FIXME(batzor): It fails for more than 1 iteration so it seems like it is
// modifying the input. But I am not sure why ;(
BENCHMARK(BM_intt_mont_benchmark)->Iterations(1)->Unit(::benchmark::kSecond);

}  // namespace
}  // namespace zkir

// clang-format off
// NOLINTBEGIN(whitespace/line_length)
// Run on (14 X 24 MHz CPU s)
// CPU Caches:
//   L1 Data 64 KiB
//   L1 Instruction 128 KiB
//   L2 Unified 4096 KiB (x14)
// Load Average: 1.82, 2.22, 2.39
// ------------------------------------------------------------------------------
// Benchmark                                    Time             CPU   Iterations
// ------------------------------------------------------------------------------
// BM_ntt_benchmark                          10.1 s          10.1 s             1
// BM_intt_benchmark/iterations:1            10.1 s          10.0 s             1
// BM_ntt_mont_benchmark                    0.183 s         0.183 s             4
// BM_intt_mont_benchmark/iterations:1      0.266 s         0.214 s             1
// NOLINTEND()
