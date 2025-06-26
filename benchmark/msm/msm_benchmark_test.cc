#include <climits>
#include <random>

#include "benchmark/BenchmarkUtils.h"
#include "benchmark/benchmark.h"
#include "gtest/gtest.h"

// BENCH: CHANGE TO DESIRED DEGREE
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

extern "C" void _mlir_ciface_msm_3(Memref<i256> *scalars, Memref<i256> *points);
extern "C" void _mlir_ciface_msm_4(Memref<i256> *scalars, Memref<i256> *points);
extern "C" void _mlir_ciface_msm_5(Memref<i256> *scalars, Memref<i256> *points);
extern "C" void _mlir_ciface_msm_6(Memref<i256> *scalars, Memref<i256> *points);
extern "C" void _mlir_ciface_msm_7(Memref<i256> *scalars, Memref<i256> *points);
extern "C" void _mlir_ciface_msm_8(Memref<i256> *scalars, Memref<i256> *points);
extern "C" void _mlir_ciface_msm_9(Memref<i256> *scalars, Memref<i256> *points);
extern "C" void _mlir_ciface_msm_10(Memref<i256> *scalars,
                                    Memref<i256> *points);
extern "C" void _mlir_ciface_msm_11(Memref<i256> *scalars,
                                    Memref<i256> *points);
extern "C" void _mlir_ciface_msm_12(Memref<i256> *scalars,
                                    Memref<i256> *points);
extern "C" void _mlir_ciface_msm_13(Memref<i256> *scalars,
                                    Memref<i256> *points);
extern "C" void _mlir_ciface_msm_14(Memref<i256> *scalars,
                                    Memref<i256> *points);
extern "C" void _mlir_ciface_msm_15(Memref<i256> *scalars,
                                    Memref<i256> *points);
extern "C" void _mlir_ciface_msm_16(Memref<i256> *scalars,
                                    Memref<i256> *points);
extern "C" void _mlir_ciface_msm_17(Memref<i256> *scalars,
                                    Memref<i256> *points);
extern "C" void _mlir_ciface_msm_18(Memref<i256> *scalars,
                                    Memref<i256> *points);
extern "C" void _mlir_ciface_msm_19(Memref<i256> *scalars,
                                    Memref<i256> *points);
extern "C" void _mlir_ciface_msm_20(Memref<i256> *scalars,
                                    Memref<i256> *points);
extern "C" void _mlir_ciface_msm_21(Memref<i256> *scalars,
                                    Memref<i256> *points);
extern "C" void _mlir_ciface_msm_22(Memref<i256> *scalars,
                                    Memref<i256> *points);
extern "C" void _mlir_ciface_msm_23(Memref<i256> *scalars,
                                    Memref<i256> *points);
extern "C" void _mlir_ciface_msm_24(Memref<i256> *scalars,
                                    Memref<i256> *points);
extern "C" void _mlir_ciface_msm_25(Memref<i256> *scalars,
                                    Memref<i256> *points);
extern "C" void _mlir_ciface_msm_26(Memref<i256> *scalars,
                                    Memref<i256> *points);
extern "C" void _mlir_ciface_msm_27(Memref<i256> *scalars,
                                    Memref<i256> *points);
extern "C" void _mlir_ciface_msm_28(Memref<i256> *scalars,
                                    Memref<i256> *points);
extern "C" void _mlir_ciface_msm_29(Memref<i256> *scalars,
                                    Memref<i256> *points);
extern "C" void _mlir_ciface_msm_30(Memref<i256> *scalars,
                                    Memref<i256> *points);

void BM_msm_benchmark(::benchmark::State &state,
                      void (*func)(Memref<i256> *, Memref<i256> *)) {
  Memref<i256> scalars(NUM_SCALARMULS, 1);
  fillWithRandom(&scalars, kPrimeScalar);
  Memref<i256> points(NUM_SCALARMULS, 2);
  fillWithRandomPoints(&points, kPrimeBase);

  for (auto _ : state) {
    func(&scalars, &points);
  }
}
void BM_msm_benchmark_3(::benchmark::State &state) {
  BM_msm_benchmark(state, _mlir_ciface_msm_3);
}

void BM_msm_benchmark_4(::benchmark::State &state) {
  BM_msm_benchmark(state, _mlir_ciface_msm_4);
}

void BM_msm_benchmark_5(::benchmark::State &state) {
  BM_msm_benchmark(state, _mlir_ciface_msm_5);
}

void BM_msm_benchmark_6(::benchmark::State &state) {
  BM_msm_benchmark(state, _mlir_ciface_msm_6);
}

void BM_msm_benchmark_7(::benchmark::State &state) {
  BM_msm_benchmark(state, _mlir_ciface_msm_7);
}

void BM_msm_benchmark_8(::benchmark::State &state) {
  BM_msm_benchmark(state, _mlir_ciface_msm_8);
}

void BM_msm_benchmark_9(::benchmark::State &state) {
  BM_msm_benchmark(state, _mlir_ciface_msm_9);
}

void BM_msm_benchmark_10(::benchmark::State &state) {
  BM_msm_benchmark(state, _mlir_ciface_msm_10);
}

void BM_msm_benchmark_11(::benchmark::State &state) {
  BM_msm_benchmark(state, _mlir_ciface_msm_11);
}

void BM_msm_benchmark_12(::benchmark::State &state) {
  BM_msm_benchmark(state, _mlir_ciface_msm_12);
}

void BM_msm_benchmark_13(::benchmark::State &state) {
  BM_msm_benchmark(state, _mlir_ciface_msm_13);
}

void BM_msm_benchmark_14(::benchmark::State &state) {
  BM_msm_benchmark(state, _mlir_ciface_msm_14);
}

void BM_msm_benchmark_15(::benchmark::State &state) {
  BM_msm_benchmark(state, _mlir_ciface_msm_15);
}

void BM_msm_benchmark_16(::benchmark::State &state) {
  BM_msm_benchmark(state, _mlir_ciface_msm_16);
}

void BM_msm_benchmark_17(::benchmark::State &state) {
  BM_msm_benchmark(state, _mlir_ciface_msm_17);
}

void BM_msm_benchmark_18(::benchmark::State &state) {
  BM_msm_benchmark(state, _mlir_ciface_msm_18);
}

void BM_msm_benchmark_19(::benchmark::State &state) {
  BM_msm_benchmark(state, _mlir_ciface_msm_19);
}

void BM_msm_benchmark_20(::benchmark::State &state) {
  BM_msm_benchmark(state, _mlir_ciface_msm_20);
}

void BM_msm_benchmark_21(::benchmark::State &state) {
  BM_msm_benchmark(state, _mlir_ciface_msm_21);
}

void BM_msm_benchmark_22(::benchmark::State &state) {
  BM_msm_benchmark(state, _mlir_ciface_msm_22);
}

void BM_msm_benchmark_23(::benchmark::State &state) {
  BM_msm_benchmark(state, _mlir_ciface_msm_23);
}

void BM_msm_benchmark_24(::benchmark::State &state) {
  BM_msm_benchmark(state, _mlir_ciface_msm_24);
}

void BM_msm_benchmark_25(::benchmark::State &state) {
  BM_msm_benchmark(state, _mlir_ciface_msm_25);
}

void BM_msm_benchmark_26(::benchmark::State &state) {
  BM_msm_benchmark(state, _mlir_ciface_msm_26);
}

void BM_msm_benchmark_27(::benchmark::State &state) {
  BM_msm_benchmark(state, _mlir_ciface_msm_27);
}

void BM_msm_benchmark_28(::benchmark::State &state) {
  BM_msm_benchmark(state, _mlir_ciface_msm_28);
}

void BM_msm_benchmark_29(::benchmark::State &state) {
  BM_msm_benchmark(state, _mlir_ciface_msm_29);
}

void BM_msm_benchmark_30(::benchmark::State &state) {
  BM_msm_benchmark(state, _mlir_ciface_msm_30);
}

// BENCH: COMMENT OUT BENCHMARKS THAT ARE FAR OUTSIDE THE EXPECTED BEST BITS PER
// WINDOW NOTE THAT HIGHER BITS PER WINDOW ARE EXPECTED TO BE SLOWER

BENCHMARK(BM_msm_benchmark_3)->Unit(::benchmark::kMillisecond);
BENCHMARK(BM_msm_benchmark_4)->Unit(::benchmark::kMillisecond);
BENCHMARK(BM_msm_benchmark_5)->Unit(::benchmark::kMillisecond);
BENCHMARK(BM_msm_benchmark_6)->Unit(::benchmark::kMillisecond);
BENCHMARK(BM_msm_benchmark_7)->Unit(::benchmark::kMillisecond);
BENCHMARK(BM_msm_benchmark_8)->Unit(::benchmark::kMillisecond);
BENCHMARK(BM_msm_benchmark_9)->Unit(::benchmark::kMillisecond);
BENCHMARK(BM_msm_benchmark_10)->Unit(::benchmark::kMillisecond);
BENCHMARK(BM_msm_benchmark_11)->Unit(::benchmark::kMillisecond);
BENCHMARK(BM_msm_benchmark_12)->Unit(::benchmark::kMillisecond);
BENCHMARK(BM_msm_benchmark_13)->Unit(::benchmark::kMillisecond);
BENCHMARK(BM_msm_benchmark_14)->Unit(::benchmark::kMillisecond);
BENCHMARK(BM_msm_benchmark_15)->Unit(::benchmark::kMillisecond);
BENCHMARK(BM_msm_benchmark_16)->Unit(::benchmark::kMillisecond);
BENCHMARK(BM_msm_benchmark_17)->Unit(::benchmark::kMillisecond);
BENCHMARK(BM_msm_benchmark_18)->Unit(::benchmark::kMillisecond);
BENCHMARK(BM_msm_benchmark_19)->Unit(::benchmark::kMillisecond);
BENCHMARK(BM_msm_benchmark_20)->Unit(::benchmark::kMillisecond);
BENCHMARK(BM_msm_benchmark_21)->Unit(::benchmark::kMillisecond);
BENCHMARK(BM_msm_benchmark_22)->Unit(::benchmark::kMillisecond);
BENCHMARK(BM_msm_benchmark_23)->Unit(::benchmark::kMillisecond);
// BENCHMARK(BM_msm_benchmark_24)->Unit(::benchmark::kMillisecond);
// BENCHMARK(BM_msm_benchmark_25)->Unit(::benchmark::kMillisecond);
// BENCHMARK(BM_msm_benchmark_26)->Unit(::benchmark::kMillisecond);
// BENCHMARK(BM_msm_benchmark_27)->Unit(::benchmark::kMillisecond);
// BENCHMARK(BM_msm_benchmark_28)->Unit(::benchmark::kMillisecond);
// BENCHMARK(BM_msm_benchmark_29)->Unit(::benchmark::kMillisecond);
// BENCHMARK(BM_msm_benchmark_30)->Unit(::benchmark::kMillisecond);

}  // namespace
}  // namespace zkir

// clang-format off
// NOLINTBEGIN(whitespace/line_length)

// $bazel run -c opt //benchmark/msm:msm_benchmark_test_serial
// 2025-06-26 tested on M4 Pro
//
// Run on (14 X 24 MHz CPU s)
// CPU Caches:
//   L1 Data 64 KiB
//   L1 Instruction 128 KiB
//   L2 Unified 4096 KiB (x14)
// Load Average: 7.82, 13.83, 10.35
// -------------------------------------------------------------------------
// Benchmark                               Time             CPU   Iterations
// -------------------------------------------------------------------------
// BM_msm_benchmark/iterations:20       2786 ms         2761 ms           20

// $bazel run -c opt //benchmark/msm:msm_benchmark_test_parallel
// 2025-06-26 tested on M4 Pro
//
// Run on (14 X 24 MHz CPU s)
// CPU Caches:
//   L1 Data 64 KiB
//   L1 Instruction 128 KiB
//   L2 Unified 4096 KiB (x14)
// Load Average: 22.03, 18.19, 11.02
// -------------------------------------------------------------------------
// Benchmark                               Time             CPU   Iterations
// -------------------------------------------------------------------------
// BM_msm_benchmark/iterations:20        609 ms          485 ms           20
// NOLINTEND()
