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
#include "gtest/gtest.h"
#include "mlir/ExecutionEngine/MemRefUtils.h"
#include "mlir/Support/LLVM.h"

#define NUM_COEFFS (1 << 20)

namespace mlir::zkir::benchmark {
namespace {

using i256 = BigInt<4>;

// `kPrime` =
// 21888242871839275222246405745257275088548364400416034343698204186575808495617
const i256 kPrime = i256::fromHexString(
    "0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001");

// Set up the random number generator.
std::mt19937_64 rng(std::random_device{}()); // NOLINT(whitespace/braces)
std::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);

// Set the element to a random number in [0, `kPrime`).
void fillWithRandom(i256 &elem, ArrayRef<int64_t> coords) {
  elem = i256::randomLT(kPrime, rng, dist);
}

extern "C" void _mlir_ciface_ntt(StridedMemRefType<i256, 1> *buffer);
extern "C" void _mlir_ciface_intt(StridedMemRefType<i256, 1> *buffer);

extern "C" void _mlir_ciface_ntt_mont(StridedMemRefType<i256, 1> *buffer);
extern "C" void _mlir_ciface_intt_mont(StridedMemRefType<i256, 1> *buffer);

template <bool kIsMont>
void BM_ntt_benchmark(::benchmark::State &state) {
  OwningMemRef<i256, 1> input(/*shape=*/{NUM_COEFFS}, /*shapeAlloc=*/{},
                              /*init=*/fillWithRandom);

  OwningMemRef<i256, 1> ntt(/*shape=*/{NUM_COEFFS}, /*shapeAlloc=*/{});
  for (auto _ : state) {
    state.PauseTiming();
    memcpy((*ntt).data, (*input).data, sizeof(i256) * NUM_COEFFS);
    state.ResumeTiming();
    if constexpr (kIsMont) {
      _mlir_ciface_ntt_mont(&*ntt);
    } else {
      _mlir_ciface_ntt(&*ntt);
    }
  }

  if constexpr (kIsMont) {
    _mlir_ciface_intt_mont(&*ntt);
  } else {
    _mlir_ciface_intt(&*ntt);
  }

  for (int i = 0; i < NUM_COEFFS; i++) {
    EXPECT_EQ((*ntt)[i], (*input)[i]);
  }
}

template <bool kIsMont>
void BM_intt_benchmark(::benchmark::State &state) {
  OwningMemRef<i256, 1> input(/*shape=*/{NUM_COEFFS}, /*shapeAlloc=*/{},
                              /*init=*/fillWithRandom);

  OwningMemRef<i256, 1> ntt(/*shape=*/{NUM_COEFFS}, /*shapeAlloc=*/{});
  memcpy((*ntt).data, (*input).data, sizeof(i256) * NUM_COEFFS);
  if constexpr (kIsMont) {
    _mlir_ciface_ntt_mont(&*ntt);
  } else {
    _mlir_ciface_ntt(&*ntt);
  }

  OwningMemRef<i256, 1> intt(/*shape=*/{NUM_COEFFS}, /*shapeAlloc=*/{});
  for (auto _ : state) {
    state.PauseTiming();
    memcpy((*intt).data, (*ntt).data, sizeof(i256) * NUM_COEFFS);
    state.ResumeTiming();
    if constexpr (kIsMont) {
      _mlir_ciface_intt_mont(&*intt);
    } else {
      _mlir_ciface_intt(&*intt);
    }
  }

  for (int i = 0; i < NUM_COEFFS; i++) {
    EXPECT_EQ((*intt)[i], (*input)[i]);
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

} // namespace
} // namespace mlir::zkir::benchmark

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
