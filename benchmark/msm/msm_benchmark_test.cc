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
#include "mlir/ExecutionEngine/MemRefUtils.h"
#include "mlir/Support/LLVM.h"

#define NUM_SCALARMULS (1 << 20)

namespace mlir::zkir::benchmark {
namespace {

using i256 = BigInt<4>;

// `kPrime` =
// 21888242871839275222246405745257275088548364400416034343698204186575808495617
const i256 kPrimeBase = i256::fromHexString(
    "0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47");

const i256 kPrimeScalar = i256::fromHexString(
    "0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001");

// Set up the random number generator.
std::mt19937_64 rng(std::random_device{}()); // NOLINT(whitespace/braces)
std::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);

template <bool kIsScalar>
void fillWithRandom(i256 &elem, ArrayRef<int64_t> coords) {
  if constexpr (kIsScalar) {
    elem = i256::randomLT(kPrimeScalar, rng, dist);
  } else {
    elem = i256::randomLT(kPrimeBase, rng, dist);
  }
}

extern "C" void _mlir_ciface_msm_serial(StridedMemRefType<i256, 1> *scalars,
                                        StridedMemRefType<i256, 2> *points);
extern "C" void _mlir_ciface_msm_parallel(StridedMemRefType<i256, 1> *scalars,
                                          StridedMemRefType<i256, 2> *points);

template <bool kIsParallel>
void BM_msm_benchmark(::benchmark::State &state) {
  OwningMemRef<i256, 1> scalars(/*shape=*/{NUM_SCALARMULS}, /*shapeAlloc=*/{},
                                /*init=*/fillWithRandom<true>);
  OwningMemRef<i256, 2> points(/*shape=*/{NUM_SCALARMULS, 2},
                               /*shapeAlloc=*/{},
                               /*init=*/fillWithRandom<false>);

  for (auto _ : state) {
    if constexpr (kIsParallel) {
      _mlir_ciface_msm_parallel(&*scalars, &*points);
    } else {
      _mlir_ciface_msm_serial(&*scalars, &*points);
    }
  }
}

BENCHMARK_TEMPLATE(BM_msm_benchmark, /*kIsParallel=*/false)
    ->Iterations(20)
    ->Unit(::benchmark::kMillisecond)
    ->Name("msm_serial");

BENCHMARK_TEMPLATE(BM_msm_benchmark, /*kIsParallel=*/true)
    ->Iterations(20)
    ->Unit(::benchmark::kMillisecond)
    ->Name("msm_parallel");

} // namespace
} // namespace mlir::zkir::benchmark

// clang-format off
// NOLINTBEGIN(whitespace/line_length)

// 2025-08-07T01:40:36+00:00
// Run on AMD Ryzen 9 9950X3D (32 X 5501.43 MHz CPU s)
// CPU Caches:
//   L1 Data 48 KiB (x16)
//   L1 Instruction 32 KiB (x16)
//   L2 Unified 1024 KiB (x16)
//   L3 Unified 98304 KiB (x2)
// Load Average: 3.91, 4.78, 7.12
// ---------------------------------------------------------------------
// Benchmark                           Time             CPU   Iterations
// ---------------------------------------------------------------------
// msm_serial/iterations:20         2348 ms         2348 ms           20
// msm_parallel/iterations:20        276 ms          276 ms           20
// NOLINTEND()
