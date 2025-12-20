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

#include "benchmark/benchmark.h"
#include "mlir/ExecutionEngine/MemRefUtils.h"
#include "mlir/Support/LLVM.h"
#include "zk_dtypes/include/elliptic_curve/bn/bn254/fr.h"
#include "zk_dtypes/include/elliptic_curve/bn/bn254/g1.h"

#define NUM_SCALARMULS (1 << 20)

namespace mlir::zkir::benchmark {
namespace {

using AffinePoint = zk_dtypes::bn254::G1AffinePoint;
using ScalarField = zk_dtypes::bn254::Fr;

extern "C" void
_mlir_ciface_msm_serial(StridedMemRefType<ScalarField, 1> *scalars,
                        StridedMemRefType<AffinePoint, 1> *points);
extern "C" void
_mlir_ciface_msm_parallel(StridedMemRefType<ScalarField, 1> *scalars,
                          StridedMemRefType<AffinePoint, 1> *points);

template <bool kIsParallel>
void BM_msm_benchmark(::benchmark::State &state) {
  OwningMemRef<ScalarField, 1> scalars(
      /*shape=*/{NUM_SCALARMULS},
      /*shapeAlloc=*/{},
      /*init=*/[](ScalarField &elem, ArrayRef<int64_t> coords) {
        elem = ScalarField::Random();
      });
  OwningMemRef<AffinePoint, 1> points(
      /*shape=*/{NUM_SCALARMULS},
      /*shapeAlloc=*/{},
      /*init=*/[](AffinePoint &elem, ArrayRef<int64_t> coords) {
        // MSM performance doesn't depend on the points, so we can use the
        // generator.
        elem = AffinePoint::Generator();
      });

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
