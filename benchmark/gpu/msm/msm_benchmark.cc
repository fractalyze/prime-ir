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
#include "cuda_runtime_api.h" // NOLINT(build/include_subdir)
#include "mlir/ExecutionEngine/MemRefUtils.h"
#include "mlir/Support/LLVM.h"
#include "utils/cuda/CudaUtils.h"
#include "zk_dtypes/include/elliptic_curve/bn/bn254/fr.h"
#include "zk_dtypes/include/elliptic_curve/bn/bn254/g1.h"

#define NUM_SCALARMULS (1 << 20)

namespace mlir::zkir::benchmark {
namespace {

using AffinePoint = zk_dtypes::bn254::G1AffinePoint;
using ScalarField = zk_dtypes::bn254::Fr;

extern "C" void _mlir_ciface_msm_cpu(StridedMemRefType<ScalarField, 1> *scalars,
                                     StridedMemRefType<AffinePoint, 1> *points);
extern "C" void _mlir_ciface_msm_gpu(StridedMemRefType<ScalarField, 1> *scalars,
                                     StridedMemRefType<AffinePoint, 1> *points);

template <bool kIsGPU>
void BM_msm_benchmark(::benchmark::State &state) {
  OwningMemRef<ScalarField, 1> hScalars(
      /*shape=*/{NUM_SCALARMULS}, /*shapeAlloc=*/{},
      /*init=*/[](ScalarField &elem, ArrayRef<int64_t> coords) {
        elem = ScalarField::Random();
      });
  OwningMemRef<AffinePoint, 1> hPoints(
      /*shape=*/{NUM_SCALARMULS},
      /*shapeAlloc=*/{},
      /*init=*/[](AffinePoint &elem, ArrayRef<int64_t> coords) {
        // MSM performance doesn't depend on the points, so we can use the
        // generator.
        elem = AffinePoint::Generator();
      });

  const size_t scalarBytes = sizeof(ScalarField) * NUM_SCALARMULS;
  const size_t pointBytes = sizeof(AffinePoint) * NUM_SCALARMULS;

  if constexpr (kIsGPU) {
    auto dScalarBuf =
        ::zkir::utils::makeCudaUnique<ScalarField>(NUM_SCALARMULS);
    auto dPointBuf = ::zkir::utils::makeCudaUnique<AffinePoint>(NUM_SCALARMULS);

    // Stage host input to device once
    CHECK_CUDA_ERROR(cudaMemcpy(dScalarBuf.get(), hScalars->data, scalarBytes,
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(dPointBuf.get(), hPoints->data, pointBytes,
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Build device-backed memrefs (same shape/stride as host)
    StridedMemRefType<ScalarField, 1> dScalarRef{/*basePtr=*/dScalarBuf.get(),
                                                 /*data=*/dScalarBuf.get(),
                                                 /*offset=*/0,
                                                 /*sizes=*/{NUM_SCALARMULS},
                                                 /*strides=*/{1}};
    StridedMemRefType<AffinePoint, 1> dPointRef{/*basePtr=*/dPointBuf.get(),
                                                /*data=*/dPointBuf.get(),
                                                /*offset=*/0,
                                                /*sizes=*/{NUM_SCALARMULS},
                                                /*strides=*/{1}};

    for (auto _ : state) {
      _mlir_ciface_msm_gpu(&dScalarRef, &dPointRef);
    }
  } else {
    for (auto _ : state) {
      _mlir_ciface_msm_cpu(&*hScalars, &*hPoints);
    }
  }
}

BENCHMARK_TEMPLATE(BM_msm_benchmark, /*kIsGPU=*/false)
    ->Iterations(5)
    ->Unit(::benchmark::kMillisecond)
    ->Name("msm_cpu");

BENCHMARK_TEMPLATE(BM_msm_benchmark, /*kIsGPU=*/true)
    ->Iterations(5)
    ->Unit(::benchmark::kMillisecond)
    ->Name("msm_gpu");

} // namespace
} // namespace mlir::zkir::benchmark

// clang-format off
// NOLINTBEGIN(whitespace/line_length)
//
// 2025-08-20T04:37:37+00:00
// Run on AMD Ryzen 9 9950X3D (32 X 624 MHz CPU s)
// CPU Caches:
//   L1 Data 48 KiB (x16)
//   L1 Instruction 32 KiB (x16)
//   L2 Unified 1024 KiB (x16)
//   L3 Unified 98304 KiB (x2)
// Load Average: 0.95, 2.65, 1.79
// ---------------------------------------------------------------
// Benchmark                     Time             CPU   Iterations
// ---------------------------------------------------------------
// msm_cpu/iterations:5        316 ms          316 ms            5
// msm_gpu/iterations:5       5369 ms         5368 ms            5
// NOLINTEND()
