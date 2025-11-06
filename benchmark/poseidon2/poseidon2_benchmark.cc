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

#include "benchmark/benchmark.h"
#include "mlir/ExecutionEngine/MemRefUtils.h"
#include "mlir/Support/LLVM.h"

namespace mlir::zkir::benchmark {
namespace {

using Vector4xI32 = int32_t __attribute__((vector_size(16)));

extern "C" void
_mlir_ciface_permute_10000(StridedMemRefType<uint32_t, 1> *input);
extern "C" void
_mlir_ciface_packed_permute_10000(StridedMemRefType<Vector4xI32, 1> *input);

// Set each element to 0.
void fillWithZero(uint32_t &elem, [[maybe_unused]] ArrayRef<int64_t> coords) {
  elem = 0;
}

void fillVectorWithZero(Vector4xI32 &data,
                        [[maybe_unused]] ArrayRef<int64_t> coords) {
  data = Vector4xI32{0};
}

template <bool kIsPacked>
void BM_permute_10000_benchmark(::benchmark::State &state) {
  if constexpr (kIsPacked) {
    OwningMemRef<Vector4xI32, 1> input(/*shape=*/{16}, /*shapeAlloc=*/{},
                                       /*init=*/fillVectorWithZero);
    for (auto _ : state) {
      _mlir_ciface_packed_permute_10000(&*input);
    }
  } else {
    OwningMemRef<uint32_t, 1> input(/*shape=*/{16}, /*shapeAlloc=*/{},
                                    /*init=*/fillWithZero);
    for (auto _ : state) {
      _mlir_ciface_permute_10000(&*input);
    }
  }
}

BENCHMARK_TEMPLATE(BM_permute_10000_benchmark, /*kIsPacked=*/false)
    ->Unit(::benchmark::kMillisecond)
    ->Name("permute_10000");
BENCHMARK_TEMPLATE(BM_permute_10000_benchmark, /*kIsPacked=*/true)
    ->Unit(::benchmark::kMillisecond)
    ->Name("permute_packed_10000");

} // namespace
} // namespace mlir::zkir::benchmark

// clang-format off
// NOLINTBEGIN(whitespace/line_length)
//
// 2025-11-13T06:49:58+00:00
// Run on (14 X 24 MHz CPU s)
// CPU Caches:
//   L1 Data 64 KiB
//   L1 Instruction 128 KiB
//   L2 Unified 4096 KiB (x14)
// Load Average: 1.79, 2.08, 1.98
// ---------------------------------------------------------------
// Benchmark                     Time             CPU   Iterations
// ---------------------------------------------------------------
// permute_10000              5.44 ms         5.44 ms          129
// permute_packed_10000       5.58 ms         5.58 ms          125
// NOLINTEND()
