/* Copyright 2025 The PrimeIR Authors.

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
#include "zkbench/benchmark_context.h"
#include "zkbench/hash.h"

namespace mlir::prime_ir::benchmark {
namespace {

using Vector16xI32 = int32_t __attribute__((vector_size(64)));

extern "C" void
_mlir_ciface_permute_10000(StridedMemRefType<uint32_t, 1> *input);
extern "C" void
_mlir_ciface_packed_permute_10000(StridedMemRefType<Vector16xI32, 1> *input);

// Set each element to 0.
void fillWithZero(uint32_t &elem, [[maybe_unused]] ArrayRef<int64_t> coords) {
  elem = 0;
}

void fillVectorWithZero(Vector16xI32 &data,
                        [[maybe_unused]] ArrayRef<int64_t> coords) {
  data = Vector16xI32{0};
}

template <bool kIsPacked>
void BM_permute_10000_benchmark(::benchmark::State &state) {
  const char *bench_name = kIsPacked ? "permute_packed_10000" : "permute_10000";

  if constexpr (kIsPacked) {
    OwningMemRef<Vector16xI32, 1> input(/*shape=*/{16}, /*shapeAlloc=*/{},
                                        /*init=*/fillVectorWithZero);

    std::string input_hash = zkbench::ComputeArrayHash(
        reinterpret_cast<const uint32_t *>((*input).data), 16 * 16);

    for (auto _ : state) {
      _mlir_ciface_packed_permute_10000(&*input);
    }

    std::string output_hash = zkbench::ComputeArrayHash(
        reinterpret_cast<const uint32_t *>((*input).data), 16 * 16);

    zkbench::BenchmarkContext::SetTestVectors(bench_name, input_hash,
                                              output_hash, /*verified=*/true);
  } else {
    OwningMemRef<uint32_t, 1> input(/*shape=*/{16}, /*shapeAlloc=*/{},
                                    /*init=*/fillWithZero);

    std::string input_hash = zkbench::ComputeArrayHash((*input).data, 16);

    for (auto _ : state) {
      _mlir_ciface_permute_10000(&*input);
    }

    std::string output_hash = zkbench::ComputeArrayHash((*input).data, 16);

    zkbench::BenchmarkContext::SetTestVectors(bench_name, input_hash,
                                              output_hash, /*verified=*/true);
  }

  zkbench::BenchmarkContext::SetMetadata(
      bench_name,
      {{"field", "M31"}, {"iterations", 10000}, {"packed", kIsPacked}});
}

BENCHMARK_TEMPLATE(BM_permute_10000_benchmark, /*kIsPacked=*/false)
    ->Unit(::benchmark::kMillisecond)
    ->Name("permute_10000");
BENCHMARK_TEMPLATE(BM_permute_10000_benchmark, /*kIsPacked=*/true)
    ->Unit(::benchmark::kMillisecond)
    ->Name("permute_packed_10000");

} // namespace
} // namespace mlir::prime_ir::benchmark

// clang-format off
// NOLINTBEGIN(whitespace/line_length)
//
// 2025-11-12T11:25:38+00:00
// Run on AMD Ryzen 9 9950X3D (32 X 5479.99 MHz CPU s)
// CPU Caches:
//   L1 Data 48 KiB (x16)
//   L1 Instruction 32 KiB (x16)
//   L2 Unified 1024 KiB (x16)
//   L3 Unified 98304 KiB (x2)
// Load Average: 0.58, 0.38, 0.33
// ---------------------------------------------------------------
// Benchmark                     Time             CPU   Iterations
// ---------------------------------------------------------------
// permute_10000              5.68 ms         5.68 ms          123
// permute_packed_10000       7.36 ms         7.36 ms           95
// NOLINTEND()
