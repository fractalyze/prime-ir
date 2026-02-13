/* Copyright 2026 The PrimeIR Authors.

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

constexpr int64_t kBufferSize = 1048576; // 1M elements for prime field

// ==============================================================================
// Prime field function declarations
// ==============================================================================

// Scalar versions
// After buffer-results-to-out-params: (result, input) for single input
// After buffer-results-to-out-params: (result, a, b) for two inputs
extern "C" void
_mlir_ciface_square_buffer(StridedMemRefType<uint32_t, 1> *result,
                           StridedMemRefType<uint32_t, 1> *input);
extern "C" void _mlir_ciface_add_buffers(StridedMemRefType<uint32_t, 1> *result,
                                         StridedMemRefType<uint32_t, 1> *a,
                                         StridedMemRefType<uint32_t, 1> *b);
extern "C" void _mlir_ciface_mul_add_buffers(
    StridedMemRefType<uint32_t, 1> *result, StridedMemRefType<uint32_t, 1> *a,
    StridedMemRefType<uint32_t, 1> *b, StridedMemRefType<uint32_t, 1> *c);

// Vectorized versions (same signature, different compilation)
extern "C" void
_mlir_ciface_vec_square_buffer(StridedMemRefType<uint32_t, 1> *result,
                               StridedMemRefType<uint32_t, 1> *input);
extern "C" void
_mlir_ciface_vec_add_buffers(StridedMemRefType<uint32_t, 1> *result,
                             StridedMemRefType<uint32_t, 1> *a,
                             StridedMemRefType<uint32_t, 1> *b);
extern "C" void _mlir_ciface_vec_mul_add_buffers(
    StridedMemRefType<uint32_t, 1> *result, StridedMemRefType<uint32_t, 1> *a,
    StridedMemRefType<uint32_t, 1> *b, StridedMemRefType<uint32_t, 1> *c);

void fillWithValue(uint32_t &elem, ArrayRef<int64_t> coords) {
  elem = static_cast<uint32_t>(coords[0] % 1000 + 1);
}

// Square buffer benchmarks
void BM_square_scalar(::benchmark::State &state) {
  OwningMemRef<uint32_t, 1> input({kBufferSize}, {}, fillWithValue);
  OwningMemRef<uint32_t, 1> result({kBufferSize}, {}, fillWithValue);

  std::string input_hash =
      zkbench::ComputeArrayHash((*input).data, kBufferSize);

  for (auto _ : state) {
    _mlir_ciface_square_buffer(&*result, &*input);
  }

  std::string output_hash =
      zkbench::ComputeArrayHash((*result).data, kBufferSize);
  zkbench::BenchmarkContext::SetTestVectors("BM_square_scalar", input_hash,
                                            output_hash, /*verified=*/true);
  zkbench::BenchmarkContext::SetMetadata("BM_square_scalar",
                                         {{"field", "M31"},
                                          {"buffer_size", kBufferSize},
                                          {"operation", "square"},
                                          {"vectorized", false}});
}

void BM_square_vectorized(::benchmark::State &state) {
  OwningMemRef<uint32_t, 1> input({kBufferSize}, {}, fillWithValue);
  OwningMemRef<uint32_t, 1> result({kBufferSize}, {}, fillWithValue);

  std::string input_hash =
      zkbench::ComputeArrayHash((*input).data, kBufferSize);

  for (auto _ : state) {
    _mlir_ciface_vec_square_buffer(&*result, &*input);
  }

  std::string output_hash =
      zkbench::ComputeArrayHash((*result).data, kBufferSize);
  zkbench::BenchmarkContext::SetTestVectors("BM_square_vectorized", input_hash,
                                            output_hash, /*verified=*/true);
  zkbench::BenchmarkContext::SetMetadata("BM_square_vectorized",
                                         {{"field", "M31"},
                                          {"buffer_size", kBufferSize},
                                          {"operation", "square"},
                                          {"vectorized", true}});
}

// Add buffers benchmarks
void BM_add_scalar(::benchmark::State &state) {
  OwningMemRef<uint32_t, 1> a({kBufferSize}, {}, fillWithValue);
  OwningMemRef<uint32_t, 1> b({kBufferSize}, {}, fillWithValue);
  OwningMemRef<uint32_t, 1> result({kBufferSize}, {}, fillWithValue);

  std::string input_hash = zkbench::ComputeArrayHash((*a).data, kBufferSize);

  for (auto _ : state) {
    _mlir_ciface_add_buffers(&*result, &*a, &*b);
  }

  std::string output_hash =
      zkbench::ComputeArrayHash((*result).data, kBufferSize);
  zkbench::BenchmarkContext::SetTestVectors("BM_add_scalar", input_hash,
                                            output_hash, /*verified=*/true);
  zkbench::BenchmarkContext::SetMetadata("BM_add_scalar",
                                         {{"field", "M31"},
                                          {"buffer_size", kBufferSize},
                                          {"operation", "add"},
                                          {"vectorized", false}});
}

void BM_add_vectorized(::benchmark::State &state) {
  OwningMemRef<uint32_t, 1> a({kBufferSize}, {}, fillWithValue);
  OwningMemRef<uint32_t, 1> b({kBufferSize}, {}, fillWithValue);
  OwningMemRef<uint32_t, 1> result({kBufferSize}, {}, fillWithValue);

  std::string input_hash = zkbench::ComputeArrayHash((*a).data, kBufferSize);

  for (auto _ : state) {
    _mlir_ciface_vec_add_buffers(&*result, &*a, &*b);
  }

  std::string output_hash =
      zkbench::ComputeArrayHash((*result).data, kBufferSize);
  zkbench::BenchmarkContext::SetTestVectors("BM_add_vectorized", input_hash,
                                            output_hash, /*verified=*/true);
  zkbench::BenchmarkContext::SetMetadata("BM_add_vectorized",
                                         {{"field", "M31"},
                                          {"buffer_size", kBufferSize},
                                          {"operation", "add"},
                                          {"vectorized", true}});
}

// Mul-add buffers benchmarks
void BM_mul_add_scalar(::benchmark::State &state) {
  OwningMemRef<uint32_t, 1> a({kBufferSize}, {}, fillWithValue);
  OwningMemRef<uint32_t, 1> b({kBufferSize}, {}, fillWithValue);
  OwningMemRef<uint32_t, 1> c({kBufferSize}, {}, fillWithValue);
  OwningMemRef<uint32_t, 1> result({kBufferSize}, {}, fillWithValue);

  std::string input_hash = zkbench::ComputeArrayHash((*a).data, kBufferSize);

  for (auto _ : state) {
    _mlir_ciface_mul_add_buffers(&*result, &*a, &*b, &*c);
  }

  std::string output_hash =
      zkbench::ComputeArrayHash((*result).data, kBufferSize);
  zkbench::BenchmarkContext::SetTestVectors("BM_mul_add_scalar", input_hash,
                                            output_hash, /*verified=*/true);
  zkbench::BenchmarkContext::SetMetadata("BM_mul_add_scalar",
                                         {{"field", "M31"},
                                          {"buffer_size", kBufferSize},
                                          {"operation", "mul_add"},
                                          {"vectorized", false}});
}

void BM_mul_add_vectorized(::benchmark::State &state) {
  OwningMemRef<uint32_t, 1> a({kBufferSize}, {}, fillWithValue);
  OwningMemRef<uint32_t, 1> b({kBufferSize}, {}, fillWithValue);
  OwningMemRef<uint32_t, 1> c({kBufferSize}, {}, fillWithValue);
  OwningMemRef<uint32_t, 1> result({kBufferSize}, {}, fillWithValue);

  std::string input_hash = zkbench::ComputeArrayHash((*a).data, kBufferSize);

  for (auto _ : state) {
    _mlir_ciface_vec_mul_add_buffers(&*result, &*a, &*b, &*c);
  }

  std::string output_hash =
      zkbench::ComputeArrayHash((*result).data, kBufferSize);
  zkbench::BenchmarkContext::SetTestVectors("BM_mul_add_vectorized", input_hash,
                                            output_hash, /*verified=*/true);
  zkbench::BenchmarkContext::SetMetadata("BM_mul_add_vectorized",
                                         {{"field", "M31"},
                                          {"buffer_size", kBufferSize},
                                          {"operation", "mul_add"},
                                          {"vectorized", true}});
}

BENCHMARK(BM_square_scalar)->Unit(::benchmark::kMillisecond);
BENCHMARK(BM_square_vectorized)->Unit(::benchmark::kMillisecond);
BENCHMARK(BM_add_scalar)->Unit(::benchmark::kMillisecond);
BENCHMARK(BM_add_vectorized)->Unit(::benchmark::kMillisecond);
BENCHMARK(BM_mul_add_scalar)->Unit(::benchmark::kMillisecond);
BENCHMARK(BM_mul_add_vectorized)->Unit(::benchmark::kMillisecond);

} // namespace
} // namespace mlir::prime_ir::benchmark

// 2026-02-05T08:46:00+00:00
// Run on (32 X 624 MHz CPU s)
// CPU Caches:
//   L1 Data 48 KiB (x16)
//   L1 Instruction 32 KiB (x16)
//   L2 Unified 1024 KiB (x16)
//   L3 Unified 98304 KiB (x2)
// Load Average: 1.08, 7.58, 12.29
// ----------------------------------------------------------------
// Benchmark                      Time             CPU   Iterations
// ----------------------------------------------------------------
// BM_square_scalar           0.581 ms        0.581 ms         1191
// BM_square_vectorized       0.583 ms        0.582 ms         1215
// BM_add_scalar              0.387 ms        0.387 ms         1782
// BM_add_vectorized          0.387 ms        0.387 ms         1811
// BM_mul_add_scalar          0.847 ms        0.847 ms          824
// BM_mul_add_vectorized      0.845 ms        0.845 ms          825
