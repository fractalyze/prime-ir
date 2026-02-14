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

#include <cstdint>
#include <cstring>
#include <random>

#include "benchmark/benchmark.h"
#include "mlir/ExecutionEngine/MemRefUtils.h"
#include "zk_dtypes/include/field/binary_field.h"
#include "zkbench/benchmark_context.h"
#include "zkbench/hash.h"

namespace mlir::prime_ir::benchmark {
namespace {

using uint128_t = zk_dtypes::BigInt<2>;

using BinaryFieldT3 = zk_dtypes::BinaryFieldT3;
using BinaryFieldT6 = zk_dtypes::BinaryFieldT6;
using BinaryFieldT7 = zk_dtypes::BinaryFieldT7;

// MLIR MemRef descriptor for 1D memref<16xi8>
using MemRef1D16xi8 = StridedMemRefType<uint8_t, 1>;

// =============================================================================
// BF64 Baseline implementations (generic Karatsuba)
// =============================================================================
extern "C" uint64_t _mlir_ciface_bf64_mul_baseline(uint64_t a, uint64_t b);

// =============================================================================
// BF128 Baseline implementations (generic Karatsuba)
// =============================================================================
extern "C" uint128_t _mlir_ciface_bf128_mul_baseline(uint128_t a, uint128_t b);

// =============================================================================
// BF8 Baseline implementations (generic Karatsuba)
// =============================================================================
extern "C" uint8_t _mlir_ciface_bf8_mul_baseline(uint8_t a, uint8_t b);
extern "C" uint8_t _mlir_ciface_bf8_inverse_baseline(uint8_t a);

// =============================================================================
// x86 PCLMULQDQ specialized implementations
// =============================================================================
#if defined(PRIME_IR_X86)
extern "C" uint64_t _mlir_ciface_bf64_mul_x86(uint64_t a, uint64_t b);
extern "C" uint128_t _mlir_ciface_bf128_mul_x86(uint128_t a, uint128_t b);

// BF8 packed 16x GFNI multiplication
// Uses memref interface for SIMD vectors
extern "C" void _mlir_ciface_bf8x16_mul_gfni(MemRef1D16xi8 *a, MemRef1D16xi8 *b,
                                             MemRef1D16xi8 *c);
#endif

// =============================================================================
// ARM PMULL specialized implementations
// =============================================================================
#if defined(PRIME_IR_ARM)
extern "C" uint64_t _mlir_ciface_bf64_mul_arm(uint64_t a, uint64_t b);
extern "C" uint128_t _mlir_ciface_bf128_mul_arm(uint128_t a, uint128_t b);
extern "C" uint64_t _mlir_ciface_bf64_square_arm(uint64_t a);
extern "C" uint128_t _mlir_ciface_bf128_square_arm(uint128_t a);

// BF8 packed 16x PMULL multiplication
// Uses memref interface for SIMD vectors
extern "C" void _mlir_ciface_bf8x16_mul_pmull(MemRef1D16xi8 *a,
                                              MemRef1D16xi8 *b,
                                              MemRef1D16xi8 *c);

// BF128 Polyval Montgomery reduction (faster than Tower polynomial)
extern "C" uint128_t _mlir_ciface_bf128_mul_arm_polyval(uint128_t a,
                                                        uint128_t b);
#endif

// =============================================================================
// BF64 Benchmarks
// =============================================================================
void BM_bf64_mul_baseline(::benchmark::State &state) {
  auto a = BinaryFieldT6::Random();
  auto b = BinaryFieldT6::Random();
  BinaryFieldT6 result;

  uint64_t inputs[] = {a.value(), b.value()};
  std::string input_hash = zkbench::ComputeArrayHash(inputs, 2);

  for (auto _ : state) {
    result = _mlir_ciface_bf64_mul_baseline(a.value(), b.value());
    ::benchmark::DoNotOptimize(result);
    a += result;
  }
  state.SetItemsProcessed(state.iterations());

  uint64_t out = result.value();
  std::string output_hash = zkbench::ComputeArrayHash(&out, 1);
  zkbench::BenchmarkContext::SetTestVectors("BM_bf64_mul_baseline", input_hash,
                                            output_hash, /*verified=*/true);
  zkbench::BenchmarkContext::SetMetadata(
      "BM_bf64_mul_baseline", {{"field", "BF64"}, {"arch", "baseline"}});
}

#if defined(PRIME_IR_X86)
void BM_bf64_mul_x86(::benchmark::State &state) {
  auto a = BinaryFieldT6::Random();
  auto b = BinaryFieldT6::Random();
  BinaryFieldT6 result;

  uint64_t inputs[] = {a.value(), b.value()};
  std::string input_hash = zkbench::ComputeArrayHash(inputs, 2);

  for (auto _ : state) {
    result = _mlir_ciface_bf64_mul_x86(a.value(), b.value());
    ::benchmark::DoNotOptimize(result);
    a += result;
  }
  state.SetItemsProcessed(state.iterations());

  uint64_t out = result.value();
  std::string output_hash = zkbench::ComputeArrayHash(&out, 1);
  zkbench::BenchmarkContext::SetTestVectors("BM_bf64_mul_x86", input_hash,
                                            output_hash, /*verified=*/true);
  zkbench::BenchmarkContext::SetMetadata("BM_bf64_mul_x86",
                                         {{"field", "BF64"}, {"arch", "x86"}});
}
#endif

#if defined(PRIME_IR_ARM)
void BM_bf64_mul_arm(::benchmark::State &state) {
  auto a = BinaryFieldT6::Random();
  auto b = BinaryFieldT6::Random();
  BinaryFieldT6 result;

  uint64_t inputs[] = {a.value(), b.value()};
  std::string input_hash = zkbench::ComputeArrayHash(inputs, 2);

  for (auto _ : state) {
    result = _mlir_ciface_bf64_mul_arm(a.value(), b.value());
    ::benchmark::DoNotOptimize(result);
    a += result;
  }
  state.SetItemsProcessed(state.iterations());

  uint64_t out = result.value();
  std::string output_hash = zkbench::ComputeArrayHash(&out, 1);
  zkbench::BenchmarkContext::SetTestVectors("BM_bf64_mul_arm", input_hash,
                                            output_hash, /*verified=*/true);
  zkbench::BenchmarkContext::SetMetadata("BM_bf64_mul_arm",
                                         {{"field", "BF64"}, {"arch", "arm"}});
}
#endif

// =============================================================================
// BF128 Benchmarks
// =============================================================================

void BM_bf128_mul_baseline(::benchmark::State &state) {
  auto a = BinaryFieldT7::Random();
  auto b = BinaryFieldT7::Random();
  BinaryFieldT7 result;

  std::string input_hash = zkbench::ComputeArrayHash(&a, 1);

  for (auto _ : state) {
    result = _mlir_ciface_bf128_mul_baseline(a.value(), b.value());
    ::benchmark::DoNotOptimize(result);
    a += result;
  }
  state.SetItemsProcessed(state.iterations());

  std::string output_hash = zkbench::ComputeArrayHash(&result, 1);
  zkbench::BenchmarkContext::SetTestVectors("BM_bf128_mul_baseline", input_hash,
                                            output_hash, /*verified=*/true);
  zkbench::BenchmarkContext::SetMetadata(
      "BM_bf128_mul_baseline", {{"field", "BF128"}, {"arch", "baseline"}});
}

#if defined(PRIME_IR_X86)
void BM_bf128_mul_x86(::benchmark::State &state) {
  auto a = BinaryFieldT7::Random();
  auto b = BinaryFieldT7::Random();
  BinaryFieldT7 result;

  std::string input_hash = zkbench::ComputeArrayHash(&a, 1);

  for (auto _ : state) {
    result = _mlir_ciface_bf128_mul_x86(a.value(), b.value());
    ::benchmark::DoNotOptimize(result);
    a += result;
  }
  state.SetItemsProcessed(state.iterations());

  std::string output_hash = zkbench::ComputeArrayHash(&result, 1);
  zkbench::BenchmarkContext::SetTestVectors("BM_bf128_mul_x86", input_hash,
                                            output_hash, /*verified=*/true);
  zkbench::BenchmarkContext::SetMetadata("BM_bf128_mul_x86",
                                         {{"field", "BF128"}, {"arch", "x86"}});
}
#endif

#if defined(PRIME_IR_ARM)
void BM_bf128_mul_arm(::benchmark::State &state) {
  auto a = BinaryFieldT7::Random();
  auto b = BinaryFieldT7::Random();
  BinaryFieldT7 result;

  std::string input_hash = zkbench::ComputeArrayHash(&a, 1);

  for (auto _ : state) {
    result = _mlir_ciface_bf128_mul_arm(a.value(), b.value());
    ::benchmark::DoNotOptimize(result);
    a += result;
  }
  state.SetItemsProcessed(state.iterations());

  std::string output_hash = zkbench::ComputeArrayHash(&result, 1);
  zkbench::BenchmarkContext::SetTestVectors("BM_bf128_mul_arm", input_hash,
                                            output_hash, /*verified=*/true);
  zkbench::BenchmarkContext::SetMetadata("BM_bf128_mul_arm",
                                         {{"field", "BF128"}, {"arch", "arm"}});
}

void BM_bf128_mul_arm_polyval(::benchmark::State &state) {
  auto a = BinaryFieldT7::Random();
  auto b = BinaryFieldT7::Random();
  BinaryFieldT7 result;

  for (auto _ : state) {
    result = _mlir_ciface_bf128_mul_arm_polyval(a.value(), b.value());
    ::benchmark::DoNotOptimize(result);
    a += result;
  }
  state.SetItemsProcessed(state.iterations());
}

void BM_bf64_square_arm(::benchmark::State &state) {
  auto a = BinaryFieldT6::Random();
  BinaryFieldT6 result;

  for (auto _ : state) {
    result = _mlir_ciface_bf64_square_arm(a.value());
    ::benchmark::DoNotOptimize(result);
    a += result;
  }
  state.SetItemsProcessed(state.iterations());
}

void BM_bf128_square_arm(::benchmark::State &state) {
  auto a = BinaryFieldT7::Random();
  BinaryFieldT7 result;

  for (auto _ : state) {
    result = _mlir_ciface_bf128_square_arm(a.value());
    ::benchmark::DoNotOptimize(result);
    a += result;
  }
  state.SetItemsProcessed(state.iterations());
}
#endif

// =============================================================================
// BF8 Benchmarks
// =============================================================================
void BM_bf8_mul_baseline(::benchmark::State &state) {
  auto a = BinaryFieldT3::Random();
  auto b = BinaryFieldT3::Random();

  uint8_t inputs[] = {a.value(), b.value()};
  std::string input_hash = zkbench::ComputeArrayHash(inputs, 2);

  uint8_t last_result = 0;
  for (auto _ : state) {
    last_result = _mlir_ciface_bf8_mul_baseline(a.value(), b.value());
    ::benchmark::DoNotOptimize(last_result);
    a = BinaryFieldT3(last_result);
  }
  state.SetItemsProcessed(state.iterations());

  std::string output_hash = zkbench::ComputeArrayHash(&last_result, 1);
  zkbench::BenchmarkContext::SetTestVectors("BM_bf8_mul_baseline", input_hash,
                                            output_hash, /*verified=*/true);
  zkbench::BenchmarkContext::SetMetadata(
      "BM_bf8_mul_baseline", {{"field", "BF8"}, {"arch", "baseline"}});
}

#if defined(PRIME_IR_X86)
void BM_bf8x16_mul_gfni(::benchmark::State &state) {
  // Aligned buffers for SIMD (128-bit = 16 bytes)
  alignas(16) uint8_t a_data[16], b_data[16], c_data[16];

  // Initialize with random values
  for (int i = 0; i < 16; ++i) {
    a_data[i] = BinaryFieldT3::Random().value();
    b_data[i] = BinaryFieldT3::Random().value();
  }

  std::string input_hash = zkbench::ComputeArrayHash(a_data, 16);

  // Create memref descriptors
  MemRef1D16xi8 a_memref = {a_data, a_data, 0, {16}, {1}};
  MemRef1D16xi8 b_memref = {b_data, b_data, 0, {16}, {1}};
  MemRef1D16xi8 c_memref = {c_data, c_data, 0, {16}, {1}};

  for (auto _ : state) {
    _mlir_ciface_bf8x16_mul_gfni(&a_memref, &b_memref, &c_memref);
    ::benchmark::DoNotOptimize(c_data);
    // Swap a and c to prevent optimization and create data dependency
    std::memcpy(a_data, c_data, 16);
  }
  // 16 multiplications per call
  state.SetItemsProcessed(state.iterations() * 16);

  std::string output_hash = zkbench::ComputeArrayHash(c_data, 16);
  zkbench::BenchmarkContext::SetTestVectors("BM_bf8x16_mul_gfni", input_hash,
                                            output_hash, /*verified=*/true);
  zkbench::BenchmarkContext::SetMetadata(
      "BM_bf8x16_mul_gfni",
      {{"field", "BF8"}, {"arch", "x86_gfni"}, {"packed", 16}});
}
#endif

#if defined(PRIME_IR_ARM)
void BM_bf8x16_mul_pmull(::benchmark::State &state) {
  // Aligned buffers for SIMD (128-bit = 16 bytes)
  alignas(16) uint8_t a_data[16], b_data[16], c_data[16];

  // Initialize with random values
  for (int i = 0; i < 16; ++i) {
    a_data[i] = BinaryFieldT3::Random().value();
    b_data[i] = BinaryFieldT3::Random().value();
  }

  std::string input_hash = zkbench::ComputeArrayHash(a_data, 16);

  // Create memref descriptors
  MemRef1D16xi8 a_memref = {a_data, a_data, 0, {16}, {1}};
  MemRef1D16xi8 b_memref = {b_data, b_data, 0, {16}, {1}};
  MemRef1D16xi8 c_memref = {c_data, c_data, 0, {16}, {1}};

  for (auto _ : state) {
    _mlir_ciface_bf8x16_mul_pmull(&a_memref, &b_memref, &c_memref);
    ::benchmark::DoNotOptimize(c_data);
    // Swap a and c to prevent optimization and create data dependency
    std::memcpy(a_data, c_data, 16);
  }
  // 16 multiplications per call
  state.SetItemsProcessed(state.iterations() * 16);

  std::string output_hash = zkbench::ComputeArrayHash(c_data, 16);
  zkbench::BenchmarkContext::SetTestVectors("BM_bf8x16_mul_pmull", input_hash,
                                            output_hash, /*verified=*/true);
  zkbench::BenchmarkContext::SetMetadata(
      "BM_bf8x16_mul_pmull",
      {{"field", "BF8"}, {"arch", "arm_pmull"}, {"packed", 16}});
}
#endif

// =============================================================================
// BF8 Inverse Benchmarks
// =============================================================================
void BM_bf8_inverse_baseline(::benchmark::State &state) {
  auto a = BinaryFieldT3::Random();
  // Ensure non-zero input (inverse of 0 is undefined)
  if (a.IsZero())
    a = BinaryFieldT3::One();

  for (auto _ : state) {
    auto result = _mlir_ciface_bf8_inverse_baseline(a.value());
    ::benchmark::DoNotOptimize(result);
    // Chain results to prevent optimization, but keep non-zero
    a = BinaryFieldT3(result);
    if (a.IsZero())
      a = BinaryFieldT3::One();
  }
  state.SetItemsProcessed(state.iterations());
}

// =============================================================================
// Register benchmarks
// =============================================================================
BENCHMARK(BM_bf64_mul_baseline);
#if defined(PRIME_IR_X86)
BENCHMARK(BM_bf64_mul_x86);
#endif
#if defined(PRIME_IR_ARM)
BENCHMARK(BM_bf64_mul_arm);
#endif

BENCHMARK(BM_bf128_mul_baseline);
#if defined(PRIME_IR_X86)
BENCHMARK(BM_bf128_mul_x86);
#endif
#if defined(PRIME_IR_ARM)
BENCHMARK(BM_bf128_mul_arm);
BENCHMARK(BM_bf128_mul_arm_polyval);
BENCHMARK(BM_bf64_square_arm);
BENCHMARK(BM_bf128_square_arm);
#endif

BENCHMARK(BM_bf8_mul_baseline);
#if defined(PRIME_IR_X86)
BENCHMARK(BM_bf8x16_mul_gfni);
#endif
#if defined(PRIME_IR_ARM)
BENCHMARK(BM_bf8x16_mul_pmull);
#endif

BENCHMARK(BM_bf8_inverse_baseline);

} // namespace
} // namespace mlir::prime_ir::benchmark

// clang-format off
// 2026-01-28T08:55:52+00:00
// Run on (32 X 4352 MHz CPU s)
// CPU Caches:
//   L1 Data 48 KiB (x16)
//   L1 Instruction 32 KiB (x16)
//   L2 Unified 1024 KiB (x16)
//   L3 Unified 98304 KiB (x2)
// Load Average: 1.55, 4.66, 7.77
// ----------------------------------------------------------------
// Benchmark                      Time             CPU   Iterations
// ----------------------------------------------------------------
// BM_bf64_mul_baseline         383 ns          383 ns      1824239
// BM_bf64_mul_x86             3.49 ns         3.49 ns    200760775
// BM_bf128_mul_baseline       1189 ns         1189 ns        59079
// BM_bf128_mul_x86            5.40 ns         5.40 ns    129165900
// BM_bf8_mul_baseline         9.63 ns         9.63 ns      7178489
// BM_bf8x16_mul_gfni          4.80 ns         4.80 ns    145784305
// clang-format on

// clang-format off
// 2026-01-29T05:20:58+00:00
// Run on (14 X 24 MHz CPU s)
// CPU Caches:
//   L1 Data 64 KiB
//   L1 Instruction 128 KiB
//   L2 Unified 4096 KiB (x14)
// Load Average: 22.48, 11.89, 7.08
// -------------------------------------------------------------------
// Benchmark                         Time             CPU   Iterations
// -------------------------------------------------------------------
// BM_bf64_mul_baseline            192 ns          191 ns      3607225
// BM_bf64_mul_arm                6.01 ns         6.00 ns    113719438
// BM_bf128_mul_baseline           613 ns          604 ns      1065676
// BM_bf128_mul_arm               8.51 ns         8.50 ns     91708263
// BM_bf128_mul_arm_polyval       12.6 ns         12.5 ns     53374406
// BM_bf64_square_arm             5.90 ns         5.89 ns    115364965
// BM_bf128_square_arm            7.89 ns         7.86 ns     88692920
// BM_bf8_mul_baseline            8.04 ns         8.00 ns     87628156
// BM_bf8x16_mul_pmull            4.44 ns         4.41 ns    163183857
// BM_bf8_inverse_baseline        3.44 ns         3.42 ns    566985258
// clang-format on
