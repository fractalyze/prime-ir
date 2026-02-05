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

// BF8 packed 16x PMULL multiplication
// Uses memref interface for SIMD vectors
extern "C" void _mlir_ciface_bf8x16_mul_pmull(MemRef1D16xi8 *a,
                                              MemRef1D16xi8 *b,
                                              MemRef1D16xi8 *c);
#endif

// =============================================================================
// BF64 Benchmarks
// =============================================================================
void BM_bf64_mul_baseline(::benchmark::State &state) {
  auto a = BinaryFieldT6::Random();
  auto b = BinaryFieldT6::Random();
  BinaryFieldT6 result;

  for (auto _ : state) {
    result = _mlir_ciface_bf64_mul_baseline(a.value(), b.value());
    ::benchmark::DoNotOptimize(result);
    a += result;
  }
  state.SetItemsProcessed(state.iterations());
}

#if defined(PRIME_IR_X86)
void BM_bf64_mul_x86(::benchmark::State &state) {
  auto a = BinaryFieldT6::Random();
  auto b = BinaryFieldT6::Random();
  BinaryFieldT6 result;

  for (auto _ : state) {
    result = _mlir_ciface_bf64_mul_x86(a.value(), b.value());
    ::benchmark::DoNotOptimize(result);
    a += result;
  }
  state.SetItemsProcessed(state.iterations());
}
#endif

#if defined(PRIME_IR_ARM)
void BM_bf64_mul_arm(::benchmark::State &state) {
  auto a = BinaryFieldT6::Random();
  auto b = BinaryFieldT6::Random();
  BinaryFieldT6 result;

  for (auto _ : state) {
    result = _mlir_ciface_bf64_mul_arm(a.value(), b.value());
    ::benchmark::DoNotOptimize(result);
    a += result;
  }
  state.SetItemsProcessed(state.iterations());
}
#endif

// =============================================================================
// BF128 Benchmarks
// =============================================================================

void BM_bf128_mul_baseline(::benchmark::State &state) {
  auto a = BinaryFieldT7::Random();
  auto b = BinaryFieldT7::Random();
  BinaryFieldT7 result;

  for (auto _ : state) {
    result = _mlir_ciface_bf128_mul_baseline(a.value(), b.value());
    ::benchmark::DoNotOptimize(result);
    a += result;
  }
  state.SetItemsProcessed(state.iterations());
}

#if defined(PRIME_IR_X86)
void BM_bf128_mul_x86(::benchmark::State &state) {
  auto a = BinaryFieldT7::Random();
  auto b = BinaryFieldT7::Random();
  BinaryFieldT7 result;

  for (auto _ : state) {
    result = _mlir_ciface_bf128_mul_x86(a.value(), b.value());
    ::benchmark::DoNotOptimize(result);
    a += result;
  }
  state.SetItemsProcessed(state.iterations());
}
#endif

#if defined(PRIME_IR_ARM)
void BM_bf128_mul_arm(::benchmark::State &state) {
  auto a = BinaryFieldT7::Random();
  auto b = BinaryFieldT7::Random();
  BinaryFieldT7 result;

  for (auto _ : state) {
    result = _mlir_ciface_bf128_mul_arm(a.value(), b.value());
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

  for (auto _ : state) {
    auto result = _mlir_ciface_bf8_mul_baseline(a.value(), b.value());
    ::benchmark::DoNotOptimize(result);
    a = BinaryFieldT3(result);
  }
  state.SetItemsProcessed(state.iterations());
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
}
#endif

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
#endif

BENCHMARK(BM_bf8_mul_baseline);
#if defined(PRIME_IR_X86)
BENCHMARK(BM_bf8x16_mul_gfni);
#endif
#if defined(PRIME_IR_ARM)
BENCHMARK(BM_bf8x16_mul_pmull);
#endif

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
// 2026-01-28T15:34:01+00:00
// Run on (14 X 24 MHz CPU s)
// CPU Caches:
//   L1 Data 64 KiB
//   L1 Instruction 128 KiB
//   L2 Unified 4096 KiB (x14)
// Load Average: 3.41, 3.18, 3.26
// ----------------------------------------------------------------
// Benchmark                      Time             CPU   Iterations
// ----------------------------------------------------------------
// BM_bf64_mul_baseline         184 ns          180 ns      3857026
// BM_bf64_mul_arm             5.84 ns         5.83 ns    118151436
// BM_bf128_mul_baseline        567 ns          565 ns      1250156
// BM_bf128_mul_arm            7.57 ns         7.56 ns     91668631
// BM_bf8_mul_baseline         7.41 ns         7.41 ns     94597151
// BM_bf8x16_mul_pmull         4.02 ns         4.01 ns    172203417
// clang-format on
