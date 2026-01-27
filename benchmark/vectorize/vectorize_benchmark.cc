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
#include "zk_dtypes/include/field/babybear/babybear4.h"
#include "zk_dtypes/include/field/mersenne31/mersenne312.h"

namespace mlir::prime_ir::benchmark {
namespace {

constexpr int64_t kBufferSize = 1048576;    // 1M elements for prime field
constexpr int64_t kExt2BufferSize = 262144; // 256K elements for quadratic ext
constexpr int64_t kExt4BufferSize = 65536;  // 64K elements for quartic ext

using Mersenne31X2 = zk_dtypes::Mersenne312;
using BabybearX4 = zk_dtypes::Babybear4;

// ==============================================================================
// Prime field function declarations
// ==============================================================================

// Scalar versions
extern "C" void _mlir_ciface_square_buffer(StridedMemRefType<uint32_t, 1> *buf);
extern "C" void _mlir_ciface_add_buffers(StridedMemRefType<uint32_t, 1> *a,
                                         StridedMemRefType<uint32_t, 1> *b,
                                         StridedMemRefType<uint32_t, 1> *c);
extern "C" void _mlir_ciface_mul_add_buffers(StridedMemRefType<uint32_t, 1> *a,
                                             StridedMemRefType<uint32_t, 1> *b,
                                             StridedMemRefType<uint32_t, 1> *c);

// Vectorized versions (same functions, different compilation)
extern "C" void
_mlir_ciface_vec_square_buffer(StridedMemRefType<uint32_t, 1> *buf);
extern "C" void _mlir_ciface_vec_add_buffers(StridedMemRefType<uint32_t, 1> *a,
                                             StridedMemRefType<uint32_t, 1> *b,
                                             StridedMemRefType<uint32_t, 1> *c);
extern "C" void
_mlir_ciface_vec_mul_add_buffers(StridedMemRefType<uint32_t, 1> *a,
                                 StridedMemRefType<uint32_t, 1> *b,
                                 StridedMemRefType<uint32_t, 1> *c);

// ==============================================================================
// Extension field function declarations
// ==============================================================================

// Quadratic extension field (scalar)
extern "C" void
_mlir_ciface_ext2_square_buffer(StridedMemRefType<Mersenne31X2, 1> *buf);
extern "C" void
_mlir_ciface_ext2_mul_buffers(StridedMemRefType<Mersenne31X2, 1> *a,
                              StridedMemRefType<Mersenne31X2, 1> *b,
                              StridedMemRefType<Mersenne31X2, 1> *c);

// Quadratic extension field (vectorized)
extern "C" void
_mlir_ciface_vec_ext2_square_buffer(StridedMemRefType<Mersenne31X2, 1> *buf);
extern "C" void
_mlir_ciface_vec_ext2_mul_buffers(StridedMemRefType<Mersenne31X2, 1> *a,
                                  StridedMemRefType<Mersenne31X2, 1> *b,
                                  StridedMemRefType<Mersenne31X2, 1> *c);

// Quartic extension field (scalar)
extern "C" void
_mlir_ciface_ext4_square_buffer(StridedMemRefType<BabybearX4, 1> *buf);
extern "C" void
_mlir_ciface_ext4_mul_buffers(StridedMemRefType<BabybearX4, 1> *a,
                              StridedMemRefType<BabybearX4, 1> *b,
                              StridedMemRefType<BabybearX4, 1> *c);

// Quartic extension field (vectorized)
extern "C" void
_mlir_ciface_vec_ext4_square_buffer(StridedMemRefType<BabybearX4, 1> *buf);
extern "C" void
_mlir_ciface_vec_ext4_mul_buffers(StridedMemRefType<BabybearX4, 1> *a,
                                  StridedMemRefType<BabybearX4, 1> *b,
                                  StridedMemRefType<BabybearX4, 1> *c);

void fillWithValue(uint32_t &elem, ArrayRef<int64_t> coords) {
  elem = static_cast<uint32_t>(coords[0] % 1000 + 1);
}

// Square buffer benchmarks
void BM_square_scalar(::benchmark::State &state) {
  OwningMemRef<uint32_t, 1> buf({kBufferSize}, {}, fillWithValue);
  for (auto _ : state) {
    _mlir_ciface_square_buffer(&*buf);
  }
}

void BM_square_vectorized(::benchmark::State &state) {
  OwningMemRef<uint32_t, 1> buf({kBufferSize}, {}, fillWithValue);
  for (auto _ : state) {
    _mlir_ciface_vec_square_buffer(&*buf);
  }
}

// Add buffers benchmarks
void BM_add_scalar(::benchmark::State &state) {
  OwningMemRef<uint32_t, 1> a({kBufferSize}, {}, fillWithValue);
  OwningMemRef<uint32_t, 1> b({kBufferSize}, {}, fillWithValue);
  OwningMemRef<uint32_t, 1> c({kBufferSize}, {}, fillWithValue);
  for (auto _ : state) {
    _mlir_ciface_add_buffers(&*a, &*b, &*c);
  }
}

void BM_add_vectorized(::benchmark::State &state) {
  OwningMemRef<uint32_t, 1> a({kBufferSize}, {}, fillWithValue);
  OwningMemRef<uint32_t, 1> b({kBufferSize}, {}, fillWithValue);
  OwningMemRef<uint32_t, 1> c({kBufferSize}, {}, fillWithValue);
  for (auto _ : state) {
    _mlir_ciface_vec_add_buffers(&*a, &*b, &*c);
  }
}

// Mul-add buffers benchmarks
void BM_mul_add_scalar(::benchmark::State &state) {
  OwningMemRef<uint32_t, 1> a({kBufferSize}, {}, fillWithValue);
  OwningMemRef<uint32_t, 1> b({kBufferSize}, {}, fillWithValue);
  OwningMemRef<uint32_t, 1> c({kBufferSize}, {}, fillWithValue);
  for (auto _ : state) {
    _mlir_ciface_mul_add_buffers(&*a, &*b, &*c);
  }
}

void BM_mul_add_vectorized(::benchmark::State &state) {
  OwningMemRef<uint32_t, 1> a({kBufferSize}, {}, fillWithValue);
  OwningMemRef<uint32_t, 1> b({kBufferSize}, {}, fillWithValue);
  OwningMemRef<uint32_t, 1> c({kBufferSize}, {}, fillWithValue);
  for (auto _ : state) {
    _mlir_ciface_vec_mul_add_buffers(&*a, &*b, &*c);
  }
}

BENCHMARK(BM_square_scalar)->Unit(::benchmark::kMillisecond);
BENCHMARK(BM_square_vectorized)->Unit(::benchmark::kMillisecond);
BENCHMARK(BM_add_scalar)->Unit(::benchmark::kMillisecond);
BENCHMARK(BM_add_vectorized)->Unit(::benchmark::kMillisecond);
BENCHMARK(BM_mul_add_scalar)->Unit(::benchmark::kMillisecond);
BENCHMARK(BM_mul_add_vectorized)->Unit(::benchmark::kMillisecond);

// ==============================================================================
// Extension Field Benchmarks
// ==============================================================================

void fillExt2WithValue(Mersenne31X2 &elem, ArrayRef<int64_t> coords) {
  uint32_t base = static_cast<uint32_t>(coords[0] % 1000 + 1);
  elem[0] = base;
  elem[1] = base + 1;
}

void fillExt4WithValue(BabybearX4 &elem, ArrayRef<int64_t> coords) {
  uint32_t base = static_cast<uint32_t>(coords[0] % 1000 + 1);
  elem[0] = base;
  elem[1] = base + 1;
  elem[2] = base + 2;
  elem[3] = base + 3;
}

// Quadratic extension field square benchmarks
void BM_ext2_square_scalar(::benchmark::State &state) {
  OwningMemRef<Mersenne31X2, 1> buf({kExt2BufferSize}, {}, fillExt2WithValue);
  for (auto _ : state) {
    _mlir_ciface_ext2_square_buffer(&*buf);
  }
}

void BM_ext2_square_vectorized(::benchmark::State &state) {
  OwningMemRef<Mersenne31X2, 1> buf({kExt2BufferSize}, {}, fillExt2WithValue);
  for (auto _ : state) {
    _mlir_ciface_vec_ext2_square_buffer(&*buf);
  }
}

// Quadratic extension field multiply benchmarks
void BM_ext2_mul_scalar(::benchmark::State &state) {
  OwningMemRef<Mersenne31X2, 1> a({kExt2BufferSize}, {}, fillExt2WithValue);
  OwningMemRef<Mersenne31X2, 1> b({kExt2BufferSize}, {}, fillExt2WithValue);
  OwningMemRef<Mersenne31X2, 1> c({kExt2BufferSize}, {}, fillExt2WithValue);
  for (auto _ : state) {
    _mlir_ciface_ext2_mul_buffers(&*a, &*b, &*c);
  }
}

void BM_ext2_mul_vectorized(::benchmark::State &state) {
  OwningMemRef<Mersenne31X2, 1> a({kExt2BufferSize}, {}, fillExt2WithValue);
  OwningMemRef<Mersenne31X2, 1> b({kExt2BufferSize}, {}, fillExt2WithValue);
  OwningMemRef<Mersenne31X2, 1> c({kExt2BufferSize}, {}, fillExt2WithValue);
  for (auto _ : state) {
    _mlir_ciface_vec_ext2_mul_buffers(&*a, &*b, &*c);
  }
}

// Quartic extension field square benchmarks
void BM_ext4_square_scalar(::benchmark::State &state) {
  OwningMemRef<BabybearX4, 1> buf({kExt4BufferSize}, {}, fillExt4WithValue);
  for (auto _ : state) {
    _mlir_ciface_ext4_square_buffer(&*buf);
  }
}

void BM_ext4_square_vectorized(::benchmark::State &state) {
  OwningMemRef<BabybearX4, 1> buf({kExt4BufferSize}, {}, fillExt4WithValue);
  for (auto _ : state) {
    _mlir_ciface_vec_ext4_square_buffer(&*buf);
  }
}

// Quartic extension field multiply benchmarks
void BM_ext4_mul_scalar(::benchmark::State &state) {
  OwningMemRef<BabybearX4, 1> a({kExt4BufferSize}, {}, fillExt4WithValue);
  OwningMemRef<BabybearX4, 1> b({kExt4BufferSize}, {}, fillExt4WithValue);
  OwningMemRef<BabybearX4, 1> c({kExt4BufferSize}, {}, fillExt4WithValue);
  for (auto _ : state) {
    _mlir_ciface_ext4_mul_buffers(&*a, &*b, &*c);
  }
}

void BM_ext4_mul_vectorized(::benchmark::State &state) {
  OwningMemRef<BabybearX4, 1> a({kExt4BufferSize}, {}, fillExt4WithValue);
  OwningMemRef<BabybearX4, 1> b({kExt4BufferSize}, {}, fillExt4WithValue);
  OwningMemRef<BabybearX4, 1> c({kExt4BufferSize}, {}, fillExt4WithValue);
  for (auto _ : state) {
    _mlir_ciface_vec_ext4_mul_buffers(&*a, &*b, &*c);
  }
}

BENCHMARK(BM_ext2_square_scalar)->Unit(::benchmark::kMillisecond);
BENCHMARK(BM_ext2_square_vectorized)->Unit(::benchmark::kMillisecond);
BENCHMARK(BM_ext2_mul_scalar)->Unit(::benchmark::kMillisecond);
BENCHMARK(BM_ext2_mul_vectorized)->Unit(::benchmark::kMillisecond);
BENCHMARK(BM_ext4_square_scalar)->Unit(::benchmark::kMillisecond);
BENCHMARK(BM_ext4_square_vectorized)->Unit(::benchmark::kMillisecond);
BENCHMARK(BM_ext4_mul_scalar)->Unit(::benchmark::kMillisecond);
BENCHMARK(BM_ext4_mul_vectorized)->Unit(::benchmark::kMillisecond);

} // namespace
} // namespace mlir::prime_ir::benchmark

// 2026-01-27T15:42:04+00:00
// Run on (32 X 624 MHz CPU s)
// CPU Caches:
//   L1 Data 48 KiB (x16)
//   L1 Instruction 32 KiB (x16)
//   L2 Unified 1024 KiB (x16)
//   L3 Unified 98304 KiB (x2)
// Load Average: 24.43, 20.69, 15.88
// --------------------------------------------------------------------
// Benchmark                          Time             CPU   Iterations
// --------------------------------------------------------------------
// BM_square_scalar                 487 ms          486 ms            2
// BM_square_vectorized            77.7 ms         77.7 ms            9
// BM_add_scalar                    297 ms          297 ms            2
// BM_add_vectorized               93.1 ms         93.1 ms            8
// BM_mul_add_scalar                882 ms          882 ms            1
// BM_mul_add_vectorized            130 ms          129 ms            5
// BM_ext2_square_scalar            552 ms          552 ms            1
// BM_ext2_square_vectorized        552 ms          552 ms            1
// BM_ext2_mul_scalar               780 ms          780 ms            1
// BM_ext2_mul_vectorized           787 ms          787 ms            1
// BM_ext4_square_scalar            143 ms          143 ms            5
// BM_ext4_square_vectorized        146 ms          146 ms            5
// BM_ext4_mul_scalar               164 ms          164 ms            4
// BM_ext4_mul_vectorized           164 ms          164 ms            4
