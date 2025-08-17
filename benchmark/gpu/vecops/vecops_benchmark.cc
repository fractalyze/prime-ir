#include <climits>
#include <cstring>
#include <random>

#include "benchmark/BenchmarkUtils.h"
#include "benchmark/benchmark.h"
#include "mlir/ExecutionEngine/MemRefUtils.h"
#include "mlir/Support/LLVM.h"

#define NUM_COEFFS (1 << 20)

namespace mlir::zkir::benchmark {
namespace {

using i64 = BigInt<1>;

extern "C" void _mlir_ciface_matvec_cpu(StridedMemRefType<i64, 2> *mat,
                                        StridedMemRefType<i64, 1> *vec,
                                        StridedMemRefType<i64, 1> *out);
extern "C" void _mlir_ciface_matvec_gpu(StridedMemRefType<i64, 2> *mat,
                                        StridedMemRefType<i64, 1> *vec,
                                        StridedMemRefType<i64, 1> *out);

// `kPrime` = 9223372036836950017
const i64 kPrime = i64({9223372036836950017});

// Set up the random number generator.
std::mt19937_64 rng(std::random_device{}()); // NOLINT(whitespace/braces)
std::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);

// Set the element to random number in [0, kPrime).
void fillWithRandom(i64 &elem, ArrayRef<int64_t> coords) {
  elem = i64::randomLT(kPrime, rng, dist);
}

template <bool kIsGPU>
void BM_matvec_benchmark(::benchmark::State &state) {
  OwningMemRef<i64, 2> mat({NUM_COEFFS, 100}, {}, fillWithRandom);
  OwningMemRef<i64, 1> vec({100}, {}, fillWithRandom);
  OwningMemRef<i64, 1> out({NUM_COEFFS}, {}, {});

  for (auto _ : state) {
    if constexpr (kIsGPU) {
      _mlir_ciface_matvec_gpu(&*mat, &*vec, &*out);
    } else {
      _mlir_ciface_matvec_cpu(&*mat, &*vec, &*out);
    }
  }
}

BENCHMARK_TEMPLATE(BM_matvec_benchmark, /*kIsGPU=*/false)
    ->Unit(::benchmark::kMillisecond)
    ->Name("matvec_cpu");
BENCHMARK_TEMPLATE(BM_matvec_benchmark, /*kIsGPU=*/true)
    ->Unit(::benchmark::kMillisecond)
    ->Name("matvec_gpu");

} // namespace
} // namespace mlir::zkir::benchmark

// clang-format off
// NOLINTBEGIN(whitespace/line_length)
//
// 2025-08-07T08:22:22+00:00
// Run on AMD Ryzen 9 9950X3D (32 X 624 MHz CPU s)
// CPU Caches:
//   L1 Data 48 KiB (x16)
//   L1 Instruction 32 KiB (x16)
//   L2 Unified 1024 KiB (x16)
//   L3 Unified 98304 KiB (x2)
// Load Average: 6.08, 21.10, 15.97
// -----------------------------------------------------
// Benchmark           Time             CPU   Iterations
// -----------------------------------------------------
// matvec_cpu       25.6 ms         25.6 ms           27
// matvec_gpu       57.9 ms         57.9 ms            9
// NOLINTEND()
