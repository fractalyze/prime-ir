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

const i256 kPrimeScalar = i256::fromHexString(
    "0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001");

void fillWithRandom(i256 &elem, ArrayRef<int64_t> coords) {
  // Set up the random number generator.
  static std::mt19937_64 rng(
      std::random_device{}()); // NOLINT(whitespace/braces)
  static std::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);
  elem = i256::randomLT(kPrimeScalar, rng, dist);
}

extern "C" void
_mlir_ciface_generate_points(StridedMemRefType<i256, 0> *rand_scalar,
                             StridedMemRefType<i256, 2> *out);
extern "C" void _mlir_ciface_gpu_msm(StridedMemRefType<i256, 1> *scalars,
                                     StridedMemRefType<i256, 2> *points);

void BM_msm_benchmark(::benchmark::State &state) {
  OwningMemRef<i256, 1> scalars(/*shape=*/{NUM_SCALARMULS}, /*shapeAlloc=*/{},
                                /*init=*/fillWithRandom);
  OwningMemRef<i256, 0> randScalar(/*shape=*/{}, /*shapeAlloc=*/{},
                                   /*init=*/fillWithRandom);
  OwningMemRef<i256, 2> points(/*shape=*/{NUM_SCALARMULS, 2},
                               /*shapeAlloc=*/{});

  _mlir_ciface_generate_points(&*randScalar, &*points);

  for (auto _ : state) {
    _mlir_ciface_gpu_msm(&*scalars, &*points);
  }
}

BENCHMARK(BM_msm_benchmark)
    ->Iterations(20)
    ->Unit(::benchmark::kMillisecond)
    ->Name("gpu_msm");

} // namespace
} // namespace mlir::zkir::benchmark

// clang-format off
// NOLINTBEGIN(whitespace/line_length)

// 2025-10-01T03:30:55+00:00
// Run on (32 X 624 MHz CPU s)
// CPU Caches:
//   L1 Data 48 KiB (x16)
//   L1 Instruction 32 KiB (x16)
//   L2 Unified 1024 KiB (x16)
//   L3 Unified 98304 KiB (x2)
// Load Average: 0.24, 0.21, 0.27
// ----------------------------------------------------------------
// Benchmark                      Time             CPU   Iterations
// ----------------------------------------------------------------
// gpu_msm/iterations:20        505 ms          504 ms           20
// NOLINTEND()
