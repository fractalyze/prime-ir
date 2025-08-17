#include "benchmark/BenchmarkUtils.h"
#include "benchmark/benchmark.h"
#include "mlir/ExecutionEngine/MemRefUtils.h"
#include "mlir/Support/LLVM.h"

#define N (1 << 20)

namespace mlir::zkir::benchmark {
namespace {

using i256 = BigInt<4>;

// `kPrime` =
// 21888242871839275222246405745257275088548364400416034343698204186575808495617
const i256 kPrime = i256::fromHexString(
    "0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001");

extern "C" void _mlir_ciface_mul(StridedMemRefType<i256, 1> *input1,
                                 StridedMemRefType<i256, 1> *input2,
                                 StridedMemRefType<i256, 0> *output);
extern "C" void _mlir_ciface_mont_mul(StridedMemRefType<i256, 1> *input1,
                                      StridedMemRefType<i256, 1> *input2,
                                      StridedMemRefType<i256, 0> *output);

// Set up the random number generator.
std::mt19937_64 rng(std::random_device{}()); // NOLINT(whitespace/braces)
std::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);

// Set each element to a random number in [0, `kPrime`).
void fillWithRandom(i256 &elem, ArrayRef<int64_t> coords) {
  elem = i256::randomLT(kPrime, rng, dist);
}

template <bool kIsMont>
void BM_mul_benchmark(::benchmark::State &state) {
  OwningMemRef<i256, 1> input1(/*shape=*/{N}, /*shapeAlloc=*/{},
                               /*init=*/fillWithRandom);
  OwningMemRef<i256, 1> input2(/*shape=*/{N}, /*shapeAlloc=*/{},
                               /*init=*/fillWithRandom);
  OwningMemRef<i256, 0> output(/*shape=*/{}, /*shapeAlloc=*/{}, /*init=*/{});
  for (auto _ : state) {
    if constexpr (kIsMont) {
      _mlir_ciface_mont_mul(&*input1, &*input2, &*output);
    } else {
      _mlir_ciface_mul(&*input1, &*input2, &*output);
    }
  }
}

BENCHMARK_TEMPLATE(BM_mul_benchmark, /*kIsMont=*/false)
    ->Unit(::benchmark::kMillisecond)
    ->Name("mul");
BENCHMARK_TEMPLATE(BM_mul_benchmark, /*kIsMont=*/true)
    ->Unit(::benchmark::kMillisecond)
    ->Name("mont_mul");

} // namespace
} // namespace mlir::zkir::benchmark

// clang-format off
// NOLINTBEGIN(whitespace/line_length)
//
// 2025-08-07T01:48:24+00:00
// Run on AMD Ryzen 9 9950X3D (32 X 5479.99 MHz CPU s)
// CPU Caches:
//   L1 Data 48 KiB (x16)
//   L1 Instruction 32 KiB (x16)
//   L2 Unified 1024 KiB (x16)
//   L3 Unified 98304 KiB (x2)
// Load Average: 0.63, 2.23, 4.92
// -----------------------------------------------------
// Benchmark           Time             CPU   Iterations
// -----------------------------------------------------
// mul              2031 ms         2031 ms            1
// mont_mul         13.1 ms         13.1 ms           55
// NOLINTEND()
