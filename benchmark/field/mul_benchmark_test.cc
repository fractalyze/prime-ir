#include "benchmark/BenchmarkUtils.h"
#include "benchmark/benchmark.h"
#include "gtest/gtest.h"

namespace zkir {
namespace {

using ::zkir::benchmark::Memref;

struct i256 {
  uint64_t limbs[4];  // 4 x 64 = 256 bits
};

extern "C" void _mlir_ciface_mul(Memref<i256> *output, Memref<i256> *input);
extern "C" void _mlir_ciface_mont_mul(Memref<i256> *output,
                                      Memref<i256> *input);

void BM_mul_benchmark(::benchmark::State &state) {
  Memref<i256> input(1, 1);

  input.pget(0, 0)->limbs[0] = 0x0032131ffffffffff;
  input.pget(0, 0)->limbs[1] = 0x0032131ffffffffff;
  input.pget(0, 0)->limbs[2] = 0x0032131ffffffffff;
  input.pget(0, 0)->limbs[3] = 0x0032131ffffffffff;

  Memref<i256> output(1, 1);
  for (auto _ : state) {
    _mlir_ciface_mul(&output, &input);
  }
}

BENCHMARK(BM_mul_benchmark);

void BM_mont_mul_benchmark(::benchmark::State &state) {
  Memref<i256> input(1, 1);

  input.pget(0, 0)->limbs[0] = 0x0032131ffffffffff;
  input.pget(0, 0)->limbs[1] = 0x0032131ffffffffff;
  input.pget(0, 0)->limbs[2] = 0x0032131ffffffffff;
  input.pget(0, 0)->limbs[3] = 0x0032131ffffffffff;

  Memref<i256> mont_output(1, 1);
  for (auto _ : state) {
    _mlir_ciface_mont_mul(&mont_output, &input);
  }
}

BENCHMARK(BM_mont_mul_benchmark);

}  // namespace
}  // namespace zkir

// Run on (14 X 24 MHz CPU s)
// CPU Caches:
//   L1 Data 64 KiB
//   L1 Instruction 128 KiB
//   L2 Unified 4096 KiB (x14)
// Load Average: 7.70, 6.06, 6.06
// ----------------------------------------------------------------
// Benchmark                      Time             CPU   Iterations
// ----------------------------------------------------------------
// BM_mul_benchmark            2575 ns         2457 ns       294375
// BM_mont_mul_benchmark       30.9 ns         30.2 ns     23041778
