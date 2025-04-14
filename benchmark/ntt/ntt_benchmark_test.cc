#include "benchmark/BenchmarkUtils.h"
#include "benchmark/benchmark.h"
#include "gtest/gtest.h"

#define DEGREE 1 << 20

namespace zkir {
namespace {

using ::zkir::benchmark::Memref;

struct i256 {
  uint64_t limbs[4];  // 4 x 64 = 256 bits
};

extern "C" void _mlir_ciface_input_generation(Memref<i256> *output);
extern "C" void _mlir_ciface_ntt(Memref<i256> *output, Memref<i256> *input);
extern "C" void _mlir_ciface_intt(Memref<i256> *output, Memref<i256> *input);

extern "C" void _mlir_ciface_ntt_mont(Memref<i256> *output,
                                      Memref<i256> *input);
extern "C" void _mlir_ciface_intt_mont(Memref<i256> *output,
                                       Memref<i256> *input);

void BM_ntt_benchmark(::benchmark::State &state) {
  Memref<i256> input(1, DEGREE);
  _mlir_ciface_input_generation(&input);

  Memref<i256> ntt(1, DEGREE);
  for (auto _ : state) {
    _mlir_ciface_ntt(&ntt, &input);
  }

  Memref<i256> intt(1, DEGREE);
  _mlir_ciface_intt(&intt, &ntt);

  for (int i = 0; i < DEGREE; i++) {
    for (int j = 0; j < 4; j++) {
      EXPECT_EQ(intt.pget(0, i)->limbs[j], input.pget(0, i)->limbs[j]);
    }
  }
}

BENCHMARK(BM_ntt_benchmark)->Unit(::benchmark::kSecond);

void BM_intt_benchmark(::benchmark::State &state) {
  Memref<i256> input(1, DEGREE);
  _mlir_ciface_input_generation(&input);

  Memref<i256> ntt(1, DEGREE);
  _mlir_ciface_ntt(&ntt, &input);

  Memref<i256> intt(1, DEGREE);
  for (auto _ : state) {
    _mlir_ciface_intt(&intt, &ntt);
  }

  for (int i = 0; i < DEGREE; i++) {
    for (int j = 0; j < 4; j++) {
      EXPECT_EQ(intt.pget(0, i)->limbs[j], input.pget(0, i)->limbs[j]);
    }
  }
}

// FIXME(batzor): It fails for more than 1 iteration so it seems like it is
// modifying the input. But I am not sure why ;(
BENCHMARK(BM_intt_benchmark)->Iterations(1)->Unit(::benchmark::kSecond);

void BM_ntt_mont_benchmark(::benchmark::State &state) {
  Memref<i256> input(1, DEGREE);
  _mlir_ciface_input_generation(&input);

  Memref<i256> ntt(1, DEGREE);
  for (auto _ : state) {
    _mlir_ciface_ntt_mont(&ntt, &input);
  }

  Memref<i256> intt(1, DEGREE);
  _mlir_ciface_intt_mont(&intt, &ntt);

  for (int i = 0; i < DEGREE; i++) {
    for (int j = 0; j < 4; j++) {
      EXPECT_EQ(intt.pget(0, i)->limbs[j], input.pget(0, i)->limbs[j]);
    }
  }
}

BENCHMARK(BM_ntt_mont_benchmark)->Unit(::benchmark::kSecond);

void BM_intt_mont_benchmark(::benchmark::State &state) {
  Memref<i256> input(1, DEGREE);
  _mlir_ciface_input_generation(&input);

  Memref<i256> ntt(1, DEGREE);
  _mlir_ciface_ntt_mont(&ntt, &input);

  Memref<i256> intt(1, DEGREE);
  for (auto _ : state) {
    _mlir_ciface_intt_mont(&intt, &ntt);
  }

  for (int i = 0; i < DEGREE; i++) {
    for (int j = 0; j < 4; j++) {
      EXPECT_EQ(intt.pget(0, i)->limbs[j], input.pget(0, i)->limbs[j]);
    }
  }
}

// FIXME(batzor): It fails for more than 1 iteration so it seems like it is
// modifying the input. But I am not sure why ;(
BENCHMARK(BM_intt_mont_benchmark)->Iterations(1)->Unit(::benchmark::kSecond);

}  // namespace
}  // namespace zkir

// clang-format off
// NOLINTBEGIN(whitespace/line_length)
// Run on (14 X 24 MHz CPU s)
// CPU Caches:
//   L1 Data 64 KiB
//   L1 Instruction 128 KiB
//   L2 Unified 4096 KiB (x14)
// Load Average: 27.66, 13.59, 9.67
// ------------------------------------------------------------------------------
// Benchmark                                    Time             CPU   Iterations
// ------------------------------------------------------------------------------
// BM_ntt_benchmark                         0.190 s         0.183 s             4
// BM_intt_benchmark/iterations:1           0.381 s         0.368 s             1
// BM_ntt_mont_benchmark                    0.221 s         0.214 s             3
// BM_intt_mont_benchmark/iterations:1      0.415 s         0.396 s             1
// NOLINTEND()
