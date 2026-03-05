/* Copyright 2026 The ZKX Authors.

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

// Shared deterministic test-vector utilities for sumcheck benchmarks and tests.
// Provides deterministic polynomial / claimed-sum generation for the
// E · (A · B − C) combine function using zk_dtype native field types.

#ifndef ZKX_TOOLS_STABLEHLO_RUNNER_SUMCHECK_TESTLIB_H_
#define ZKX_TOOLS_STABLEHLO_RUNNER_SUMCHECK_TESTLIB_H_

#include <cstdint>
#include <thread>

#include "Eigen/ThreadPool"

#include "zk_dtypes/include/field/babybear/babybear.h"
#include "zkx/literal.h"
#include "zkx/primitive_util.h"

namespace zkx {
namespace sumcheck_testlib {

// Converts a Montgomery-form raw value to its standard representation.
inline uint32_t FromMontgomery(uint32_t mont_val) {
  return zk_dtypes::BabybearMont::FromUnchecked(mont_val).MontReduce().value();
}

// Runs `fn(shard_begin, shard_end)` in parallel over [0, total) using an
// Eigen thread pool. Each shard gets a contiguous range.
template <typename Fn>
void ParallelFor(int64_t total, Fn&& fn) {
  int num_threads = std::thread::hardware_concurrency();
  if (num_threads <= 1 || total <= num_threads) {
    fn(0, total);
    return;
  }
  Eigen::ThreadPool pool(num_threads);
  Eigen::Barrier barrier(num_threads);
  int64_t block = (total + num_threads - 1) / num_threads;
  for (int t = 0; t < num_threads; ++t) {
    int64_t begin = t * block;
    int64_t end = std::min(begin + block, total);
    pool.Schedule([&fn, &barrier, begin, end]() {
      fn(begin, end);
      barrier.Notify();
    });
  }
  barrier.Wait();
}

// Fills a polynomial literal with deterministic values using the same formula
// as ICICLE's create_deterministic_polys: poly[k][i] = ((k*size+i+42) % P + 1)
// Assumes shape is [num_polys, poly_size].
inline void FillDeterministicPolys(Literal& literal) {
  const auto& shape = literal.shape();
  if (shape.rank() != 2) return;
  int64_t num_polys = shape.dimensions(0);
  int64_t poly_size = shape.dimensions(1);
  auto* data = static_cast<uint32_t*>(literal.untyped_data());
  primitive_util::ArrayTypeSwitch<void>(
      [&](auto primitive_type_constant) {
        using NativeT = primitive_util::NativeTypeOf<primitive_type_constant>;
        if constexpr (primitive_util::IsPrimeFieldType(
                          primitive_type_constant) &&
                      sizeof(NativeT) == sizeof(uint32_t)) {
          int64_t total = num_polys * poly_size;
          NativeT step(poly_size);
          ParallelFor(total, [&](int64_t begin, int64_t end) {
            for (int64_t idx = begin; idx < end; ++idx) {
              int64_t k = idx / poly_size;
              int64_t i = idx % poly_size;
              NativeT val = NativeT(43) + NativeT(i) + NativeT(k) * step;
              data[idx] = val.value();
            }
          });
        }
      },
      shape.element_type());
}

// Fills challenge vector with deterministic values: challenge[i] = i + 1.
inline void FillDeterministicChallenges(Literal& literal) {
  const auto& shape = literal.shape();
  if (shape.rank() != 1) return;
  auto* data = static_cast<uint32_t*>(literal.untyped_data());
  primitive_util::ArrayTypeSwitch<void>(
      [&](auto primitive_type_constant) {
        using NativeT = primitive_util::NativeTypeOf<primitive_type_constant>;
        if constexpr (primitive_util::IsPrimeFieldType(
                          primitive_type_constant) &&
                      sizeof(NativeT) == sizeof(uint32_t)) {
          int64_t n = shape.dimensions(0);
          ParallelFor(n, [&](int64_t begin, int64_t end) {
            NativeT val(begin + 1);
            for (int64_t i = begin; i < end; ++i) {
              data[i] = val.value();
              val += NativeT::One();
            }
          });
        }
      },
      shape.element_type());
}

// Fills a scalar literal (rank 0) with a deterministic value.
inline void FillDeterministicScalar(Literal& literal, uint32_t val) {
  const auto& shape = literal.shape();
  if (shape.rank() != 0) return;
  auto* data = static_cast<uint32_t*>(literal.untyped_data());
  primitive_util::ArrayTypeSwitch<void>(
      [&](auto primitive_type_constant) {
        using NativeT = primitive_util::NativeTypeOf<primitive_type_constant>;
        if constexpr (primitive_util::IsPrimeFieldType(
                          primitive_type_constant) &&
                      sizeof(NativeT) == sizeof(uint32_t)) {
          data[0] = NativeT(val).value();
        }
      },
      shape.element_type());
}

// Computes claimed_sum = Σ E[i] · (A[i] · B[i] − C[i]) from deterministic
// polys. poly[k][i] = k * size + i + 43 (in field arithmetic).
inline uint32_t ComputeDeterministicClaimedSum(int64_t poly_size) {
  using F = zk_dtypes::BabybearMont;
  F sum(0);
  F step(poly_size);
  F base(43);  // 0 + 42 + 1
  for (int64_t i = 0; i < poly_size; ++i) {
    // poly[k] = base + k * step.
    F a = base;
    F b = a + step;
    F c = b + step;
    F e = c + step;
    sum += e * (a * b - c);
    base += F::One();
  }
  return sum.MontReduce().value();
}

}  // namespace sumcheck_testlib
}  // namespace zkx

#endif  // ZKX_TOOLS_STABLEHLO_RUNNER_SUMCHECK_TESTLIB_H_
