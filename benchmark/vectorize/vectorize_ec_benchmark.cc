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
#include "zk_dtypes/include/elliptic_curve/bn/bn254/fr.h"
#include "zk_dtypes/include/elliptic_curve/bn/bn254/g1.h"

namespace mlir::prime_ir::benchmark {
namespace {

using AffinePoint = zk_dtypes::bn254::G1AffinePoint;
using JacobianPoint = zk_dtypes::bn254::G1JacobianPoint;
using ScalarField = zk_dtypes::bn254::Fr;

constexpr int64_t kECBufferSize = 16384;       // 16K points for double/add
constexpr int64_t kScalarMulBufferSize = 1024; // 1K points for scalar_mul

// ==============================================================================
// EC function declarations (scalar versions)
// ==============================================================================

extern "C" void
_mlir_ciface_ec_double_buffer(StridedMemRefType<JacobianPoint, 1> *points);

extern "C" void
_mlir_ciface_ec_add_buffer(StridedMemRefType<JacobianPoint, 1> *jac,
                           StridedMemRefType<AffinePoint, 1> *aff,
                           StridedMemRefType<JacobianPoint, 1> *out);

extern "C" void
_mlir_ciface_ec_scalar_mul_buffer(StridedMemRefType<ScalarField, 1> *scalars,
                                  StridedMemRefType<AffinePoint, 1> *points,
                                  StridedMemRefType<JacobianPoint, 1> *out);

// ==============================================================================
// EC function declarations (vectorized versions)
// ==============================================================================

extern "C" void
_mlir_ciface_vec_ec_double_buffer(StridedMemRefType<JacobianPoint, 1> *points);

extern "C" void
_mlir_ciface_vec_ec_add_buffer(StridedMemRefType<JacobianPoint, 1> *jac,
                               StridedMemRefType<AffinePoint, 1> *aff,
                               StridedMemRefType<JacobianPoint, 1> *out);

extern "C" void _mlir_ciface_vec_ec_scalar_mul_buffer(
    StridedMemRefType<ScalarField, 1> *scalars,
    StridedMemRefType<AffinePoint, 1> *points,
    StridedMemRefType<JacobianPoint, 1> *out);

// ==============================================================================
// Initialization helpers
// ==============================================================================

void fillWithGenerator(AffinePoint &elem, ArrayRef<int64_t> coords) {
  elem = AffinePoint::Generator();
}

void fillWithJacobianGenerator(JacobianPoint &elem, ArrayRef<int64_t> coords) {
  // Convert generator to Jacobian form (Z = 1)
  auto gen = AffinePoint::Generator();
  elem = gen.ToJacobian();
}

void fillWithRandomScalar(ScalarField &elem, ArrayRef<int64_t> coords) {
  elem = ScalarField::Random();
}

// ==============================================================================
// EC Double Benchmarks
// ==============================================================================

void BM_ec_double_scalar(::benchmark::State &state) {
  OwningMemRef<JacobianPoint, 1> points({kECBufferSize}, {},
                                        fillWithJacobianGenerator);
  for (auto _ : state) {
    _mlir_ciface_ec_double_buffer(&*points);
  }
}

void BM_ec_double_vectorized(::benchmark::State &state) {
  OwningMemRef<JacobianPoint, 1> points({kECBufferSize}, {},
                                        fillWithJacobianGenerator);
  for (auto _ : state) {
    _mlir_ciface_vec_ec_double_buffer(&*points);
  }
}

// ==============================================================================
// EC Add Benchmarks
// ==============================================================================

void BM_ec_add_scalar(::benchmark::State &state) {
  OwningMemRef<JacobianPoint, 1> jac({kECBufferSize}, {},
                                     fillWithJacobianGenerator);
  OwningMemRef<AffinePoint, 1> aff({kECBufferSize}, {}, fillWithGenerator);
  OwningMemRef<JacobianPoint, 1> out({kECBufferSize}, {},
                                     fillWithJacobianGenerator);
  for (auto _ : state) {
    _mlir_ciface_ec_add_buffer(&*jac, &*aff, &*out);
  }
}

void BM_ec_add_vectorized(::benchmark::State &state) {
  OwningMemRef<JacobianPoint, 1> jac({kECBufferSize}, {},
                                     fillWithJacobianGenerator);
  OwningMemRef<AffinePoint, 1> aff({kECBufferSize}, {}, fillWithGenerator);
  OwningMemRef<JacobianPoint, 1> out({kECBufferSize}, {},
                                     fillWithJacobianGenerator);
  for (auto _ : state) {
    _mlir_ciface_vec_ec_add_buffer(&*jac, &*aff, &*out);
  }
}

// ==============================================================================
// EC Scalar Multiplication Benchmarks
// ==============================================================================

void BM_ec_scalar_mul_scalar(::benchmark::State &state) {
  OwningMemRef<ScalarField, 1> scalars({kScalarMulBufferSize}, {},
                                       fillWithRandomScalar);
  OwningMemRef<AffinePoint, 1> points({kScalarMulBufferSize}, {},
                                      fillWithGenerator);
  OwningMemRef<JacobianPoint, 1> out({kScalarMulBufferSize}, {},
                                     fillWithJacobianGenerator);
  for (auto _ : state) {
    _mlir_ciface_ec_scalar_mul_buffer(&*scalars, &*points, &*out);
  }
}

void BM_ec_scalar_mul_vectorized(::benchmark::State &state) {
  OwningMemRef<ScalarField, 1> scalars({kScalarMulBufferSize}, {},
                                       fillWithRandomScalar);
  OwningMemRef<AffinePoint, 1> points({kScalarMulBufferSize}, {},
                                      fillWithGenerator);
  OwningMemRef<JacobianPoint, 1> out({kScalarMulBufferSize}, {},
                                     fillWithJacobianGenerator);
  for (auto _ : state) {
    _mlir_ciface_vec_ec_scalar_mul_buffer(&*scalars, &*points, &*out);
  }
}

BENCHMARK(BM_ec_double_scalar)->Unit(::benchmark::kMillisecond);
BENCHMARK(BM_ec_double_vectorized)->Unit(::benchmark::kMillisecond);
BENCHMARK(BM_ec_add_scalar)->Unit(::benchmark::kMillisecond);
BENCHMARK(BM_ec_add_vectorized)->Unit(::benchmark::kMillisecond);
BENCHMARK(BM_ec_scalar_mul_scalar)->Unit(::benchmark::kMillisecond);
BENCHMARK(BM_ec_scalar_mul_vectorized)->Unit(::benchmark::kMillisecond);

} // namespace
} // namespace mlir::prime_ir::benchmark

// 2026-01-27T15:44:10+00:00
// Run on (32 X 624 MHz CPU s)
// CPU Caches:
//   L1 Data 48 KiB (x16)
//   L1 Instruction 32 KiB (x16)
//   L2 Unified 1024 KiB (x16)
//   L3 Unified 98304 KiB (x2)
// Load Average: 5.36, 14.59, 14.25
// ----------------------------------------------------------------------
// Benchmark                            Time             CPU   Iterations
// ----------------------------------------------------------------------
// BM_ec_double_scalar                207 ms          207 ms            3
// BM_ec_double_vectorized            207 ms          207 ms            3
// BM_ec_add_scalar                   420 ms          420 ms            2
// BM_ec_add_vectorized               419 ms          419 ms            2
// BM_ec_scalar_mul_scalar            712 ms          712 ms            1
// BM_ec_scalar_mul_vectorized        715 ms          715 ms            1
