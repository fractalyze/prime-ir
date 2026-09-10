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

// Asserts that prime_ir resolves, compiles and links as a non-root Bazel
// module. The dialect exercise is deliberately trivial: prime_ir's own suite
// covers behaviour, and what is under test here is dependency resolution.
//
// Loading the dialects reaches what a consumer cannot declare for itself: MLIR,
// which arrives from the patched @llvm-project that prime_ir's module extension
// fetches, and zk_dtypes, which the field and curve types are built on.

#include "gtest/gtest.h"
#include "mlir/IR/MLIRContext.h"
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveDialect.h"
#include "prime_ir/Dialect/Field/IR/FieldDialect.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithDialect.h"

namespace {

namespace ec = mlir::prime_ir::elliptic_curve;
namespace field = mlir::prime_ir::field;
namespace mod_arith = mlir::prime_ir::mod_arith;

TEST(ConsumePrimeIrTest, DialectsLoad) {
  mlir::MLIRContext context;

  auto *fieldDialect = context.getOrLoadDialect<field::FieldDialect>();
  auto *ecDialect = context.getOrLoadDialect<ec::EllipticCurveDialect>();
  auto *modArithDialect =
      context.getOrLoadDialect<mod_arith::ModArithDialect>();

  EXPECT_EQ(fieldDialect->getNamespace(), "field");
  EXPECT_EQ(ecDialect->getNamespace(), "elliptic_curve");
  EXPECT_EQ(modArithDialect->getNamespace(), "mod_arith");
}

} // namespace
