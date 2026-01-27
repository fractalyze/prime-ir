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

#include "prime_ir/Dialect/Field/IR//ExtensionFieldOperation.h"

#include "gtest/gtest.h"
#include "prime_ir/Dialect/Field/IR/FieldDialect.h"
#include "prime_ir/Utils/ZkDtypes.h"
#include "zk_dtypes/include/field/babybear/babybear4.h"
#include "zk_dtypes/include/field/goldilocks/goldilocks3.h"
#include "zk_dtypes/include/field/mersenne31/mersenne312.h"
#include "zk_dtypes/include/field/mersenne31/mersenne314.h"

namespace mlir::prime_ir::field {

template <typename ExtF>
class ExtensionFieldOperationTest : public testing::Test {
public:
  // Use the type trait to get the correct ExtensionFieldOperation type
  // (handles both non-tower and tower extensions)
  using EfOp = typename detail::ZkDtypeToExtensionFieldOp<ExtF>::type;

  static void SetUpTestSuite() { context.loadDialect<FieldDialect>(); }

  void runBinaryOperationTest(
      std::function<ExtF(const ExtF &, const ExtF &)> f_operation,
      std::function<EfOp(const EfOp &, const EfOp &)> ef_operation,
      bool bMustBeNonZero = false) {
    auto a = ExtF::Random();
    auto b = ExtF::Random();
    if (bMustBeNonZero) {
      while (b.IsZero()) {
        b = ExtF::Random();
      }
    }
    runBinaryOperationTest(f_operation, ef_operation, a, b);
  }

  void runBinaryOperationTest(
      std::function<ExtF(const ExtF &, const ExtF &)> f_operation,
      std::function<EfOp(const EfOp &, const EfOp &)> ef_operation,
      const ExtF &a, const ExtF &b) {
    auto efA = EfOp::fromZkDtype(&context, a);
    auto efB = EfOp::fromZkDtype(&context, b);
    EXPECT_EQ(EfOp::fromZkDtype(&context, f_operation(a, b)),
              ef_operation(efA, efB));
  }

  void runUnaryOperationTest(std::function<ExtF(const ExtF &)> f_operation,
                             std::function<EfOp(const EfOp &)> ef_operation,
                             bool aMustBeNonZero = false) {
    auto a = ExtF::Random();
    if (aMustBeNonZero) {
      while (a.IsZero()) {
        a = ExtF::Random();
      }
    }
    runUnaryOperationTest(f_operation, ef_operation, a);
  }

  void runUnaryOperationTest(std::function<ExtF(const ExtF &)> f_operation,
                             std::function<EfOp(const EfOp &)> ef_operation,
                             const ExtF &a) {
    auto efA = EfOp::fromZkDtype(&context, a);
    EXPECT_EQ(EfOp::fromZkDtype(&context, f_operation(a)), ef_operation(efA));
  }

  static MLIRContext context;
};

template <typename F>
MLIRContext ExtensionFieldOperationTest<F>::context;

using ExtensionFieldTypes = testing::Types<
    // degree = 2
    zk_dtypes::Mersenne312,
    // degree = 3
    zk_dtypes::Goldilocks3,
    // degree = 4
    zk_dtypes::Babybear4,
    // degree = 2 x 2
    zk_dtypes::Mersenne314>;
TYPED_TEST_SUITE(ExtensionFieldOperationTest, ExtensionFieldTypes);

//===----------------------------------------------------------------------===//
// Binary Operations
//===----------------------------------------------------------------------===//

TYPED_TEST(ExtensionFieldOperationTest, Add) {
  using ExtensionFieldType = TypeParam;

  this->runBinaryOperationTest(
      [](const ExtensionFieldType &a, const ExtensionFieldType &b) {
        return a + b;
      },
      [](const auto &a, const auto &b) { return a + b; });
}

TYPED_TEST(ExtensionFieldOperationTest, Sub) {
  using ExtensionFieldType = TypeParam;

  this->runBinaryOperationTest(
      [](const ExtensionFieldType &a, const ExtensionFieldType &b) {
        return a - b;
      },
      [](const auto &a, const auto &b) { return a - b; });
}

TYPED_TEST(ExtensionFieldOperationTest, Mul) {
  using ExtensionFieldType = TypeParam;

  this->runBinaryOperationTest(
      [](const ExtensionFieldType &a, const ExtensionFieldType &b) {
        return a * b;
      },
      [](const auto &a, const auto &b) { return a * b; });
}

TYPED_TEST(ExtensionFieldOperationTest, Div) {
  using ExtensionFieldType = TypeParam;

  this->runBinaryOperationTest(
      [](const ExtensionFieldType &a, const ExtensionFieldType &b) {
        return a / b;
      },
      [](const auto &a, const auto &b) { return a / b; },
      /*bMustBeNonZero=*/true);
}

TYPED_TEST(ExtensionFieldOperationTest, Negate) {
  using ExtensionFieldType = TypeParam;

  this->runUnaryOperationTest([](const ExtensionFieldType &a) { return -a; },
                              [](const auto &a) { return -a; });
}

TYPED_TEST(ExtensionFieldOperationTest, Double) {
  using ExtensionFieldType = TypeParam;

  this->runUnaryOperationTest(
      [](const ExtensionFieldType &a) { return a.Double(); },
      [](const auto &a) { return a.dbl(); });
}

TYPED_TEST(ExtensionFieldOperationTest, Square) {
  using ExtensionFieldType = TypeParam;

  this->runUnaryOperationTest(
      [](const ExtensionFieldType &a) { return a.Square(); },
      [](const auto &a) { return a.square(); });
}

TYPED_TEST(ExtensionFieldOperationTest, Power) {
  using ExtensionFieldType = TypeParam;

  this->runUnaryOperationTest(
      [](const ExtensionFieldType &a) { return a.Pow(3); },
      [](const auto &a) { return a.power(convertToAPInt(3)); });
}

TYPED_TEST(ExtensionFieldOperationTest, Inverse) {
  using ExtensionFieldType = TypeParam;

  this->runUnaryOperationTest(
      [](const ExtensionFieldType &a) { return a.Inverse(); },
      [](const auto &a) { return a.inverse(); },
      /*aMustBeNonZero=*/true);
}

TYPED_TEST(ExtensionFieldOperationTest, ZeroAndOne) {
  using ExtensionFieldType = TypeParam;
  using EfOp = typename detail::ZkDtypeToExtensionFieldOp<ExtensionFieldType>::type;

  auto zero = ExtensionFieldType::Zero();
  auto efZero = EfOp::fromZkDtype(&this->context, zero);
  EXPECT_TRUE(efZero.isZero());
  EXPECT_FALSE(efZero.isOne());
  EXPECT_EQ(efZero, efZero.getZero());

  auto one = ExtensionFieldType::One();
  auto efOne = EfOp::fromZkDtype(&this->context, one);
  EXPECT_FALSE(efOne.isZero());
  EXPECT_TRUE(efOne.isOne());
  EXPECT_EQ(efOne, efOne.getOne());

  auto rnd = ExtensionFieldType::Random();
  while (rnd.IsZero() || rnd.IsOne()) {
    rnd = ExtensionFieldType::Random();
  }
  auto efRnd = EfOp::fromZkDtype(&this->context, rnd);
  EXPECT_FALSE(efRnd.isZero());
  EXPECT_FALSE(efRnd.isOne());
}

} // namespace mlir::prime_ir::field
