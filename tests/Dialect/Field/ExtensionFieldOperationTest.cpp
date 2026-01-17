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
#include "zk_dtypes/include/elliptic_curve/bn/bn254/fq2.h"
#include "zk_dtypes/include/field/babybear/babybear4.h"
#include "zk_dtypes/include/field/goldilocks/goldilocks3.h"

namespace mlir::prime_ir::field {

template <typename ExtF>
class ExtensionFieldOperationTest : public testing::Test {
public:
  static constexpr uint32_t N = ExtF::Config::kDegreeOverBaseField;
  using F = typename ExtF::Config::BaseField;

  static void SetUpTestSuite() { context.loadDialect<FieldDialect>(); }

  void runBinaryOperationTest(
      std::function<ExtF(const ExtF &, const ExtF &)> f_operation,
      std::function<
          ExtensionFieldOperation<N>(const ExtensionFieldOperation<N> &,
                                     const ExtensionFieldOperation<N> &)>
          ef_operation,
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
      std::function<
          ExtensionFieldOperation<N>(const ExtensionFieldOperation<N> &,
                                     const ExtensionFieldOperation<N> &)>
          ef_operation,
      const ExtF &a, const ExtF &b) {
    auto efA = ExtensionFieldOperation<N>::fromZkDtype(&context, a);
    auto efB = ExtensionFieldOperation<N>::fromZkDtype(&context, b);
    EXPECT_EQ(
        ExtensionFieldOperation<N>::fromZkDtype(&context, f_operation(a, b)),
        ef_operation(efA, efB));
  }

  void runUnaryOperationTest(std::function<ExtF(const ExtF &)> f_operation,
                             std::function<ExtensionFieldOperation<N>(
                                 const ExtensionFieldOperation<N> &)>
                                 ef_operation,
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
                             std::function<ExtensionFieldOperation<N>(
                                 const ExtensionFieldOperation<N> &)>
                                 ef_operation,
                             const ExtF &a) {
    auto efA = ExtensionFieldOperation<N>::fromZkDtype(&context, a);
    EXPECT_EQ(ExtensionFieldOperation<N>::fromZkDtype(&context, f_operation(a)),
              ef_operation(efA));
  }

  static MLIRContext context;
};

template <typename F>
MLIRContext ExtensionFieldOperationTest<F>::context;

using ExtensionFieldTypes = testing::Types<
    // modulus bits = 2³¹
    // modulus.getBitWidth() == 32
    // modulus.getActiveBits() == 31
    zk_dtypes::Babybear4, zk_dtypes::Babybear4Std,
    // modulus bits = 2⁶⁴
    // modulus.getBitWidth() == 64
    // modulus.getActiveBits() == 64
    zk_dtypes::Goldilocks3, zk_dtypes::Goldilocks3Std,
    // modulus bits = 2²⁵⁴
    // modulus.getBitWidth() == 254
    // modulus.getActiveBits() == 254
    zk_dtypes::bn254::Fq2, zk_dtypes::bn254::Fq2Std>;
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
  static constexpr uint32_t N =
      ExtensionFieldType::Config::kDegreeOverBaseField;

  auto zero = ExtensionFieldType::Zero();
  auto efZero = ExtensionFieldOperation<N>::fromZkDtype(&this->context, zero);
  EXPECT_TRUE(efZero.isZero());
  EXPECT_FALSE(efZero.isOne());
  EXPECT_EQ(efZero, efZero.getZero());

  auto one = ExtensionFieldType::One();
  auto efOne = ExtensionFieldOperation<N>::fromZkDtype(&this->context, one);
  EXPECT_FALSE(efOne.isZero());
  EXPECT_TRUE(efOne.isOne());
  EXPECT_EQ(efOne, efOne.getOne());

  auto rnd = ExtensionFieldType::Random();
  while (rnd.IsZero() || rnd.IsOne()) {
    rnd = ExtensionFieldType::Random();
  }
  auto efRnd = ExtensionFieldOperation<N>::fromZkDtype(&this->context, rnd);
  EXPECT_FALSE(efRnd.isZero());
  EXPECT_FALSE(efRnd.isOne());
}

} // namespace mlir::prime_ir::field
