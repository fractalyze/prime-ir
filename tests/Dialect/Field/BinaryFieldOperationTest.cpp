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

#include "prime_ir/Dialect/Field/IR/BinaryFieldOperation.h"

#include "gtest/gtest.h"
#include "prime_ir/Dialect/Field/IR/FieldDialect.h"
#include "prime_ir/Utils/ZkDtypes.h"
#include "zk_dtypes/include/field/binary_field.h"

namespace mlir::prime_ir::field {

template <typename F>
class BinaryFieldOperationTest : public testing::Test {
public:
  static void SetUpTestSuite() { context.loadDialect<FieldDialect>(); }

  void runBinaryOperationTest(
      std::function<F(const F &, const F &)> f_operation,
      std::function<BinaryFieldOperation(const BinaryFieldOperation &,
                                         const BinaryFieldOperation &)>
          b_operation,
      bool bMustBeNonZero = false) {
    auto a = F::Random();
    auto b = F::Random();
    if (bMustBeNonZero) {
      while (b.IsZero()) {
        b = F::Random();
      }
    }
    runBinaryOperationTest(f_operation, b_operation, a, b);
  }

  void runBinaryOperationTest(
      std::function<F(const F &, const F &)> f_operation,
      std::function<BinaryFieldOperation(const BinaryFieldOperation &,
                                         const BinaryFieldOperation &)>
          b_operation,
      const F &a, const F &b) {
    auto modA = BinaryFieldOperation::fromZkDtype(&context, a);
    auto modB = BinaryFieldOperation::fromZkDtype(&context, b);
    EXPECT_EQ(BinaryFieldOperation::fromZkDtype(&context, f_operation(a, b)),
              b_operation(modA, modB));
  }

  void runUnaryOperationTest(
      std::function<F(const F &)> f_operation,
      std::function<BinaryFieldOperation(const BinaryFieldOperation &)>
          b_operation,
      bool aMustBeNonZero = false) {
    auto a = F::Random();
    if (aMustBeNonZero) {
      while (a.IsZero()) {
        a = F::Random();
      }
    }
    runUnaryOperationTest(f_operation, b_operation, a);
  }

  void runUnaryOperationTest(
      std::function<F(const F &)> f_operation,
      std::function<BinaryFieldOperation(const BinaryFieldOperation &)>
          b_operation,
      const F &a) {
    auto bfA = BinaryFieldOperation::fromZkDtype(&this->context, a);
    EXPECT_EQ(BinaryFieldOperation::fromZkDtype(&this->context, f_operation(a)),
              b_operation(bfA));
  }

  static MLIRContext context;
};

template <typename T>
MLIRContext BinaryFieldOperationTest<T>::context;

using BinaryFieldTypes =
    testing::Types<zk_dtypes::BinaryFieldT0,  // GF(2)     - 1 bit
                   zk_dtypes::BinaryFieldT4,  // GF(2^16)  - 16 bits
                   zk_dtypes::BinaryFieldT7>; // GF(2^128) - 128 bits
TYPED_TEST_SUITE(BinaryFieldOperationTest, BinaryFieldTypes);

//===----------------------------------------------------------------------===//
// Binary Operations
//===----------------------------------------------------------------------===//

TYPED_TEST(BinaryFieldOperationTest, Add) {
  using BinaryFieldType = TypeParam;

  this->runBinaryOperationTest(
      [](const BinaryFieldType &a, const BinaryFieldType &b) { return a + b; },
      [](const BinaryFieldOperation &a, const BinaryFieldOperation &b) {
        return a + b;
      });
}

TYPED_TEST(BinaryFieldOperationTest, Sub) {
  using BinaryFieldType = TypeParam;

  this->runBinaryOperationTest(
      [](const BinaryFieldType &a, const BinaryFieldType &b) { return a - b; },
      [](const BinaryFieldOperation &a, const BinaryFieldOperation &b) {
        return a - b;
      });
}

TYPED_TEST(BinaryFieldOperationTest, Mul) {
  using BinaryFieldType = TypeParam;

  this->runBinaryOperationTest(
      [](const BinaryFieldType &a, const BinaryFieldType &b) { return a * b; },
      [](const BinaryFieldOperation &a, const BinaryFieldOperation &b) {
        return a * b;
      });
}

TYPED_TEST(BinaryFieldOperationTest, Div) {
  using BinaryFieldType = TypeParam;

  this->runBinaryOperationTest(
      [](const BinaryFieldType &a, const BinaryFieldType &b) { return a / b; },
      [](const BinaryFieldOperation &a, const BinaryFieldOperation &b) {
        return a * b.inverse();
      },
      /*bMustBeNonZero=*/true);
}

TYPED_TEST(BinaryFieldOperationTest, Cmp) {
  using BinaryFieldType = TypeParam;

  auto a = BinaryFieldType::Random();
  auto b = BinaryFieldType::Random();
  auto bfA = BinaryFieldOperation::fromZkDtype(&this->context, a);
  auto bfB = BinaryFieldOperation::fromZkDtype(&this->context, b);

  EXPECT_EQ(a < b, bfA < bfB);
  EXPECT_EQ(a <= b, bfA <= bfB);
  EXPECT_EQ(a > b, bfA > bfB);
  EXPECT_EQ(a >= b, bfA >= bfB);
  EXPECT_EQ(a == b, bfA == bfB);
  EXPECT_EQ(a != b, bfA != bfB);
}

//===----------------------------------------------------------------------===//
// Unary Operations
//===----------------------------------------------------------------------===//

TYPED_TEST(BinaryFieldOperationTest, Negate) {
  using BinaryFieldType = TypeParam;

  // In characteristic 2, -a = a
  this->runUnaryOperationTest([](const BinaryFieldType &a) { return -a; },
                              [](const BinaryFieldOperation &a) { return a; });
}

TYPED_TEST(BinaryFieldOperationTest, Double) {
  using BinaryFieldType = TypeParam;

  // In characteristic 2, a + a = 0
  this->runUnaryOperationTest(
      [](const BinaryFieldType &a) { return a.Double(); },
      [](const BinaryFieldOperation &a) { return a.dbl(); });
}

TYPED_TEST(BinaryFieldOperationTest, Square) {
  using BinaryFieldType = TypeParam;

  this->runUnaryOperationTest(
      [](const BinaryFieldType &a) { return a.Square(); },
      [](const BinaryFieldOperation &a) { return a.square(); });
}

TYPED_TEST(BinaryFieldOperationTest, Power) {
  using BinaryFieldType = TypeParam;

  uint32_t exponents[] = {
      0,
      1,
      static_cast<uint32_t>(
          static_cast<uint64_t>(BinaryFieldType::Random().value())),
  };

  for (uint32_t exponent : exponents) {
    // NOTE: Technically, power operation is not unary operation. However, we
    // can still test the power operations using runUnaryOperationTest.
    this->runUnaryOperationTest(
        [exponent](const BinaryFieldType &a) { return a.Pow(exponent); },
        [exponent](const BinaryFieldOperation &a) {
          return a.power(convertToAPInt(exponent, BinaryFieldType::kBitWidth));
        });
  }
}

TYPED_TEST(BinaryFieldOperationTest, Inverse) {
  using BinaryFieldType = TypeParam;

  this->runUnaryOperationTest(
      [](const BinaryFieldType &a) { return a.Inverse(); },
      [](const BinaryFieldOperation &a) { return a.inverse(); },
      /*aMustBeNonZero=*/true);
}

TYPED_TEST(BinaryFieldOperationTest, ZeroAndOne) {
  using BinaryFieldType = TypeParam;

  auto zero = BinaryFieldType::Zero();
  auto bfZero = BinaryFieldOperation::fromZkDtype(&this->context, zero);
  EXPECT_TRUE(bfZero.isZero());
  EXPECT_FALSE(bfZero.isOne());
  EXPECT_EQ(bfZero, bfZero.getZero());

  auto one = BinaryFieldType::One();
  auto bfOne = BinaryFieldOperation::fromZkDtype(&this->context, one);
  EXPECT_FALSE(bfOne.isZero());
  EXPECT_TRUE(bfOne.isOne());
  EXPECT_EQ(bfOne, bfOne.getOne());

  if constexpr (std::is_same_v<BinaryFieldType, zk_dtypes::BinaryFieldT0>) {
    GTEST_SKIP() << "Skip this test because this field only has 0 or 1";
  }
  auto rnd = BinaryFieldType::Random();
  while (rnd.IsZero() || rnd.IsOne()) {
    rnd = BinaryFieldType::Random();
  }
  auto bfRnd = BinaryFieldOperation::fromZkDtype(&this->context, rnd);
  EXPECT_FALSE(bfRnd.isZero());
  EXPECT_FALSE(bfRnd.isOne());
}

} // namespace mlir::prime_ir::field
