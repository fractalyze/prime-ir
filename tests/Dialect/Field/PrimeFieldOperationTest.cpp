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

#include "prime_ir/Dialect/Field/IR/PrimeFieldOperation.h"

#include "gtest/gtest.h"
#include "llvm/ADT/bit.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "prime_ir/Dialect/Field/IR/FieldDialect.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"
#include "prime_ir/Utils/ZkDtypes.h"
#include "zk_dtypes/include/elliptic_curve/bn/bn254/fr.h"
#include "zk_dtypes/include/field/babybear/babybear.h"
#include "zk_dtypes/include/field/goldilocks/goldilocks.h"

namespace mlir::prime_ir::field {

template <typename F>
class PrimeFieldOperationTest : public testing::Test {
public:
  static void SetUpTestSuite() {
    context.loadDialect<FieldDialect>();
    auto modulusBits = llvm::bit_ceil(F::Config::kModulusBits);
    IntegerAttr modulus =
        IntegerAttr::get(IntegerType::get(&context, modulusBits),
                         convertToAPInt(F::Config::kModulus, modulusBits));
    pfType = PrimeFieldType::get(&context, modulus, F::kUseMontgomery);
  }

  void runBinaryOperationTest(
      std::function<F(const F &, const F &)> f_operation,
      std::function<PrimeFieldOperation(const PrimeFieldOperation &,
                                        const PrimeFieldOperation &)>
          p_operation,
      bool bMustBeNonZero = false) {
    auto a = F::Random();
    auto b = F::Random();
    if (bMustBeNonZero) {
      while (b.IsZero()) {
        b = F::Random();
      }
    }
    runBinaryOperationTest(f_operation, p_operation, a, b);
  }

  void runBinaryOperationTest(
      std::function<F(const F &, const F &)> f_operation,
      std::function<PrimeFieldOperation(const PrimeFieldOperation &,
                                        const PrimeFieldOperation &)>
          p_operation,
      const F &a, const F &b) {
    auto pfA =
        PrimeFieldOperation::fromUnchecked(convertToAPInt(a.value()), pfType);
    auto pfB =
        PrimeFieldOperation::fromUnchecked(convertToAPInt(b.value()), pfType);
    auto c = f_operation(a, b);
    EXPECT_EQ(convertToAPInt(c.value()),
              static_cast<APInt>(p_operation(pfA, pfB)));
  }

  void runUnaryOperationTest(
      std::function<F(const F &)> f_operation,
      std::function<PrimeFieldOperation(const PrimeFieldOperation &)>
          p_operation,
      bool aMustBeNonZero = false) {
    auto a = F::Random();
    if (aMustBeNonZero) {
      while (a.IsZero()) {
        a = F::Random();
      }
    }
    runUnaryOperationTest(f_operation, p_operation, a);
  }

  void runUnaryOperationTest(
      std::function<F(const F &)> f_operation,
      std::function<PrimeFieldOperation(const PrimeFieldOperation &)>
          p_operation,
      const F &a) {
    auto pfA =
        PrimeFieldOperation::fromUnchecked(convertToAPInt(a.value()), pfType);
    auto c = f_operation(a);
    EXPECT_EQ(convertToAPInt(c.value()), static_cast<APInt>(p_operation(pfA)));
  }

  static MLIRContext context;
  static PrimeFieldType pfType;
};

template <typename F>
MLIRContext PrimeFieldOperationTest<F>::context;

template <typename F>
PrimeFieldType PrimeFieldOperationTest<F>::pfType;

using PrimeFieldTypes = testing::Types<
    // modulus bits = 2³¹
    // modulus.getBitWidth() == 32
    // modulus.getActiveBits() == 31
    zk_dtypes::Babybear, zk_dtypes::BabybearStd,
    // modulus bits = 2⁶⁴
    // modulus.getBitWidth() == 64
    // modulus.getActiveBits() == 64
    zk_dtypes::Goldilocks, zk_dtypes::GoldilocksStd,
    // modulus bits = 2²⁵⁴
    // modulus.getBitWidth() == 254
    // modulus.getActiveBits() == 254
    zk_dtypes::bn254::Fr, zk_dtypes::bn254::FrStd>;
TYPED_TEST_SUITE(PrimeFieldOperationTest, PrimeFieldTypes);

//===----------------------------------------------------------------------===//
// Binary Operations
//===----------------------------------------------------------------------===//

TYPED_TEST(PrimeFieldOperationTest, Add) {
  using PrimeFieldType = TypeParam;

  this->runBinaryOperationTest(
      [](const PrimeFieldType &a, const PrimeFieldType &b) { return a + b; },
      [](const PrimeFieldOperation &a, const PrimeFieldOperation &b) {
        return a + b;
      });
}

TYPED_TEST(PrimeFieldOperationTest, AddOverflow) {
  using PrimeFieldType = TypeParam;

  this->runBinaryOperationTest(
      [](const PrimeFieldType &a, const PrimeFieldType &b) { return a + b; },
      [](const PrimeFieldOperation &a, const PrimeFieldOperation &b) {
        return a + b;
      },
      PrimeFieldType::Max(), PrimeFieldType::Random());
}

TYPED_TEST(PrimeFieldOperationTest, Sub) {
  using PrimeFieldType = TypeParam;

  this->runBinaryOperationTest(
      [](const PrimeFieldType &a, const PrimeFieldType &b) { return a - b; },
      [](const PrimeFieldOperation &a, const PrimeFieldOperation &b) {
        return a - b;
      });
}

TYPED_TEST(PrimeFieldOperationTest, SubOverflow) {
  using PrimeFieldType = TypeParam;

  this->runBinaryOperationTest(
      [](const PrimeFieldType &a, const PrimeFieldType &b) { return a - b; },
      [](const PrimeFieldOperation &a, const PrimeFieldOperation &b) {
        return a - b;
      },
      PrimeFieldType::Zero(), PrimeFieldType::Random());
}

TYPED_TEST(PrimeFieldOperationTest, Mul) {
  using PrimeFieldType = TypeParam;

  this->runBinaryOperationTest(
      [](const PrimeFieldType &a, const PrimeFieldType &b) { return a * b; },
      [](const PrimeFieldOperation &a, const PrimeFieldOperation &b) {
        return a * b;
      });
}

TYPED_TEST(PrimeFieldOperationTest, Div) {
  using PrimeFieldType = TypeParam;

  this->runBinaryOperationTest(
      [](const PrimeFieldType &a, const PrimeFieldType &b) { return a / b; },
      [](const PrimeFieldOperation &a, const PrimeFieldOperation &b) {
        return a / b;
      },
      /*bMustBeNonZero=*/true);
}

TYPED_TEST(PrimeFieldOperationTest, Cmp) {
  using PrimeFieldType = TypeParam;

  auto a = PrimeFieldType::Random();
  auto b = PrimeFieldType::Random();
  auto pfA = PrimeFieldOperation::fromUnchecked(convertToAPInt(a.value()),
                                                this->pfType);
  auto pfB = PrimeFieldOperation::fromUnchecked(convertToAPInt(b.value()),
                                                this->pfType);

  EXPECT_EQ(a < b, pfA < pfB);
  EXPECT_EQ(a <= b, pfA <= pfB);
  EXPECT_EQ(a > b, pfA > pfB);
  EXPECT_EQ(a >= b, pfA >= pfB);
  EXPECT_EQ(a == b, pfA == pfB);
  EXPECT_EQ(a != b, pfA != pfB);
}

//===----------------------------------------------------------------------===//
// Unary Operations
//===----------------------------------------------------------------------===//

TYPED_TEST(PrimeFieldOperationTest, Negate) {
  using PrimeFieldType = TypeParam;

  this->runUnaryOperationTest([](const PrimeFieldType &a) { return -a; },
                              [](const PrimeFieldOperation &a) { return -a; });
}

TYPED_TEST(PrimeFieldOperationTest, NegateZero) {
  using PrimeFieldType = TypeParam;

  this->runUnaryOperationTest([](const PrimeFieldType &a) { return -a; },
                              [](const PrimeFieldOperation &a) { return -a; },
                              PrimeFieldType::Zero());
}

TYPED_TEST(PrimeFieldOperationTest, Double) {
  using PrimeFieldType = TypeParam;

  this->runUnaryOperationTest(
      [](const PrimeFieldType &a) { return a.Double(); },
      [](const PrimeFieldOperation &a) { return a.dbl(); });
}

TYPED_TEST(PrimeFieldOperationTest, Square) {
  using PrimeFieldType = TypeParam;

  this->runUnaryOperationTest(
      [](const PrimeFieldType &a) { return a.Square(); },
      [](const PrimeFieldOperation &a) { return a.square(); });
}

TYPED_TEST(PrimeFieldOperationTest, Power) {
  using PrimeFieldType = TypeParam;

  uint32_t exponents[] = {
      0,
      1,
      static_cast<uint32_t>(
          static_cast<uint64_t>(PrimeFieldType::Random().value())),
  };

  for (uint32_t exponent : exponents) {
    // NOTE: Technically, power operation is not unary operation. However, we
    // can still test the power operations using runUnaryOperationTest.
    this->runUnaryOperationTest(
        [exponent](const PrimeFieldType &a) { return a.Pow(exponent); },
        [exponent](const PrimeFieldOperation &a) {
          auto modulusBits =
              llvm::bit_ceil(PrimeFieldType::Config::kModulusBits);
          return a.power(convertToAPInt(exponent, modulusBits));
        });
  }
}

TYPED_TEST(PrimeFieldOperationTest, Inverse) {
  using PrimeFieldType = TypeParam;

  this->runUnaryOperationTest(
      [](const PrimeFieldType &a) { return a.Inverse(); },
      [](const PrimeFieldOperation &a) { return a.inverse(); },
      /*aMustBeNonZero=*/true);
}

TYPED_TEST(PrimeFieldOperationTest, FromMont) {
  using PrimeFieldType = TypeParam;

  if constexpr (!PrimeFieldType::kUseMontgomery) {
    GTEST_SKIP() << "Non-Montgomery field is not supported";
  } else {
    this->runUnaryOperationTest(
        [](const PrimeFieldType &a) {
          return PrimeFieldType::FromUnchecked(a.MontReduce().value());
        },
        [](const PrimeFieldOperation &a) { return a.fromMont(); });
  }
}

// TODO(chokobole): Re-enable this test once a mechanism for obtaining a
// MontType from a StdType is implemented.
//
// Note: This conversion is primarily intended for testing internal
// representation consistency and is not required for production workflows.
// Disabling this test for now as it lacks the necessary type-mapping helpers.
TYPED_TEST(PrimeFieldOperationTest, DISABLED_ToMont) {
  using PrimeFieldType = TypeParam;

  if constexpr (PrimeFieldType::kUseMontgomery) {
    GTEST_SKIP() << "Montgomery field is not supported";
  } else {
    this->runUnaryOperationTest(
        [](const PrimeFieldType &a) { return PrimeFieldType(a.value()); },
        [](const PrimeFieldOperation &a) { return a.toMont(); });
  }
}

TYPED_TEST(PrimeFieldOperationTest, ZeroAndOne) {
  using PrimeFieldType = TypeParam;

  auto zero = PrimeFieldType::Zero();
  auto pfZero = PrimeFieldOperation::fromUnchecked(convertToAPInt(zero.value()),
                                                   this->pfType);
  EXPECT_TRUE(pfZero.isZero());
  EXPECT_FALSE(pfZero.isOne());
  EXPECT_EQ(pfZero, pfZero.getZero());

  auto one = PrimeFieldType::One();
  auto pfOne = PrimeFieldOperation::fromUnchecked(convertToAPInt(one.value()),
                                                  this->pfType);
  EXPECT_FALSE(pfOne.isZero());
  EXPECT_TRUE(pfOne.isOne());
  EXPECT_EQ(pfOne, pfOne.getOne());

  auto rnd = PrimeFieldType::Random();
  while (rnd.IsZero() || rnd.IsOne()) {
    rnd = PrimeFieldType::Random();
  }
  auto pfRnd = PrimeFieldOperation::fromUnchecked(convertToAPInt(rnd.value()),
                                                  this->pfType);
  EXPECT_FALSE(pfRnd.isZero());
  EXPECT_FALSE(pfRnd.isOne());
}

} // namespace mlir::prime_ir::field
