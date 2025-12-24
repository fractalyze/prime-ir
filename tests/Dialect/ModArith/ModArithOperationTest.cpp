/* Copyright 2025 The ZKIR Authors.

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

#include "zkir/Dialect/ModArith/IR/ModArithOperation.h"

#include "gtest/gtest.h"
#include "llvm/ADT/bit.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "zk_dtypes/include/field/babybear/babybear.h"
#include "zk_dtypes/include/field/goldilocks/goldilocks.h"
#include "zkir/Dialect/ModArith/IR/ModArithDialect.h"
#include "zkir/Dialect/ModArith/IR/ModArithTypes.h"
#include "zkir/Utils/ZkDtypes.h"

namespace mlir::zkir::mod_arith {

template <typename F>
class ModArithOperationTest : public testing::Test {
public:
  static void SetUpTestSuite() {
    context.loadDialect<ModArithDialect>();
    auto modulusBits = llvm::bit_ceil(F::Config::kModulusBits);
    IntegerAttr modulus =
        IntegerAttr::get(IntegerType::get(&context, modulusBits),
                         convertToAPInt(F::Config::kModulus, modulusBits));
    modArithType = ModArithType::get(&context, modulus, F::kUseMontgomery);
  }

  void runBinaryOperationTest(
      std::function<F(const F &, const F &)> f_operation,
      std::function<ModArithOperation(const ModArithOperation &,
                                      const ModArithOperation &)>
          m_operation,
      bool bMustBeNonZero = false) {
    auto a = F::Random();
    auto b = F::Random();
    if (!bMustBeNonZero) {
      while (b.IsZero()) {
        b = F::Random();
      }
    }
    runBinaryOperationTest(f_operation, m_operation, a, b);
  }

  void runBinaryOperationTest(
      std::function<F(const F &, const F &)> f_operation,
      std::function<ModArithOperation(const ModArithOperation &,
                                      const ModArithOperation &)>
          m_operation,
      const F &a, const F &b) {
    ModArithOperation modA =
        ModArithOperation(convertToAPInt(a.value()), modArithType);
    ModArithOperation modB =
        ModArithOperation(convertToAPInt(b.value()), modArithType);
    auto c = f_operation(a, b);
    EXPECT_EQ(convertToAPInt(c.value()), m_operation(modA, modB));
  }

  void runUnaryOperationTest(
      std::function<F(const F &)> f_operation,
      std::function<ModArithOperation(const ModArithOperation &)> m_operation,
      bool aMustBeNonZero = false) {
    auto a = F::Random();
    if (!aMustBeNonZero) {
      while (a.IsZero()) {
        a = F::Random();
      }
    }
    runUnaryOperationTest(f_operation, m_operation, a);
  }

  void runUnaryOperationTest(
      std::function<F(const F &)> f_operation,
      std::function<ModArithOperation(const ModArithOperation &)> m_operation,
      const F &a) {
    ModArithOperation modA =
        ModArithOperation(convertToAPInt(a.value()), modArithType);
    auto c = f_operation(a);
    EXPECT_EQ(convertToAPInt(c.value()), m_operation(modA));
  }

  static MLIRContext context;
  static ModArithType modArithType;
};

template <typename F>
MLIRContext ModArithOperationTest<F>::context;

template <typename F>
ModArithType ModArithOperationTest<F>::modArithType;

using PrimeFieldTypes = testing::Types<
    // modulus bits = 2³¹
    // modulus.getBitWidth() == 32
    // modulus.getActiveBits() == 31
    zk_dtypes::Babybear,
    // modulus bits = 2⁶⁴
    // modulus.getBitWidth() == 64
    // modulus.getActiveBits() == 64
    zk_dtypes::Goldilocks>;
TYPED_TEST_SUITE(ModArithOperationTest, PrimeFieldTypes);

//===----------------------------------------------------------------------===//
// Binary Operations
//===----------------------------------------------------------------------===//

TYPED_TEST(ModArithOperationTest, Add) {
  using PrimeFieldType = TypeParam;

  this->runBinaryOperationTest(
      [](const PrimeFieldType &a, const PrimeFieldType &b) { return a + b; },
      [](const ModArithOperation &a, const ModArithOperation &b) {
        return a + b;
      });
}

TYPED_TEST(ModArithOperationTest, AddOverflow) {
  using PrimeFieldType = TypeParam;

  this->runBinaryOperationTest(
      [](const PrimeFieldType &a, const PrimeFieldType &b) { return a + b; },
      [](const ModArithOperation &a, const ModArithOperation &b) {
        return a + b;
      },
      PrimeFieldType::Max(), PrimeFieldType::Random());
}

TYPED_TEST(ModArithOperationTest, Sub) {
  using PrimeFieldType = TypeParam;

  this->runBinaryOperationTest(
      [](const PrimeFieldType &a, const PrimeFieldType &b) { return a - b; },
      [](const ModArithOperation &a, const ModArithOperation &b) {
        return a - b;
      });
}

TYPED_TEST(ModArithOperationTest, SubOverflow) {
  using PrimeFieldType = TypeParam;

  this->runBinaryOperationTest(
      [](const PrimeFieldType &a, const PrimeFieldType &b) { return a - b; },
      [](const ModArithOperation &a, const ModArithOperation &b) {
        return a - b;
      },
      PrimeFieldType::Zero(), PrimeFieldType::Random());
}

TYPED_TEST(ModArithOperationTest, Mul) {
  using PrimeFieldType = TypeParam;

  this->runBinaryOperationTest(
      [](const PrimeFieldType &a, const PrimeFieldType &b) { return a * b; },
      [](const ModArithOperation &a, const ModArithOperation &b) {
        return a * b;
      });
}

TYPED_TEST(ModArithOperationTest, Div) {
  using PrimeFieldType = TypeParam;

  this->runBinaryOperationTest(
      [](const PrimeFieldType &a, const PrimeFieldType &b) { return *(a / b); },
      [](const ModArithOperation &a, const ModArithOperation &b) {
        return a / b;
      },
      /*bMustBeNonZero=*/true);
}

TYPED_TEST(ModArithOperationTest, Cmp) {
  using PrimeFieldType = TypeParam;

  auto a = PrimeFieldType::Random();
  auto b = PrimeFieldType::Random();
  ModArithOperation modA =
      ModArithOperation(convertToAPInt(a.value()), this->modArithType);
  ModArithOperation modB =
      ModArithOperation(convertToAPInt(b.value()), this->modArithType);

  EXPECT_EQ(a < b, modA < modB);
  EXPECT_EQ(a <= b, modA <= modB);
  EXPECT_EQ(a > b, modA > modB);
  EXPECT_EQ(a >= b, modA >= modB);
  EXPECT_EQ(a == b, modA == modB);
  EXPECT_EQ(a != b, modA != modB);
}

//===----------------------------------------------------------------------===//
// Unary Operations
//===----------------------------------------------------------------------===//

TYPED_TEST(ModArithOperationTest, Negate) {
  using PrimeFieldType = TypeParam;

  this->runUnaryOperationTest([](const PrimeFieldType &a) { return -a; },
                              [](const ModArithOperation &a) { return -a; });
}

TYPED_TEST(ModArithOperationTest, NegateZero) {
  using PrimeFieldType = TypeParam;

  this->runUnaryOperationTest([](const PrimeFieldType &a) { return -a; },
                              [](const ModArithOperation &a) { return -a; },
                              PrimeFieldType::Zero());
}

TYPED_TEST(ModArithOperationTest, Double) {
  using PrimeFieldType = TypeParam;

  this->runUnaryOperationTest(
      [](const PrimeFieldType &a) { return a.Double(); },
      [](const ModArithOperation &a) { return a.Double(); });
}

TYPED_TEST(ModArithOperationTest, Square) {
  using PrimeFieldType = TypeParam;

  this->runUnaryOperationTest(
      [](const PrimeFieldType &a) { return a.Square(); },
      [](const ModArithOperation &a) { return a.Square(); });
}

TYPED_TEST(ModArithOperationTest, Power) {
  using PrimeFieldType = TypeParam;

  uint32_t exponents[] = {
      0,
      1,
      static_cast<uint32_t>(PrimeFieldType::Random().value()),
  };

  for (uint32_t exponent : exponents) {
    // NOTE: Technically, power operation is not unary operation. However, we
    // can still test the power operations using runUnaryOperationTest.
    this->runUnaryOperationTest(
        [exponent](const PrimeFieldType &a) { return a.Pow(exponent); },
        [exponent](const ModArithOperation &a) {
          auto modulusBits =
              llvm::bit_ceil(PrimeFieldType::Config::kModulusBits);
          return a.Power(convertToAPInt(exponent, modulusBits));
        });
  }
}

TYPED_TEST(ModArithOperationTest, Inverse) {
  using PrimeFieldType = TypeParam;

  this->runUnaryOperationTest(
      [](const PrimeFieldType &a) { return *a.Inverse(); },
      [](const ModArithOperation &a) { return a.Inverse(); },
      /*aMustBeNonZero=*/true);
}

TYPED_TEST(ModArithOperationTest, FromMont) {
  using PrimeFieldType = TypeParam;

  this->runUnaryOperationTest(
      [](const PrimeFieldType &a) {
        return PrimeFieldType::FromUnchecked(a.MontReduce().value());
      },
      [](const ModArithOperation &a) { return a.FromMont(); });
}

TYPED_TEST(ModArithOperationTest, ToMont) {
  using PrimeFieldType = TypeParam;

  this->runUnaryOperationTest(
      [](const PrimeFieldType &a) { return PrimeFieldType(a.value()); },
      [](const ModArithOperation &a) { return a.ToMont(); });
}

TYPED_TEST(ModArithOperationTest, IsZero) {
  using PrimeFieldType = TypeParam;

  auto zero = PrimeFieldType::Zero();
  ModArithOperation modZero =
      ModArithOperation(convertToAPInt(zero.value()), this->modArithType);
  EXPECT_TRUE(modZero.isZero());
  EXPECT_FALSE(modZero.isOne());

  auto one = PrimeFieldType::One();
  ModArithOperation modOne =
      ModArithOperation(convertToAPInt(one.value()), this->modArithType);
  EXPECT_FALSE(modOne.isZero());
  EXPECT_TRUE(modOne.isOne());

  auto rnd = PrimeFieldType::Random();
  while (rnd.IsZero() || rnd.IsOne()) {
    rnd = PrimeFieldType::Random();
  }
  ModArithOperation modRnd =
      ModArithOperation(convertToAPInt(rnd.value()), this->modArithType);
  EXPECT_FALSE(modRnd.isZero());
  EXPECT_FALSE(modRnd.isOne());
}

} // namespace mlir::zkir::mod_arith
