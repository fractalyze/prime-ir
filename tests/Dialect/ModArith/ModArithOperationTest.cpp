/* Copyright 2025 The PrimeIR Authors.

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

#include "prime_ir/Dialect/ModArith/IR/ModArithOperation.h"

#include "gtest/gtest.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/bit.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithAttributes.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithDialect.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithTypes.h"
#include "zk_dtypes/include/elliptic_curve/bn/bn254/fr.h"
#include "zk_dtypes/include/field/babybear/babybear.h"
#include "zk_dtypes/include/field/goldilocks/goldilocks.h"

namespace mlir::prime_ir::mod_arith {

template <typename F>
class ModArithOperationTest : public testing::Test {
public:
  static void SetUpTestSuite() { context.loadDialect<ModArithDialect>(); }

  void runBinaryOperationTest(
      std::function<F(const F &, const F &)> f_operation,
      std::function<ModArithOperation(const ModArithOperation &,
                                      const ModArithOperation &)>
          m_operation,
      bool bMustBeNonZero = false) {
    auto a = F::Random();
    auto b = F::Random();
    if (bMustBeNonZero) {
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
    auto modA = ModArithOperation::fromZkDtype(&context, a);
    auto modB = ModArithOperation::fromZkDtype(&context, b);
    EXPECT_EQ(ModArithOperation::fromZkDtype(&context, f_operation(a, b)),
              m_operation(modA, modB));
  }

  void runUnaryOperationTest(
      std::function<F(const F &)> f_operation,
      std::function<ModArithOperation(const ModArithOperation &)> m_operation,
      bool aMustBeNonZero = false) {
    auto a = F::Random();
    if (aMustBeNonZero) {
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
    auto modA = ModArithOperation::fromZkDtype(&context, a);
    EXPECT_EQ(ModArithOperation::fromZkDtype(&context, f_operation(a)),
              m_operation(modA));
  }

  static MLIRContext context;
};

template <typename F>
MLIRContext ModArithOperationTest<F>::context;

using PrimeFieldTypes = testing::Types<
    // modulus bits = 2³¹
    // modulus.getBitWidth() == 32
    // modulus.getActiveBits() == 31
    zk_dtypes::BabybearMont, zk_dtypes::Babybear,
    // modulus bits = 2⁶⁴
    // modulus.getBitWidth() == 64
    // modulus.getActiveBits() == 64
    zk_dtypes::GoldilocksMont, zk_dtypes::Goldilocks,
    // modulus bits = 2²⁵⁴
    // modulus.getBitWidth() == 254
    // modulus.getActiveBits() == 254
    zk_dtypes::bn254::FrMont, zk_dtypes::bn254::Fr>;
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
      [](const PrimeFieldType &a, const PrimeFieldType &b) { return a / b; },
      [](const ModArithOperation &a, const ModArithOperation &b) {
        return a / b;
      },
      /*bMustBeNonZero=*/true);
}

TYPED_TEST(ModArithOperationTest, Cmp) {
  using PrimeFieldType = TypeParam;

  auto a = PrimeFieldType::Random();
  auto b = PrimeFieldType::Random();
  auto modA = ModArithOperation::fromZkDtype(&this->context, a);
  auto modB = ModArithOperation::fromZkDtype(&this->context, b);

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
      [](const ModArithOperation &a) { return a.dbl(); });
}

TYPED_TEST(ModArithOperationTest, Square) {
  using PrimeFieldType = TypeParam;

  this->runUnaryOperationTest(
      [](const PrimeFieldType &a) { return a.Square(); },
      [](const ModArithOperation &a) { return a.square(); });
}

TYPED_TEST(ModArithOperationTest, Power) {
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
        [exponent](const ModArithOperation &a) {
          auto modulusBits =
              llvm::bit_ceil(PrimeFieldType::Config::kModulusBits);
          return a.power(convertToAPInt(exponent, modulusBits));
        });
  }
}

TYPED_TEST(ModArithOperationTest, Inverse) {
  using PrimeFieldType = TypeParam;

  this->runUnaryOperationTest(
      [](const PrimeFieldType &a) { return a.Inverse(); },
      [](const ModArithOperation &a) { return a.inverse(); },
      /*aMustBeNonZero=*/true);
}

TYPED_TEST(ModArithOperationTest, InverseOfZero) {
  using PrimeFieldType = TypeParam;

  // inv(0) = 0 by ZK convention.
  auto zero = PrimeFieldType::Zero();
  auto modArithZero = ModArithOperation::fromZkDtype(&this->context, zero);
  auto result = modArithZero.inverse();
  auto expected = ModArithOperation::fromZkDtype(&this->context, zero);
  EXPECT_EQ(result, expected);
}

TYPED_TEST(ModArithOperationTest, FromMont) {
  using PrimeFieldType = TypeParam;

  if constexpr (!PrimeFieldType::kUseMontgomery) {
    GTEST_SKIP() << "Non-Montgomery field is not supported";
  } else {
    auto a = PrimeFieldType::Random();
    auto modA = ModArithOperation::fromZkDtype(&this->context, a);
    EXPECT_EQ(ModArithOperation::fromZkDtype(&this->context, a.MontReduce()),
              modA.fromMont());
  }
}

// TODO(chokobole): Re-enable this test once a mechanism for obtaining a
// MontType from a StdType is implemented.
//
// Note: This conversion is primarily intended for testing internal
// representation consistency and is not required for production workflows.
// Disabling this test for now as it lacks the necessary type-mapping helpers.
TYPED_TEST(ModArithOperationTest, DISABLED_ToMont) {
  using PrimeFieldType = TypeParam;

  if constexpr (PrimeFieldType::kUseMontgomery) {
    GTEST_SKIP() << "Montgomery field is not supported";
  } else {
    auto a = PrimeFieldType::Random();
    auto modA = ModArithOperation::fromZkDtype(&this->context, a);
    EXPECT_EQ(ModArithOperation::fromZkDtype(&this->context,
                                             PrimeFieldType(a.value())),
              modA.toMont());
  }
}

TYPED_TEST(ModArithOperationTest, ZeroAndOne) {
  using PrimeFieldType = TypeParam;

  auto zero = PrimeFieldType::Zero();
  auto modZero = ModArithOperation::fromZkDtype(&this->context, zero);
  EXPECT_TRUE(modZero.isZero());
  EXPECT_FALSE(modZero.isOne());
  EXPECT_EQ(modZero, modZero.getZero());

  auto one = PrimeFieldType::One();
  auto modOne = ModArithOperation::fromZkDtype(&this->context, one);
  EXPECT_FALSE(modOne.isZero());
  EXPECT_TRUE(modOne.isOne());
  EXPECT_EQ(modOne, modOne.getOne());

  auto rnd = PrimeFieldType::Random();
  while (rnd.IsZero() || rnd.IsOne()) {
    rnd = PrimeFieldType::Random();
  }
  auto modRnd = ModArithOperation::fromZkDtype(&this->context, rnd);
  EXPECT_FALSE(modRnd.isZero());
  EXPECT_FALSE(modRnd.isOne());
}

// MontgomeryAttrStorage::construct used to reduce b = 2⁶⁴ modulo the modulus
// TRUNCATED to 65 bits, so bReduced — and every constant derived from it
// (R, R², R⁻¹, b⁻¹) — was corrupted for any multi-limb modulus whose bit 64
// is clear. When bit 64 is set, 2⁶⁴ mod (modulus mod 2⁶⁵) happens to equal
// 2⁶⁴, which masked the bug for BN254 and secp256k1 (the typed tests above).
// Pin the derived constants against their definitions for two bit-64-clear
// moduli: the BLS12-381 and secp256r1 scalar fields.
TEST(MontgomeryAttrDerivedConstantsTest, Bit64ClearModulus) {
  MLIRContext context;
  context.loadDialect<ModArithDialect>();

  const char *moduliHex[] = {
      // BLS12-381 Fr
      "73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001",
      // secp256r1 Fr
      "ffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551",
  };
  for (const char *hex : moduliHex) {
    APInt modulus(256, hex, 16);
    ASSERT_FALSE(modulus[64]) << "test wants a bit-64-clear modulus";
    auto modAttr = IntegerAttr::get(IntegerType::get(&context, 256), modulus);
    ModArithType modType = ModArithType::get(&context, modAttr);
    MontgomeryAttr mont = modType.getMontgomeryAttr();

    // Work in 512 bits so products cannot overflow.
    APInt mod512 = modulus.zext(512);
    auto mulMod = [&](const APInt &a, const APInt &b) {
      return (a.zext(512) * b.zext(512)).urem(mod512).trunc(256);
    };
    APInt one(256, 1);

    // R = 2²⁵⁶ mod n, computed independently of the attr.
    APInt r = APInt::getOneBitSet(512, 256).urem(mod512).trunc(256);
    EXPECT_EQ(mont.getR().getValue(), r) << hex;
    EXPECT_EQ(mont.getRSquared().getValue(), mulMod(r, r)) << hex;
    EXPECT_EQ(mulMod(mont.getRInv().getValue(), r), one) << hex;
    // bInv · 2⁶⁴ ≡ 1 (mod n).
    APInt b64 = APInt::getOneBitSet(256, 64);
    EXPECT_EQ(mulMod(mont.getBInv().getValue(), b64), one) << hex;
  }
}

} // namespace mlir::prime_ir::mod_arith
