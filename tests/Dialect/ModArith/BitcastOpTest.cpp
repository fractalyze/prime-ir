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

#include "gtest/gtest.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithDialect.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithOps.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithTypes.h"

namespace mlir::prime_ir::mod_arith {

class BitcastOpTest : public testing::Test {
protected:
  void SetUp() override {
    context.loadDialect<ModArithDialect>();
    builder = std::make_unique<OpBuilder>(&context);
  }

  MLIRContext context;
  std::unique_ptr<OpBuilder> builder;
};

// Test Montgomery form bitcast compatibility
TEST_F(BitcastOpTest, MontgomeryFormBitcastScalar) {
  auto modulus = IntegerAttr::get(builder->getI32Type(), 7);
  auto stdType = ModArithType::get(&context, modulus, false);
  auto montType = ModArithType::get(&context, modulus, true);

  // Standard to Montgomery should be compatible
  EXPECT_TRUE(
      BitcastOp::areCastCompatible(TypeRange{stdType}, TypeRange{montType}));
  // Montgomery to Standard should be compatible
  EXPECT_TRUE(
      BitcastOp::areCastCompatible(TypeRange{montType}, TypeRange{stdType}));
  // Same type is allowed (will be folded away)
  EXPECT_TRUE(
      BitcastOp::areCastCompatible(TypeRange{stdType}, TypeRange{stdType}));
  EXPECT_TRUE(
      BitcastOp::areCastCompatible(TypeRange{montType}, TypeRange{montType}));
}

TEST_F(BitcastOpTest, MontgomeryFormBitcastTensor) {
  auto modulus = IntegerAttr::get(builder->getI32Type(), 7);
  auto stdType = ModArithType::get(&context, modulus, false);
  auto montType = ModArithType::get(&context, modulus, true);

  auto tensorStd = RankedTensorType::get({4}, stdType);
  auto tensorMont = RankedTensorType::get({4}, montType);

  // Tensor conversion should be compatible
  EXPECT_TRUE(BitcastOp::areCastCompatible(TypeRange{tensorStd},
                                           TypeRange{tensorMont}));
  EXPECT_TRUE(BitcastOp::areCastCompatible(TypeRange{tensorMont},
                                           TypeRange{tensorStd}));
}

TEST_F(BitcastOpTest, MontgomeryFormBitcastShapeMismatch) {
  auto modulus = IntegerAttr::get(builder->getI32Type(), 7);
  auto stdType = ModArithType::get(&context, modulus, false);
  auto montType = ModArithType::get(&context, modulus, true);

  auto tensor4 = RankedTensorType::get({4}, stdType);
  auto tensor5 = RankedTensorType::get({5}, montType);

  // Mismatched shapes should not be compatible
  EXPECT_FALSE(
      BitcastOp::areCastCompatible(TypeRange{tensor4}, TypeRange{tensor5}));
}

// Test ModArith <-> Integer bitcast compatibility
TEST_F(BitcastOpTest, ModArithToIntegerScalar) {
  auto modulus = IntegerAttr::get(builder->getI32Type(), 7);
  auto modArithType = ModArithType::get(&context, modulus, false);
  auto intType = builder->getI32Type();

  // ModArith to Integer should be compatible
  EXPECT_TRUE(BitcastOp::areCastCompatible(TypeRange{modArithType},
                                           TypeRange{intType}));
  // Integer to ModArith should be compatible
  EXPECT_TRUE(BitcastOp::areCastCompatible(TypeRange{intType},
                                           TypeRange{modArithType}));
}

TEST_F(BitcastOpTest, ModArithToIntegerTensor) {
  auto modulus = IntegerAttr::get(builder->getI32Type(), 7);
  auto modArithType = ModArithType::get(&context, modulus, false);
  auto intType = builder->getI32Type();

  auto tensorModArith = RankedTensorType::get({4}, modArithType);
  auto tensorInt = RankedTensorType::get({4}, intType);

  // Tensor conversion should be compatible
  EXPECT_TRUE(BitcastOp::areCastCompatible(TypeRange{tensorModArith},
                                           TypeRange{tensorInt}));
  EXPECT_TRUE(BitcastOp::areCastCompatible(TypeRange{tensorInt},
                                           TypeRange{tensorModArith}));
}

TEST_F(BitcastOpTest, IntegerToIntegerNotCompatible) {
  auto intType = builder->getI32Type();

  // Integer to Integer should not be compatible (should use arith.bitcast)
  EXPECT_FALSE(
      BitcastOp::areCastCompatible(TypeRange{intType}, TypeRange{intType}));
}

TEST_F(BitcastOpTest, DifferentModuliSameBitwidthCompatible) {
  auto modulus7 = IntegerAttr::get(builder->getI32Type(), 7);
  auto modulus11 = IntegerAttr::get(builder->getI32Type(), 11);
  auto type7 = ModArithType::get(&context, modulus7, false);
  auto type11 = ModArithType::get(&context, modulus11, false);

  // Different moduli with same bitwidth should be compatible
  EXPECT_TRUE(
      BitcastOp::areCastCompatible(TypeRange{type7}, TypeRange{type11}));

  // Also test with Montgomery form
  auto type7Mont = ModArithType::get(&context, modulus7, true);
  auto type11Mont = ModArithType::get(&context, modulus11, true);
  EXPECT_TRUE(BitcastOp::areCastCompatible(TypeRange{type7Mont},
                                           TypeRange{type11Mont}));

  // Cross Montgomery/standard with different moduli should also work
  EXPECT_TRUE(
      BitcastOp::areCastCompatible(TypeRange{type7}, TypeRange{type11Mont}));
}

TEST_F(BitcastOpTest, BitwidthMismatchNotCompatible) {
  auto modulus7i32 = IntegerAttr::get(builder->getI32Type(), 7);
  auto modulus7i64 = IntegerAttr::get(builder->getI64Type(), 7);
  auto type32 = ModArithType::get(&context, modulus7i32, false);
  auto type64 = ModArithType::get(&context, modulus7i64, false);

  // Different bitwidths should not be compatible
  EXPECT_FALSE(
      BitcastOp::areCastCompatible(TypeRange{type32}, TypeRange{type64}));
}

TEST_F(BitcastOpTest, ScalarToTensorNotCompatible) {
  auto modulus = IntegerAttr::get(builder->getI32Type(), 7);
  auto scalarType = ModArithType::get(&context, modulus, false);
  auto tensorType = RankedTensorType::get({1}, scalarType);

  // Scalar to tensor should not be compatible
  EXPECT_FALSE(BitcastOp::areCastCompatible(TypeRange{scalarType},
                                            TypeRange{tensorType}));
  EXPECT_FALSE(BitcastOp::areCastCompatible(TypeRange{tensorType},
                                            TypeRange{scalarType}));
}

} // namespace mlir::prime_ir::mod_arith
