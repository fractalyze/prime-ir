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
#include "prime_ir/Dialect/Field/IR/FieldDialect.h"
#include "prime_ir/Dialect/Field/IR/FieldOps.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"

namespace mlir::prime_ir::field {

class BitcastOpTest : public testing::Test {
protected:
  void SetUp() override {
    context.loadDialect<FieldDialect>();
    builder = std::make_unique<OpBuilder>(&context);
  }

  MLIRContext context;
  std::unique_ptr<OpBuilder> builder;
};

// Test Montgomery form bitcast compatibility for prime fields
TEST_F(BitcastOpTest, PrimeFieldMontgomeryFormBitcastScalar) {
  auto modulus = IntegerAttr::get(builder->getI32Type(), 7);
  auto stdType = PrimeFieldType::get(&context, modulus, false);
  auto montType = PrimeFieldType::get(&context, modulus, true);

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

TEST_F(BitcastOpTest, PrimeFieldMontgomeryFormBitcastTensor) {
  auto modulus = IntegerAttr::get(builder->getI32Type(), 7);
  auto stdType = PrimeFieldType::get(&context, modulus, false);
  auto montType = PrimeFieldType::get(&context, modulus, true);

  auto tensorStd = RankedTensorType::get({4}, stdType);
  auto tensorMont = RankedTensorType::get({4}, montType);

  // Tensor conversion should be compatible
  EXPECT_TRUE(BitcastOp::areCastCompatible(TypeRange{tensorStd},
                                           TypeRange{tensorMont}));
  EXPECT_TRUE(BitcastOp::areCastCompatible(TypeRange{tensorMont},
                                           TypeRange{tensorStd}));
}

// Note: ExtensionField Montgomery form bitcast is tested in bitcast.mlir
// C++ unittest is complex due to nonResidue Montgomery form conversion
// requirements

// Test tensor reinterpret bitcast: extension field <-> prime field
TEST_F(BitcastOpTest, TensorReinterpretBitcast) {
  auto modulus = IntegerAttr::get(builder->getI32Type(), 7);
  auto nonResidue = IntegerAttr::get(builder->getI32Type(), 3);

  auto pfType = PrimeFieldType::get(&context, modulus, false);
  auto efType = ExtensionFieldType::get(&context, 3, pfType, nonResidue);

  // 1 EF3 element = 3 PF elements
  auto tensorEF = RankedTensorType::get({1}, efType);
  auto tensorPF = RankedTensorType::get({3}, pfType);

  // Extension field to prime field should be compatible
  EXPECT_TRUE(
      BitcastOp::areCastCompatible(TypeRange{tensorEF}, TypeRange{tensorPF}));
  // Prime field to extension field should be compatible
  EXPECT_TRUE(
      BitcastOp::areCastCompatible(TypeRange{tensorPF}, TypeRange{tensorEF}));
}

TEST_F(BitcastOpTest, TensorReinterpretBitcastMultiElement) {
  auto modulus = IntegerAttr::get(builder->getI32Type(), 7);
  auto nonResidue = IntegerAttr::get(builder->getI32Type(), 3);

  auto pfType = PrimeFieldType::get(&context, modulus, false);
  auto efType = ExtensionFieldType::get(&context, 2, pfType, nonResidue);

  // 4 EF2 elements = 8 PF elements
  auto tensorEF = RankedTensorType::get({4}, efType);
  auto tensorPF = RankedTensorType::get({8}, pfType);

  EXPECT_TRUE(
      BitcastOp::areCastCompatible(TypeRange{tensorEF}, TypeRange{tensorPF}));
  EXPECT_TRUE(
      BitcastOp::areCastCompatible(TypeRange{tensorPF}, TypeRange{tensorEF}));
}

TEST_F(BitcastOpTest, TensorReinterpretBitcastElementCountMismatch) {
  auto modulus = IntegerAttr::get(builder->getI32Type(), 7);
  auto nonResidue = IntegerAttr::get(builder->getI32Type(), 3);

  auto pfType = PrimeFieldType::get(&context, modulus, false);
  auto efType = ExtensionFieldType::get(&context, 3, pfType, nonResidue);

  // 1 EF3 element = 3 PF elements, but we have 4 PF elements (mismatch)
  auto tensorEF = RankedTensorType::get({1}, efType);
  auto tensorPF = RankedTensorType::get({4}, pfType);

  EXPECT_FALSE(
      BitcastOp::areCastCompatible(TypeRange{tensorEF}, TypeRange{tensorPF}));
}

TEST_F(BitcastOpTest, TensorReinterpretBitcastDifferentModuliSameBitwidth) {
  auto modulus7 = IntegerAttr::get(builder->getI32Type(), 7);
  auto modulus11 = IntegerAttr::get(builder->getI32Type(), 11);
  auto nonResidue = IntegerAttr::get(builder->getI32Type(), 3);

  auto pf7 = PrimeFieldType::get(&context, modulus7, false);
  auto pf11 = PrimeFieldType::get(&context, modulus11, false);
  auto ef7 = ExtensionFieldType::get(&context, 3, pf7, nonResidue);

  // Different base field moduli with same total bitwidth should be compatible
  auto tensorEF = RankedTensorType::get({1}, ef7);
  auto tensorPF = RankedTensorType::get({3}, pf11);

  EXPECT_TRUE(
      BitcastOp::areCastCompatible(TypeRange{tensorEF}, TypeRange{tensorPF}));
}

// Test PrimeField <-> Integer bitcast (not tensor reinterpret)
TEST_F(BitcastOpTest, PrimeFieldToIntegerNotTensorReinterpret) {
  auto modulus = IntegerAttr::get(builder->getI32Type(), 7);
  auto pfType = PrimeFieldType::get(&context, modulus, false);
  auto intType = builder->getI32Type();

  // Scalar conversion should be compatible
  EXPECT_TRUE(
      BitcastOp::areCastCompatible(TypeRange{pfType}, TypeRange{intType}));
  EXPECT_TRUE(
      BitcastOp::areCastCompatible(TypeRange{intType}, TypeRange{pfType}));

  // Same type tensor is also allowed (will be folded away)
  auto tensorPF = RankedTensorType::get({3}, pfType);
  EXPECT_TRUE(
      BitcastOp::areCastCompatible(TypeRange{tensorPF}, TypeRange{tensorPF}));
}

TEST_F(BitcastOpTest, DifferentPrimeFieldModuliSameBitwidthCompatible) {
  auto modulus7 = IntegerAttr::get(builder->getI32Type(), 7);
  auto modulus11 = IntegerAttr::get(builder->getI32Type(), 11);
  auto pf7 = PrimeFieldType::get(&context, modulus7, false);
  auto pf11 = PrimeFieldType::get(&context, modulus11, false);

  // Different moduli with same bitwidth should be compatible
  EXPECT_TRUE(BitcastOp::areCastCompatible(TypeRange{pf7}, TypeRange{pf11}));

  // Also test with Montgomery form
  auto pf7Mont = PrimeFieldType::get(&context, modulus7, true);
  auto pf11Mont = PrimeFieldType::get(&context, modulus11, true);
  EXPECT_TRUE(
      BitcastOp::areCastCompatible(TypeRange{pf7Mont}, TypeRange{pf11Mont}));

  // Cross Montgomery/standard with different moduli should also work
  EXPECT_TRUE(
      BitcastOp::areCastCompatible(TypeRange{pf7}, TypeRange{pf11Mont}));
}

// Test tower extension field bitcast (extension over extension)
TEST_F(BitcastOpTest, TowerExtensionFieldMontgomeryFormBitcastScalar) {
  auto modulus = IntegerAttr::get(builder->getI32Type(), 7);
  // Use separate non-residues: 6 for EF2 (quadratic), 2 for tower (cubic)
  // These values are validated: 6^3 ≡ 6 (mod 7) ≠ 1, and 2^16 ≠ 1 in EF2
  auto nonResidueEF2 = IntegerAttr::get(builder->getI32Type(), 6);
  auto nonResidueTower = IntegerAttr::get(builder->getI32Type(), 2);

  auto pfStd = PrimeFieldType::get(&context, modulus, false);
  auto pfMont = PrimeFieldType::get(&context, modulus, true);

  // Create EF2 over PF (standard and Montgomery)
  auto ef2Std = ExtensionFieldType::get(&context, 2, pfStd, nonResidueEF2);
  auto ef2Mont = ExtensionFieldType::get(&context, 2, pfMont, nonResidueEF2);

  // Create tower: EF3 over EF2 (Fp6 as tower)
  auto towerStd = ExtensionFieldType::get(&context, 3, ef2Std, nonResidueTower);
  auto towerMont =
      ExtensionFieldType::get(&context, 3, ef2Mont, nonResidueTower);

  // Standard to Montgomery should be compatible
  EXPECT_TRUE(
      BitcastOp::areCastCompatible(TypeRange{towerStd}, TypeRange{towerMont}));
  // Montgomery to Standard should be compatible
  EXPECT_TRUE(
      BitcastOp::areCastCompatible(TypeRange{towerMont}, TypeRange{towerStd}));
  // Same type is allowed (will be folded away)
  EXPECT_TRUE(
      BitcastOp::areCastCompatible(TypeRange{towerStd}, TypeRange{towerStd}));
  EXPECT_TRUE(
      BitcastOp::areCastCompatible(TypeRange{towerMont}, TypeRange{towerMont}));
}

TEST_F(BitcastOpTest, TowerExtensionFieldMontgomeryFormBitcastTensor) {
  auto modulus = IntegerAttr::get(builder->getI32Type(), 7);
  auto nonResidueEF2 = IntegerAttr::get(builder->getI32Type(), 6);
  auto nonResidueTower = IntegerAttr::get(builder->getI32Type(), 2);

  auto pfStd = PrimeFieldType::get(&context, modulus, false);
  auto pfMont = PrimeFieldType::get(&context, modulus, true);

  auto ef2Std = ExtensionFieldType::get(&context, 2, pfStd, nonResidueEF2);
  auto ef2Mont = ExtensionFieldType::get(&context, 2, pfMont, nonResidueEF2);

  auto towerStd = ExtensionFieldType::get(&context, 3, ef2Std, nonResidueTower);
  auto towerMont =
      ExtensionFieldType::get(&context, 3, ef2Mont, nonResidueTower);

  auto tensorStd = RankedTensorType::get({4}, towerStd);
  auto tensorMont = RankedTensorType::get({4}, towerMont);

  // Tensor conversion should be compatible
  EXPECT_TRUE(BitcastOp::areCastCompatible(TypeRange{tensorStd},
                                           TypeRange{tensorMont}));
  EXPECT_TRUE(BitcastOp::areCastCompatible(TypeRange{tensorMont},
                                           TypeRange{tensorStd}));
}

// Test tensor reinterpret bitcast: tower extension <-> prime field
TEST_F(BitcastOpTest, TowerExtensionToPrimeFieldReinterpretBitcast) {
  auto modulus = IntegerAttr::get(builder->getI32Type(), 7);
  auto nonResidueEF2 = IntegerAttr::get(builder->getI32Type(), 6);
  auto nonResidueTower = IntegerAttr::get(builder->getI32Type(), 2);

  auto pfType = PrimeFieldType::get(&context, modulus, false);
  auto ef2Type = ExtensionFieldType::get(&context, 2, pfType, nonResidueEF2);
  // Tower: EF3 over EF2 = Fp6
  auto towerType =
      ExtensionFieldType::get(&context, 3, ef2Type, nonResidueTower);

  // 1 tower element (Fp6) = 6 PF elements
  auto tensorTower = RankedTensorType::get({1}, towerType);
  auto tensorPF = RankedTensorType::get({6}, pfType);

  // Tower extension to prime field should be compatible
  EXPECT_TRUE(BitcastOp::areCastCompatible(TypeRange{tensorTower},
                                           TypeRange{tensorPF}));
  // Prime field to tower extension should be compatible
  EXPECT_TRUE(BitcastOp::areCastCompatible(TypeRange{tensorPF},
                                           TypeRange{tensorTower}));
}

// Test tensor reinterpret bitcast: different tower structures with same total
// degree
TEST_F(BitcastOpTest, TowerExtensionDifferentStructuresSameDegree) {
  auto modulus = IntegerAttr::get(builder->getI32Type(), 7);
  auto nonResidueEF2 = IntegerAttr::get(builder->getI32Type(), 6);
  auto nonResidueEF3 = IntegerAttr::get(builder->getI32Type(), 3);
  // For tower1 (EF3 over EF2): 2^16 ≠ 1 in EF2 (q=49, (q-1)/3=16)
  auto nonResidueTower1 = IntegerAttr::get(builder->getI32Type(), 2);
  // For tower2 (EF2 over EF3): 5^171 ≠ 1 in EF3 (q=343, (q-1)/2=171)
  // Note: 2 is a square in Fp7 (3²=2), so we need a non-square like 5
  auto nonResidueTower2 = IntegerAttr::get(builder->getI32Type(), 5);

  auto pfType = PrimeFieldType::get(&context, modulus, false);
  auto ef2Type = ExtensionFieldType::get(&context, 2, pfType, nonResidueEF2);
  auto ef3Type = ExtensionFieldType::get(&context, 3, pfType, nonResidueEF3);

  // Tower 1: EF3 over EF2 = Fp6
  auto tower1Type =
      ExtensionFieldType::get(&context, 3, ef2Type, nonResidueTower1);
  // Tower 2: EF2 over EF3 = Fp6
  auto tower2Type =
      ExtensionFieldType::get(&context, 2, ef3Type, nonResidueTower2);

  // Both have total degree 6 over prime
  auto tensorTower1 = RankedTensorType::get({2}, tower1Type);
  auto tensorTower2 = RankedTensorType::get({2}, tower2Type);

  // Different tower structures with same total degree should be compatible
  EXPECT_TRUE(BitcastOp::areCastCompatible(TypeRange{tensorTower1},
                                           TypeRange{tensorTower2}));
  EXPECT_TRUE(BitcastOp::areCastCompatible(TypeRange{tensorTower2},
                                           TypeRange{tensorTower1}));
}

TEST_F(BitcastOpTest, TowerExtensionReinterpretBitcastMultiElement) {
  auto modulus = IntegerAttr::get(builder->getI32Type(), 7);
  auto nonResidueEF2 = IntegerAttr::get(builder->getI32Type(), 6);
  auto nonResidueTower = IntegerAttr::get(builder->getI32Type(), 2);

  auto pfType = PrimeFieldType::get(&context, modulus, false);
  auto ef2Type = ExtensionFieldType::get(&context, 2, pfType, nonResidueEF2);
  // Tower: EF3 over EF2 = Fp6 (using EF3 over EF2 instead of EF2 over EF2
  // because no Fp element works as a non-square in Fp2 for p=7)
  auto towerType =
      ExtensionFieldType::get(&context, 3, ef2Type, nonResidueTower);

  // 2 tower elements (Fp6 each) = 12 PF elements
  auto tensorTower = RankedTensorType::get({2}, towerType);
  auto tensorPF = RankedTensorType::get({12}, pfType);

  EXPECT_TRUE(BitcastOp::areCastCompatible(TypeRange{tensorTower},
                                           TypeRange{tensorPF}));
  EXPECT_TRUE(BitcastOp::areCastCompatible(TypeRange{tensorPF},
                                           TypeRange{tensorTower}));
}

TEST_F(BitcastOpTest, TowerExtensionReinterpretBitcastElementCountMismatch) {
  auto modulus = IntegerAttr::get(builder->getI32Type(), 7);
  auto nonResidueEF2 = IntegerAttr::get(builder->getI32Type(), 6);
  auto nonResidueTower = IntegerAttr::get(builder->getI32Type(), 2);

  auto pfType = PrimeFieldType::get(&context, modulus, false);
  auto ef2Type = ExtensionFieldType::get(&context, 2, pfType, nonResidueEF2);
  // Tower: EF3 over EF2 = Fp6
  auto towerType =
      ExtensionFieldType::get(&context, 3, ef2Type, nonResidueTower);

  // 1 tower element (Fp6) = 6 PF elements, but we have 5 PF elements (mismatch)
  auto tensorTower = RankedTensorType::get({1}, towerType);
  auto tensorPF = RankedTensorType::get({5}, pfType);

  EXPECT_FALSE(BitcastOp::areCastCompatible(TypeRange{tensorTower},
                                            TypeRange{tensorPF}));
}

} // namespace mlir::prime_ir::field
