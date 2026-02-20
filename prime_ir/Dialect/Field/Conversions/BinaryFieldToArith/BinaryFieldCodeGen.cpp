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

#include "prime_ir/Dialect/Field/Conversions/BinaryFieldToArith/BinaryFieldCodeGen.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "prime_ir/Dialect/Field/Conversions/BinaryFieldToArith/BinaryFieldTables.h"

namespace mlir::prime_ir::field {

BinaryFieldCodeGen::BinaryFieldCodeGen(BinaryFieldType bfType, Value value,
                                       ImplicitLocOpBuilder &builder)
    : bfType_(bfType), value_(value), builder_(builder) {}

BinaryFieldCodeGen
BinaryFieldCodeGen::operator+(const BinaryFieldCodeGen &other) const {
  // In characteristic 2, addition is XOR
  Value result = builder_.create<arith::XOrIOp>(value_, other.value_);
  return BinaryFieldCodeGen(bfType_, result, builder_);
}

BinaryFieldCodeGen
BinaryFieldCodeGen::operator-(const BinaryFieldCodeGen &other) const {
  // In characteristic 2, subtraction is the same as addition (XOR)
  return *this + other;
}

BinaryFieldCodeGen BinaryFieldCodeGen::operator-() const {
  // In characteristic 2, negation is identity: -a = a
  return *this;
}

BinaryFieldCodeGen BinaryFieldCodeGen::dbl() const {
  // In characteristic 2, doubling is zero: 2a = a + a = 0
  return zero(bfType_, builder_);
}

BinaryFieldCodeGen
BinaryFieldCodeGen::operator*(const BinaryFieldCodeGen &other) const {
  unsigned towerLevel = bfType_.getTowerLevel();
  Value result = mulTower(value_, other.value_, towerLevel);
  return BinaryFieldCodeGen(bfType_, result, builder_);
}

BinaryFieldCodeGen BinaryFieldCodeGen::square() const {
  unsigned towerLevel = bfType_.getTowerLevel();
  Value result = squareTower(value_, towerLevel);
  return BinaryFieldCodeGen(bfType_, result, builder_);
}

BinaryFieldCodeGen BinaryFieldCodeGen::inverse() const {
  unsigned towerLevel = bfType_.getTowerLevel();

  // For bf<8> (tower level 3), use lookup table for O(1) inverse
  if (towerLevel == 3) {
    return inverseLookupTable();
  }

  // Use Fermat's little theorem: a⁻¹ = a^(2ⁿ - 2) in GF(2ⁿ)
  // where n = 2^towerLevel
  unsigned bitWidth = bfType_.getBitWidth();

  // Start with a²
  Value result = squareTower(value_, towerLevel);
  Value power = result;

  // Compute a^(2ⁿ - 2) using repeated squaring
  // a^(2ⁿ - 2) = a^(2 + 4 + 8 + ... + 2ⁿ⁻¹)
  for (unsigned i = 2; i < bitWidth; ++i) {
    power = squareTower(power, towerLevel);
    result = mulTower(result, power, towerLevel);
  }

  return BinaryFieldCodeGen(bfType_, result, builder_);
}

BinaryFieldCodeGen BinaryFieldCodeGen::inverseLookupTable() const {
  // Generate lookup table as a global constant memref and use it
  // For now, generate inline switch-case style code using arith ops
  // Future optimization: use LLVM global constant + load

  // Create the lookup table as a series of constants
  // We'll use a branchless approach with masking for each possible value

  // For SIMD friendliness, we could use vector shuffles, but for scalar
  // bf<8>, we'll generate a direct index into the table using arith ops.

  // Simple approach: create constant array and index into it
  // Use LLVM's constant array with memref.global + memref.load

  // For simplicity in this baseline implementation, use scf.index_switch
  // which will be lowered to an efficient jump table

  auto indexValue =
      builder_.create<arith::IndexCastUIOp>(builder_.getIndexType(), value_);

  // Create switch with all 256 cases
  SmallVector<int64_t> caseValues;

  for (int i = 0; i < 256; ++i) {
    caseValues.push_back(i);
  }

  // Generate scf.index_switch
  auto switchOp = builder_.create<scf::IndexSwitchOp>(
      TypeRange{builder_.getI8Type()}, indexValue, caseValues,
      caseValues.size());

  // Generate case regions
  for (int i = 0; i < 256; ++i) {
    Region &caseRegion = switchOp.getCaseRegions()[i];
    Block *caseBlock = builder_.createBlock(&caseRegion);
    builder_.setInsertionPointToStart(caseBlock);
    Value inverseVal =
        builder_.create<arith::ConstantIntOp>(kBinaryTower8bInverseTable[i], 8);
    builder_.create<scf::YieldOp>(inverseVal);
  }

  // Generate default region (return 0 for safety, should never be reached)
  Region &defaultRegion = switchOp.getDefaultRegion();
  Block *defaultBlock = builder_.createBlock(&defaultRegion);
  builder_.setInsertionPointToStart(defaultBlock);
  Value defaultVal = builder_.create<arith::ConstantIntOp>(0, 8);
  builder_.create<scf::YieldOp>(defaultVal);

  // Move insertion point back after switch
  builder_.setInsertionPointAfter(switchOp);

  return BinaryFieldCodeGen(bfType_, switchOp.getResult(0), builder_);
}

BinaryFieldCodeGen BinaryFieldCodeGen::constant(BinaryFieldType bfType,
                                                uint64_t val,
                                                ImplicitLocOpBuilder &builder) {
  IntegerType intType = bfType.getStorageType();
  Value c = builder.create<arith::ConstantIntOp>(static_cast<int64_t>(val),
                                                 intType.getWidth());
  return BinaryFieldCodeGen(bfType, c, builder);
}

BinaryFieldCodeGen BinaryFieldCodeGen::zero(BinaryFieldType bfType,
                                            ImplicitLocOpBuilder &builder) {
  return constant(bfType, 0, builder);
}

BinaryFieldCodeGen BinaryFieldCodeGen::one(BinaryFieldType bfType,
                                           ImplicitLocOpBuilder &builder) {
  return constant(bfType, 1, builder);
}

uint64_t BinaryFieldCodeGen::getTowerAlpha(unsigned towerLevel) const {
  // Tower α constants are defined in BinaryFieldTables.h
  // These are the elements satisfying x² + x + α = 0
  // at each tower level. Level 0 has no α (it's GF(2)).
  assert(towerLevel >= 1 && towerLevel <= 7 && "Tower level must be 1-7 for α");
  return kTowerAlphas[towerLevel];
}

Value BinaryFieldCodeGen::mulTower(Value a, Value b,
                                   unsigned towerLevel) const {
  // Base case: tower level 0 is GF(2), multiplication is AND
  if (towerLevel == 0) {
    return builder_.create<arith::AndIOp>(a, b);
  }

  // Recursive case: use Karatsuba-style multiplication
  // For GF(2^(2^k)), we have elements a = a₀ + a₁*x and b = b₀ + b₁*x
  // where x² + x + α = ₀
  //
  // Product: a*b = (a₀*b₀ + a₁*b₁*α) + (a₀*b₁ + a₁*b₀ + a₁*b₁)*x
  //
  // Using Karatsuba:
  // m₀ = a₀*b₀
  // m₁ = a₁*b₁
  // m₂ = (a₀+a₁)*(b₀+b₁)
  // result_lo = m₀ + m₁*α
  // result_hi = m₂ + m₀ + m₁  (= a₀*b₁ + a₁*b₀ + a₁*b₁)

  unsigned halfBits = 1u << (towerLevel - 1);
  IntegerType halfType = IntegerType::get(builder_.getContext(), halfBits);
  IntegerType fullType = IntegerType::get(builder_.getContext(), halfBits * 2);

  // Extract lower and upper halves
  Value a0 = builder_.create<arith::TruncIOp>(halfType, a);
  Value b0 = builder_.create<arith::TruncIOp>(halfType, b);

  Value halfBitsConst = builder_.create<arith::ConstantIntOp>(
      static_cast<int64_t>(halfBits), fullType.getWidth());
  Value a1 = builder_.create<arith::TruncIOp>(
      halfType, builder_.create<arith::ShRUIOp>(a, halfBitsConst));
  Value b1 = builder_.create<arith::TruncIOp>(
      halfType, builder_.create<arith::ShRUIOp>(b, halfBitsConst));

  // Karatsuba products
  Value m0 = mulTower(a0, b0, towerLevel - 1);
  Value m1 = mulTower(a1, b1, towerLevel - 1);
  Value a0Xa1 = builder_.create<arith::XOrIOp>(a0, a1);
  Value b0Xb1 = builder_.create<arith::XOrIOp>(b0, b1);
  Value m2 = mulTower(a0Xa1, b0Xb1, towerLevel - 1);

  // Get α for this tower level
  // TowerAlpha<k> is the α used in GF(2^(2ᵏ))
  uint64_t alpha = getTowerAlpha(towerLevel);
  Value alphaConst = builder_.create<arith::ConstantIntOp>(
      static_cast<int64_t>(alpha), halfType.getWidth());

  // result_lo = m₀ + m₁*α
  Value m1Alpha = mulTower(m1, alphaConst, towerLevel - 1);
  Value resultLo = builder_.create<arith::XOrIOp>(m0, m1Alpha);

  // result_hi = m₂ + m₀ + m₁
  Value m2Xm0 = builder_.create<arith::XOrIOp>(m2, m0);
  Value resultHi = builder_.create<arith::XOrIOp>(m2Xm0, m1);

  // Combine: result = result_lo | (result_hi << halfBits)
  Value resultLoExt = builder_.create<arith::ExtUIOp>(fullType, resultLo);
  Value resultHiExt = builder_.create<arith::ExtUIOp>(fullType, resultHi);
  Value resultHiShifted =
      builder_.create<arith::ShLIOp>(resultHiExt, halfBitsConst);
  return builder_.create<arith::OrIOp>(resultLoExt, resultHiShifted);
}

Value BinaryFieldCodeGen::squareTower(Value a, unsigned towerLevel) const {
  // Base case: tower level 0 is GF(2), squaring is identity
  if (towerLevel == 0) {
    return a;
  }

  // For GF(2^(2ᵏ)), element a = a₀ + a1*x where x² + x + α = 0
  // a² = a₀² + a₁²*x² = a₀² + a₁²*(x + α)
  //     = (a₀² + a₁²*α) + a₁²*x

  unsigned halfBits = 1u << (towerLevel - 1);
  IntegerType halfType = IntegerType::get(builder_.getContext(), halfBits);
  IntegerType fullType = IntegerType::get(builder_.getContext(), halfBits * 2);

  // Extract lower and upper halves
  Value a0 = builder_.create<arith::TruncIOp>(halfType, a);
  Value halfBitsConst = builder_.create<arith::ConstantIntOp>(
      static_cast<int64_t>(halfBits), fullType.getWidth());
  Value a1 = builder_.create<arith::TruncIOp>(
      halfType, builder_.create<arith::ShRUIOp>(a, halfBitsConst));

  // Recursive squaring
  Value a0Sq = squareTower(a0, towerLevel - 1);
  Value a1Sq = squareTower(a1, towerLevel - 1);

  // Get α for this tower level
  // TowerAlpha<k> is the α used in GF(2^(2ᵏ))
  uint64_t alpha = getTowerAlpha(towerLevel);
  Value alphaConst = builder_.create<arith::ConstantIntOp>(
      static_cast<int64_t>(alpha), halfType.getWidth());

  // result_lo = a₀² + a1²*α
  Value a1SqAlpha = mulTower(a1Sq, alphaConst, towerLevel - 1);
  Value resultLo = builder_.create<arith::XOrIOp>(a0Sq, a1SqAlpha);

  // result_hi = a1²
  Value resultHi = a1Sq;

  // Combine: result = result_lo | (result_hi << halfBits)
  Value resultLoExt = builder_.create<arith::ExtUIOp>(fullType, resultLo);
  Value resultHiExt = builder_.create<arith::ExtUIOp>(fullType, resultHi);
  Value resultHiShifted =
      builder_.create<arith::ShLIOp>(resultHiExt, halfBitsConst);
  return builder_.create<arith::OrIOp>(resultLoExt, resultHiShifted);
}

} // namespace mlir::prime_ir::field
