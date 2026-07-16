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
#include "mlir/IR/TypeUtilities.h"
#include "prime_ir/Dialect/Field/Conversions/BinaryFieldToArith/BinaryFieldTables.h"

namespace mlir::prime_ir::field {

namespace {

// Rebuild `like`'s shape (if any) around a new integer element type.
Type cloneWithElementType(Type like, IntegerType elementType) {
  if (auto shaped = dyn_cast<ShapedType>(like)) {
    return shaped.clone(elementType);
  }
  return elementType;
}

} // namespace

BinaryFieldCodeGen::BinaryFieldCodeGen(BinaryFieldType bfType, Value value,
                                       ImplicitLocOpBuilder &builder)
    : bfType_(bfType), value_(value), builder_(builder) {
  // Values arrive at the (byte-rounded) storage width; the tower algorithms
  // run at the element width.
  auto elementTy = IntegerType::get(bfType.getContext(), bfType.getBitWidth());
  auto valueTy = cast<IntegerType>(getElementTypeOrSelf(value.getType()));
  if (valueTy.getWidth() > elementTy.getWidth()) {
    value_ = arith::TruncIOp::create(
        builder, cloneWithElementType(value.getType(), elementTy), value);
  }
}

Value BinaryFieldCodeGen::getValue() const {
  // Widen back from the element width to the storage width.
  IntegerType storageTy = bfType_.getStorageType();
  auto valueTy = cast<IntegerType>(getElementTypeOrSelf(value_.getType()));
  if (valueTy.getWidth() < storageTy.getWidth()) {
    return arith::ExtUIOp::create(
        builder_, cloneWithElementType(value_.getType(), storageTy), value_);
  }
  return value_;
}

BinaryFieldCodeGen
BinaryFieldCodeGen::operator+(const BinaryFieldCodeGen &other) const {
  // In characteristic 2, addition is XOR
  Value result = arith::XOrIOp::create(builder_, value_, other.value_);
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

  // Levels ≥ 4: recursive tower descent. An inlined Fermat chain here is
  // (n−1) squares + (n−2) full Karatsuba muls and grows ~6× per level —
  // GF(2⁶⁴) is ~0.5M arith ops per scalar inverse, which blows up compile
  // time in every consumer pipeline. Descent is I(k) = I(k−1) + O(M(k−1)).
  if (towerLevel >= 4) {
    Value result = inverseTower(value_, towerLevel);
    return BinaryFieldCodeGen(bfType_, result, builder_);
  }

  // Levels ≤ 2 (n ≤ 4 bits): Fermat's little theorem, a⁻¹ = a^(2ⁿ - 2).
  // At these widths the chain is at most 3 squares + 2 muls — smaller than
  // the descent's constant overhead.
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

Value BinaryFieldCodeGen::inverseTower(Value a, unsigned towerLevel) const {
  // Base case: GF(2⁸) table lookup (0 → 0, matching binius invert_or_zero).
  if (towerLevel == 3) {
    return inverseLookupTable8b(a);
  }

  // Fan-Paar descent for a = a₀ + a₁·X over GF(2^(2ᵏ⁻¹)), X² = β·X + 1:
  // the Frobenius conjugate is ā = (a₀ + β·a₁) + a₁·X (X ↦ X + β, the other
  // root), and the norm N = a·ā = a₀² + β·a₀·a₁ + a₁² lies in the subfield
  // (its X coefficient cancels in characteristic 2). Then a⁻¹ = ā·N⁻¹, so
  // one level costs 3 muls + 2 squares + 2 β-muls + one recursive inverse.
  // a = 0 gives N = 0 and propagates 0 up from the base case.
  unsigned halfBits = 1u << (towerLevel - 1);
  IntegerType halfType = IntegerType::get(builder_.getContext(), halfBits);
  IntegerType fullType = IntegerType::get(builder_.getContext(), halfBits * 2);

  Value a0 = arith::TruncIOp::create(builder_, halfType, a);
  Value halfBitsConst = arith::ConstantIntOp::create(
      builder_, static_cast<int64_t>(halfBits), fullType.getWidth());
  Value a1 = arith::TruncIOp::create(
      builder_, halfType, arith::ShRUIOp::create(builder_, a, halfBitsConst));

  // conj_lo = a₀ + β·a₁ (the conjugate's X coefficient is a₁ unchanged).
  Value betaA1 = mulXTower(a1, towerLevel - 1);
  Value conjLo = arith::XOrIOp::create(builder_, a0, betaA1);

  // N = a₀² + β·(a₀·a₁) + a₁²
  Value a0a1 = mulTower(a0, a1, towerLevel - 1);
  Value betaA0a1 = mulXTower(a0a1, towerLevel - 1);
  Value a0Sq = squareTower(a0, towerLevel - 1);
  Value a1Sq = squareTower(a1, towerLevel - 1);
  Value norm = arith::XOrIOp::create(
      builder_, arith::XOrIOp::create(builder_, a0Sq, betaA0a1), a1Sq);

  Value normInv = inverseTower(norm, towerLevel - 1);

  // a⁻¹ = (a₀ + β·a₁)·N⁻¹ + a₁·N⁻¹·X
  Value resultLo = mulTower(conjLo, normInv, towerLevel - 1);
  Value resultHi = mulTower(a1, normInv, towerLevel - 1);

  Value resultLoExt = arith::ExtUIOp::create(builder_, fullType, resultLo);
  Value resultHiExt = arith::ExtUIOp::create(builder_, fullType, resultHi);
  Value resultHiShifted =
      arith::ShLIOp::create(builder_, resultHiExt, halfBitsConst);
  return arith::OrIOp::create(builder_, resultLoExt, resultHiShifted);
}

BinaryFieldCodeGen BinaryFieldCodeGen::inverseLookupTable() const {
  return BinaryFieldCodeGen(bfType_, inverseLookupTable8b(value_), builder_);
}

Value BinaryFieldCodeGen::inverseLookupTable8b(Value a) const {
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
      arith::IndexCastUIOp::create(builder_, builder_.getIndexType(), a);

  // Create switch with all 256 cases
  SmallVector<int64_t> caseValues;

  for (int i = 0; i < 256; ++i) {
    caseValues.push_back(i);
  }

  // Generate scf.index_switch
  auto switchOp =
      scf::IndexSwitchOp::create(builder_, TypeRange{builder_.getI8Type()},
                                 indexValue, caseValues, caseValues.size());

  // Generate case regions
  for (int i = 0; i < 256; ++i) {
    Region &caseRegion = switchOp.getCaseRegions()[i];
    Block *caseBlock = builder_.createBlock(&caseRegion);
    builder_.setInsertionPointToStart(caseBlock);
    Value inverseVal = arith::ConstantIntOp::create(
        builder_, kBinaryTower8bInverseTable[i], 8);
    scf::YieldOp::create(builder_, inverseVal);
  }

  // Generate default region (return 0 for safety, should never be reached)
  Region &defaultRegion = switchOp.getDefaultRegion();
  Block *defaultBlock = builder_.createBlock(&defaultRegion);
  builder_.setInsertionPointToStart(defaultBlock);
  Value defaultVal = arith::ConstantIntOp::create(builder_, 0, 8);
  scf::YieldOp::create(builder_, defaultVal);

  // Move insertion point back after switch
  builder_.setInsertionPointAfter(switchOp);

  return switchOp.getResult(0);
}

BinaryFieldCodeGen BinaryFieldCodeGen::constant(BinaryFieldType bfType,
                                                uint64_t val,
                                                ImplicitLocOpBuilder &builder) {
  Value c = arith::ConstantIntOp::create(builder, static_cast<int64_t>(val),
                                         bfType.getBitWidth());
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

Value BinaryFieldCodeGen::mulXTower(Value a, unsigned towerLevel) const {
  // Multiply by the generator Xₖ of GF(2^(2ᵏ)) = subfield[X]/(X² + βₖ₋₁·X + 1):
  //   (a₀ + a₁·X)·X = a₁ + (a₀ + βₖ₋₁·a₁)·X
  // where βₖ₋₁·(·) recurses as mulXTower one level down. Base: the generator of
  // GF(2) is 1, so multiply-by-generator is the identity.
  if (towerLevel == 0) {
    return a;
  }

  unsigned halfBits = 1u << (towerLevel - 1);
  IntegerType halfType = IntegerType::get(builder_.getContext(), halfBits);
  IntegerType fullType = IntegerType::get(builder_.getContext(), halfBits * 2);

  Value a0 = arith::TruncIOp::create(builder_, halfType, a);
  Value halfBitsConst = arith::ConstantIntOp::create(
      builder_, static_cast<int64_t>(halfBits), fullType.getWidth());
  Value a1 = arith::TruncIOp::create(
      builder_, halfType, arith::ShRUIOp::create(builder_, a, halfBitsConst));

  // lo = a₁,  hi = a₀ + βₖ₋₁·a₁
  Value resultLo = a1;
  Value resultHi =
      arith::XOrIOp::create(builder_, a0, mulXTower(a1, towerLevel - 1));

  Value resultLoExt = arith::ExtUIOp::create(builder_, fullType, resultLo);
  Value resultHiExt = arith::ExtUIOp::create(builder_, fullType, resultHi);
  Value resultHiShifted =
      arith::ShLIOp::create(builder_, resultHiExt, halfBitsConst);
  return arith::OrIOp::create(builder_, resultLoExt, resultHiShifted);
}

Value BinaryFieldCodeGen::mulTower(Value a, Value b,
                                   unsigned towerLevel) const {
  // Base case: tower level 0 is GF(2), multiplication is AND
  if (towerLevel == 0) {
    return arith::AndIOp::create(builder_, a, b);
  }

  // Recursive case: Karatsuba over the Fan-Paar tower. For GF(2^(2ᵏ)) with
  // a = a₀ + a₁*x, b = b₀ + b₁*x and reduction x² = βₖ₋₁*x + 1:
  //
  //   a*b = (a₀*b₀ + a₁*b₁) + (a₀*b₁ + a₁*b₀ + βₖ₋₁*a₁*b₁)*x
  //
  // Using Karatsuba products m₀ = a₀*b₀, m₁ = a₁*b₁, m₂ = (a₀+a₁)*(b₀+b₁):
  // result_lo = m₀ + m₁                    (constant term 1)
  // result_hi = (m₂ + m₀ + m₁) + βₖ₋₁*m₁   (m₂+m₀+m₁ = a₀b₁+a₁b₀; βₖ₋₁* = mulX)

  unsigned halfBits = 1u << (towerLevel - 1);
  IntegerType halfType = IntegerType::get(builder_.getContext(), halfBits);
  IntegerType fullType = IntegerType::get(builder_.getContext(), halfBits * 2);

  // Extract lower and upper halves
  Value a0 = arith::TruncIOp::create(builder_, halfType, a);
  Value b0 = arith::TruncIOp::create(builder_, halfType, b);

  Value halfBitsConst = arith::ConstantIntOp::create(
      builder_, static_cast<int64_t>(halfBits), fullType.getWidth());
  Value a1 = arith::TruncIOp::create(
      builder_, halfType, arith::ShRUIOp::create(builder_, a, halfBitsConst));
  Value b1 = arith::TruncIOp::create(
      builder_, halfType, arith::ShRUIOp::create(builder_, b, halfBitsConst));

  // Karatsuba products
  Value m0 = mulTower(a0, b0, towerLevel - 1);
  Value m1 = mulTower(a1, b1, towerLevel - 1);
  Value a0Xa1 = arith::XOrIOp::create(builder_, a0, a1);
  Value b0Xb1 = arith::XOrIOp::create(builder_, b0, b1);
  Value m2 = mulTower(a0Xa1, b0Xb1, towerLevel - 1);

  // result_lo = m₀ + m₁
  Value resultLo = arith::XOrIOp::create(builder_, m0, m1);

  // result_hi = (m₂ + m₀ + m₁) + βₖ₋₁·m₁
  Value m2Xm0 = arith::XOrIOp::create(builder_, m2, m0);
  Value crossTerm = arith::XOrIOp::create(builder_, m2Xm0, m1);
  Value resultHi =
      arith::XOrIOp::create(builder_, crossTerm, mulXTower(m1, towerLevel - 1));

  // Combine: result = result_lo | (result_hi << halfBits)
  Value resultLoExt = arith::ExtUIOp::create(builder_, fullType, resultLo);
  Value resultHiExt = arith::ExtUIOp::create(builder_, fullType, resultHi);
  Value resultHiShifted =
      arith::ShLIOp::create(builder_, resultHiExt, halfBitsConst);
  return arith::OrIOp::create(builder_, resultLoExt, resultHiShifted);
}

Value BinaryFieldCodeGen::squareTower(Value a, unsigned towerLevel) const {
  // Base case: tower level 0 is GF(2), squaring is identity
  if (towerLevel == 0) {
    return a;
  }

  // For GF(2^(2ᵏ)), element a = a₀ + a1*x where x² = βₖ₋₁*x + 1
  // a² = a₀² + a₁²*x² = a₀² + a₁²*(βₖ₋₁*x + 1)
  //     = (a₀² + a₁²) + βₖ₋₁*a₁²*x

  unsigned halfBits = 1u << (towerLevel - 1);
  IntegerType halfType = IntegerType::get(builder_.getContext(), halfBits);
  IntegerType fullType = IntegerType::get(builder_.getContext(), halfBits * 2);

  // Extract lower and upper halves
  Value a0 = arith::TruncIOp::create(builder_, halfType, a);
  Value halfBitsConst = arith::ConstantIntOp::create(
      builder_, static_cast<int64_t>(halfBits), fullType.getWidth());
  Value a1 = arith::TruncIOp::create(
      builder_, halfType, arith::ShRUIOp::create(builder_, a, halfBitsConst));

  // Recursive squaring
  Value a0Sq = squareTower(a0, towerLevel - 1);
  Value a1Sq = squareTower(a1, towerLevel - 1);

  // result_lo = a₀² + a₁²          (constant term 1)
  Value resultLo = arith::XOrIOp::create(builder_, a0Sq, a1Sq);

  // result_hi = βₖ₋₁·a₁²
  Value resultHi = mulXTower(a1Sq, towerLevel - 1);

  // Combine: result = result_lo | (result_hi << halfBits)
  Value resultLoExt = arith::ExtUIOp::create(builder_, fullType, resultLo);
  Value resultHiExt = arith::ExtUIOp::create(builder_, fullType, resultHi);
  Value resultHiShifted =
      arith::ShLIOp::create(builder_, resultHiExt, halfBitsConst);
  return arith::OrIOp::create(builder_, resultLoExt, resultHiShifted);
}

} // namespace mlir::prime_ir::field
