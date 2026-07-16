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

#ifndef PRIME_IR_DIALECT_FIELD_CONVERSIONS_BINARYFIELDTOARITH_BINARYFIELDCODEGEN_H_
#define PRIME_IR_DIALECT_FIELD_CONVERSIONS_BINARYFIELDTOARITH_BINARYFIELDCODEGEN_H_

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Value.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"

namespace mlir::prime_ir::field {

// Code generator for binary field (GF(2^n)) operations.
//
// Binary fields use tower field construction where GF(2^(2ᵏ)) is built
// recursively as degree-2 extensions over GF(2^(2ᵏ⁻¹)). This enables
// efficient Karatsuba-style multiplication.
//
// The operations are:
// - Addition: XOR (arith.xori)
// - Subtraction: XOR (same as addition in characteristic 2)
// - Negation: Identity (in characteristic 2, -a = a)
// - Doubling: Zero (2a = a + a = 0 in characteristic 2)
// - Multiplication: Karatsuba tower multiplication
// - Squaring: Optimized using Frobenius endomorphism
// - Inverse: recursive tower descent via the subfield norm (levels ≥ 4),
//   lookup table (level 3), Fermat's little theorem (levels ≤ 2)
class BinaryFieldCodeGen {
public:
  // `value` may arrive on the byte-rounded carrier type (see getCarrierType);
  // the constructor truncates it down to the logical storage width so the
  // tower algorithms always run on i(2^level).
  BinaryFieldCodeGen(BinaryFieldType bfType, Value value,
                     ImplicitLocOpBuilder &builder);

  // Get the underlying value, widened back to the byte-rounded carrier.
  Value getValue() const;

  // The integer type binary-field values are carried in at tensor/function
  // boundaries: the logical storage type byte-rounded up (i8 for levels
  // 0-2). XLA and zk_dtypes store sub-byte binary fields byte-per-element
  // (PRED-style), and the shared sub-byte tensor packing in consumers keys
  // on raw i2/i4 tensor element types — so those types must never appear at
  // a buffer boundary.
  static IntegerType getCarrierType(BinaryFieldType type);

  // Get the field type
  BinaryFieldType getType() const { return bfType_; }

  // Addition (XOR)
  BinaryFieldCodeGen operator+(const BinaryFieldCodeGen &other) const;

  // Subtraction (same as addition in char 2)
  BinaryFieldCodeGen operator-(const BinaryFieldCodeGen &other) const;

  // Negation (identity in char 2)
  BinaryFieldCodeGen operator-() const;

  // Multiplication using Karatsuba tower algorithm
  BinaryFieldCodeGen operator*(const BinaryFieldCodeGen &other) const;

  // Doubling (returns zero in char 2)
  BinaryFieldCodeGen dbl() const;

  // Squaring using optimized Frobenius-based algorithm
  BinaryFieldCodeGen square() const;

  // Multiplicative inverse using Fermat's little theorem
  BinaryFieldCodeGen inverse() const;

  // Create a constant
  static BinaryFieldCodeGen constant(BinaryFieldType bfType, uint64_t value,
                                     ImplicitLocOpBuilder &builder);

  // Create zero
  static BinaryFieldCodeGen zero(BinaryFieldType bfType,
                                 ImplicitLocOpBuilder &builder);

  // Create one
  static BinaryFieldCodeGen one(BinaryFieldType bfType,
                                ImplicitLocOpBuilder &builder);

private:
  BinaryFieldType bfType_;
  Value value_;
  ImplicitLocOpBuilder &builder_;

  // Recursive Karatsuba multiplication for tower level k
  Value mulTower(Value a, Value b, unsigned towerLevel) const;

  // Recursive squaring for tower level k
  Value squareTower(Value a, unsigned towerLevel) const;

  // Multiply a tower-level-k element by that level's generator βₖ (the root X
  // of the level-k defining polynomial). This is the Fan-Paar tower's
  // "multiply by βₖ₋₁" one level up: the linear coefficient of the reduction
  // X² + βₖ₋₁·X + 1 is a subfield element, not a stored constant, so it is
  // realized recursively rather than as a multiply by an α table entry.
  Value mulXTower(Value a, unsigned towerLevel) const;

  // Recursive tower-descent inverse for tower level k ≥ 3: one inverse a
  // level down plus a constant number of level-(k−1) muls/squares, so the
  // generated IR grows linearly in level instead of the ~6ᵏ of an inlined
  // Fermat chain. Bottoms out in the level-3 lookup table.
  Value inverseTower(Value a, unsigned towerLevel) const;

  // Lookup table-based inverse for bf<8> (tower level 3)
  // Uses O(1) table lookup instead of O(n) Fermat computation
  BinaryFieldCodeGen inverseLookupTable() const;

  // Same lookup on an explicit level-3 (i8) value — the descent base case.
  Value inverseLookupTable8b(Value a) const;
};

} // namespace mlir::prime_ir::field

// NOLINTNEXTLINE(whitespace/line_length)
#endif // PRIME_IR_DIALECT_FIELD_CONVERSIONS_BINARYFIELDTOARITH_BINARYFIELDCODEGEN_H_
