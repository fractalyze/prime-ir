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
// - Inverse: Fermat's little theorem: a⁻¹ = a^(2ⁿ - 2)
class BinaryFieldCodeGen {
public:
  BinaryFieldCodeGen(BinaryFieldType bfType, Value value,
                     ImplicitLocOpBuilder &builder);

  // Get the underlying value
  Value getValue() const { return value_; }

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

  // Get the α constant for tower level k
  // Alpha is the element satisfying x² + x + α = 0
  uint64_t getTowerAlpha(unsigned towerLevel) const;

  // Lookup table-based inverse for bf<8> (tower level 3)
  // Uses O(1) table lookup instead of O(n) Fermat computation
  BinaryFieldCodeGen inverseLookupTable() const;
};

} // namespace mlir::prime_ir::field

// NOLINTNEXTLINE(whitespace/line_length)
#endif // PRIME_IR_DIALECT_FIELD_CONVERSIONS_BINARYFIELDTOARITH_BINARYFIELDCODEGEN_H_
