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

#ifndef PRIME_IR_MODARITH_IR_MODARITH_CANONICALIZATION_TD
#define PRIME_IR_MODARITH_IR_MODARITH_CANONICALIZATION_TD

include "mlir/IR/PatternBase.td"
include "prime_ir/Dialect/ModArith/IR/ModArithOps.td"

//===----------------------------------------------------------------------===//
// Helper Definitions
//===----------------------------------------------------------------------===//

// Scalar constant matchers
def IsZero : Constraint<CPred<"isEqualTo($0, $1, 0)">>;
def IsOne : Constraint<CPred<"isEqualTo($0, $1, 1)">>;
def IsTwo : Constraint<CPred<"isEqualTo($0, $1, 2)">>;
def IsThree : Constraint<CPred<"isEqualTo($0, $1, 3)">>;
def IsFour : Constraint<CPred<"isEqualTo($0, $1, 4)">>;
def IsFive : Constraint<CPred<"isEqualTo($0, $1, 5)">>;
def IsSix : Constraint<CPred<"isEqualTo($0, $1, 6)">>;
def IsSeven : Constraint<CPred<"isEqualTo($0, $1, 7)">>;
def IsEight : Constraint<CPred<"isEqualTo($0, $1, 8)">>;
def IsNine : Constraint<CPred<"isEqualTo($0, $1, 9)">>;
def IsNegativeOne : Constraint<CPred<"isNegativeOf($0, $1, 1)">>;
def IsNegativeTwo : Constraint<CPred<"isNegativeOf($0, $1, 2)">>;
def IsNegativeThree : Constraint<CPred<"isNegativeOf($0, $1, 3)">>;
def IsNegativeFour : Constraint<CPred<"isNegativeOf($0, $1, 4)">>;
def IsNegativeFive : Constraint<CPred<"isNegativeOf($0, $1, 5)">>;
def IsNegativeSix : Constraint<CPred<"isNegativeOf($0, $1, 6)">>;
def IsNegativeSeven : Constraint<CPred<"isNegativeOf($0, $1, 7)">>;
def IsNegativeEight : Constraint<CPred<"isNegativeOf($0, $1, 8)">>;
def IsNegativeNine : Constraint<CPred<"isNegativeOf($0, $1, 9)">>;

// Constant materialization
def GetAsConstant : NativeCodeCall<"$_builder.create<ConstantOp>(odsLoc, $0.getType(), cast<TypedAttr>($1))">;
def GetZeroAttr : NativeCodeCall<"$_builder.getZeroAttr(cast<IntegerType>($0))">;
def GetIntegerType : NativeCodeCall<"cast<ModArithType>($0.getType()).getStorageType()">;

//===----------------------------------------------------------------------===//
// Include Canonicalization Pattern Files
//===----------------------------------------------------------------------===//

include "prime_ir/Utils/FieldCanonicalization.td"

//===----------------------------------------------------------------------===//
// Instantiate Patterns for ModArith Dialect
//===----------------------------------------------------------------------===//

// ModArith is a Ring (no multiplicative inverse), so we use:
// - Additive patterns (Add, Sub, Neg, Double)
// - Multiplicative patterns (Mul, Square) - no Inverse
// - Strength reduction and distributivity patterns

// 1. Additive patterns (Add, Sub, Neg, Double)
defm ModArith : FieldAdditiveCorePatterns<
    ModArith_AddOp, ModArith_SubOp, ModArith_NegateOp, ModArith_DoubleOp>;
defm ModArith : FieldAdditiveConstantPatterns<
    ModArith_AddOp, ModArith_SubOp>;

// 2. Multiplicative patterns (Mul, Square)
defm ModArith : FieldMultiplicativeCorePatterns<
    ModArith_MulOp, ModArith_SquareOp>;
defm ModArith : FieldMultiplicativeConstantPatterns<
    ModArith_MulOp>;

// Note: ModArith does not have InverseOp, so we skip FieldInversePatterns

// 3. Strength reduction patterns (mul by small constants -> double/neg)
defm ModArith : FieldStrengthReductionPatterns<
    ModArith_AddOp, ModArith_MulOp, ModArith_DoubleOp, ModArith_NegateOp>;

// 4. Distributivity patterns
defm ModArith : FieldDistributivityPatterns<
    ModArith_AddOp, ModArith_SubOp, ModArith_MulOp, ModArith_SquareOp, ModArith_DoubleOp>;

#endif // PRIME_IR_MODARITH_IR_MODARITH_CANONICALIZATION_TD
