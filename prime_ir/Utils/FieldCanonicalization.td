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

#ifndef PRIME_IR_UTILS_FIELD_CANONICALIZATION_TD
#define PRIME_IR_UTILS_FIELD_CANONICALIZATION_TD

//===----------------------------------------------------------------------===//
// FieldCanonicalizationPatterns
//===----------------------------------------------------------------------===//
//
// Canonicalization patterns for field structures (PrimeField, ExtensionField,
// ModArith). A field has:
// - Additive group: AddOp, SubOp, NegOp, DoubleOp, Zero
// - Multiplicative group: MulOp, SquareOp, InverseOp, One
// - Distributivity: a * (b + c) = a*b + a*c
//
// These patterns use direct operation construction (e.g., AddOp instead of
// CreateAddOp) since field operations have SameOperandsAndResultType trait.
//
// Requirements for the including file:
// 1. Define `GetAsConstant`: A helper to materialize a constant of the same
//    type as a given value from an attribute.
// 2. Define constant matchers: `IsZero`, `IsOne`, `IsTwo`, `IsNegativeOne`,
//    `IsNegativeTwo`, `IsNegativeThree`, `IsNegativeFour`.
// 3. Define `GetZeroAttr` and `GetIntegerType` for the zero constant pattern.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// FieldAdditiveCorePatterns
//===----------------------------------------------------------------------===//
//
// Pure structural patterns for additive operations (Add, Sub, Neg, Double).
//
//===----------------------------------------------------------------------===//
multiclass FieldAdditiveCorePatterns<Op AddOp, Op SubOp, Op NegOp, Op DoubleOp> {
    //===--------------------------------------------------------------------===//
    // AddOp Core Patterns
    //===--------------------------------------------------------------------===//

    // x + x -> double(x)
    def AddSelfIsDouble :
        Pat<(AddOp $x, $x),
            (DoubleOp $x)>;

    // (-x) + (-y) -> -(x + y)
    def AddBothNegated :
        Pat<(AddOp (NegOp $x), (NegOp $y)),
            (NegOp (AddOp $x, $y))>;

    // (x - y) + y -> x
    def AddAfterSub :
        Pat<(AddOp (SubOp $x, $y), $y),
            (replaceWithValue $x)>;

    // (-x) + y -> y - x
    def AddAfterNegLhs :
        Pat<(AddOp (NegOp $x), $y),
            (SubOp $y, $x)>;

    // x + (-y) -> x - y
    def AddAfterNegRhs :
        Pat<(AddOp $x, (NegOp $y)),
            (SubOp $x, $y)>;

    //===--------------------------------------------------------------------===//
    // SubOp Core Patterns
    //===--------------------------------------------------------------------===//

    // (x + y) - x -> y
    def SubLhsAfterAdd :
        Pat<(SubOp (AddOp $x, $y), $x),
            (replaceWithValue $y)>;

    // (x + y) - y -> x
    def SubRhsAfterAdd :
        Pat<(SubOp (AddOp $x, $y), $y),
            (replaceWithValue $x)>;

    // (x - y) - x -> -y
    def SubLhsAfterSub :
        Pat<(SubOp (SubOp $x, $y), $x),
            (NegOp $y)>;

    // (-x) - y -> -(x + y)
    def SubAfterNegLhs :
        Pat<(SubOp (NegOp $x), $y),
            (NegOp (AddOp $x, $y))>;

    // x - (-y) -> x + y
    def SubAfterNegRhs :
        Pat<(SubOp $x, (NegOp $y)),
            (AddOp $x, $y)>;

    // (-x) - (-y) -> y - x
    def SubBothNegated :
        Pat<(SubOp (NegOp $x), (NegOp $y)),
            (SubOp $y, $x)>;
}

//===----------------------------------------------------------------------===//
// FieldAdditiveConstantPatterns
//===----------------------------------------------------------------------===//
//
// Patterns involving constant values for additive operations.
//
//===----------------------------------------------------------------------===//
multiclass FieldAdditiveConstantPatterns<Op AddOp, Op SubOp> {
    //===--------------------------------------------------------------------===//
    // AddOp Constant Patterns
    //===--------------------------------------------------------------------===//

    // (x + c0) + c1 -> x + (c0 + c1)
    def AddConstantTwice :
        Pat<(AddOp (AddOp $x, (ConstantLikeMatcher TypedAttrInterface:$c0)),
                    (ConstantLikeMatcher TypedAttrInterface:$c1)),
            (AddOp $x, (AddOp (GetAsConstant $x, $c0),
                              (GetAsConstant $x, $c1)))>;

    // (c0 - x) + c1 -> (c0 + c1) - x
    def AddConstantToSubLhs :
        Pat<(AddOp (SubOp (ConstantLikeMatcher TypedAttrInterface:$c0), $x),
                    (ConstantLikeMatcher TypedAttrInterface:$c1)),
            (SubOp (AddOp (GetAsConstant $x, $c0),
                          (GetAsConstant $x, $c1)),
                   $x)>;

    // (x - c0) + c1 -> x + (c1 - c0)
    def AddConstantToSubRhs :
        Pat<(AddOp (SubOp $x, (ConstantLikeMatcher TypedAttrInterface:$c0)),
                    (ConstantLikeMatcher TypedAttrInterface:$c1)),
            (AddOp $x,
                   (SubOp (GetAsConstant $x, $c1),
                          (GetAsConstant $x, $c0)))>;

    //===--------------------------------------------------------------------===//
    // SubOp Constant Patterns
    //===--------------------------------------------------------------------===//

    // (x + c0) - c1 -> x + (c0 - c1)
    def SubConstantFromAdd :
        Pat<(SubOp (AddOp $x, (ConstantLikeMatcher TypedAttrInterface:$c0)),
                    (ConstantLikeMatcher TypedAttrInterface:$c1)),
            (AddOp $x,
                   (SubOp (GetAsConstant $x, $c0),
                          (GetAsConstant $x, $c1)))>;

    // (c0 - x) - c1 -> (c0 - c1) - x
    def SubConstantTwiceLhs :
        Pat<(SubOp (SubOp (ConstantLikeMatcher TypedAttrInterface:$c0), $x),
                    (ConstantLikeMatcher TypedAttrInterface:$c1)),
            (SubOp (SubOp (GetAsConstant $x, $c0),
                          (GetAsConstant $x, $c1)),
                   $x)>;

    // (x - c0) - c1 -> x - (c0 + c1)
    def SubConstantTwiceRhs :
        Pat<(SubOp (SubOp $x, (ConstantLikeMatcher TypedAttrInterface:$c0)),
                    (ConstantLikeMatcher TypedAttrInterface:$c1)),
            (SubOp $x,
                   (AddOp (GetAsConstant $x, $c0),
                          (GetAsConstant $x, $c1)))>;

    // c0 - (x + c1) -> (c0 - c1) - x
    def SubAddFromConstant :
        Pat<(SubOp (ConstantLikeMatcher TypedAttrInterface:$c0),
                    (AddOp $x, (ConstantLikeMatcher TypedAttrInterface:$c1))),
            (SubOp (SubOp (GetAsConstant $x, $c0),
                          (GetAsConstant $x, $c1)),
                   $x)>;

    // c0 - (c1 - x) -> x + (c0 - c1)
    def SubSubFromConstantLhs :
        Pat<(SubOp (ConstantLikeMatcher TypedAttrInterface:$c0),
                    (SubOp (ConstantLikeMatcher TypedAttrInterface:$c1), $x)),
            (AddOp $x,
                   (SubOp (GetAsConstant $x, $c0),
                          (GetAsConstant $x, $c1)))>;

    // c0 - (x - c1) -> (c0 + c1) - x
    def SubSubFromConstantRhs :
        Pat<(SubOp (ConstantLikeMatcher TypedAttrInterface:$c0),
                    (SubOp $x, (ConstantLikeMatcher TypedAttrInterface:$c1))),
            (SubOp (AddOp (GetAsConstant $x, $c0),
                          (GetAsConstant $x, $c1)),
                   $x)>;

    // x - x -> 0
    def SubSelfIsZero :
        Pat<(SubOp $x, $x),
            (GetAsConstant $x, (GetZeroAttr (GetIntegerType $x)))>;
}

//===----------------------------------------------------------------------===//
// FieldMultiplicativeCorePatterns
//===----------------------------------------------------------------------===//
//
// Pure structural patterns for multiplicative operations (Mul, Square).
//
//===----------------------------------------------------------------------===//
multiclass FieldMultiplicativeCorePatterns<Op MulOp, Op SquareOp> {
    // x * x -> square(x)
    def MulSelfIsSquare :
        Pat<(MulOp $x, $x),
            (SquareOp $x)>;
}

//===----------------------------------------------------------------------===//
// FieldMultiplicativeConstantPatterns
//===----------------------------------------------------------------------===//
//
// Patterns involving constant values for multiplicative operations.
//
//===----------------------------------------------------------------------===//
multiclass FieldMultiplicativeConstantPatterns<Op MulOp> {
    // (x * c0) * c1 -> x * (c0 * c1)
    def MulConstantTwice :
        Pat<(MulOp (MulOp $x, (ConstantLikeMatcher TypedAttrInterface:$c0)),
                    (ConstantLikeMatcher TypedAttrInterface:$c1)),
            (MulOp $x,
                   (MulOp (GetAsConstant $x, $c0),
                          (GetAsConstant $x, $c1)))>;

    // (x * c0) * (y * c1) -> (x * y) * (c0 * c1)
    def MulOfMulByConstant :
        Pat<(MulOp (MulOp $x, (ConstantLikeMatcher TypedAttrInterface:$c0)),
                    (MulOp $y, (ConstantLikeMatcher TypedAttrInterface:$c1))),
            (MulOp (MulOp $x, $y),
                   (MulOp (GetAsConstant $x, $c0),
                          (GetAsConstant $y, $c1)))>;
}

//===----------------------------------------------------------------------===//
// FieldStrengthReductionPatterns
//===----------------------------------------------------------------------===//
//
// Patterns that replace multiplication by small constants with cheaper
// additive operations (DoubleOp, NegOp).
//
//===----------------------------------------------------------------------===//
multiclass FieldStrengthReductionPatterns<Op AddOp, Op MulOp, Op DoubleOp, Op NegOp> {
    // x * 2 -> double(x)
    def MulByTwoIsDouble :
        Pat<(MulOp $x, (ConstantLikeMatcher TypedAttrInterface:$c2)),
            (DoubleOp $x),
            [(IsTwo $c2, $x)]>;

    // x * (-1) -> -x
    def MulNegativeOneRhs :
        Pat<(MulOp $x, (ConstantLikeMatcher TypedAttrInterface:$c0)),
            (NegOp $x),
            [(IsNegativeOne $c0, $x)]>;

    // x * (-2) -> -double(x)
    def MulNegativeTwoRhs :
        Pat<(MulOp $x, (ConstantLikeMatcher TypedAttrInterface:$c0)),
            (NegOp (DoubleOp $x)),
            [(IsNegativeTwo $c0, $x)]>;

    // x * (-3) -> -(x + double(x))
    def MulNegativeThreeRhs :
        Pat<(MulOp $x, (ConstantLikeMatcher TypedAttrInterface:$c0)),
            (NegOp (AddOp (DoubleOp $x), $x)),
            [(IsNegativeThree $c0, $x)]>;

    // x * (-4) -> -double(double(x))
    def MulNegativeFourRhs :
        Pat<(MulOp $x, (ConstantLikeMatcher TypedAttrInterface:$c0)),
            (NegOp (DoubleOp (DoubleOp $x))),
            [(IsNegativeFour $c0, $x)]>;
}

//===----------------------------------------------------------------------===//
// FieldDistributivityPatterns
//===----------------------------------------------------------------------===//
//
// Patterns that exploit the distributivity of multiplication over addition.
//
//===----------------------------------------------------------------------===//
multiclass FieldDistributivityPatterns<Op AddOp, Op SubOp, Op MulOp, Op SquareOp, Op DoubleOp> {
    //===--------------------------------------------------------------------===//
    // Factoring Patterns (Reverse Distributivity)
    //===--------------------------------------------------------------------===//

    // xy + xz -> x(y + z)
    def FactorMulAdd :
        Pat<(AddOp (MulOp $x, $y), (MulOp $x, $z)),
            (MulOp $x, (AddOp $y, $z))>;

    // xy - xz -> x(y - z)
    def FactorMulSub :
        Pat<(SubOp (MulOp $x, $y), (MulOp $x, $z)),
            (MulOp $x, (SubOp $y, $z))>;

    //===--------------------------------------------------------------------===//
    // Quadratic Identities
    //===--------------------------------------------------------------------===//

    // x^2 - y^2 -> (x - y)(x + y)
    def SubAfterSquareBoth :
        Pat<(SubOp (SquareOp $x), (SquareOp $y)),
            (MulOp (SubOp $x, $y), (AddOp $x, $y))>;

    // (x + y)^2 - x^2 - y^2 -> 2xy
    def SubAfterSumSquare :
        Pat<(SubOp (SubOp (SquareOp (AddOp $x, $y)), (SquareOp $x)),
                   (SquareOp $y)),
            (DoubleOp (MulOp $x, $y))>;

    //===--------------------------------------------------------------------===//
    // Expansion Patterns (Constant Distribution)
    //===--------------------------------------------------------------------===//

    // (x + c0) * c1 -> x * c1 + c0 * c1
    def MulAddDistributeConstant :
        Pat<(MulOp (AddOp $x, (ConstantLikeMatcher TypedAttrInterface:$c0)),
                   (ConstantLikeMatcher TypedAttrInterface:$c1)),
            (AddOp (MulOp $x, (GetAsConstant $x, $c1)),
                   (MulOp (GetAsConstant $x, $c0), (GetAsConstant $x, $c1)))>;

    // (x - c0) * c1 -> x * c1 - c0 * c1
    def MulSubDistributeConstantRhs :
        Pat<(MulOp (SubOp $x, (ConstantLikeMatcher TypedAttrInterface:$c0)),
                   (ConstantLikeMatcher TypedAttrInterface:$c1)),
            (SubOp (MulOp $x, (GetAsConstant $x, $c1)),
                   (MulOp (GetAsConstant $x, $c0), (GetAsConstant $x, $c1)))>;

    // (c0 - x) * c1 -> c0 * c1 - x * c1
    def MulSubDistributeConstantLhs :
        Pat<(MulOp (SubOp (ConstantLikeMatcher TypedAttrInterface:$c0), $x),
                   (ConstantLikeMatcher TypedAttrInterface:$c1)),
            (SubOp (MulOp (GetAsConstant $x, $c0), (GetAsConstant $x, $c1)),
                   (MulOp $x, (GetAsConstant $x, $c1)))>;
}

#endif // PRIME_IR_UTILS_FIELD_CANONICALIZATION_TD
