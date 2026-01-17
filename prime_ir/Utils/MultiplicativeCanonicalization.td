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

#ifndef PRIME_IR_UTILS_MULTIPLICATIVE_CANONICALIZATION_TD
#define PRIME_IR_UTILS_MULTIPLICATIVE_CANONICALIZATION_TD

//===----------------------------------------------------------------------===//
// MultiplicativeCanonicalizationPatterns
//===----------------------------------------------------------------------===//
//
// This multiclass provides common algebraic simplifications for multiplicative
// operations. It targets operations that satisfy associative and commutative
// properties (e.g., PrimeField, ExtensionField, and ModArith Integer).
//
// Requirements for the including file:
// 1. Define `GetAsConstant`: A helper to materialize a constant of the same
//    type as a given value.
// 2. Define Matchers: `IsTwo`, `IsNegativeOne`, `IsNegativeTwo`, and
//    `IsNegativeFour` must be defined to check constant values.
//
// Assumptions:
// - MulOp is commutative and associative.
// - SquareOp(x) is a specialized version of x * x (Strength Reduction).
// - DoubleOp(x) is a specialized version of x * 2.
//===----------------------------------------------------------------------===//
multiclass MultiplicativeCanonicalizationPatterns<Op MulOp, Op SquareOp, Op DoubleOp,
                                                  Op NegateOp> {
    //===--------------------------------------------------------------------===//
    // MulOp
    //===--------------------------------------------------------------------===//

    // --- Associativity/commutativity ---
    // MulOp is commutative and associative; rewrite to place constants on the RHS.

    // x * x -> square(x)
    def MulSelfIsSquare :
        Pat<(MulOp $x, $x),
            (SquareOp $x)>;

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

    // x * 2 -> double(x)
    def MulByTwoIsDouble :
        Pat<(MulOp $x, (ConstantLikeMatcher TypedAttrInterface:$c2)),
            (DoubleOp $x),
            [(IsTwo $c2, $x)]>;

    // x * (-1) -> negate(x)
    def MulNegativeOneRhs :
        Pat<(MulOp $x, (ConstantLikeMatcher TypedAttrInterface:$c0)),
            (NegateOp $x),
            [(IsNegativeOne $c0, $x)]>;

    // x * (-2) -> negate(double(x))
    def MulNegativeTwoRhs :
        Pat<(MulOp $x, (ConstantLikeMatcher TypedAttrInterface:$c0)),
            (NegateOp (DoubleOp $x)),
            [(IsNegativeTwo $c0, $x)]>;

    // x * (-4) -> negate(double(double(x)))
    def MulNegativeFourRhs :
        Pat<(MulOp $x, (ConstantLikeMatcher TypedAttrInterface:$c0)),
            (NegateOp (DoubleOp (DoubleOp $x))),
            [(IsNegativeFour $c0, $x)]>;

}

#endif // PRIME_IR_UTILS_MULTIPLICATIVE_CANONICALIZATION_TD
