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

#ifndef PRIME_IR_UTILS_DISTRIBUTIVITY_CANONICALIZATION_TD
#define PRIME_IR_UTILS_DISTRIBUTIVITY_CANONICALIZATION_TD

//===----------------------------------------------------------------------===//
// DistributivityCanonicalizationPatterns
//===----------------------------------------------------------------------===//
//
// This multiclass provides algebraic transformations that involve both
// additive and multiplicative group properties (Distributivity).
//
// Requirements for the including file:
// 1. Define `GetAsConstant`: A helper to materialize a constant of the same
//    type as a given value.
// 2. Define `IsNegativeThree`: A constraint to match the constant value -3.
//
// Assumptions:
// - Operations follow standard ring axioms (Distributivity of Mul over Add/Sub).
// - MulOp is commutative.
//===----------------------------------------------------------------------===//

multiclass DistributivityCanonicalizationPatterns<
    Op AddOp, Op SubOp, Op MulOp, Op SquareOp, Op DoubleOp, Op NegateOp> {

    //===--------------------------------------------------------------------===//
    // Factoring Patterns (Reverse Distributivity)
    //===--------------------------------------------------------------------===//
    // Goal: Reduce the number of operations by factoring out common terms.

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

    // x² - y² -> (x - y)(x + y)
    def SubAfterSquareBoth :
        Pat<(SubOp (SquareOp $x), (SquareOp $y)),
            (MulOp (SubOp $x, $y), (AddOp $x, $y))>;

    // (x + y)² - x² - y² -> 2xy
    def SubAfterSumSquare :
        Pat<(SubOp (SubOp (SquareOp (AddOp $x, $y)), (SquareOp $x)),
                        (SquareOp $y)),
            (DoubleOp (MulOp $x, $y))>;

    //===--------------------------------------------------------------------===//
    // Expansion Patterns (Constant Distribution)
    //===--------------------------------------------------------------------===//
    // Goal: Distribute constants to potentially enable further constant folding
    // down the expression tree.

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

    //===--------------------------------------------------------------------===//
    // Strength Reduction (Complex Constants)
    //===--------------------------------------------------------------------===//

    // x * (-3) -> negate(x + double(x))
    // Note: Defined here because it requires AddOp, unlike MultiplicativePatterns.
    def MulNegativeThreeRhs :
        Pat<(MulOp $x, (ConstantLikeMatcher TypedAttrInterface:$c0)),
            (NegateOp (AddOp (DoubleOp $x), $x)),
            [(IsNegativeThree $c0, $x)]>;
}

#endif // PRIME_IR_UTILS_DISTRIBUTIVITY_CANONICALIZATION_TD
