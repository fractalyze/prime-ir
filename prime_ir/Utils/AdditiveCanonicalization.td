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

#ifndef PRIME_IR_UTILS_ADDITIVE_CANONICALIZATION_TD
#define PRIME_IR_UTILS_ADDITIVE_CANONICALIZATION_TD

// The including file must define `GetAsConstant`.
// Assumptions for the patterns below:
// - AddOp/SubOp are associative and commutative where noted.
// - NegOp is the additive inverse (additive group structure).
// - Distributivity between MulOp and AddOp/SubOp when factoring.
multiclass AdditiveCanonicalizationPatterns<
    Op AddOp, Op SubOp, Op NegOp, Op MulOp, Op DoubleOp, Op SquareOp> {
    //===--------------------------------------------------------------------===//
    // AddOp
    //===--------------------------------------------------------------------===//

    // --- Associativity/commutativity ---
    // AddOp is commutative and associative; rewrite to place constants on the RHS.

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

    // --- Additive identity/duplication ---
    // x + x -> double(x)
    def AddSelfIsDouble :
        Pat<(AddOp $x, $x),
            (DoubleOp $x)>;

    // --- Additive inverse ---
    // (-x) + (-y) -> -(x + y)
    def AddBothNegated :
        Pat<(AddOp (NegOp $x), (NegOp $y)),
            (NegOp (AddOp $x, $y))>;

    // x - y + y -> x
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

    // --- Distributivity ---
    // xy + xz -> x(y + z)
    def FactorMulAdd :
        Pat<(AddOp (MulOp $x, $y), (MulOp $x, $z)),
            (MulOp $x, (AddOp $y, $z))>;

    //===--------------------------------------------------------------------===//
    // SubOp
    //===--------------------------------------------------------------------===//

    // --- Associativity/commutativity ---
    // AddOp is commutative and associative; rewrite to place constants on the RHS.
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

    // --- Additive inverse/identity ---

    // x - x -> 0
    def SubSelfIsZero :
        Pat<(SubOp $x, $x),
            (GetAsConstant $x, (GetZeroAttr (GetIntegerType $x)))>;

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

    // --- Distributivity ---
    // x² - y² -> (x - y)(x + y)
    def SubAfterSquareBoth :
        Pat<(SubOp (SquareOp $x), (SquareOp $y)),
            (MulOp (SubOp $x, $y), (AddOp $x, $y))>;

    // (x + y)² - x² - y² -> 2xy
    def SubAfterSumSquare :
        Pat<(SubOp (SubOp (SquareOp (AddOp $x, $y)), (SquareOp $x)),
                    (SquareOp $y)),
            (DoubleOp (MulOp $x, $y))>;

    // xy - xz -> x(y - z)
    def FactorMulSub :
        Pat<(SubOp (MulOp $x, $y), (MulOp $x, $z)),
            (MulOp $x, (SubOp $y, $z))>;
}

#endif // PRIME_IR_UTILS_ADDITIVE_CANONICALIZATION_TD
