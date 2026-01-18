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

//===----------------------------------------------------------------------===//
// AdditiveCanonicalizationPatterns
//===----------------------------------------------------------------------===//
//
// This multiclass provides common algebraic simplifications for types that
// satisfy additive group properties (e.g., Elliptic Curve Point, PrimeField,
// and ModArith Integer).
//
// Requirements for the including file:
// 1. Define `GetAsConstant`: A helper to materialize a constant of the same
//    type as a given value.
// 2. Define `CreateAddOp` / `CreateSubOp`: NativeCodeCall helpers.
//    - For operations with 'SameOperandsAndResultType', use:
//      def CreateAddOp : NativeCodeCall<"$_builder.create<AddOp>(odsLoc, $0, $1)">;
//    - For operations that require explicit result types, use the original
//      operation's result type. The helper should accept an optional third
//      parameter for the result type, or infer it from the operands.
//      Example: def CreateAddOp : NativeCodeCall<"...">; (see implementations)
//
// Assumptions:
// - AddOp is commutative and associative.
// - NegOp is the additive inverse.
// - DoubleOp(x) is equivalent to AddOp(x, x).
//===----------------------------------------------------------------------===//
multiclass AdditiveCanonicalizationPatterns<
    Op AddOp, Op SubOp, Op NegOp, Op DoubleOp> {
    //===--------------------------------------------------------------------===//
    // AddOp
    //===--------------------------------------------------------------------===//

    // --- Associativity/commutativity ---
    // AddOp is commutative and associative; rewrite to place constants on the RHS.

    // (x + c0) + c1 -> x + (c0 + c1)
    def AddConstantTwice :
        Pat<(AddOp (AddOp $x, (ConstantLikeMatcher TypedAttrInterface:$c0)),
                    (ConstantLikeMatcher TypedAttrInterface:$c1)),
            (CreateAddOp $x, (CreateAddOp (GetAsConstant $x, $c0),
                                (GetAsConstant $x, $c1)))>;

    // (c0 - x) + c1 -> (c0 + c1) - x
    def AddConstantToSubLhs :
        Pat<(AddOp (SubOp (ConstantLikeMatcher TypedAttrInterface:$c0), $x),
                    (ConstantLikeMatcher TypedAttrInterface:$c1)),
            (CreateSubOp (CreateAddOp (GetAsConstant $x, $c0),
                            (GetAsConstant $x, $c1)),
                    $x)>;

    // (x - c0) + c1 -> x + (c1 - c0)
    def AddConstantToSubRhs :
        Pat<(AddOp (SubOp $x, (ConstantLikeMatcher TypedAttrInterface:$c0)),
                    (ConstantLikeMatcher TypedAttrInterface:$c1)),
            (CreateAddOp $x,
                    (CreateSubOp (GetAsConstant $x, $c1),
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
            (NegOp (CreateAddOp $x, $y))>;

    // x - y + y -> x
    def AddAfterSub :
        Pat<(AddOp (SubOp $x, $y), $y),
            (replaceWithValue $x)>;

    // (-x) + y -> y - x
    def AddAfterNegLhs :
        Pat<(AddOp (NegOp $x), $y),
            (CreateSubOp $y, $x)>;

    // x + (-y) -> x - y
    def AddAfterNegRhs :
        Pat<(AddOp $x, (NegOp $y)),
            (CreateSubOp $x, $y)>;

    //===--------------------------------------------------------------------===//
    // SubOp
    //===--------------------------------------------------------------------===//

    // --- Associativity/commutativity ---
    // AddOp is commutative and associative; rewrite to place constants on the RHS.
    // (x + c0) - c1 -> x + (c0 - c1)
    def SubConstantFromAdd :
        Pat<(SubOp (AddOp $x, (ConstantLikeMatcher TypedAttrInterface:$c0)),
                    (ConstantLikeMatcher TypedAttrInterface:$c1)),
            (CreateAddOp $x,
                    (CreateSubOp (GetAsConstant $x, $c0),
                            (GetAsConstant $x, $c1)))>;

    // (c0 - x) - c1 -> (c0 - c1) - x
    def SubConstantTwiceLhs :
        Pat<(SubOp (SubOp (ConstantLikeMatcher TypedAttrInterface:$c0), $x),
                    (ConstantLikeMatcher TypedAttrInterface:$c1)),
            (CreateSubOp (CreateSubOp (GetAsConstant $x, $c0),
                            (GetAsConstant $x, $c1)),
                    $x)>;

    // (x - c0) - c1 -> x - (c0 + c1)
    def SubConstantTwiceRhs :
        Pat<(SubOp (SubOp $x, (ConstantLikeMatcher TypedAttrInterface:$c0)),
                    (ConstantLikeMatcher TypedAttrInterface:$c1)),
            (CreateSubOp $x,
                    (CreateAddOp (GetAsConstant $x, $c0),
                            (GetAsConstant $x, $c1)))>;

    // c0 - (x + c1) -> (c0 - c1) - x
    def SubAddFromConstant :
        Pat<(SubOp (ConstantLikeMatcher TypedAttrInterface:$c0),
                    (AddOp $x, (ConstantLikeMatcher TypedAttrInterface:$c1))),
            (CreateSubOp (CreateSubOp (GetAsConstant $x, $c0),
                            (GetAsConstant $x, $c1)),
                    $x)>;

    // c0 - (c1 - x) -> x + (c0 - c1)
    def SubSubFromConstantLhs :
        Pat<(SubOp (ConstantLikeMatcher TypedAttrInterface:$c0),
                    (SubOp (ConstantLikeMatcher TypedAttrInterface:$c1), $x)),
            (CreateAddOp $x,
                    (CreateSubOp (GetAsConstant $x, $c0),
                            (GetAsConstant $x, $c1)))>;

    // c0 - (x - c1) -> (c0 + c1) - x
    def SubSubFromConstantRhs :
        Pat<(SubOp (ConstantLikeMatcher TypedAttrInterface:$c0),
                    (SubOp $x, (ConstantLikeMatcher TypedAttrInterface:$c1))),
            (CreateSubOp (CreateAddOp (GetAsConstant $x, $c0),
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
            (NegOp (CreateAddOp $x, $y))>;

    // x - (-y) -> x + y
    def SubAfterNegRhs :
        Pat<(SubOp $x, (NegOp $y)),
            (CreateAddOp $x, $y)>;

    // (-x) - (-y) -> y - x
    def SubBothNegated :
        Pat<(SubOp (NegOp $x), (NegOp $y)),
            (CreateSubOp $y, $x)>;
}

#endif // PRIME_IR_UTILS_ADDITIVE_CANONICALIZATION_TD
