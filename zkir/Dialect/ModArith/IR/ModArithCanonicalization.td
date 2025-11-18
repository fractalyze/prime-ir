/* Copyright 2025 The ZKIR Authors.

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

#ifndef ZKIR_MODARITH_IR_MODARITH_CANONICALIZATION_TD
#define ZKIR_MODARITH_IR_MODARITH_CANONICALIZATION_TD

include "mlir/IR/PatternBase.td"
include "zkir/Dialect/ModArith/IR/ModArithOps.td"

def IsZero : Constraint<CPred<"isEqualTo($0, 0)">>;
def IsOne : Constraint<CPred<"isEqualTo($0, 1)">>;
def IsTwo : Constraint<CPred<"isEqualTo($0, 2)">>;
def IsNegativeOne : Constraint<CPred<"isNegativeOf($0, $1, 1)">>;
def IsNegativeTwo : Constraint<CPred<"isNegativeOf($0, $1, 2)">>;
def IsNegativeThree : Constraint<CPred<"isNegativeOf($0, $1, 3)">>;
def IsNegativeFour : Constraint<CPred<"isNegativeOf($0, $1, 4)">>;
def GetAsConstant : NativeCodeCall<"$_builder.create<ConstantOp>(odsLoc, $0.getType(), cast<TypedAttr>($1))">;
def GetZeroAttr : NativeCodeCall<"$_builder.getZeroAttr(cast<IntegerType>($0))">;
def GetIntegerType : NativeCodeCall<"cast<ModArithType>($0.getType()).getStorageType()">;

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

// `AddOp` is commutative and will be canonicalized to have its constants appear
// as the second operand.

// (x + c0) + c1 -> x + (c0 + c1)
def AddConstantTwice :
    Pat<(ModArith_AddOp (ModArith_AddOp $x, (ConstantLikeMatcher TypedAttrInterface:$c0)), (ConstantLikeMatcher TypedAttrInterface:$c1)),
        (ModArith_AddOp $x, (ModArith_AddOp (GetAsConstant $x, $c0), (GetAsConstant $x, $c1)))>;

// (c0 - x) + c1 -> (c0 + c1) - x
def AddConstantToSubLhs :
    Pat<(ModArith_AddOp (ModArith_SubOp (ConstantLikeMatcher TypedAttrInterface:$c0), $x), (ConstantLikeMatcher TypedAttrInterface:$c1)),
        (ModArith_SubOp (ModArith_AddOp (GetAsConstant $x, $c0), (GetAsConstant $x, $c1)), $x)>;

// (x - c0) + c1 -> x + (c1 - c0)
def AddConstantToSubRhs :
    Pat<(ModArith_AddOp (ModArith_SubOp $x, (ConstantLikeMatcher TypedAttrInterface:$c0)), (ConstantLikeMatcher TypedAttrInterface:$c1)),
        (ModArith_AddOp $x, (ModArith_SubOp (GetAsConstant $x, $c1), (GetAsConstant $x, $c0)))>;

// x + x -> double(x)
def AddSelfIsDouble :
    Pat<(ModArith_AddOp $x, $x),
        (ModArith_DoubleOp $x)>;

// (-x) + (-y) -> -(x + y)
def AddBothNegated :
    Pat<(ModArith_AddOp (ModArith_NegateOp $x), (ModArith_NegateOp $y)),
        (ModArith_NegateOp (ModArith_AddOp $x, $y))>;

// x - y + y -> x
def AddAfterSub :
    Pat<(ModArith_AddOp (ModArith_SubOp $x, $y), $y),
        (replaceWithValue $x)>;

// (-x) + y -> y - x
def AddAfterNegLhs :
    Pat<(ModArith_AddOp (ModArith_NegateOp $x), $y),
        (ModArith_SubOp $y, $x)>;

// x + (-y) -> x - y
def AddAfterNegRhs :
    Pat<(ModArith_AddOp $x, (ModArith_NegateOp $y)),
        (ModArith_SubOp $x, $y)>;

// xy + xz -> x(y + z)
def FactorMulAdd :
    Pat<(ModArith_AddOp (ModArith_MulOp $x, $y), (ModArith_MulOp $x, $z)),
        (ModArith_MulOp $x, (ModArith_AddOp $y, $z))>;

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

// (x + c0) - c1 -> x + (c0 - c1)
def SubConstantFromAdd :
    Pat<(ModArith_SubOp (ModArith_AddOp $x, (ConstantLikeMatcher TypedAttrInterface:$c0)), (ConstantLikeMatcher TypedAttrInterface:$c1)),
        (ModArith_AddOp $x, (ModArith_SubOp (GetAsConstant $x, $c0), (GetAsConstant $x, $c1)))>;

// (c0 - x) - c1 -> (c0 - c1) - x
def SubConstantTwiceLhs :
    Pat<(ModArith_SubOp (ModArith_SubOp (ConstantLikeMatcher TypedAttrInterface:$c0), $x), (ConstantLikeMatcher TypedAttrInterface:$c1)),
        (ModArith_SubOp (ModArith_SubOp (GetAsConstant $x, $c0), (GetAsConstant $x, $c1)), $x)>;

// (x - c0) - c1 -> x - (c0 + c1)
def SubConstantTwiceRhs :
    Pat<(ModArith_SubOp (ModArith_SubOp $x, (ConstantLikeMatcher TypedAttrInterface:$c0)), (ConstantLikeMatcher TypedAttrInterface:$c1)),
        (ModArith_SubOp $x, (ModArith_AddOp (GetAsConstant $x, $c0), (GetAsConstant $x, $c1)))>;

// c0 - (x + c1) -> (c0 - c1) - x
def SubAddFromConstant :
    Pat<(ModArith_SubOp (ConstantLikeMatcher TypedAttrInterface:$c0), (ModArith_AddOp $x, (ConstantLikeMatcher TypedAttrInterface:$c1))),
        (ModArith_SubOp (ModArith_SubOp (GetAsConstant $x, $c0), (GetAsConstant $x, $c1)), $x)>;

// c0 - (c1 - x) -> x + (c0 - c1)
def SubSubFromConstantLhs :
    Pat<(ModArith_SubOp (ConstantLikeMatcher TypedAttrInterface:$c0), (ModArith_SubOp (ConstantLikeMatcher TypedAttrInterface:$c1), $x)),
        (ModArith_AddOp $x, (ModArith_SubOp (GetAsConstant $x, $c0), (GetAsConstant $x, $c1)))>;

// c0 - (x - c1) -> (c0 + c1) - x
def SubSubFromConstantRhs :
    Pat<(ModArith_SubOp (ConstantLikeMatcher TypedAttrInterface:$c0), (ModArith_SubOp $x, (ConstantLikeMatcher TypedAttrInterface:$c1))),
        (ModArith_SubOp (ModArith_AddOp (GetAsConstant $x, $c0), (GetAsConstant $x, $c1)), $x)>;

// x - x -> 0
def SubSelfIsZero :
    Pat<(ModArith_SubOp $x, $x),
        (GetAsConstant $x, (GetZeroAttr (GetIntegerType $x)))>;

// (x + y) - x -> y
def SubLhsAfterAdd :
    Pat<(ModArith_SubOp (ModArith_AddOp $x, $y), $x),
        (replaceWithValue $y)>;

// (x + y) - y -> x
def SubRhsAfterAdd :
    Pat<(ModArith_SubOp (ModArith_AddOp $x, $y), $y),
        (replaceWithValue $x)>;

// (x - y) - x -> -y
def SubLhsAfterSub :
    Pat<(ModArith_SubOp (ModArith_SubOp $x, $y), $x),
        (ModArith_NegateOp $y)>;

// (-x) - y -> -(x + y)
def SubAfterNegLhs :
    Pat<(ModArith_SubOp (ModArith_NegateOp $x), $y),
        (ModArith_NegateOp (ModArith_AddOp $x, $y))>;

// x - (-y) -> x + y
def SubAfterNegRhs :
    Pat<(ModArith_SubOp $x, (ModArith_NegateOp $y)),
        (ModArith_AddOp $x, $y)>;

// (-x) - (-y) -> y - x
def SubBothNegated :
    Pat<(ModArith_SubOp (ModArith_NegateOp $x), (ModArith_NegateOp $y)),
        (ModArith_SubOp $y, $x)>;

// x² - y² -> (x - y)(x + y)
def SubAfterSquareBoth :
    Pat<(ModArith_SubOp (ModArith_SquareOp $x), (ModArith_SquareOp $y)),
        (ModArith_MulOp (ModArith_SubOp $x, $y), (ModArith_AddOp $x, $y))>;

// (x + y)² - x² - y² -> 2xy
def SubAfterSumSquare :
    Pat<(ModArith_SubOp (ModArith_SubOp (ModArith_SquareOp (ModArith_AddOp $x, $y)), (ModArith_SquareOp $x)), (ModArith_SquareOp $y)),
        (ModArith_DoubleOp (ModArith_MulOp $x, $y))>;

// xy - xz -> x(y - z)
def FactorMulSub :
    Pat<(ModArith_SubOp (ModArith_MulOp $x, $y), (ModArith_MulOp $x, $z)),
        (ModArith_MulOp $x, (ModArith_SubOp $y, $z))>;

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

// `MulOp` is commutative and will be canonicalized to have its constants appear
// as the second operand.

// x * 2 -> double(x)
def MulByTwoIsDouble :
    Pat<(ModArith_MulOp $x, (ConstantLikeMatcher TypedAttrInterface:$c2)),
        (ModArith_DoubleOp $x),
        [(IsTwo $c2)]>;

// x * x -> square(x)
def MulSelfIsSquare :
    Pat<(ModArith_MulOp $x, $x),
        (ModArith_SquareOp $x)>;

// x * (-1) -> negate(x)
def MulNegativeOneRhs :
    Pat<(ModArith_MulOp $x, (ConstantLikeMatcher TypedAttrInterface:$c0)),
        (ModArith_NegateOp $x),
        [(IsNegativeOne $c0, $x)]>;

// x * (-2) -> negate(double(x))
def MulNegativeTwoRhs :
    Pat<(ModArith_MulOp $x, (ConstantLikeMatcher TypedAttrInterface:$c0)),
        (ModArith_NegateOp (ModArith_DoubleOp $x)),
        [(IsNegativeTwo $c0, $x)]>;

// x * (-3) -> negate(x + double(x))
def MulNegativeThreeRhs :
    Pat<(ModArith_MulOp $x, (ConstantLikeMatcher TypedAttrInterface:$c0)),
        (ModArith_NegateOp (ModArith_AddOp (ModArith_DoubleOp $x), $x)),
        [(IsNegativeThree $c0, $x)]>;

// x * (-4) -> negate(double(double(x)))
def MulNegativeFourRhs :
    Pat<(ModArith_MulOp $x, (ConstantLikeMatcher TypedAttrInterface:$c0)),
        (ModArith_NegateOp (ModArith_DoubleOp (ModArith_DoubleOp $x))),
        [(IsNegativeFour $c0, $x)]>;

// (x * c0) * c1 -> x * (c0 * c1)
def MulConstantTwice :
    Pat<(ModArith_MulOp (ModArith_MulOp $x, (ConstantLikeMatcher TypedAttrInterface:$c0)), (ConstantLikeMatcher TypedAttrInterface:$c1)),
        (ModArith_MulOp $x, (ModArith_MulOp (GetAsConstant $x, $c0), (GetAsConstant $x, $c1)))>;

// (x * c0) * (y * c1) -> (x * y) * (c0 * c1)
def MulOfMulByConstant :
    Pat<(ModArith_MulOp (ModArith_MulOp $x, (ConstantLikeMatcher TypedAttrInterface:$c0)), (ModArith_MulOp $y, (ConstantLikeMatcher TypedAttrInterface:$c1))),
        (ModArith_MulOp (ModArith_MulOp $x, $y), (ModArith_MulOp (GetAsConstant $x, $c0), (GetAsConstant $y, $c1)))>;

// (x + c0) * c1 -> x * c1 + c0 * c1
def MulAddDistributeConstant :
    Pat<(ModArith_MulOp (ModArith_AddOp $x, (ConstantLikeMatcher TypedAttrInterface:$c0)), (ConstantLikeMatcher TypedAttrInterface:$c1)),
        (ModArith_AddOp (ModArith_MulOp $x, (GetAsConstant $x, $c1)), (ModArith_MulOp (GetAsConstant $x, $c0), (GetAsConstant $x, $c1)))>;

// (x - c0) * c1 -> x * c1 - c0 * c1
def MulSubDistributeConstantRhs :
    Pat<(ModArith_MulOp (ModArith_SubOp $x, (ConstantLikeMatcher TypedAttrInterface:$c0)), (ConstantLikeMatcher TypedAttrInterface:$c1)),
        (ModArith_SubOp (ModArith_MulOp $x, (GetAsConstant $x, $c1)), (ModArith_MulOp (GetAsConstant $x, $c0), (GetAsConstant $x, $c1)))>;

// (c0 - x) * c1 -> c0 * c1 - x * c1
def MulSubDistributeConstantLhs :
    Pat<(ModArith_MulOp (ModArith_SubOp (ConstantLikeMatcher TypedAttrInterface:$c0), $x), (ConstantLikeMatcher TypedAttrInterface:$c1)),
        (ModArith_SubOp (ModArith_MulOp (GetAsConstant $x, $c0), (GetAsConstant $x, $c1)), (ModArith_MulOp $x, (GetAsConstant $x, $c1)))>;

#endif // ZKIR_MODARITH_IR_MODARITH_CANONICALIZATION_TD
