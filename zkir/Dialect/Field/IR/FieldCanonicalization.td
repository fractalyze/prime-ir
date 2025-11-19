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

#ifndef ZKIR_FIELD_IR_FIELD_CANONICALIZATION_TD
#define ZKIR_FIELD_IR_FIELD_CANONICALIZATION_TD

include "mlir/IR/PatternBase.td"
include "zkir/Dialect/Field/IR/FieldOps.td"
include "zkir/Dialect/TensorExt/IR/TensorExtOps.td"

def GetAsConstant : NativeCodeCall<"$_builder.create<ConstantOp>(odsLoc, $0.getType(), cast<TypedAttr>($1))">;

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

// `AddOp` is commutative and will be canonicalized to have its constants appear
// as the second operand.

// (x + c0) + c1 -> x + (c0 + c1)
def AddConstantTwice :
    Pat<(Field_AddOp (Field_AddOp $x, (ConstantLikeMatcher TypedAttrInterface:$c0)), (ConstantLikeMatcher TypedAttrInterface:$c1)),
        (Field_AddOp $x, (Field_AddOp (GetAsConstant $x, $c0), (GetAsConstant $x, $c1)))>;

// (c0 - x) + c1 -> (c0 + c1) - x
def AddConstantToSubLhs :
    Pat<(Field_AddOp (Field_SubOp (ConstantLikeMatcher TypedAttrInterface:$c0), $x), (ConstantLikeMatcher TypedAttrInterface:$c1)),
        (Field_SubOp (Field_AddOp (GetAsConstant $x, $c0), (GetAsConstant $x, $c1)), $x)>;

// (x - c0) + c1 -> x + (c1 - c0)
def AddConstantToSubRhs :
    Pat<(Field_AddOp (Field_SubOp $x, (ConstantLikeMatcher TypedAttrInterface:$c0)), (ConstantLikeMatcher TypedAttrInterface:$c1)),
        (Field_AddOp $x, (Field_SubOp (GetAsConstant $x, $c1), (GetAsConstant $x, $c0)))>;

// x + x -> double(x)
def AddSelfIsDouble :
    Pat<(Field_AddOp $x, $x),
        (Field_DoubleOp $x)>;

// (-x) + (-y) -> -(x + y)
def AddBothNegated :
    Pat<(Field_AddOp (Field_NegateOp $x), (Field_NegateOp $y)),
        (Field_NegateOp (Field_AddOp $x, $y))>;

// x - y + y -> x
def AddAfterSub :
    Pat<(Field_AddOp (Field_SubOp $x, $y), $y),
        (replaceWithValue $x)>;

// (-x) + y -> y - x
def AddAfterNegLhs :
    Pat<(Field_AddOp (Field_NegateOp $x), $y),
        (Field_SubOp $y, $x)>;

// x + (-y) -> x - y
def AddAfterNegRhs :
    Pat<(Field_AddOp $x, (Field_NegateOp $y)),
        (Field_SubOp $x, $y)>;

// xy + xz -> x(y + z)
def FactorMulAdd :
    Pat<(Field_AddOp (Field_MulOp $x, $y), (Field_MulOp $x, $z)),
        (Field_MulOp $x, (Field_AddOp $y, $z))>;

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

// (x + c0) - c1 -> x + (c0 - c1)
def SubConstantFromAdd :
    Pat<(Field_SubOp (Field_AddOp $x, (ConstantLikeMatcher TypedAttrInterface:$c0)), (ConstantLikeMatcher TypedAttrInterface:$c1)),
        (Field_AddOp $x, (Field_SubOp (GetAsConstant $x, $c0), (GetAsConstant $x, $c1)))>;

// (c0 - x) - c1 -> (c0 - c1) - x
def SubConstantTwiceLhs :
    Pat<(Field_SubOp (Field_SubOp (ConstantLikeMatcher TypedAttrInterface:$c0), $x), (ConstantLikeMatcher TypedAttrInterface:$c1)),
        (Field_SubOp (Field_SubOp (GetAsConstant $x, $c0), (GetAsConstant $x, $c1)), $x)>;

// (x - c0) - c1 -> x - (c0 + c1)
def SubConstantTwiceRhs :
    Pat<(Field_SubOp (Field_SubOp $x, (ConstantLikeMatcher TypedAttrInterface:$c0)), (ConstantLikeMatcher TypedAttrInterface:$c1)),
        (Field_SubOp $x, (Field_AddOp (GetAsConstant $x, $c0), (GetAsConstant $x, $c1)))>;

// c0 - (x + c1) -> (c0 - c1) - x
def SubAddFromConstant :
    Pat<(Field_SubOp (ConstantLikeMatcher TypedAttrInterface:$c0), (Field_AddOp $x, (ConstantLikeMatcher TypedAttrInterface:$c1))),
        (Field_SubOp (Field_SubOp (GetAsConstant $x, $c0), (GetAsConstant $x, $c1)), $x)>;

// c0 - (c1 - x) -> x + (c0 - c1)
def SubSubFromConstantLhs :
    Pat<(Field_SubOp (ConstantLikeMatcher TypedAttrInterface:$c0), (Field_SubOp (ConstantLikeMatcher TypedAttrInterface:$c1), $x)),
        (Field_AddOp $x, (Field_SubOp (GetAsConstant $x, $c0), (GetAsConstant $x, $c1)))>;

// c0 - (x - c1) -> (c0 + c1) - x
def SubSubFromConstantRhs :
    Pat<(Field_SubOp (ConstantLikeMatcher TypedAttrInterface:$c0), (Field_SubOp $x, (ConstantLikeMatcher TypedAttrInterface:$c1))),
        (Field_SubOp (Field_AddOp (GetAsConstant $x, $c0), (GetAsConstant $x, $c1)), $x)>;

// (x + y) - x -> y
def SubLhsAfterAdd :
    Pat<(Field_SubOp (Field_AddOp $x, $y), $x),
        (replaceWithValue $y)>;

// (x + y) - y -> x
def SubRhsAfterAdd :
    Pat<(Field_SubOp (Field_AddOp $x, $y), $y),
        (replaceWithValue $x)>;

// (x - y) - x -> -y
def SubLhsAfterSub :
    Pat<(Field_SubOp (Field_SubOp $x, $y), $x),
        (Field_NegateOp $y)>;

// (-x) - y -> -(x + y)
def SubAfterNegLhs :
    Pat<(Field_SubOp (Field_NegateOp $x), $y),
        (Field_NegateOp (Field_AddOp $x, $y))>;

// x - (-y) -> x + y
def SubAfterNegRhs :
    Pat<(Field_SubOp $x, (Field_NegateOp $y)),
        (Field_AddOp $x, $y)>;

// (-x) - (-y) -> y - x
def SubBothNegated :
    Pat<(Field_SubOp (Field_NegateOp $x), (Field_NegateOp $y)),
        (Field_SubOp $y, $x)>;

// x² - y² -> (x - y)(x + y)
def SubAfterSquareBoth :
    Pat<(Field_SubOp (Field_SquareOp $x), (Field_SquareOp $y)),
        (Field_MulOp (Field_SubOp $x, $y), (Field_AddOp $x, $y))>;

// (x + y)² - x² - y² -> 2xy
def SubAfterSumSquare :
    Pat<(Field_SubOp (Field_SubOp (Field_SquareOp (Field_AddOp $x, $y)), (Field_SquareOp $x)), (Field_SquareOp $y)),
        (Field_DoubleOp (Field_MulOp $x, $y))>;

// xy - xz -> x(y - z)
def FactorMulSub :
    Pat<(Field_SubOp (Field_MulOp $x, $y), (Field_MulOp $x, $z)),
        (Field_MulOp $x, (Field_SubOp $y, $z))>;

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

// `MulOp` is commutative and will be canonicalized to have its constants appear
// as the second operand.

// x * x -> square(x)
def MulSelfIsSquare :
    Pat<(Field_MulOp $x, $x),
        (Field_SquareOp $x)>;

// (x * c0) * c1 -> x * (c0 * c1)
def MulConstantTwice :
    Pat<(Field_MulOp (Field_MulOp $x, (ConstantLikeMatcher TypedAttrInterface:$c0)), (ConstantLikeMatcher TypedAttrInterface:$c1)),
        (Field_MulOp $x, (Field_MulOp (GetAsConstant $x, $c0), (GetAsConstant $x, $c1)))>;

// (x * c0) * (y * c1) -> (x * y) * (c0 * c1)
def MulOfMulByConstant :
    Pat<(Field_MulOp (Field_MulOp $x, (ConstantLikeMatcher TypedAttrInterface:$c0)), (Field_MulOp $y, (ConstantLikeMatcher TypedAttrInterface:$c1))),
        (Field_MulOp (Field_MulOp $x, $y), (Field_MulOp (GetAsConstant $x, $c0), (GetAsConstant $y, $c1)))>;

// bit_reverse(mul(bit_reverse(x), y)) -> mul(x, bit_reverse(y))
def BitReverseMulBitReverse :
    Pat<(TensorExt_BitReverseOp (Field_MulOp (TensorExt_BitReverseOp $x, $dst0), $y), $dst1),
        (Field_MulOp $x, (TensorExt_BitReverseOp $y, $y))>;

#endif // ZKIR_FIELD_IR_FIELD_CANONICALIZATION_TD
