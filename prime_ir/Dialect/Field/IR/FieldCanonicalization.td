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

#ifndef PRIME_IR_FIELD_IR_FIELD_CANONICALIZATION_TD
#define PRIME_IR_FIELD_IR_FIELD_CANONICALIZATION_TD

include "mlir/IR/PatternBase.td"
include "prime_ir/Dialect/Field/IR/FieldOps.td"

def IsZero : Constraint<CPred<"isEqualTo($0, $1, 0)">>;
def IsOne : Constraint<CPred<"isEqualTo($0, $1, 1)">>;
def IsTwo : Constraint<CPred<"isEqualTo($0, $1, 2)">>;
def IsNegativeOne : Constraint<CPred<"isNegativeOf($0, $1, 1)">>;
def IsNegativeTwo : Constraint<CPred<"isNegativeOf($0, $1, 2)">>;
def IsNegativeThree : Constraint<CPred<"isNegativeOf($0, $1, 3)">>;
def IsNegativeFour : Constraint<CPred<"isNegativeOf($0, $1, 4)">>;
def GetAsConstant : NativeCodeCall<"$_builder.create<ConstantOp>(odsLoc, $0.getType(), cast<TypedAttr>($1))">;
def GetZeroAttr : NativeCodeCall<"$_builder.getZeroAttr(cast<IntegerType>($0))">;
def GetIntegerType : NativeCodeCall<"cast<PrimeFieldType>($0.getType()).getStorageType()">;

include "prime_ir/Utils/AdditiveCanonicalization.td"
include "prime_ir/Utils/MultiplicativeCanonicalization.td"

defm Field : AdditiveCanonicalizationPatterns<
    Field_AddOp, Field_SubOp, Field_NegateOp,
    Field_MulOp, Field_DoubleOp, Field_SquareOp>;

defm Field : MultiplicativeCanonicalizationPatterns<
    Field_MulOp, Field_SquareOp, Field_DoubleOp, Field_NegateOp, Field_AddOp,
    Field_SubOp>;


#endif // PRIME_IR_FIELD_IR_FIELD_CANONICALIZATION_TD
