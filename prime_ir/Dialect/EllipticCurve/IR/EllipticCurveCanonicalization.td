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

#ifndef PRIME_IR_ELLIPTICCURVE_IR_ELLIPTICCURVE_CANONICALIZATION_TD
#define PRIME_IR_ELLIPTICCURVE_IR_ELLIPTICCURVE_CANONICALIZATION_TD

include "mlir/IR/PatternBase.td"
include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveOps.td"

def GetAsConstant : NativeCodeCall<"$_builder.create<FromCoordsOp>(odsLoc, $0.getType(), ValueRange{})">;
def GetZeroAttr : NativeCodeCall<"IntegerAttr::get(IntegerType::get($_builder.getContext(), 1), 0)">;
def GetIntegerType : NativeCodeCall<"IntegerType::get($_builder.getContext(), 1)">;

// Helper functions to create AddOp/SubOp with explicit result types.
// For elliptic curves, we need to specify the result type explicitly.
// These helpers use the result type from the original operation being matched.
// In the generated code, the lambda captures the surrounding scope with [&],
// so we can access `castedOp0` which is the original operation cast to AddOp/SubOp.
def CreateAddOp : NativeCodeCall<"[&]{ "
    "Type resultType = castedOp0.getType(); "
    "ImplicitLocOpBuilder b(odsLoc, $_builder); "
    "return b.create<AddOp>(resultType, $0, $1); "
    "}()">;
def CreateSubOp : NativeCodeCall<"[&]{ "
    "Type resultType = castedOp0.getType(); "
    "ImplicitLocOpBuilder b(odsLoc, $_builder); "
    "return b.create<SubOp>(resultType, $0, $1); "
    "}()">;

include "prime_ir/Utils/AdditiveCanonicalization.td"

defm EllipticCurve : AdditiveCanonicalizationPatterns<
    EllipticCurve_AddOp, EllipticCurve_SubOp, EllipticCurve_NegateOp, EllipticCurve_DoubleOp>;

#endif // PRIME_IR_ELLIPTICCURVE_IR_ELLIPTICCURVE_CANONICALIZATION_TD
