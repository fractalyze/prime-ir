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
include "prime_ir/Dialect/Field/IR/FieldOps.td"

//===----------------------------------------------------------------------===//
// Helper Definitions for Point Operations
//===----------------------------------------------------------------------===//

// Helper functions to create operations with explicit result types.
// For elliptic curves, we need to specify the result type explicitly since
// the result type may differ from operand types (e.g., affine + affine -> jacobian).
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
def CreateScalarMulOp : NativeCodeCall<"[&]{ "
    "Type resultType = castedOp0.getType(); "
    "ImplicitLocOpBuilder b(odsLoc, $_builder); "
    "return b.create<ScalarMulOp>(resultType, $0, $1); "
    "}()">;
def CreateNegOp : NativeCodeCall<"[&]{ "
    "Type resultType = castedOp0.getType(); "
    "ImplicitLocOpBuilder b(odsLoc, $_builder); "
    "return b.create<NegateOp>(resultType, $0); "
    "}()">;
def CreateDoubleOp : NativeCodeCall<"[&]{ "
    "Type resultType = castedOp0.getType(); "
    "ImplicitLocOpBuilder b(odsLoc, $_builder); "
    "return b.create<DoubleOp>(resultType, $0); "
    "}()">;

//===----------------------------------------------------------------------===//
// Include Canonicalization Pattern Files
//===----------------------------------------------------------------------===//

include "prime_ir/Utils/AdditiveGroupCanonicalization.td"

//===----------------------------------------------------------------------===//
// Instantiate Patterns for Elliptic Curve Dialect
//===----------------------------------------------------------------------===//

// Elliptic Curve points form an additive group with scalar multiplication.

// 1. Additive Group Core Patterns (Add, Sub, Double)
defm EllipticCurve : AdditiveGroupCorePatterns<
    EllipticCurve_AddOp, EllipticCurve_SubOp, EllipticCurve_NegateOp>;

// 2. Scalar Multiplication Distributivity (factoring)
//    s*P + s*Q -> s*(P + Q), s*P - s*Q -> s*(P - Q)
defm EllipticCurve : ScalarMulDistributivityPatterns<
    EllipticCurve_ScalarMulOp, EllipticCurve_AddOp, EllipticCurve_SubOp>;

// 3. Scalar Multiplication Associativity
//    s * (t * P) -> (s * t) * P
defm EllipticCurve : ScalarMulAssociativityPatterns<
    EllipticCurve_ScalarMulOp, Field_MulOp>;

#endif // PRIME_IR_ELLIPTICCURVE_IR_ELLIPTICCURVE_CANONICALIZATION_TD
