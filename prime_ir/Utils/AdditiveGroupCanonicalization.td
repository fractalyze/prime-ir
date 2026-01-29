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

#ifndef PRIME_IR_UTILS_ADDITIVE_GROUP_CANONICALIZATION_TD
#define PRIME_IR_UTILS_ADDITIVE_GROUP_CANONICALIZATION_TD

//===----------------------------------------------------------------------===//
// AdditiveGroupCanonicalizationPatterns
//===----------------------------------------------------------------------===//
//
// Canonicalization patterns for additive group structures with scalar
// multiplication. This file is designed for Elliptic Curve point operations.
//
// An additive group with scalar multiplication has:
// - AddOp: commutative, associative binary operation (P + Q)
// - SubOp: derived operation (P - Q = P + (-Q))
// - NegOp: additive inverse (-P)
// - DoubleOp: derived operation (2P = P + P)
// - ScalarMulOp: scalar multiplication (s * P)
// - Zero: identity element (P + 0 = P)
//
// Requirements for the including file:
// 1. Define `CreateAddOp`, `CreateSubOp`, `CreateScalarMulOp`: NativeCodeCall
//    helpers that handle result type inference (needed for EC where result
//    type may differ from operand types).
// 2. Define `GetZeroPoint`: Creates the zero/identity point.
// 3. Define scalar constant matchers: `IsScalarZero`, `IsScalarOne`,
//    `IsScalarTwo`, `IsScalarNegOne`.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// AdditiveGroupCorePatterns
//===----------------------------------------------------------------------===//
//
// Pure structural patterns for additive group operations (Add, Sub, Neg, Double).
// These patterns don't involve constants or scalar multiplication.
//
//===----------------------------------------------------------------------===//
multiclass AdditiveGroupCorePatterns<Op AddOp, Op SubOp, Op NegOp> {
    //===--------------------------------------------------------------------===//
    // AddOp Core Patterns
    //===--------------------------------------------------------------------===//

    // P + P -> double(P)
    def AddSelfIsDouble :
        Pat<(AddOp $x, $x),
            (CreateDoubleOp $x)>;

    // (-P) + (-Q) -> -(P + Q)
    def AddBothNegated :
        Pat<(AddOp (NegOp $x), (NegOp $y)),
            (CreateNegOp (CreateAddOp $x, $y))>;

    // (P - Q) + Q -> P
    def AddAfterSub :
        Pat<(AddOp (SubOp $x, $y), $y),
            (replaceWithValue $x)>;

    // (-P) + Q -> Q - P
    def AddAfterNegLhs :
        Pat<(AddOp (NegOp $x), $y),
            (CreateSubOp $y, $x)>;

    // P + (-Q) -> P - Q
    def AddAfterNegRhs :
        Pat<(AddOp $x, (NegOp $y)),
            (CreateSubOp $x, $y)>;

    //===--------------------------------------------------------------------===//
    // SubOp Core Patterns
    //===--------------------------------------------------------------------===//

    // (P + Q) - P -> Q
    def SubLhsAfterAdd :
        Pat<(SubOp (AddOp $x, $y), $x),
            (replaceWithValue $y)>;

    // (P + Q) - Q -> P
    def SubRhsAfterAdd :
        Pat<(SubOp (AddOp $x, $y), $y),
            (replaceWithValue $x)>;

    // (P - Q) - P -> -Q
    def SubLhsAfterSub :
        Pat<(SubOp (SubOp $x, $y), $x),
            (CreateNegOp $y)>;

    // (-P) - Q -> -(P + Q)
    def SubAfterNegLhs :
        Pat<(SubOp (NegOp $x), $y),
            (CreateNegOp (CreateAddOp $x, $y))>;

    // P - (-Q) -> P + Q
    def SubAfterNegRhs :
        Pat<(SubOp $x, (NegOp $y)),
            (CreateAddOp $x, $y)>;

    // (-P) - (-Q) -> Q - P
    def SubBothNegated :
        Pat<(SubOp (NegOp $x), (NegOp $y)),
            (CreateSubOp $y, $x)>;
}

//===----------------------------------------------------------------------===//
// ScalarMulDistributivityPatterns
//===----------------------------------------------------------------------===//
//
// Patterns based on distributivity of scalar multiplication.
// These factor expressions to reduce the number of expensive scalar muls.
//
//===----------------------------------------------------------------------===//
multiclass ScalarMulDistributivityPatterns<
    Op ScalarMulOp, Op PointAddOp, Op PointSubOp> {

    // s*P + s*Q -> s * (P + Q)
    // Reduces two ScalarMuls to one, beneficial when ScalarMul >> PointAdd cost.
    def FactorScalarMulAdd :
        Pat<(PointAddOp (ScalarMulOp $s, $P), (ScalarMulOp $s, $Q)),
            (CreateScalarMulOp $s, (CreateAddOp $P, $Q))>;

    // s*P - s*Q -> s * (P - Q)
    def FactorScalarMulSub :
        Pat<(PointSubOp (ScalarMulOp $s, $P), (ScalarMulOp $s, $Q)),
            (CreateScalarMulOp $s, (CreateSubOp $P, $Q))>;
}

//===----------------------------------------------------------------------===//
// ScalarMulAssociativityPatterns
//===----------------------------------------------------------------------===//
//
// Patterns based on associativity: (s * t) * P = s * (t * P)
// These require scalar field multiplication.
//
//===----------------------------------------------------------------------===//
multiclass ScalarMulAssociativityPatterns<Op ScalarMulOp, Op ScalarFieldMulOp> {

    // s * (t * P) -> (s * t) * P
    // Reduces the number of point scalar multiplications.
    def ScalarMulAssocLeft :
        Pat<(ScalarMulOp $s, (ScalarMulOp $t, $P)),
            (ScalarMulOp (ScalarFieldMulOp $s, $t), $P)>;
}

#endif // PRIME_IR_UTILS_ADDITIVE_GROUP_CANONICALIZATION_TD
