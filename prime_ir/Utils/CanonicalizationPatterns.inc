//===----------------------------------------------------------------------===//
// Pattern Suffix List for Canonicalization
//===----------------------------------------------------------------------===//
//
// This file defines macros that simplify the registration of canonicalization
// patterns. Patterns are organized by their algebraic structure:
//
// 1. ADDITIVE_GROUP: Patterns for structures supporting Add, Sub, Neg,
//    Double, and ScalarMul. Ideal for Elliptic Curve (EC) points.
//
// 2. FIELD: Patterns for Field/Ring structures supporting full arithmetic,
//    including Multiplicative and Distributive identities.
//
// Usage:
//   #include "prime_ir/Utils/CanonicalizationPatterns.inc"
//
//   // Define a helper to instantiate the pattern class
//   #define PRIME_IR_ADD_PATTERN(Name) patterns.add<Prefix##Name>(context);
//
//   // Register all additive group patterns for a specific dialect/op
//   PRIME_IR_ADDITIVE_GROUP_ADD_PATTERN_LIST(PRIME_IR_ADD_PATTERN)
//
//   #undef PRIME_IR_ADD_PATTERN
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// 1. Additive Group Patterns (for Elliptic Curve points)
//===----------------------------------------------------------------------===//

// AddOp patterns
#define PRIME_IR_ADDITIVE_GROUP_ADD_CORE_PATTERN_LIST(X) \
  X(AddSelfIsDouble)                                     \
  X(AddBothNegated)                                      \
  X(AddAfterSub)                                         \
  X(AddAfterNegLhs)                                      \
  X(AddAfterNegRhs)

#define PRIME_IR_ADDITIVE_GROUP_ADD_DISTRIBUTIVITY_PATTERN_LIST(X) \
  X(FactorScalarMulAdd)

#define PRIME_IR_ADDITIVE_GROUP_ADD_PATTERN_LIST(X)      \
  PRIME_IR_ADDITIVE_GROUP_ADD_CORE_PATTERN_LIST(X)       \
  PRIME_IR_ADDITIVE_GROUP_ADD_DISTRIBUTIVITY_PATTERN_LIST(X)

// SubOp patterns
#define PRIME_IR_ADDITIVE_GROUP_SUB_CORE_PATTERN_LIST(X) \
  X(SubLhsAfterAdd)                                      \
  X(SubRhsAfterAdd)                                      \
  X(SubLhsAfterSub)                                      \
  X(SubAfterNegLhs)                                      \
  X(SubAfterNegRhs)                                      \
  X(SubBothNegated)

#define PRIME_IR_ADDITIVE_GROUP_SUB_DISTRIBUTIVITY_PATTERN_LIST(X) \
  X(FactorScalarMulSub)

#define PRIME_IR_ADDITIVE_GROUP_SUB_PATTERN_LIST(X)      \
  PRIME_IR_ADDITIVE_GROUP_SUB_CORE_PATTERN_LIST(X)       \
  PRIME_IR_ADDITIVE_GROUP_SUB_DISTRIBUTIVITY_PATTERN_LIST(X)

// ScalarMulOp patterns
#define PRIME_IR_SCALARMUL_STRENGTH_REDUCTION_PATTERN_LIST(X) \
  X(ScalarMulByOne)                                           \
  X(ScalarMulByZero)                                          \
  X(ScalarMulByTwo)                                           \
  X(ScalarMulByNegOne)

#define PRIME_IR_SCALARMUL_DISTRIBUTIVITY_PATTERN_LIST(X) \
  X(FactorScalarMulAdd)                                   \
  X(FactorScalarMulSub)

#define PRIME_IR_SCALARMUL_ASSOCIATIVITY_PATTERN_LIST(X) \
  X(ScalarMulAssocLeft)

#define PRIME_IR_ADDITIVE_GROUP_SCALARMUL_PATTERN_LIST(X) \
  PRIME_IR_SCALARMUL_STRENGTH_REDUCTION_PATTERN_LIST(X)   \
  PRIME_IR_SCALARMUL_ASSOCIATIVITY_PATTERN_LIST(X)

//===----------------------------------------------------------------------===//
// 2. Field Patterns (for Field/ModArith)
//===----------------------------------------------------------------------===//

// AddOp patterns
#define PRIME_IR_FIELD_ADD_CORE_PATTERN_LIST(X) \
  X(AddSelfIsDouble)                            \
  X(AddBothNegated)                             \
  X(AddAfterSub)                                \
  X(AddAfterNegLhs)                             \
  X(AddAfterNegRhs)

#define PRIME_IR_FIELD_ADD_CONSTANT_PATTERN_LIST(X) \
  X(AddConstantTwice)                               \
  X(AddConstantToSubLhs)                            \
  X(AddConstantToSubRhs)

#define PRIME_IR_FIELD_ADD_DISTRIBUTIVITY_PATTERN_LIST(X) \
  X(FactorMulAdd)

#define PRIME_IR_FIELD_ADD_PATTERN_LIST(X)        \
  PRIME_IR_FIELD_ADD_CORE_PATTERN_LIST(X)         \
  PRIME_IR_FIELD_ADD_CONSTANT_PATTERN_LIST(X)     \
  PRIME_IR_FIELD_ADD_DISTRIBUTIVITY_PATTERN_LIST(X)

// SubOp patterns
#define PRIME_IR_FIELD_SUB_CORE_PATTERN_LIST(X) \
  X(SubLhsAfterAdd)                             \
  X(SubRhsAfterAdd)                             \
  X(SubLhsAfterSub)                             \
  X(SubAfterNegLhs)                             \
  X(SubAfterNegRhs)                             \
  X(SubBothNegated)

#define PRIME_IR_FIELD_SUB_CONSTANT_PATTERN_LIST(X) \
  X(SubConstantFromAdd)                             \
  X(SubConstantTwiceLhs)                            \
  X(SubConstantTwiceRhs)                            \
  X(SubAddFromConstant)                             \
  X(SubSubFromConstantLhs)                          \
  X(SubSubFromConstantRhs)                          \
  X(SubSelfIsZero)

#define PRIME_IR_FIELD_SUB_DISTRIBUTIVITY_PATTERN_LIST(X) \
  X(FactorMulSub)                                         \
  X(SubAfterSquareBoth)                                   \
  X(SubAfterSumSquare)

#define PRIME_IR_FIELD_SUB_PATTERN_LIST(X)         \
  PRIME_IR_FIELD_SUB_CORE_PATTERN_LIST(X)          \
  PRIME_IR_FIELD_SUB_CONSTANT_PATTERN_LIST(X)      \
  PRIME_IR_FIELD_SUB_DISTRIBUTIVITY_PATTERN_LIST(X)

// MulOp patterns
#define PRIME_IR_FIELD_MUL_CORE_PATTERN_LIST(X) \
  X(MulSelfIsSquare)

#define PRIME_IR_FIELD_MUL_CONSTANT_PATTERN_LIST(X) \
  X(MulConstantTwice)                               \
  X(MulOfMulByConstant)

#define PRIME_IR_FIELD_MUL_STRENGTH_REDUCTION_PATTERN_LIST(X) \
  X(MulByTwoIsDouble)                                         \
  X(MulByThreeRhs)                                            \
  X(MulByFourRhs)                                             \
  X(MulByFiveRhs)                                             \
  X(MulBySixRhs)                                              \
  X(MulBySevenRhs)                                            \
  X(MulByEightRhs)                                            \
  X(MulByNineRhs)                                             \
  X(MulNegativeOneRhs)                                        \
  X(MulNegativeTwoRhs)                                        \
  X(MulNegativeThreeRhs)                                      \
  X(MulNegativeFourRhs)                                       \
  X(MulNegativeFiveRhs)                                       \
  X(MulNegativeSixRhs)                                        \
  X(MulNegativeSevenRhs)                                      \
  X(MulNegativeEightRhs)                                      \
  X(MulNegativeNineRhs)

#define PRIME_IR_FIELD_MUL_DISTRIBUTIVITY_PATTERN_LIST(X) \
  X(MulAddDistributeConstant)                             \
  X(MulSubDistributeConstantRhs)                          \
  X(MulSubDistributeConstantLhs)

#define PRIME_IR_FIELD_MUL_PATTERN_LIST(X)              \
  PRIME_IR_FIELD_MUL_CORE_PATTERN_LIST(X)               \
  PRIME_IR_FIELD_MUL_CONSTANT_PATTERN_LIST(X)           \
  PRIME_IR_FIELD_MUL_STRENGTH_REDUCTION_PATTERN_LIST(X) \
  PRIME_IR_FIELD_MUL_DISTRIBUTIVITY_PATTERN_LIST(X)
