// Pattern suffix list for canonicalization patterns.
//
// If the target type is a field type, use the FIELD patterns.
//
// Usage:
//   #include "prime_ir/Utils/CanonicalizationPatterns.inc"
//   #define PRIME_IR_ADD_PATTERN(Name) patterns.add<Prefix##Name>(context);
//   PRIME_IR_FIELD_ADD_PATTERN_LIST(PRIME_IR_ADD_PATTERN)
//   #undef PRIME_IR_ADD_PATTERN
//
// On the other hand, if the target type is a group type, use the GROUP patterns.
//
// Usage:
//   #include "prime_ir/Utils/CanonicalizationPatterns.inc"
//   #define PRIME_IR_ADD_PATTERN(Name) patterns.add<Prefix##Name>(context);
//   PRIME_IR_GROUP_ADD_PATTERN_LIST(PRIME_IR_ADD_PATTERN)
//   #undef PRIME_IR_ADD_PATTERN
//
// GROUP patterns: Only require additive or multiplicative operations (no distributivity)
// FIELD patterns: Require both additive and multiplicative operations (distributivity)

// AddOp patterns - Group (no MulOp dependency)
#define PRIME_IR_GROUP_ADD_PATTERN_LIST(X) \
  X(AddConstantTwice)                      \
  X(AddConstantToSubLhs)                   \
  X(AddConstantToSubRhs)                   \
  X(AddSelfIsDouble)                       \
  X(AddBothNegated)                        \
  X(AddAfterSub)                           \
  X(AddAfterNegLhs)                        \
  X(AddAfterNegRhs)

// AddOp patterns - Field (requires MulOp for distributivity)
#define PRIME_IR_FIELD_ADD_PATTERN_LIST(X) \
  PRIME_IR_GROUP_ADD_PATTERN_LIST(X)       \
  X(FactorMulAdd)

// SubOp patterns - Group (no MulOp dependency)
#define PRIME_IR_GROUP_SUB_PATTERN_LIST(X) \
  X(SubConstantFromAdd)                    \
  X(SubConstantTwiceLhs)                   \
  X(SubConstantTwiceRhs)                   \
  X(SubAddFromConstant)                    \
  X(SubSubFromConstantLhs)                 \
  X(SubSubFromConstantRhs)                 \
  X(SubSelfIsZero)                         \
  X(SubLhsAfterAdd)                        \
  X(SubRhsAfterAdd)                        \
  X(SubLhsAfterSub)                        \
  X(SubAfterNegLhs)                        \
  X(SubAfterNegRhs)                        \
  X(SubBothNegated)

// SubOp patterns - Field (requires MulOp/SquareOp for distributivity)
#define PRIME_IR_FIELD_SUB_PATTERN_LIST(X) \
  PRIME_IR_GROUP_SUB_PATTERN_LIST(X)       \
  X(FactorMulSub)                          \
  X(SubAfterSquareBoth)                    \
  X(SubAfterSumSquare)

// MulOp patterns - Group (no AddOp dependency)
#define PRIME_IR_GROUP_MUL_PATTERN_LIST(X) \
  X(MulByTwoIsDouble)                      \
  X(MulSelfIsSquare)                       \
  X(MulNegativeOneRhs)                     \
  X(MulNegativeTwoRhs)                     \
  X(MulNegativeFourRhs)                    \
  X(MulConstantTwice)                      \
  X(MulOfMulByConstant)

// MulOp patterns - Field (requires AddOp/SubOp for distributivity)
#define PRIME_IR_FIELD_MUL_PATTERN_LIST(X) \
  PRIME_IR_GROUP_MUL_PATTERN_LIST(X)       \
  X(MulNegativeThreeRhs)                   \
  X(MulAddDistributeConstant)              \
  X(MulSubDistributeConstantRhs)           \
  X(MulSubDistributeConstantLhs)
