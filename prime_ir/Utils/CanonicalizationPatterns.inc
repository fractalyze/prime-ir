// Pattern suffix list for canonicalization patterns.
// Usage:
//   #include "prime_ir/Utils/CanonicalizationPatterns.inc"
//   #define PRIME_IR_ADD_PATTERN(Name) patterns.add<Prefix##Name>(context);
//   PRIME_IR_ADD_PATTERN_LIST(PRIME_IR_ADD_PATTERN)
//   #undef PRIME_IR_ADD_PATTERN

// AddOp patterns.
#define PRIME_IR_ADD_PATTERN_LIST(X) \
  X(AddConstantTwice)            \
  X(AddConstantToSubLhs)         \
  X(AddConstantToSubRhs)         \
  X(AddSelfIsDouble)             \
  X(AddBothNegated)              \
  X(AddAfterSub)                 \
  X(AddAfterNegLhs)              \
  X(AddAfterNegRhs)              \
  X(FactorMulAdd)

// SubOp patterns.
#define PRIME_IR_SUB_PATTERN_LIST(X) \
  X(SubConstantFromAdd)          \
  X(SubConstantTwiceLhs)         \
  X(SubConstantTwiceRhs)         \
  X(SubAddFromConstant)          \
  X(SubSubFromConstantLhs)       \
  X(SubSubFromConstantRhs)       \
  X(SubSelfIsZero)               \
  X(SubLhsAfterAdd)              \
  X(SubRhsAfterAdd)              \
  X(SubLhsAfterSub)              \
  X(SubAfterNegLhs)              \
  X(SubAfterNegRhs)              \
  X(SubBothNegated)              \
  X(SubAfterSquareBoth)          \
  X(SubAfterSumSquare)           \
  X(FactorMulSub)

// MulOp patterns.
#define PRIME_IR_MUL_PATTERN_LIST(X) \
  X(MulByTwoIsDouble)            \
  X(MulSelfIsSquare)             \
  X(MulNegativeOneRhs)           \
  X(MulNegativeTwoRhs)           \
  X(MulNegativeThreeRhs)         \
  X(MulNegativeFourRhs)          \
  X(MulConstantTwice)            \
  X(MulOfMulByConstant)          \
  X(MulAddDistributeConstant)    \
  X(MulSubDistributeConstantRhs) \
  X(MulSubDistributeConstantLhs)
