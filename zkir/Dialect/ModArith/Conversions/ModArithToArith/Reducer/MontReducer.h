#ifndef ZKIR_DIALECT_MODARITH_CONVERSIONS_MODARITHTOARITH_REDUCER_MONTREDUCER_H_
#define ZKIR_DIALECT_MODARITH_CONVERSIONS_MODARITHTOARITH_REDUCER_MONTREDUCER_H_

#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Value.h"
#include "zkir/Dialect/ModArith/IR/ModArithTypes.h" // IWYU pragma: keep

namespace mlir {
class TypedAttr;
}

namespace mlir::zkir::mod_arith {

// Helper class to perform Montgomery reduction operations.
// This class encapsulates the logic for generating MLIR operations
// that implement the REDC (Montgomery Reduction) algorithm.
class MontReducer {
public:
  // Constructs a MontReducer with the given builder and ModArithType.
  // Extracts the modulus and Montgomery parameters from the type.
  explicit MontReducer(ImplicitLocOpBuilder &b, ModArithType modArithType);

  // Performs Montgomery reduction on the given input values.
  // Given T = tLow + tHigh * 2ʷ (where w is the modulus bit width),
  // computes T * R⁻¹ mod n, where R is the Montgomery radix.
  Value reduce(Value tLow, Value tHigh);

  // Gets the canonical form from an input value in [0, 2n).
  Value getCanonicalFromExtended(Value input);

  // Gets the canonical form from an input value in [0, 2n) and an overflow
  // flag.
  Value getCanonicalFromExtended(Value input, Value overflow);

  // Gets the canonical difference of two values in modular arithmetic.
  // Computes (lhs - rhs) mod n, returning a value in [0, n).
  // Assumes that both are in the range [0, n).
  Value getCanonicalDiff(Value lhs, Value rhs);

private:
  // Creates a properly typed constant for the modulus based on the input type.
  // Handles splatting for vector types automatically.
  Value createModulusConst(Type inputType);

  // Performs single-limb Montgomery reduction.
  // Used when the modulus fits in a single limb (2ʷ > modulus).
  Value reduceSingleLimb(Value tLow, Value tHigh);

  // Performs multi-limb Montgomery reduction.
  // Used when the modulus requires multiple limbs (2ʷ <= modulus).
  Value reduceMultiLimb(Value tLow, Value tHigh);

  // Checks if the input is from a signed multiplication.
  bool isFromSignedMul(Value input);

  ImplicitLocOpBuilder &b_;
  TypedAttr modAttr_;
  MontgomeryAttr montAttr_;
};

} // namespace mlir::zkir::mod_arith

// NOLINTNEXTLINE(whitespace/line_length)
#endif // ZKIR_DIALECT_MODARITH_CONVERSIONS_MODARITHTOARITH_REDUCER_MONTREDUCER_H_
