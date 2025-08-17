#include "zkir/Utils/APIntUtils.h"

#include <utility>

namespace mlir::zkir {

using namespace llvm;

/// Cloned after upstream removal in
/// https://github.com/llvm/llvm-project/pull/87644
/// Computes the multiplicative inverse of this APInt for a given modulus. The
/// iterative extended Euclidean algorithm is used to solve for this value,
/// however we simplify it to speed up calculating only the inverse, and take
/// advantage of div+rem calculations. We also use some tricks to avoid copying
/// (potentially large) APInts around.
APInt multiplicativeInverse(const APInt &x, const APInt &modulus) {
  assert(x.ult(modulus) && "This APInt must be smaller than the modulus");
  // Using the properties listed at the following web page (accessed 06/21/08):
  //   http://www.numbertheory.org/php/euclid.html
  // (especially the properties numbered 3, 4 and 9) it can be proved that
  // BitWidth bits suffice for all the computations in the algorithm implemented
  // below. More precisely, this number of bits suffice if the multiplicative
  // inverse exists, but may not suffice for the general extended Euclidean
  // algorithm.

  auto BitWidth = x.getBitWidth();
  APInt r[2] = {modulus, x};
  APInt t[2] = {APInt(BitWidth, 0), APInt(BitWidth, 1)};
  APInt q(BitWidth, 0);

  unsigned i;
  for (i = 0; r[i ^ 1] != 0; i ^= 1) {
    // An overview of the math without the confusing bit-flipping:
    // q = r[i-2] / r[i-1]
    // r[i] = r[i-2] % r[i-1]
    // t[i] = t[i-2] - t[i-1] * q
    x.udivrem(r[i], r[i ^ 1], q, r[i]);
    t[i] -= t[i ^ 1] * q;
  }

  // If this APInt and the modulus are not coprime, there is no multiplicative
  // inverse, so return 0. We check this by looking at the next-to-last
  // remainder, which is the gcd(*this,modulus) as calculated by the Euclidean
  // algorithm.
  if (r[i] != 1)
    return APInt(BitWidth, 0);

  // The next-to-last t is the multiplicative inverse.  However, we are
  // interested in a positive inverse. Calculate a positive one from a negative
  // one if necessary. A simple addition of the modulus suffices because
  // abs(t[i]) is known to be less than *this/2 (see the link above).
  if (t[i].isNegative())
    t[i] += modulus;

  return std::move(t[i]);
}

// Compute `_x` * `_y` (mod `_modulus`).
APInt mulMod(const APInt &_x, const APInt &_y, const APInt &_modulus) {
  assert(_x.getBitWidth() == _y.getBitWidth() &&
         "expected same bitwidth of operands");
  auto intermediateBitwidth = _modulus.getBitWidth() * 2;
  APInt x = _x.zext(intermediateBitwidth);
  APInt y = _y.zext(intermediateBitwidth);
  APInt modulus = _modulus.zext(intermediateBitwidth);
  APInt res = (x * y).urem(modulus);
  return res.trunc(_x.getBitWidth());
}

// Compute `_base` ^ `exp` (mod `_modulus`) using exponentiation by squaring.
APInt expMod(const APInt &_base, unsigned exp, const APInt &_modulus) {
  // Ensure _base and _modulus have the same bitwidth.
  assert(_base.getBitWidth() == _modulus.getBitWidth() &&
         "expected same bitwidth for base and modulus");

  // Extend bitwidth for intermediate computations.
  auto intermediateBitwidth = _modulus.getBitWidth() * 2;
  APInt base = _base.zext(intermediateBitwidth);
  APInt modulus = _modulus.zext(intermediateBitwidth);

  APInt result(intermediateBitwidth, 1);
  while (true) {
    // If the lowest bit of exp is 1, multiply result with the current base.
    if (exp % 2 == 1)
      result = (result * base).urem(modulus);

    // Right-shift exponent by 1 (divide by 2).
    exp = exp >> 1;
    if (exp == 0)
      break;
    // Square the `base` (mod `modulus`).
    base = (base * base).urem(modulus);
  }
  // Truncate the result to the modulus bitwidth.
  return result.trunc(_modulus.getBitWidth());
}

} // namespace mlir::zkir
