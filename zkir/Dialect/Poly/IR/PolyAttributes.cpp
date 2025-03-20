#include "zkir/Dialect/Poly/IR/PolyAttributes.h"

#include <utility>

#include "llvm/Support/ThreadPool.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::zkir::poly {

/// Cloned after upstream removal in
/// https://github.com/llvm/llvm-project/pull/87644
///
/// Computes the multiplicative inverse of this APInt for a given modulo. The
/// iterative extended Euclidean algorithm is used to solve for this value,
/// however we simplify it to speed up calculating only the inverse, and take
/// advantage of div+rem calculations. We also use some tricks to avoid copying
/// (potentially large) APInts around.
/// WARNING: a value of '0' may be returned,
///          signifying that no multiplicative inverse exists!
static APInt multiplicativeInverse(const APInt &x, const APInt &modulo) {
  assert(x.ult(modulo) && "This APInt must be smaller than the modulo");
  // Using the properties listed at the following web page (accessed 06/21/08):
  //   http://www.numbertheory.org/php/euclid.html
  // (especially the properties numbered 3, 4 and 9) it can be proved that
  // BitWidth bits suffice for all the computations in the algorithm implemented
  // below. More precisely, this number of bits suffice if the multiplicative
  // inverse exists, but may not suffice for the general extended Euclidean
  // algorithm.

  auto BitWidth = x.getBitWidth();
  APInt r[2] = {modulo, x};
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

  // If this APInt and the modulo are not coprime, there is no multiplicative
  // inverse, so return 0. We check this by looking at the next-to-last
  // remainder, which is the gcd(*this,modulo) as calculated by the Euclidean
  // algorithm.
  if (r[i] != 1) return APInt(BitWidth, 0);

  // The next-to-last t is the multiplicative inverse.  However, we are
  // interested in a positive inverse. Calculate a positive one from a negative
  // one if necessary. A simple addition of the modulo suffices because
  // abs(t[i]) is known to be less than *this/2 (see the link above).
  if (t[i].isNegative()) t[i] += modulo;

  return std::move(t[i]);
}

// Multiply two integers x, y modulo mod.
static APInt mulMod(const APInt &_x, const APInt &_y, const APInt &_mod) {
  assert(_x.getBitWidth() == _y.getBitWidth() &&
         "expected same bitwidth of operands");
  auto intermediateBitwidth = _mod.getBitWidth() * 2;
  APInt x = _x.zext(intermediateBitwidth);
  APInt y = _y.zext(intermediateBitwidth);
  APInt mod = _mod.zext(intermediateBitwidth);
  APInt res = (x * y).urem(mod);
  return res.trunc(_x.getBitWidth());
}

// Compute the first degree powers of root modulo mod.
static void precomputeRoots(APInt root, const APInt &mod, unsigned degree,
                            SmallVector<APInt> &roots,
                            SmallVector<APInt> &invRoots) {
  unsigned kBitWidth = llvm::bit_width(degree);

  // Precompute powers-of-two: `powerOfTwo[k]` = `root^(2^k)` mod `mod`.
  SmallVector<APInt> powerOfTwo(kBitWidth);
  powerOfTwo[0] = root;
  for (unsigned k = 1; k < kBitWidth; k++) {
    powerOfTwo[k] = mulMod(powerOfTwo[k - 1], powerOfTwo[k - 1], mod);
  }

  // Prepare the result vector.
  roots.resize(degree);
  invRoots.resize(degree);
  roots[0] = APInt(root.getBitWidth(), 1);     // Identity element.
  invRoots[0] = APInt(root.getBitWidth(), 1);  // Identity element.

  llvm::StdThreadPool pool(llvm::hardware_concurrency());

  // For each exponent i, decompose it into powers of two using its binary
  // representation.
  for (unsigned i = 1; i < degree; i++) {
    pool.async([&, i] {
      APInt result(root.getBitWidth(), 1);  // Identity element.
      unsigned exp = i;
      unsigned bit = 0;
      while (exp > 0) {
        if (exp & 1) result = mulMod(result, powerOfTwo[bit], mod);
        exp >>= 1;
        bit++;
      }
      roots[i] = result;
      invRoots[degree - i] = result;
    });
  }

  // Wait for all scheduled tasks to complete.
  pool.wait();
}

zkir::field::PrimeFieldAttr PrimitiveRootAttr::getRoot() const {
  return getImpl()->root;
}

zkir::field::PrimeFieldAttr PrimitiveRootAttr::getInvRoot() const {
  return getImpl()->invRoot;
}

IntegerAttr PrimitiveRootAttr::getDegree() const { return getImpl()->degree; }

zkir::field::PrimeFieldAttr PrimitiveRootAttr::getInvDegree() const {
  return getImpl()->invDegree;
}

DenseElementsAttr PrimitiveRootAttr::getRoots() const {
  return getImpl()->roots;
}

DenseElementsAttr PrimitiveRootAttr::getInvRoots() const {
  return getImpl()->invRoots;
}

namespace detail {

PrimitiveRootAttrStorage *PrimitiveRootAttrStorage::construct(
    AttributeStorageAllocator &allocator, KeyTy &&key) {
  // Extract the root and degree from the key.
  zkir::field::PrimeFieldAttr root = std::get<0>(key);
  IntegerAttr degree = std::get<1>(key);

  APInt mod = root.getType().getModulus().getValue();
  APInt rootVal = root.getValue().getValue();
  APInt invRootVal = multiplicativeInverse(rootVal, mod);
  APInt invDegreeVal = multiplicativeInverse(
      degree.getValue().zextOrTrunc(mod.getBitWidth()), mod);

  field::PrimeFieldAttr invDegree =
      field::PrimeFieldAttr::get(root.getType(), invDegreeVal);
  zkir::field::PrimeFieldAttr invRoot =
      zkir::field::PrimeFieldAttr::get(root.getType(), invRootVal);

  // Compute the exponent table.
  SmallVector<APInt> roots, invRoots;
  precomputeRoots(rootVal, mod, degree.getInt(), roots, invRoots);

  // Create a ranked tensor type for the exponents attribute.
  auto tensorType = RankedTensorType::get(
      {degree.getInt()}, root.getType().getModulus().getType());

  // Create the DenseIntElementsAttr from the computed exponent values.
  DenseElementsAttr rootsAttr = DenseElementsAttr::get(tensorType, roots);
  DenseElementsAttr invRootsAttr = DenseElementsAttr::get(tensorType, invRoots);
  return new (allocator.allocate<PrimitiveRootAttrStorage>())
      PrimitiveRootAttrStorage(std::move(degree), std::move(invDegree),
                               std::move(root), std::move(invRoot),
                               std::move(rootsAttr), std::move(invRootsAttr));
}

}  // namespace detail
}  // namespace mlir::zkir::poly
