#include "zkir/Dialect/Poly/IR/PolyAttributes.h"

#include <utility>

#include "llvm/Support/ThreadPool.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "zkir/Dialect/Field/IR/FieldOps.h"
#include "zkir/Utils/APIntUtils.h"

namespace mlir::zkir::poly {

// Compute the first degree powers of root modulo mod.
static void precomputeRoots(APInt root, const APInt &mod, unsigned degree,
                            SmallVector<APInt> &roots,
                            SmallVector<APInt> &invRoots,
                            std::optional<IntegerAttr> montgomeryR) {
  unsigned kBitWidth = llvm::bit_width(degree);

  // Precompute powers-of-two: `powerOfTwo[k]` = `root^(2^k)` mod `mod`.
  SmallVector<APInt> powerOfTwo(kBitWidth);
  powerOfTwo[0] = root;
  for (unsigned k = 1; k < kBitWidth; k++) {
    powerOfTwo[k] = mulMod(powerOfTwo[k - 1], powerOfTwo[k - 1], mod);
  }

  // Coset factor
  APInt coset(mod.getBitWidth(), 1);
  if (montgomeryR) {
    coset = montgomeryR->getValue();
  }

  // Prepare the result vector.
  roots.resize(degree);
  invRoots.resize(degree);
  roots[0] = coset;
  invRoots[0] = coset;

  llvm::StdThreadPool pool(llvm::hardware_concurrency());

  // For each exponent i, decompose it into powers of two using its binary
  // representation.
  for (unsigned i = 1; i < degree; i++) {
    pool.async([&, i] {
      APInt result(root.getBitWidth(), 1); // Identity element.
      unsigned exp = i;
      unsigned bit = 0;
      while (exp > 0) {
        if (exp & 1)
          result = mulMod(result, powerOfTwo[bit], mod);
        exp >>= 1;
        bit++;
      }

      roots[i] = mulMod(coset, result, mod);
      invRoots[degree - i] = mulMod(coset, result, mod);
    });
  }

  // Wait for all scheduled tasks to complete.
  pool.wait();
}

field::RootOfUnityAttr PrimitiveRootAttr::getRootOfUnity() const {
  return getImpl()->rootOfUnity;
}

field::PrimeFieldAttr PrimitiveRootAttr::getRoot() const {
  return getImpl()->root;
}

field::PrimeFieldAttr PrimitiveRootAttr::getInvRoot() const {
  return getImpl()->invRoot;
}

IntegerAttr PrimitiveRootAttr::getDegree() const {
  return getImpl()->rootOfUnity.getDegree();
}

field::PrimeFieldAttr PrimitiveRootAttr::getInvDegree() const {
  return getImpl()->invDegree;
}

DenseElementsAttr PrimitiveRootAttr::getRoots() const {
  return getImpl()->roots;
}

DenseElementsAttr PrimitiveRootAttr::getInvRoots() const {
  return getImpl()->invRoots;
}

mod_arith::MontgomeryAttr PrimitiveRootAttr::getMontgomery() const {
  return getImpl()->montgomery;
}

namespace detail {

PrimitiveRootAttrStorage *
PrimitiveRootAttrStorage::construct(AttributeStorageAllocator &allocator,
                                    KeyTy &&key) {
  // Extract the root and degree from the key.
  field::RootOfUnityAttr rootOfUnity = std::get<0>(key);
  field::PrimeFieldAttr root = rootOfUnity.getRoot();
  IntegerAttr degree = rootOfUnity.getDegree();
  mod_arith::MontgomeryAttr montgomery = std::get<1>(key);

  std::optional<IntegerAttr> montgomeryR;
  if (montgomery != mod_arith::MontgomeryAttr()) {
    montgomeryR = montgomery.getR();
  }

  APInt mod = root.getType().getModulus().getValue();
  APInt rootVal = root.getValue().getValue();
  APInt invRootVal = multiplicativeInverse(rootVal, mod);
  APInt invDegreeVal = multiplicativeInverse(
      degree.getValue().zextOrTrunc(mod.getBitWidth()), mod);

  if (montgomeryR) {
    invDegreeVal = mulMod(invDegreeVal, montgomeryR->getValue(), mod);
  }

  if (montgomery != mod_arith::MontgomeryAttr()) {
    APInt rootMontVal =
        mulMod(root.getValue().getValue(), montgomery.getR().getValue(), mod);
    root = field::PrimeFieldAttr::get(
        cast<field::PrimeFieldType>(
            field::getMontgomeryFormType(root.getType())),
        rootMontVal);
  }
  field::PrimeFieldAttr invDegree =
      field::PrimeFieldAttr::get(root.getType(), invDegreeVal);
  field::PrimeFieldAttr invRoot =
      field::PrimeFieldAttr::get(root.getType(), invRootVal);

  // Compute the exponent table.
  SmallVector<APInt> roots, invRoots;
  precomputeRoots(rootVal, mod, degree.getInt(), roots, invRoots, montgomeryR);
  // Create a ranked tensor type for the exponents attribute.
  auto tensorType =
      RankedTensorType::get({degree.getInt()}, root.getType().getStorageType());

  // Create the DenseIntElementsAttr from the computed exponent values.
  DenseElementsAttr rootsAttr = DenseElementsAttr::get(tensorType, roots);
  DenseElementsAttr invRootsAttr = DenseElementsAttr::get(tensorType, invRoots);
  return new (allocator.allocate<PrimitiveRootAttrStorage>())
      PrimitiveRootAttrStorage(std::move(rootOfUnity), std::move(root),
                               std::move(invDegree), std::move(invRoot),
                               std::move(rootsAttr), std::move(invRootsAttr),
                               std::move(montgomery));
}

} // namespace detail
} // namespace mlir::zkir::poly
