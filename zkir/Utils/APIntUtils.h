#ifndef ZKIR_UTILS_APINTUTILS_H_
#define ZKIR_UTILS_APINTUTILS_H_

#include "llvm/ADT/APInt.h"

namespace mlir::zkir {

// Compute `x` ^ -1 (mod `modulus`).
/// WARNING: a value of '0' may be returned,
///          signifying that no multiplicative inverse exists!
llvm::APInt multiplicativeInverse(const llvm::APInt &x,
                                  const llvm::APInt &modulus);

// Compute `x` * `y` (mod `modulus`).
llvm::APInt mulMod(const llvm::APInt &x, const llvm::APInt &y,
                   const llvm::APInt &modulus);

// Compute `base` ^ `exp` (mod `modulus`).
llvm::APInt expMod(const llvm::APInt &base, unsigned exp,
                   const llvm::APInt &modulus);
}  // namespace mlir::zkir

#endif  // ZKIR_UTILS_APINTUTILS_H_
