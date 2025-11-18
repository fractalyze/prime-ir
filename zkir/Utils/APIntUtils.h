/* Copyright 2025 The ZKIR Authors.

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
} // namespace mlir::zkir

#endif // ZKIR_UTILS_APINTUTILS_H_
