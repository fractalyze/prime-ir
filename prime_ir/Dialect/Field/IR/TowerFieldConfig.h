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

#ifndef PRIME_IR_DIALECT_FIELD_IR_TOWERFIELDCONFIG_H_
#define PRIME_IR_DIALECT_FIELD_IR_TOWERFIELDCONFIG_H_

#include "llvm/ADT/SmallVector.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"

namespace mlir::prime_ir::field {

// Returns a signature encoding the tower structure.
// e.g., Fp12 = ((Fp2)^3)^2 -> {2, 3, 2}  (top degree first)
// e.g., Fp6 = (Fp2)^3 -> {3, 2}
// e.g., Fp2 -> {2}
inline SmallVector<unsigned, 4> getTowerSignature(ExtensionFieldType efType) {
  SmallVector<unsigned, 4> signature;
  Type current = efType;
  while (auto ef = dyn_cast<ExtensionFieldType>(current)) {
    signature.push_back(ef.getDegree());
    current = ef.getBaseField();
  }
  return signature;
}

// Macro to dispatch to the correct tower extension type based on signature.
// Args:
//   SIG_VAR: variable name holding SmallVector<unsigned, 4> signature
//   ACTION: macro that takes (Signature, TypeName) and produces an action
//   NON_TOWER_SUFFIX: suffix for non-tower types (e.g., Op or CodeGen)
//   TOWER_SUFFIX: suffix for tower types (e.g., Op or CodeGen)
//
// clang-format off
#define DISPATCH_TOWER_BY_SIGNATURE(SIG_VAR, ACTION, NON_TOWER_SUFFIX, TOWER_SUFFIX) \
  /* Non-tower extension fields */                                                   \
  if (SIG_VAR == SmallVector<unsigned, 4>{2}) {                                      \
    ACTION({2}, Quadratic##NON_TOWER_SUFFIX)                                         \
  } else if (SIG_VAR == SmallVector<unsigned, 4>{3}) {                               \
    ACTION({3}, Cubic##NON_TOWER_SUFFIX)                                             \
  } else if (SIG_VAR == SmallVector<unsigned, 4>{4}) {                               \
    ACTION({4}, Quartic##NON_TOWER_SUFFIX)                                           \
  }                                                                                  \
  /* Depth-1 tower extension fields */                                               \
  else if (SIG_VAR == SmallVector<unsigned, 4>({2, 2})) {                            \
    ACTION(({2, 2}), TowerQuadraticOverQuadratic##TOWER_SUFFIX)                      \
  } else if (SIG_VAR == SmallVector<unsigned, 4>({3, 2})) {                          \
    ACTION(({3, 2}), TowerCubicOverQuadratic##TOWER_SUFFIX)                          \
  } else if (SIG_VAR == SmallVector<unsigned, 4>({4, 2})) {                          \
    ACTION(({4, 2}), TowerQuarticOverQuadratic##TOWER_SUFFIX)                        \
  } else if (SIG_VAR == SmallVector<unsigned, 4>({2, 3})) {                          \
    ACTION(({2, 3}), TowerQuadraticOverCubic##TOWER_SUFFIX)                          \
  }                                                                                  \
  /* Depth-2 tower extension fields */                                               \
  else if (SIG_VAR == SmallVector<unsigned, 4>({2, 3, 2})) {                         \
    ACTION(({2, 3, 2}), TowerQuadraticOverCubicOverQuadratic##TOWER_SUFFIX)          \
  } else if (SIG_VAR == SmallVector<unsigned, 4>({3, 2, 2})) {                       \
    ACTION(({3, 2, 2}), TowerCubicOverQuadraticOverQuadratic##TOWER_SUFFIX)          \
  } else if (SIG_VAR == SmallVector<unsigned, 4>({2, 2, 2})) {                       \
    ACTION(({2, 2, 2}), TowerQuadraticOverQuadraticOverQuadratic##TOWER_SUFFIX)      \
  } else if (SIG_VAR == SmallVector<unsigned, 4>({2, 2, 3})) {                       \
    ACTION(({2, 2, 3}), TowerQuadraticOverQuadraticOverCubic##TOWER_SUFFIX)          \
  }                                                                                  \
  /* Depth-3 tower extension fields */                                               \
  else if (SIG_VAR == SmallVector<unsigned, 4>({2, 2, 3, 2})) {                      \
    ACTION(({2, 2, 3, 2}),                                                           \
           TowerQuadraticOverQuadraticOverCubicOverQuadratic##TOWER_SUFFIX)          \
  } else {                                                                           \
    llvm_unreachable("Unsupported tower extension field configuration");             \
  }
// clang-format on

// Macro to generate variant type with all tower field types.
// Expands to: std::variant<PrimeT, QuadraticT, CubicT, ..., Fp24T>
// Args:
//   PrimeT: the prime field type
//   NON_TOWER_SUFFIX: suffix for non-tower types (e.g., ExtensionFieldOperation)
//   TOWER_SUFFIX: suffix for tower types (e.g., Op)
//
// clang-format off
#define TOWER_FIELD_VARIANT(PrimeT, NON_TOWER_SUFFIX, TOWER_SUFFIX)           \
  std::variant<PrimeT,                                                        \
               /* Non-tower extension fields */                               \
               Quadratic##NON_TOWER_SUFFIX,                                   \
               Cubic##NON_TOWER_SUFFIX,                                       \
               Quartic##NON_TOWER_SUFFIX,                                     \
               /* Depth-1 tower extension fields */                           \
               TowerQuadraticOverQuadratic##TOWER_SUFFIX,                     \
               TowerCubicOverQuadratic##TOWER_SUFFIX,                         \
               TowerQuarticOverQuadratic##TOWER_SUFFIX,                       \
               TowerQuadraticOverCubic##TOWER_SUFFIX,                         \
               /* Depth-2 tower extension fields */                           \
               TowerQuadraticOverCubicOverQuadratic##TOWER_SUFFIX,            \
               TowerCubicOverQuadraticOverQuadratic##TOWER_SUFFIX,            \
               TowerQuadraticOverQuadraticOverQuadratic##TOWER_SUFFIX,        \
               TowerQuadraticOverQuadraticOverCubic##TOWER_SUFFIX,            \
               /* Depth-3 tower extension fields */                           \
               TowerQuadraticOverQuadraticOverCubicOverQuadratic##TOWER_SUFFIX>
// clang-format on

} // namespace mlir::prime_ir::field

#endif // PRIME_IR_DIALECT_FIELD_IR_TOWERFIELDCONFIG_H_
