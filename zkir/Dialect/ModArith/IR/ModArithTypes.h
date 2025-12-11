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

#ifndef ZKIR_DIALECT_MODARITH_IR_MODARITHTYPES_H_
#define ZKIR_DIALECT_MODARITH_IR_MODARITHTYPES_H_

// IWYU pragma: begin_keep
// Headers needed for ModArithTypes.h.inc
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "zkir/Dialect/ModArith/IR/ModArithAttributes.h"
// IWYU pragma: end_keep

#define GET_TYPEDEF_CLASSES
#include "zkir/Dialect/ModArith/IR/ModArithTypes.h.inc"

namespace mlir::zkir::mod_arith {
inline unsigned getIntOrModArithBitWidth(Type type) {
  assert(llvm::isa<ModArithType>(type) || llvm::isa<IntegerType>(type));
  if (auto modArithType = dyn_cast<ModArithType>(type)) {
    return modArithType.getStorageBitWidth();
  }
  return cast<IntegerType>(type).getWidth();
}

IntegerAttr getAttrAsStandardForm(IntegerAttr modulus, IntegerAttr attr);
IntegerAttr getAttrAsMontgomeryForm(IntegerAttr modulus, IntegerAttr attr);
} // namespace mlir::zkir::mod_arith

#endif // ZKIR_DIALECT_MODARITH_IR_MODARITHTYPES_H_
