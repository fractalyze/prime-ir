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

#ifndef ZKIR_DIALECT_MODARITH_IR_MODARITHOPS_H_
#define ZKIR_DIALECT_MODARITH_IR_MODARITHOPS_H_

#include "zkir/Dialect/ModArith/IR/ModArithTypes.h"

// IWYU pragma: begin_keep
// Headers needed for ModArithOps.h.inc
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
// IWYU pragma: end_keep

#define GET_OP_CLASSES
#include "zkir/Dialect/ModArith/IR/ModArithOps.h.inc"

namespace mlir::zkir::mod_arith {

template <typename OpType>
inline ModArithType getResultModArithType(OpType op) {
  return cast<ModArithType>(getElementTypeOrSelf(op.getType()));
}

template <typename OpType>
inline ModArithType getOperandModArithType(OpType op) {
  return cast<ModArithType>(getElementTypeOrSelf(op.getOperand().getType()));
}

template <typename OpType>
inline IntegerType getResultIntegerType(OpType op) {
  return cast<IntegerType>(getElementTypeOrSelf(op.getType()));
}

template <typename OpType>
inline IntegerType getOperandIntegerType(OpType op) {
  return cast<IntegerType>(getElementTypeOrSelf(op.getOperand().getType()));
}

Type getStandardFormType(Type type);
Type getMontgomeryFormType(Type type);

} // namespace mlir::zkir::mod_arith

#endif // ZKIR_DIALECT_MODARITH_IR_MODARITHOPS_H_
