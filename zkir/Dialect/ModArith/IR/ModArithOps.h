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
