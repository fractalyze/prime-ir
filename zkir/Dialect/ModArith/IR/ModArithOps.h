#ifndef ZKIR_DIALECT_MODARITH_IR_MODARITHOPS_H_
#define ZKIR_DIALECT_MODARITH_IR_MODARITHOPS_H_

#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project
#include "zkir/Dialect/ModArith/IR/ModArithAttributes.h"
#include "zkir/Dialect/ModArith/IR/ModArithDialect.h"
#include "zkir/Dialect/ModArith/IR/ModArithTypes.h"

#define GET_OP_CLASSES
#include "zkir/Dialect/ModArith/IR/ModArithOps.h.inc"

namespace mlir {
namespace zkir {
namespace mod_arith {

template <typename OpType>
inline ModArithType getResultModArithType(OpType op) {
  return cast<ModArithType>(getElementTypeOrSelf(op.getResult().getType()));
}

template <typename OpType>
inline ModArithType getOperandModArithType(OpType op) {
  return cast<ModArithType>(getElementTypeOrSelf(op.getOperand().getType()));
}

template <typename OpType>
inline IntegerType getResultIntegerType(OpType op) {
  return cast<IntegerType>(getElementTypeOrSelf(op.getResult().getType()));
}

template <typename OpType>
inline IntegerType getOperandIntegerType(OpType op) {
  return cast<IntegerType>(getElementTypeOrSelf(op.getOperand().getType()));
}

}  // namespace mod_arith
}  // namespace zkir
}  // namespace mlir

#endif  // ZKIR_DIALECT_MODARITH_IR_MODARITHOPS_H_
