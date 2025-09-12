#ifndef ZKIR_DIALECT_MODARITH_IR_MODARITHTYPES_H_
#define ZKIR_DIALECT_MODARITH_IR_MODARITHTYPES_H_

// IWYU pragma: begin_keep
// Headers needed for ModArithTypes.h.inc
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
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
} // namespace mlir::zkir::mod_arith

#endif // ZKIR_DIALECT_MODARITH_IR_MODARITHTYPES_H_
