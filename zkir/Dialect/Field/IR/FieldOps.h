#ifndef ZKIR_DIALECT_FIELD_IR_FIELDOPS_H_
#define ZKIR_DIALECT_FIELD_IR_FIELDOPS_H_

#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project
#include "zkir/Dialect/Field/IR/FieldAttributes.h"
#include "zkir/Dialect/Field/IR/FieldDialect.h"
#include "zkir/Dialect/Field/IR/FieldTypes.h"

#define GET_OP_CLASSES
#include "zkir/Dialect/Field/IR/FieldOps.h.inc"

namespace mlir {
namespace zkir {
namespace field {

template <typename OpType>
PrimeFieldType getResultPrimeFieldType(OpType op) {
  return cast<PrimeFieldType>(getElementTypeOrSelf(op.getResult().getType()));
}

}  // namespace field
}  // namespace zkir
}  // namespace mlir

#endif  // ZKIR_DIALECT_FIELD_IR_FIELDOPS_H_
