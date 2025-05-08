#ifndef ZKIR_DIALECT_FIELD_IR_FIELDOPS_H_
#define ZKIR_DIALECT_FIELD_IR_FIELDOPS_H_

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "zkir/Dialect/Field/IR/FieldAttributes.h"
#include "zkir/Dialect/Field/IR/FieldDialect.h"
#include "zkir/Dialect/Field/IR/FieldTypes.h"
#include "zkir/Dialect/ModArith/IR/ModArithAttributes.h"
#include "zkir/Utils/OpUtils.h"

#define GET_OP_CLASSES
#include "zkir/Dialect/Field/IR/FieldOps.h.inc"

namespace mlir::zkir::field {

template <typename OpType>
PrimeFieldType getResultPrimeFieldType(OpType op) {
  return cast<PrimeFieldType>(getElementTypeOrSelf(op.getResult().getType()));
}

}  // namespace mlir::zkir::field

#endif  // ZKIR_DIALECT_FIELD_IR_FIELDOPS_H_
