#ifndef ZKIR_DIALECT_FIELD_IR_FIELDOPS_H_
#define ZKIR_DIALECT_FIELD_IR_FIELDOPS_H_

#include "zkir/Dialect/Field/IR/FieldAttributes.h"
#include "zkir/Dialect/Field/IR/FieldTypes.h"

// IWYU pragma: begin_keep
// Headers needed for FieldOps.h.inc
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
// IWYU pragma: end_keep

#define GET_OP_CLASSES
#include "zkir/Dialect/Field/IR/FieldOps.h.inc"

namespace mlir::zkir::field {

template <typename OpType>
PrimeFieldType getResultPrimeFieldType(OpType op) {
  return cast<PrimeFieldType>(getElementTypeOrSelf(op.getResult().getType()));
}

PrimeFieldAttr getAttrAsStandardForm(PrimeFieldAttr attr);
PrimeFieldAttr getAttrAsMontgomeryForm(PrimeFieldAttr attr);

Type getStandardFormType(Type type);
Type getMontgomeryFormType(Type type);

}  // namespace mlir::zkir::field

#endif  // ZKIR_DIALECT_FIELD_IR_FIELDOPS_H_
