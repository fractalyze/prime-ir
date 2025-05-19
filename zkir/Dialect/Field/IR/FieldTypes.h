#ifndef ZKIR_DIALECT_FIELD_IR_FIELDTYPES_H_
#define ZKIR_DIALECT_FIELD_IR_FIELDTYPES_H_

#include "zkir/Dialect/Field/IR/FieldDialect.h"

namespace mlir::zkir::field {
class PrimeFieldAttr;
}

#define GET_TYPEDEF_CLASSES
#include "zkir/Dialect/Field/IR/FieldTypes.h.inc"

#endif  // ZKIR_DIALECT_FIELD_IR_FIELDTYPES_H_
