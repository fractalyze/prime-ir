#ifndef ZKIR_DIALECT_FIELD_IR_FIELDTYPES_H_
#define ZKIR_DIALECT_FIELD_IR_FIELDTYPES_H_

#include "mlir/IR/Types.h"

// IWYU pragma: begin_keep
// Headers needed for FieldTypes.h.inc
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
// IWYU pragma: end_keep

namespace mlir::zkir::field {

class PrimeFieldAttr;

bool isMontgomery(Type type);
unsigned getIntOrPrimeFieldBitWidth(Type type);

} // namespace mlir::zkir::field

#define GET_TYPEDEF_CLASSES
#include "zkir/Dialect/Field/IR/FieldTypes.h.inc"

#endif // ZKIR_DIALECT_FIELD_IR_FIELDTYPES_H_
