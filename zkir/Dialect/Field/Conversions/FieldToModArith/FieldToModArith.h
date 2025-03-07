#ifndef ZKIR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_FIELDTOMODARITH_H_
#define ZKIR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_FIELDTOMODARITH_H_

#include "mlir/include/mlir/Pass/Pass.h"

namespace mlir::zkir::field {

#define GEN_PASS_DECL
#include "zkir/Dialect/Field/Conversions/FieldToModArith/FieldToModArith.h.inc"

#define GEN_PASS_REGISTRATION
#include "zkir/Dialect/Field/Conversions/FieldToModArith/FieldToModArith.h.inc"  // NOLINT(build/include)

}  // namespace mlir::zkir::field

#endif  // ZKIR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_FIELDTOMODARITH_H_
