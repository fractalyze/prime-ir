#ifndef ZKIR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_FIELDTOMODARITH_H_
#define ZKIR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_FIELDTOMODARITH_H_

// IWYU pragma: begin_keep
// Headers needed for FieldToModArith.h.inc
#include "mlir/Pass/Pass.h"
#include "zkir/Dialect/ModArith/IR/ModArithDialect.h"
// IWYU pragma: end_keep

namespace mlir::zkir::field {

#define GEN_PASS_DECL
#include "zkir/Dialect/Field/Conversions/FieldToModArith/FieldToModArith.h.inc"

#define GEN_PASS_REGISTRATION
#include "zkir/Dialect/Field/Conversions/FieldToModArith/FieldToModArith.h.inc" // NOLINT(build/include)

} // namespace mlir::zkir::field

#endif // ZKIR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_FIELDTOMODARITH_H_
