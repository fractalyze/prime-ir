#ifndef ZKIR_DIALECT_POLY_CONVERSIONS_POLYTOFIELD_POLYTOFIELD_H_
#define ZKIR_DIALECT_POLY_CONVERSIONS_POLYTOFIELD_POLYTOFIELD_H_

#include "mlir/Pass/Pass.h"
#include "zkir/Dialect/Field/IR/FieldDialect.h"

namespace mlir::zkir::poly {

#define GEN_PASS_DECL
#include "zkir/Dialect/Poly/Conversions/PolyToField/PolyToField.h.inc"

#define GEN_PASS_REGISTRATION
#include "zkir/Dialect/Poly/Conversions/PolyToField/PolyToField.h.inc"  // NOLINT(build/include)

}  // namespace mlir::zkir::poly

#endif  // ZKIR_DIALECT_POLY_CONVERSIONS_POLYTOFIELD_POLYTOFIELD_H_
