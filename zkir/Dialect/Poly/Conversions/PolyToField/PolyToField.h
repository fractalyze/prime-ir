#ifndef ZKIR_DIALECT_POLY_CONVERSIONS_POLYTOFIELD_POLYTOFIELD_H_
#define ZKIR_DIALECT_POLY_CONVERSIONS_POLYTOFIELD_POLYTOFIELD_H_

// IWYU pragma: begin_keep
// Headers needed for PolyToField.h.inc
#include "mlir/Pass/Pass.h"
#include "zkir/Dialect/Field/IR/FieldDialect.h"
#include "zkir/Dialect/TensorExt/IR/TensorExtDialect.h"
// IWYU pragma: end_keep

namespace mlir::zkir::poly {

#define GEN_PASS_DECL
#include "zkir/Dialect/Poly/Conversions/PolyToField/PolyToField.h.inc"

#define GEN_PASS_REGISTRATION
#include "zkir/Dialect/Poly/Conversions/PolyToField/PolyToField.h.inc"  // NOLINT(build/include)

}  // namespace mlir::zkir::poly

#endif  // ZKIR_DIALECT_POLY_CONVERSIONS_POLYTOFIELD_POLYTOFIELD_H_
