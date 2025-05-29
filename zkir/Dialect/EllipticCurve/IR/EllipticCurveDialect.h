#ifndef ZKIR_DIALECT_ELLIPTICCURVE_IR_ELLIPTICCURVEDIALECT_H_
#define ZKIR_DIALECT_ELLIPTICCURVE_IR_ELLIPTICCURVEDIALECT_H_

#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "zkir/Dialect/Field/IR/FieldAttributes.h"
#include "zkir/Dialect/Field/IR/FieldTypes.h"

// Generated headers (block clang-format from messing up order)
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveDialect.h.inc"

namespace mlir::zkir::elliptic_curve {
// Helper functions for the Elliptic Curve dialect
class ShortWeierstrassAttr;

size_t getNumCoordsFromPointLike(Type pointLike);
ShortWeierstrassAttr getCurveFromPointLike(Type pointLike);
}  // namespace mlir::zkir::elliptic_curve

#endif  // ZKIR_DIALECT_ELLIPTICCURVE_IR_ELLIPTICCURVEDIALECT_H_
