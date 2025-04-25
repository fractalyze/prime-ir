#ifndef ZKIR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_ELLIPTICCURVETOFIELD_H_
#define ZKIR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_ELLIPTICCURVETOFIELD_H_

#include "mlir/Pass/Pass.h"

namespace mlir::zkir::elliptic_curve {

#define GEN_PASS_DECL
#include "zkir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/EllipticCurveToField.h.inc"

#define GEN_PASS_REGISTRATION
#include "zkir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/EllipticCurveToField.h.inc"  // NOLINT(build/include)

}  // namespace mlir::zkir::elliptic_curve

#endif  // ZKIR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_ELLIPTICCURVETOFIELD_H_
