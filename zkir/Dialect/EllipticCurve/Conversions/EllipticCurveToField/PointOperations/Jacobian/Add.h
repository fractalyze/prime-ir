#ifndef ZKIR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_POINTOPERATIONS_JACOBIAN_ADD_H_
#define ZKIR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_POINTOPERATIONS_JACOBIAN_ADD_H_

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveAttributes.h"

namespace mlir::zkir::elliptic_curve {

SmallVector<Value> jacobianAdd(ValueRange p1, ValueRange p2,
                               ShortWeierstrassAttr curve,
                               ImplicitLocOpBuilder &b);

}  // namespace mlir::zkir::elliptic_curve

#endif  // ZKIR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_POINTOPERATIONS_JACOBIAN_ADD_H_
