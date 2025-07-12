#ifndef ZKIR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_POINTOPERATIONS_XYZZ_DOUBLE_H_
#define ZKIR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_POINTOPERATIONS_XYZZ_DOUBLE_H_

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveAttributes.h"

namespace mlir::zkir::elliptic_curve {

SmallVector<Value> xyzzDouble(ValueRange point, ShortWeierstrassAttr curve,
                              ImplicitLocOpBuilder &b);

}  // namespace mlir::zkir::elliptic_curve

#endif  // ZKIR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_POINTOPERATIONS_XYZZ_DOUBLE_H_
