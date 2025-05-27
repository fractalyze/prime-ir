#ifndef ZKIR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_POINTOPERATIONS_XYZZ_ADD_H_
#define ZKIR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_POINTOPERATIONS_XYZZ_ADD_H_

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveAttributes.h"
#include "zkir/Dialect/Field/IR/FieldOps.h"

namespace mlir::zkir::elliptic_curve {

SmallVector<Value> xyzzAdd(ValueRange p1, ValueRange p2,
                           ShortWeierstrassAttr curve, ImplicitLocOpBuilder &b);

}  // namespace mlir::zkir::elliptic_curve

#endif  // ZKIR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_POINTOPERATIONS_XYZZ_ADD_H_
