#ifndef ZKIR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_POINTOPERATIONS_XYZZ_DOUBLE_H_
#define ZKIR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_POINTOPERATIONS_XYZZ_DOUBLE_H_

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveTypes.h"
#include "zkir/Dialect/Field/IR/FieldOps.h"
#include "zkir/Dialect/Field/IR/FieldTypes.h"

namespace mlir::zkir::elliptic_curve {

Value affineToXYZZDouble(const Value &point, Type inputType,
                         ImplicitLocOpBuilder &b);
Value xyzzDouble(const Value &point, Type inputType, ImplicitLocOpBuilder &b);

}  // namespace mlir::zkir::elliptic_curve

#endif  // ZKIR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_POINTOPERATIONS_XYZZ_DOUBLE_H_
