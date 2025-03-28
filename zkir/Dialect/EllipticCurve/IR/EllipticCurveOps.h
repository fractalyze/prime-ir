#ifndef ZKIR_DIALECT_ELLIPTICCURVE_IR_ELLIPTICCURVEOPS_H_
#define ZKIR_DIALECT_ELLIPTICCURVE_IR_ELLIPTICCURVEOPS_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveDialect.h"
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveTypes.h"
#include "zkir/Dialect/Field/IR/FieldAttributes.h"

#define GET_OP_CLASSES
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveOps.h.inc"

#endif  // ZKIR_DIALECT_ELLIPTICCURVE_IR_ELLIPTICCURVEOPS_H_
