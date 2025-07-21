#ifndef ZKIR_DIALECT_ELLIPTICCURVE_IR_ELLIPTICCURVEOPS_H_
#define ZKIR_DIALECT_ELLIPTICCURVE_IR_ELLIPTICCURVEOPS_H_

// IWYU pragma: begin_keep
// Headers needed for EllipticCurveOps.h.inc
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
// IWYU pragma: end_keep

#define GET_OP_CLASSES
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveOps.h.inc"

#endif  // ZKIR_DIALECT_ELLIPTICCURVE_IR_ELLIPTICCURVEOPS_H_
