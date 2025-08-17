#ifndef ZKIR_DIALECT_POLY_IR_POLYOPS_H_
#define ZKIR_DIALECT_POLY_IR_POLYOPS_H_

// IWYU pragma: begin_keep
// Headers needed for PolyOps.h.inc
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "zkir/Dialect/Field/IR/FieldAttributes.h"
#include "zkir/Dialect/Poly/IR/PolyTypes.h"
// IWYU pragma: end_keep

#define GET_OP_CLASSES
#include "zkir/Dialect/Poly/IR/PolyOps.h.inc"

#endif // ZKIR_DIALECT_POLY_IR_POLYOPS_H_
