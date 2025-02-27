#ifndef ZKIR_DIALECT_POLY_IR_POLYOPS_H_
#define ZKIR_DIALECT_POLY_IR_POLYOPS_H_

#include "mlir/include/mlir/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"
#include "mlir/include/mlir/IR/BuiltinTypes.h"
#include "mlir/include/mlir/IR/Dialect.h"
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"
#include "zkir/Dialect/Field/IR/FieldTypes.h"
#include "zkir/Dialect/Poly/IR/PolyAttributes.h"
#include "zkir/Dialect/Poly/IR/PolyDialect.h"
#include "zkir/Dialect/Poly/IR/PolyTypes.h"

#define GET_OP_CLASSES
#include "zkir/Dialect/Poly/IR/PolyOps.h.inc"

#endif  // ZKIR_DIALECT_POLY_IR_POLYOPS_H_
