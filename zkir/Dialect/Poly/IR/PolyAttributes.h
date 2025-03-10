#ifndef ZKIR_DIALECT_POLY_IR_POLYATTRIBUTES_H_
#define ZKIR_DIALECT_POLY_IR_POLYATTRIBUTES_H_

#include "mlir/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "zkir/Dialect/Field/IR/FieldAttributes.h"
#include "zkir/Dialect/Poly/IR/PolyDialect.h"
#include "zkir/Dialect/Poly/IR/PolyTypes.h"

#define GET_ATTRDEF_CLASSES
#include "zkir/Dialect/Poly/IR/PolyAttributes.h.inc"

#endif  // ZKIR_DIALECT_POLY_IR_POLYATTRIBUTES_H_
