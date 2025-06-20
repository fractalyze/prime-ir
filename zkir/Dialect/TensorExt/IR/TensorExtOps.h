#ifndef ZKIR_DIALECT_TENSOREXT_IR_TENSOREXTOPS_H_
#define ZKIR_DIALECT_TENSOREXT_IR_TENSOREXTOPS_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "zkir/Dialect/TensorExt/IR/TensorExtDialect.h"

#define GET_OP_CLASSES
#include "zkir/Dialect/TensorExt/IR/TensorExtOps.h.inc"

#endif  // ZKIR_DIALECT_TENSOREXT_IR_TENSOREXTOPS_H_
