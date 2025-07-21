#ifndef ZKIR_DIALECT_TENSOREXT_IR_TENSOREXTOPS_H_
#define ZKIR_DIALECT_TENSOREXT_IR_TENSOREXTOPS_H_

// IWYU pragma: begin_keep
// Headers needed for TensorExtOps.h.inc
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
// IWYU pragma: end_keep

#define GET_OP_CLASSES
#include "zkir/Dialect/TensorExt/IR/TensorExtOps.h.inc"

#endif  // ZKIR_DIALECT_TENSOREXT_IR_TENSOREXTOPS_H_
