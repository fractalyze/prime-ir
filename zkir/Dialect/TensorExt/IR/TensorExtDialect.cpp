#include "zkir/Dialect/TensorExt/IR/TensorExtDialect.h"

// IWYU pragma: begin_keep
// Headers needed for TensorExtDialect.cpp.inc
#include "mlir/Dialect/Tensor/IR/Tensor.h"
// Headers needed for TensorExtAttributes.cpp.inc
#include "zkir/Dialect/TensorExt/IR/TensorExtAttributes.h"
// Headers needed for TensorExtOps.cpp.inc
#include "zkir/Dialect/TensorExt/IR/TensorExtOps.h"
// IWYU pragma: end_keep

// Generated definitions
#include "zkir/Dialect/TensorExt/IR/TensorExtDialect.cpp.inc"

#define GET_OP_CLASSES
#include "zkir/Dialect/TensorExt/IR/TensorExtOps.cpp.inc"

namespace mlir::zkir::tensor_ext {

void TensorExtDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "zkir/Dialect/TensorExt/IR/TensorExtAttributes.cpp.inc"  // NOLINT(build/include)
      >();
  addOperations<
#define GET_OP_LIST
#include "zkir/Dialect/TensorExt/IR/TensorExtOps.cpp.inc"  // NOLINT(build/include)
      >();
}

}  // namespace mlir::zkir::tensor_ext
