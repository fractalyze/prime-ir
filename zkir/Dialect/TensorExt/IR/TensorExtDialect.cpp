#include "zkir/Dialect/TensorExt/IR/TensorExtDialect.h"

#include "zkir/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "zkir/Dialect/TensorExt/IR/TensorExtOps.h"

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
