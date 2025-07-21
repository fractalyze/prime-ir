#ifndef ZKIR_DIALECT_TENSOREXT_CONVERSIONS_TENSOREXTTOTENSOR_TENSOREXTTOTENSOR_H_
#define ZKIR_DIALECT_TENSOREXT_CONVERSIONS_TENSOREXTTOTENSOR_TENSOREXTTOTENSOR_H_

// IWYU pragma: begin_keep
// Headers needed for TensorExtToTensor.h.inc
#include "mlir/Pass/Pass.h"
// IWYU pragma: end_keep

namespace mlir::zkir::tensor_ext {
#define GEN_PASS_DECL
#include "zkir/Dialect/TensorExt/Conversions/TensorExtToTensor/TensorExtToTensor.h.inc"

#define GEN_PASS_REGISTRATION
#include "zkir/Dialect/TensorExt/Conversions/TensorExtToTensor/TensorExtToTensor.h.inc"  // NOLINT(build/include)
}  // namespace mlir::zkir::tensor_ext

#endif  // ZKIR_DIALECT_TENSOREXT_CONVERSIONS_TENSOREXTTOTENSOR_TENSOREXTTOTENSOR_H_
