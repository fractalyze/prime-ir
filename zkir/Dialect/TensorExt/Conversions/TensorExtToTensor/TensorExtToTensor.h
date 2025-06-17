#ifndef ZKIR_DIALECT_TENSOREXT_CONVERSIONS_TENSOREXTTOTENSOR_TENSOREXTTOTENSOR_H_
#define ZKIR_DIALECT_TENSOREXT_CONVERSIONS_TENSOREXTTOTENSOR_TENSOREXTTOTENSOR_H_

#include "mlir/Pass/Pass.h"

namespace mlir::zkir::tensor_ext {
#define GEN_PASS_DECL
#include "zkir/Dialect/TensorExt/Conversions/TensorExtToTensor/TensorExtToTensor.h.inc"

#define GEN_PASS_REGISTRATION
#include "zkir/Dialect/TensorExt/Conversions/TensorExtToTensor/TensorExtToTensor.h.inc"  // NOLINT(build/include)
}  // namespace mlir::zkir::tensor_ext

#endif  // ZKIR_DIALECT_TENSOREXT_CONVERSIONS_TENSOREXTTOTENSOR_TENSOREXTTOTENSOR_H_
