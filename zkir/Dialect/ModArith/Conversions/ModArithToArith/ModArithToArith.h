#ifndef ZKIR_DIALECT_MODARITH_CONVERSIONS_MODARITHTOARITH_MODARITHTOARITH_H_
#define ZKIR_DIALECT_MODARITH_CONVERSIONS_MODARITHTOARITH_MODARITHTOARITH_H_

#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/include/mlir/Pass/Pass.h"

namespace mlir::zkir::mod_arith {

#define GEN_PASS_DECL
#include "zkir/Dialect/ModArith/Conversions/ModArithToArith/ModArithToArith.h.inc"

#define GEN_PASS_REGISTRATION
#include "zkir/Dialect/ModArith/Conversions/ModArithToArith/ModArithToArith.h.inc"  // NOLINT(build/include)

}  // namespace mlir::zkir::mod_arith

#endif  // ZKIR_DIALECT_MODARITH_CONVERSIONS_MODARITHTOARITH_MODARITHTOARITH_H_
