#ifndef ZKIR_DIALECT_MODARITH_CONVERSIONS_MODARITHTOARITH_MODARITHTOARITH_H_
#define ZKIR_DIALECT_MODARITH_CONVERSIONS_MODARITHTOARITH_MODARITHTOARITH_H_

// IWYU pragma: begin_keep
// Headers needed for ModArithToArith.h.inc
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"
// IWYU pragma: end_keep

namespace mlir::zkir::mod_arith {

#define GEN_PASS_DECL
#include "zkir/Dialect/ModArith/Conversions/ModArithToArith/ModArithToArith.h.inc"

#define GEN_PASS_REGISTRATION
#include "zkir/Dialect/ModArith/Conversions/ModArithToArith/ModArithToArith.h.inc" // NOLINT(build/include)

} // namespace mlir::zkir::mod_arith

#endif // ZKIR_DIALECT_MODARITH_CONVERSIONS_MODARITHTOARITH_MODARITHTOARITH_H_
