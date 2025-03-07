#ifndef ZKIR_DIALECT_ARITH_CONVERSIONS_ARITHTOMODARITH_ARITHTOMODARITH_H_
#define ZKIR_DIALECT_ARITH_CONVERSIONS_ARITHTOMODARITH_ARITHTOMODARITH_H_

#include "mlir/include/mlir/Pass/Pass.h"

namespace mlir::zkir::arith {

#define GEN_PASS_DECL
#include "zkir/Dialect/Arith/Conversions/ArithToModArith/ArithToModArith.h.inc"

#define GEN_PASS_REGISTRATION
#include "zkir/Dialect/Arith/Conversions/ArithToModArith/ArithToModArith.h.inc"  // NOLINT(build/include)

}  // namespace mlir::zkir::arith

#endif  // ZKIR_DIALECT_ARITH_CONVERSIONS_ARITHTOMODARITH_ARITHTOMODARITH_H_
