#ifndef ZKIR_DIALECT_ARITH_CONVERSIONS_ARITHTOMODARITH_ARITHTOMODARITH_H_
#define ZKIR_DIALECT_ARITH_CONVERSIONS_ARITHTOMODARITH_ARITHTOMODARITH_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace zkir {
namespace arith {

#define GEN_PASS_DECL
#include "zkir/Dialect/Arith/Conversions/ArithToModArith/ArithToModArith.h.inc"

#define GEN_PASS_REGISTRATION
#include "zkir/Dialect/Arith/Conversions/ArithToModArith/ArithToModArith.h.inc"  // NOLINT(build/include)

}  // namespace arith
}  // namespace zkir
}  // namespace mlir

#endif  // ZKIR_DIALECT_ARITH_CONVERSIONS_ARITHTOMODARITH_ARITHTOMODARITH_H_
