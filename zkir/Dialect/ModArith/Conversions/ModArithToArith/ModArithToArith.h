#ifndef ZKIR_DIALECT_MODARITH_CONVERSIONS_MODARITHTOARITH_MODARITHTOARITH_H_
#define ZKIR_DIALECT_MODARITH_CONVERSIONS_MODARITHTOARITH_MODARITHTOARITH_H_

#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"               // from @llvm-project

namespace mlir {
namespace zkir {
namespace mod_arith {

#define GEN_PASS_DECL
#include "zkir/Dialect/ModArith/Conversions/ModArithToArith/ModArithToArith.h.inc"

#define GEN_PASS_REGISTRATION
#include "zkir/Dialect/ModArith/Conversions/ModArithToArith/ModArithToArith.h.inc"  // NOLINT(build/include)

}  // namespace mod_arith
}  // namespace zkir
}  // namespace mlir

#endif  // ZKIR_DIALECT_MODARITH_CONVERSIONS_MODARITHTOARITH_MODARITHTOARITH_H_
