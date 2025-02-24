#include "llvm/Support/LogicalResult.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "zkir/Dialect/Arith/Conversions/ArithToModArith/ArithToModArith.h"
#include "zkir/Dialect/ModArith/Conversions/ModArithToArith/ModArithToArith.h"
#include "zkir/Dialect/ModArith/IR/ModArithDialect.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::zkir::mod_arith::ModArithDialect>();
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);

  mlir::registerAllPasses();

  // Dialect conversion passes
  mlir::zkir::mod_arith::registerModArithToArithPasses();
  mlir::zkir::arith::registerArithToModArithPasses();

  return failed(mlir::MlirOptMain(argc, argv, "ZKIR optimizer\n", registry));
}
