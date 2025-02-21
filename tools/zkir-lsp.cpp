#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"
#include "zkir/Dialect/ModArith/IR/ModArithDialect.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::zkir::mod_arith::ModArithDialect>();
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  return mlir::failed(mlir::MlirLspServerMain(argc, argv, registry));
}
