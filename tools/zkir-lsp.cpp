#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"
#include "zkir/Dialect/Field/IR/FieldDialect.h"
#include "zkir/Dialect/ModArith/IR/ModArithDialect.h"
#include "zkir/Dialect/Poly/IR/PolyDialect.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::zkir::mod_arith::ModArithDialect>();
  registry.insert<mlir::zkir::field::FieldDialect>();
  registry.insert<mlir::zkir::poly::PolyDialect>();
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  return mlir::failed(mlir::MlirLspServerMain(argc, argv, registry));
}
