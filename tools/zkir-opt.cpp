#include "llvm/Support/LogicalResult.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "zkir/Dialect/Arith/Conversions/ArithToModArith/ArithToModArith.h"
#include "zkir/Dialect/Field/Conversions/FieldToModArith/FieldToModArith.h"
#include "zkir/Dialect/Field/IR/FieldDialect.h"
#include "zkir/Dialect/ModArith/Conversions/ModArithToArith/ModArithToArith.h"
#include "zkir/Dialect/ModArith/IR/ModArithDialect.h"
#include "zkir/Dialect/Poly/Conversions/PolyToField/PolyToField.h"
#include "zkir/Dialect/Poly/IR/PolyDialect.h"
#include "zkir/Pipelines/PipelineRegistration.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::zkir::mod_arith::ModArithDialect>();
  registry.insert<mlir::zkir::field::FieldDialect>();
  registry.insert<mlir::zkir::poly::PolyDialect>();
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);

  mlir::registerAllPasses();

  // Dialect conversion passes
  mlir::zkir::mod_arith::registerModArithToArithPasses();
  mlir::zkir::arith::registerArithToModArithPasses();
  mlir::zkir::field::registerFieldToModArithPasses();
  mlir::zkir::poly::registerPolyToFieldPasses();

  mlir::PassPipelineRegistration<>(
      "poly-to-llvm", "Run passes to lower the polynomial dialect to LLVM",
      mlir::zkir::pipelines::polyToLLVMPipelineBuilder);

  return failed(mlir::MlirOptMain(argc, argv, "ZKIR optimizer\n", registry));
}
