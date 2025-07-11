#include "llvm/Support/LogicalResult.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "zkir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/EllipticCurveToField.h"
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveDialect.h"
#include "zkir/Dialect/Field/Conversions/FieldToModArith/FieldToModArith.h"
#include "zkir/Dialect/Field/IR/FieldDialect.h"
#include "zkir/Dialect/ModArith/Conversions/ModArithToArith/ModArithToArith.h"
#include "zkir/Dialect/ModArith/IR/ModArithDialect.h"
#include "zkir/Dialect/Poly/Conversions/PolyToField/PolyToField.h"
#include "zkir/Dialect/Poly/IR/PolyDialect.h"
#include "zkir/Dialect/TensorExt/Conversions/TensorExtToTensor/TensorExtToTensor.h"
#include "zkir/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "zkir/Pipelines/PipelineRegistration.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::zkir::mod_arith::ModArithDialect>();
  registry.insert<mlir::zkir::field::FieldDialect>();
  registry.insert<mlir::zkir::poly::PolyDialect>();
  registry.insert<mlir::zkir::elliptic_curve::EllipticCurveDialect>();
  registry.insert<mlir::zkir::tensor_ext::TensorExtDialect>();
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);

  mlir::registerAllPasses();

  // Dialect conversion passes
  mlir::zkir::mod_arith::registerModArithToArithPasses();
  mlir::zkir::field::registerFieldToModArithPasses();
  mlir::zkir::poly::registerPolyToFieldPasses();
  mlir::zkir::elliptic_curve::registerEllipticCurveToFieldPasses();
  mlir::zkir::tensor_ext::registerTensorExtToTensorPasses();

  mlir::PassPipelineRegistration<>(
      "field-to-llvm", "Run passes to lower the field dialect to LLVM",
      mlir::zkir::pipelines::fieldToLLVMPipelineBuilder<false>);
  mlir::PassPipelineRegistration<>(
      "field-to-omp", "Run passes to lower the field dialect to OpenMP + LLVM",
      mlir::zkir::pipelines::fieldToLLVMPipelineBuilder<true>);
  mlir::PassPipelineRegistration<>(
      "poly-to-llvm", "Run passes to lower the polynomial dialect to LLVM",
      mlir::zkir::pipelines::polyToLLVMPipelineBuilder<false>);
  mlir::PassPipelineRegistration<>(
      "poly-to-omp",
      "Run passes to lower the polynomial dialect to OpenMP + LLVM",
      mlir::zkir::pipelines::polyToLLVMPipelineBuilder<true>);
  mlir::PassPipelineRegistration<>(
      "elliptic-curve-to-llvm",
      "Run passes to lower the elliptic curve dialect to LLVM",
      mlir::zkir::pipelines::ellipticCurveToLLVMPipelineBuilder<false>);
  mlir::PassPipelineRegistration<>(
      "elliptic-curve-to-omp",
      "Run passes to lower the elliptic curve dialect to OpenMP + LLVM",
      mlir::zkir::pipelines::ellipticCurveToLLVMPipelineBuilder<true>);

  return failed(mlir::MlirOptMain(argc, argv, "ZKIR optimizer\n", registry));
}
