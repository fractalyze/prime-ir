#include "llvm/Support/LogicalResult.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "zkir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/EllipticCurveToField.h"
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveDialect.h"
#include "zkir/Dialect/Field/Conversions/FieldToModArith/FieldToModArith.h"
#include "zkir/Dialect/Field/IR/FieldDialect.h"
#include "zkir/Dialect/Field/Pipelines/Passes.h"
#include "zkir/Dialect/ModArith/Conversions/ModArithToArith/ModArithToArith.h"
#include "zkir/Dialect/ModArith/IR/ModArithDialect.h"
#include "zkir/Dialect/Poly/Conversions/PolyToField/PolyToField.h"
#include "zkir/Dialect/Poly/IR/PolyDialect.h"
#include "zkir/Dialect/TensorExt/Conversions/TensorExtToTensor/TensorExtToTensor.h"
#include "zkir/Dialect/TensorExt/IR/TensorExtDialect.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::zkir::mod_arith::ModArithDialect>();
  registry.insert<mlir::zkir::field::FieldDialect>();
  registry.insert<mlir::zkir::poly::PolyDialect>();
  registry.insert<mlir::zkir::elliptic_curve::EllipticCurveDialect>();
  registry.insert<mlir::zkir::tensor_ext::TensorExtDialect>();
  mlir::registerAllDialects(registry);
  mlir::registerGPUDialectTranslation(registry);
  mlir::registerNVVMDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerAllExtensions(registry);

  mlir::registerAllPasses();

  // Dialect conversion passes
  mlir::zkir::mod_arith::registerModArithToArithPasses();
  mlir::zkir::field::registerFieldToModArithPasses();
  mlir::zkir::poly::registerPolyToFieldPasses();
  mlir::zkir::elliptic_curve::registerEllipticCurveToFieldPasses();
  mlir::zkir::tensor_ext::registerTensorExtToTensorPasses();

  mlir::zkir::field::registerFieldPipelines();

  return failed(mlir::MlirOptMain(argc, argv, "ZKIR optimizer\n", registry));
}
