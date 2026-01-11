/* Copyright 2025 The PrimeIR Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "llvm/Support/LogicalResult.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "prime_ir/Dialect/ArithExt/Conversions/SpecializeArithToAVX/SpecializeArithToAVX.h"
#include "prime_ir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/EllipticCurveToField.h"
#include "prime_ir/Dialect/EllipticCurve/Conversions/EllipticCurveToLLVM/EllipticCurveToLLVM.h"
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveDialect.h"
#include "prime_ir/Dialect/Field/Conversions/ExtFieldToLLVM/ExtFieldToLLVM.h"
#include "prime_ir/Dialect/Field/Conversions/FieldToModArith/FieldToModArith.h"
#include "prime_ir/Dialect/Field/IR/FieldDialect.h"
#include "prime_ir/Dialect/Field/Pipelines/Passes.h"
#include "prime_ir/Dialect/ModArith/Conversions/ModArithToArith/ModArithToArith.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithDialect.h"
#include "prime_ir/Dialect/Poly/Conversions/PolyToField/PolyToField.h"
#include "prime_ir/Dialect/Poly/IR/PolyDialect.h"
#include "prime_ir/Dialect/TensorExt/Conversions/TensorExtToTensor/TensorExtToTensor.h"
#include "prime_ir/Dialect/TensorExt/IR/TensorExtDialect.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::prime_ir::mod_arith::ModArithDialect>();
  registry.insert<mlir::prime_ir::field::FieldDialect>();
  registry.insert<mlir::prime_ir::poly::PolyDialect>();
  registry.insert<mlir::prime_ir::elliptic_curve::EllipticCurveDialect>();
  registry.insert<mlir::prime_ir::tensor_ext::TensorExtDialect>();
  mlir::prime_ir::elliptic_curve::registerConvertEllipticCurveToLLVMInterface(
      registry);
  mlir::prime_ir::field::registerConvertExtFieldToLLVMInterface(registry);
  mlir::prime_ir::arith_ext::registerSpecializeArithToAVXPasses();
  mlir::registerAllDialects(registry);
  mlir::registerGPUDialectTranslation(registry);
  mlir::registerNVVMDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerAllExtensions(registry);

  mlir::registerAllPasses();

  // Dialect conversion passes
  mlir::prime_ir::mod_arith::registerModArithToArithPasses();
  mlir::prime_ir::field::registerFieldToModArithPasses();
  mlir::prime_ir::field::registerExtFieldToLLVMPasses();
  mlir::prime_ir::poly::registerPolyToFieldPasses();
  mlir::prime_ir::elliptic_curve::registerEllipticCurveToFieldPasses();
  mlir::prime_ir::elliptic_curve::registerEllipticCurveToLLVMPasses();
  mlir::prime_ir::tensor_ext::registerTensorExtToTensorPasses();

  mlir::prime_ir::field::registerFieldPipelines();

  return failed(mlir::MlirOptMain(argc, argv, "PrimeIR optimizer\n", registry));
}
