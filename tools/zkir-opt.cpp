/* Copyright 2025 The ZKIR Authors.

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
#include "zkir/Dialect/ArithExt/Conversions/SpecializeArithToAVX/SpecializeArithToAVX.h"
#include "zkir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/EllipticCurveToField.h"
#include "zkir/Dialect/EllipticCurve/Conversions/EllipticCurveToLLVM/EllipticCurveToLLVM.h"
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveDialect.h"
#include "zkir/Dialect/Field/Conversions/ExtFieldToLLVM/ExtFieldToLLVM.h"
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
  mlir::zkir::elliptic_curve::registerConvertEllipticCurveToLLVMInterface(
      registry);
  mlir::zkir::field::registerConvertExtFieldToLLVMInterface(registry);
  mlir::zkir::arith_ext::registerSpecializeArithToAVXPasses();
  mlir::registerAllDialects(registry);
  mlir::registerGPUDialectTranslation(registry);
  mlir::registerNVVMDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerAllExtensions(registry);

  mlir::registerAllPasses();

  // Dialect conversion passes
  mlir::zkir::mod_arith::registerModArithToArithPasses();
  mlir::zkir::field::registerFieldToModArithPasses();
  mlir::zkir::field::registerExtFieldToLLVMPasses();
  mlir::zkir::poly::registerPolyToFieldPasses();
  mlir::zkir::elliptic_curve::registerEllipticCurveToFieldPasses();
  mlir::zkir::elliptic_curve::registerEllipticCurveToLLVMPasses();
  mlir::zkir::tensor_ext::registerTensorExtToTensorPasses();

  mlir::zkir::field::registerFieldPipelines();

  return failed(mlir::MlirOptMain(argc, argv, "ZKIR optimizer\n", registry));
}
