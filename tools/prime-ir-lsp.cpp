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

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveDialect.h"
#include "prime_ir/Dialect/Field/IR/FieldDialect.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithDialect.h"
#include "prime_ir/Dialect/Poly/IR/PolyDialect.h"
#include "prime_ir/Dialect/TensorExt/IR/TensorExtDialect.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::prime_ir::mod_arith::ModArithDialect>();
  registry.insert<mlir::prime_ir::field::FieldDialect>();
  registry.insert<mlir::prime_ir::poly::PolyDialect>();
  registry.insert<mlir::prime_ir::elliptic_curve::EllipticCurveDialect>();
  registry.insert<mlir::prime_ir::tensor_ext::TensorExtDialect>();
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  return mlir::failed(mlir::MlirLspServerMain(argc, argv, registry));
}
