/* Copyright 2026 The ZKX Authors.

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

#include "zkx/tools/stablehlo_runner/stablehlo_utils.h"

#include <memory>
#include <string_view>
#include <utility>

#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"

#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveDialect.h"
#include "prime_ir/Dialect/Field/IR/FieldDialect.h"
#include "stablehlo/dialect/Register.h"
#include "xla/tsl/platform/statusor.h"
#include "zkx/debug_options_flags.h"
#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/mlir/utils/error_util.h"
#include "zkx/pjrt/mlir_to_hlo.h"

namespace zkx {

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ParseStablehloModule(
    std::string_view module_text, mlir::MLIRContext* context) {
  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  registry.insert<mlir::prime_ir::field::FieldDialect>();
  registry.insert<mlir::prime_ir::elliptic_curve::EllipticCurveDialect>();
  context->appendDialectRegistry(registry);
  context->loadAllAvailableDialects();

  mlir::BaseScopedDiagnosticHandler diagnostic_handler(context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(
          llvm::StringRef(module_text.data(), module_text.size()),
          mlir::ParserConfig{context});
  if (!module) {
    mlir::emitError(mlir::UnknownLoc::get(context))
        << "Failed to parse StableHLO module";
    return diagnostic_handler.ConsumeStatus();
  }
  return std::move(module);
}

absl::StatusOr<std::unique_ptr<HloModule>> ConvertStablehloToHloModule(
    mlir::ModuleOp module) {
  ZkxComputation computation;
  TF_RETURN_IF_ERROR(MlirToZkxComputation(
      module, computation, /*use_tuple_args=*/false, /*return_tuple=*/false,
      /*use_shardy=*/false));
  TF_ASSIGN_OR_RETURN(HloModuleConfig config,
                      HloModule::CreateModuleConfigFromProto(
                          computation.proto(), GetDebugOptionsFromFlags()));
  return HloModule::CreateFromProto(computation.proto(), config);
}

}  // namespace zkx
