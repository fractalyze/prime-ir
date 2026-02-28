/* Copyright 2021 The OpenXLA Authors.
Copyright 2025 The ZKX Authors.

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

#include "zkx/pjrt/mlir_to_hlo.h"

#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/Version.h"
#include "stablehlo/transforms/Passes.h"
#include "xla/tsl/platform/statusor.h"
#include "zkx/hlo/translate/mhlo_to_hlo/mlir_hlo_to_hlo.h"
#include "zkx/mlir/utils/error_util.h"
#include "zkx/mlir_hlo/mhlo/IR/register.h"
#include "zkx/mlir_hlo/mhlo/transforms/passes.h"

namespace zkx {

absl::Status MlirToZkxComputation(mlir::ModuleOp module,
                                  ZkxComputation& zkx_computation,
                                  bool use_tuple_args, bool return_tuple,
                                  bool use_shardy) {
  mlir::MLIRContext* context = module->getContext();
  mlir::BaseScopedDiagnosticHandler diagnostic_handler(context);
  {
    mlir::PassManager pm(context);
    pm.addPass(mlir::mhlo::createStablehloLegalizeToHloPass());
    // TODO(chokobole): Uncomment this. Dependency: createChloLegalizeToHloPass
    // pm.addNestedPass<mlir::func::FuncOp>(
    //     mlir::mhlo::createChloLegalizeToHloPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
    // In order to export to ZKX, we must sink constants to control flow
    // regions, since ZKX uses functional control flow.
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::mhlo::createSinkConstantsToControlFlowPass());
    if (failed(pm.run(module))) {
      VLOG(1) << "MHLO->HLO lowering passes failed.";
      module->dump();
      return diagnostic_handler.ConsumeStatus();
    }

    VLOG(5) << "MHLO module after lowering, before HLO import ";
    if (VLOG_IS_ON(5)) {
      module->dump();
    }
  }

  // TODO(b/345414638): Delete when we move Shardy as the first pass in the
  // ZKX pipeline.
  // TODO(chokobole): Uncomment this. Dependency: shardy
  // if (use_tuple_args && use_shardy) {
  //   // Shardy can't handle tuple args when round-tripping. So delay using
  //   // tuples until after Shardy is run.
  //   sdy::setFrontendAttribute(module, sdy::kUseTupleArgs,
  //                             mlir::StringAttr::get(context, "t"));
  //   use_tuple_args = false;
  // }

  // create config options use use_tuple_args, return_tuple set:
  mlir::MlirToHloConversionOptions options;
  options.use_tuple_args = use_tuple_args;
  options.return_tuple = return_tuple;

  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> hlo_module,
                      mlir::ConvertMlirHloToHloModule(module, options));

  zkx_computation = ZkxComputation(hlo_module->ToProto());
  return absl::OkStatus();
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ParseMlirModuleString(
    std::string_view mlir_module_str, mlir::MLIRContext& context) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::shape::ShapeDialect>();
  mlir::func::registerAllExtensions(registry);
  mlir::mhlo::registerAllMhloDialects(registry);
  // TODO(chokobole): Uncomment this. Dependency: sdy
  // mlir::sdy::registerAllDialects(registry);
  mlir::stablehlo::registerAllDialects(registry);
  context.appendDialectRegistry(registry);

  mlir::BaseScopedDiagnosticHandler diagnostic_handler(&context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(
          llvm::StringRef(mlir_module_str.data(), mlir_module_str.size()),
          mlir::ParserConfig{&context});
  if (!module) {
    mlir::emitError(mlir::UnknownLoc::get(&context))
        << "Failed to parse using StableHLO v"
        << mlir::vhlo::Version::getCurrentVersion() << ", "
        << "this could indicate forward incompatibility, >12w old "
           "unsupported plugin, or a portable artifact that needs to be "
           "further downgraded.";
    return diagnostic_handler.ConsumeStatus();
  }

  TF_RETURN_IF_ERROR(UpgradeVersionedStablehlo(*module));
  return std::move(module);
}

absl::Status UpgradeVersionedStablehlo(mlir::ModuleOp mlir_module) {
  // Upgrade if VHLO
  mlir::PassManager pm(mlir_module->getContext());
  mlir::stablehlo::createStablehloDeserializePipeline(pm);
  if (!mlir::succeeded(pm.run(mlir_module)))
    return absl::InvalidArgumentError("Failed to upgrade versioned StableHLO.");
  return absl::OkStatus();
}

}  // namespace zkx
