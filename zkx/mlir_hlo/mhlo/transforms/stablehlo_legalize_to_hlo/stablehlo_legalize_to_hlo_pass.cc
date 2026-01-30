/* Copyright 2022 The OpenXLA Authors.
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

#include <memory>
#include <utility>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "zkx/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "zkx/mlir_hlo/mhlo/transforms/rewriters.h"
#include "zkx/mlir_hlo/mhlo/utils/type_conversion.h"

namespace mlir::mhlo {

#define GEN_PASS_DEF_STABLEHLOLEGALIZETOHLOPASS
#include "zkx/mlir_hlo/mhlo/transforms/mhlo_passes.h.inc"

namespace {

struct StablehloLegalizeToHloPass
    : public impl::StablehloLegalizeToHloPassBase<StablehloLegalizeToHloPass> {
  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addIllegalDialect<stablehlo::StablehloDialect>();
    target.addLegalDialect<mhlo::MhloDialect>();

    stablehlo::StablehloToHloTypeConverter converter;
    RewritePatternSet patterns(&getContext());
    stablehlo::populateStablehloToHloPatterns(&patterns, &converter,
                                              &getContext());
    stablehlo::registerFuncOpsForTypeConversion(target, patterns, converter);

    // Our guiding principle is to support all StableHLO functionality in MHLO.
    // This check is here only for exceptional situations, e.g. when we added
    // a new StableHLO op and forgot to update the conversion patterns.
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<ModuleOp>>
createStablehloLegalizeToHloPass() {
  return std::make_unique<StablehloLegalizeToHloPass>();
}

} // namespace mlir::mhlo
