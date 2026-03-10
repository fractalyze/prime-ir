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

#ifndef ZKX_TOOLS_STABLEHLO_RUNNER_STABLEHLO_UTILS_H_
#define ZKX_TOOLS_STABLEHLO_RUNNER_STABLEHLO_UTILS_H_

#include <memory>
#include <string_view>

#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/literal.h"

namespace zkx {

// Parses a StableHLO MLIR module text, registering StableHLO, Field, and
// EllipticCurve dialects.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ParseStablehloModule(
    std::string_view module_text, mlir::MLIRContext* context);

// Converts a parsed StableHLO module to an HloModule via MlirToZkxComputation.
absl::StatusOr<std::unique_ptr<HloModule>> ConvertStablehloToHloModule(
    mlir::ModuleOp module);

// Fills a literal with valid random data using type-aware generation.
//
// For field types: NativeT::Random() generates valid field elements.
// For EC affine points: NativeT::Generator() (all same base point, fast).
//   Set use_random_points=true to use NativeT::Random() (scalar mul, slow).
// For BigInt types: NativeT::Random().
// For other types: fills raw bytes with seeded mt19937_64.
void FillLiteralWithRandom(Literal& literal, bool use_random_points = false);

}  // namespace zkx

#endif  // ZKX_TOOLS_STABLEHLO_RUNNER_STABLEHLO_UTILS_H_
