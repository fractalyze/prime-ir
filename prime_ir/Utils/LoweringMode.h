/* Copyright 2026 The PrimeIR Authors.

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

#ifndef PRIME_IR_UTILS_LOWERINGMODE_H_
#define PRIME_IR_UTILS_LOWERINGMODE_H_

#include "llvm/ADT/StringRef.h"

namespace mlir::prime_ir {

/// Lowering mode for complex operations in conversion passes.
///
/// This enum controls how operations are lowered during dialect conversions:
/// - Inline: All operations are expanded inline (default behavior)
/// - Intrinsic: Complex operations are lowered to function calls with
///   out-parameters to avoid large aggregate returns (useful for NVPTX)
/// - Auto: Uses heuristics to decide based on operation complexity
enum class LoweringMode {
  /// All operations are expanded inline (default).
  Inline,
  /// Complex operations are lowered to function calls.
  Intrinsic,
  /// Use heuristics to decide based on complexity.
  Auto
};

/// Parse the lowering mode from a string.
/// Returns LoweringMode::Inline for unrecognized values.
inline LoweringMode parseLoweringMode(llvm::StringRef mode) {
  if (mode == "intrinsic")
    return LoweringMode::Intrinsic;
  if (mode == "auto")
    return LoweringMode::Auto;
  // Default to inline
  return LoweringMode::Inline;
}

/// Convert lowering mode to string for debugging/logging.
inline llvm::StringRef loweringModeToString(LoweringMode mode) {
  switch (mode) {
  case LoweringMode::Inline:
    return "inline";
  case LoweringMode::Intrinsic:
    return "intrinsic";
  case LoweringMode::Auto:
    return "auto";
  }
  return "inline";
}

} // namespace mlir::prime_ir

#endif // PRIME_IR_UTILS_LOWERINGMODE_H_
