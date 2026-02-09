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

#ifndef PRIME_IR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_INTRINSICFUNCTIONGENERATOR_H_
#define PRIME_IR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_INTRINSICFUNCTIONGENERATOR_H_

#include "prime_ir/Dialect/Field/IR/FieldTypes.h"
#include "prime_ir/Utils/IntrinsicFunctionGeneratorBase.h"
#include "prime_ir/Utils/LoweringMode.h"

namespace mlir::prime_ir::field {

/// Generates and manages intrinsic functions for complex field operations.
///
/// Intrinsic functions use high-level extension field types directly in their
/// signatures. The lowering pass handles type decomposition automatically.
///
/// Example intrinsic signature for quartic mul:
///   func @__prime_ir_ext4_mul_<modulus>(
///       %a: !field.extension<...>,
///       %b: !field.extension<...>) -> !field.extension<...>
class IntrinsicFunctionGenerator
    : public IntrinsicFunctionGeneratorBase<IntrinsicFunctionGenerator> {
  using Base = IntrinsicFunctionGeneratorBase<IntrinsicFunctionGenerator>;

public:
  explicit IntrinsicFunctionGenerator(ModuleOp module) : Base(module) {}

  /// Get or create the intrinsic function for quartic extension field multiply.
  func::FuncOp getOrCreateQuarticMulFunction(ExtensionFieldType type);

  /// Get or create the intrinsic function for quartic extension field square.
  func::FuncOp getOrCreateQuarticSquareFunction(ExtensionFieldType type);

  /// Get or create the intrinsic function for quartic extension field inverse.
  func::FuncOp getOrCreateQuarticInverseFunction(ExtensionFieldType type);

  /// Emit a call to the quartic multiplication intrinsic.
  Value emitQuarticMulCall(OpBuilder &builder, Location loc,
                           ExtensionFieldType type, Value lhs, Value rhs);

  /// Emit a call to the quartic square intrinsic.
  Value emitQuarticSquareCall(OpBuilder &builder, Location loc,
                              ExtensionFieldType type, Value input);

  /// Emit a call to the quartic inverse intrinsic.
  Value emitQuarticInverseCall(OpBuilder &builder, Location loc,
                               ExtensionFieldType type, Value input);

  /// Determine if an operation should use intrinsic mode based on type and
  /// mode. Returns false if the operation is inside an intrinsic function
  /// (detected by `__prime_ir_` prefix) to prevent infinite recursion.
  static bool shouldUseIntrinsic(Operation *op, Type fieldType,
                                 LoweringMode mode);

private:
  /// Generate a mangled function name encoding the type information.
  std::string mangleFunctionName(StringRef baseName, ExtensionFieldType type);
};

} // namespace mlir::prime_ir::field

// NOLINTNEXTLINE(whitespace/line_length)
#endif // PRIME_IR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_INTRINSICFUNCTIONGENERATOR_H_
