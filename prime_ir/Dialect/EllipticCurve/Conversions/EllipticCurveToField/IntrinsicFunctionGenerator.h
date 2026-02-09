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

#ifndef PRIME_IR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_INTRINSICFUNCTIONGENERATOR_H_
#define PRIME_IR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_INTRINSICFUNCTIONGENERATOR_H_

#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveTypes.h"
#include "prime_ir/Utils/IntrinsicFunctionGeneratorBase.h"
#include "prime_ir/Utils/LoweringMode.h"

namespace mlir::prime_ir::elliptic_curve {

/// Generates and manages intrinsic functions for elliptic curve operations.
///
/// The primary use case is to reduce code size for scalar_mul operations,
/// which can generate thousands of instructions when inlined with extension
/// field coordinates (e.g., G2 points over Fp2).
///
/// Intrinsic functions use high-level point and scalar types directly in their
/// signatures. The lowering pass handles type decomposition automatically.
///
/// Example intrinsic signature for scalar_mul:
///   func @__prime_ir_ec_scalar_mul_<curve_hash>(
///       %point: !elliptic_curve.jacobian<...>,
///       %scalar: !field.prime<...>) -> !elliptic_curve.jacobian<...>
class IntrinsicFunctionGenerator
    : public IntrinsicFunctionGeneratorBase<IntrinsicFunctionGenerator> {
  using Base = IntrinsicFunctionGeneratorBase<IntrinsicFunctionGenerator>;

public:
  explicit IntrinsicFunctionGenerator(ModuleOp module) : Base(module) {}

  /// Get or create the intrinsic function for scalar multiplication.
  /// The function implements the double-and-add algorithm.
  func::FuncOp getOrCreateScalarMulFunction(Type pointType, Type scalarType);

  /// Emit a call to the scalar multiplication intrinsic.
  Value emitScalarMulCall(OpBuilder &builder, Location loc, Type pointType,
                          Type outputType, Value point, Value scalar);

  /// Determine if an operation should use intrinsic mode based on type and
  /// mode. Returns false if the operation is inside an intrinsic function
  /// (detected by `__prime_ir_` prefix) to prevent infinite recursion.
  ///
  /// Returns true for:
  /// - Intrinsic mode: always true for scalar_mul
  /// - Auto mode: true for points over extension fields (G2, etc.)
  /// - Inline mode: always false
  static bool shouldUseIntrinsic(Operation *op, Type pointType,
                                 LoweringMode mode);

private:
  /// Generate a mangled function name encoding the type information.
  std::string mangleFunctionName(StringRef baseName, Type pointType,
                                 Type scalarType);
};

} // namespace mlir::prime_ir::elliptic_curve

// NOLINTNEXTLINE(whitespace/line_length)
#endif // PRIME_IR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_INTRINSICFUNCTIONGENERATOR_H_
