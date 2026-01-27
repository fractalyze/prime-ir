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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveTypes.h"
#include "prime_ir/Utils/LoweringMode.h"

namespace mlir::prime_ir::elliptic_curve {

/// Generates and manages intrinsic functions for elliptic curve operations.
///
/// The primary use case is to reduce code size for scalar_mul operations,
/// which can generate thousands of instructions when inlined with extension
/// field coordinates (e.g., G2 points over Fp2).
///
/// Intrinsic functions use out-parameters (memref) instead of aggregate returns
/// to avoid NVPTX code generation issues with large aggregates.
///
/// Example intrinsic signature for scalar_mul:
///   func @__prime_ir_ec_scalar_mul_<curve_hash>(
///       %point_in: memref<Nx!FieldType>,   // N = num coords * field degree
///       %scalar: IntegerType,
///       %point_out: memref<Nx!FieldType>)
class IntrinsicFunctionGenerator {
public:
  explicit IntrinsicFunctionGenerator(ModuleOp module);

  /// Get or create the intrinsic function for scalar multiplication.
  /// The function implements the double-and-add algorithm.
  func::FuncOp getOrCreateScalarMulFunction(Type pointType, Type scalarType);

  /// Emit a call to the scalar multiplication intrinsic.
  /// Creates temporary memrefs, stores inputs, calls function, loads result.
  Value emitScalarMulCall(OpBuilder &builder, Location loc, Type pointType,
                          Type outputType, Value point, Value scalar);

  /// Determine if an operation should use intrinsic mode based on type and
  /// mode.
  ///
  /// Returns true for:
  /// - Intrinsic mode: always true for scalar_mul
  /// - Auto mode: true for points over extension fields (G2, etc.)
  /// - Inline mode: always false
  static bool shouldUseIntrinsic(Type pointType, LoweringMode mode);

private:
  /// Generate a mangled function name encoding the type information.
  std::string mangleFunctionName(StringRef baseName, Type pointType,
                                 Type scalarType);

  /// Get the memref type for point coordinates.
  /// For a point with N coordinates each of type FieldType,
  /// returns memref<N*degree x baseFieldType>.
  MemRefType getCoordMemRefType(Type pointType);

  /// Get the number of base field elements needed to represent a point.
  unsigned getNumBaseFieldElements(Type pointType);

  /// Generate the function body for scalar multiplication.
  void generateScalarMulBody(func::FuncOp func, Type pointType,
                             Type scalarType);

  ModuleOp module;
  SymbolTable symbolTable;
};

} // namespace mlir::prime_ir::elliptic_curve

// NOLINTNEXTLINE(whitespace/line_length)
#endif // PRIME_IR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_INTRINSICFUNCTIONGENERATOR_H_
