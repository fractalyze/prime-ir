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

#ifndef PRIME_IR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_PAIRINGOPERATIONS_PAIRINGOUTLINER_H_
#define PRIME_IR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_PAIRINGOPERATIONS_PAIRINGOUTLINER_H_

#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Value.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"
#include "prime_ir/Utils/IntrinsicFunctionGeneratorBase.h"

namespace mlir::prime_ir::elliptic_curve {

/// Generates outlined func.func helpers for pairing-specific composite ops.
///
/// Operations like CyclotomicSquare, MulBy034, and MulBy014 decompose into
/// many sub-field operations when expanded via CRTP. Without outlining, these
/// are fully unrolled at every call site, causing massive IR bloat:
///   - CyclotomicSquare: ~55 ops × 192 calls = ~10K lines
///   - MulBy034/014: ~50 ops × 316 calls = ~15K lines
///
/// This class lazily creates func.func helpers using the same codegen CRTP
/// infrastructure, and replaces inline expansion with func.call.
class PairingOutliner : public IntrinsicFunctionGeneratorBase<PairingOutliner> {
  using Base = IntrinsicFunctionGeneratorBase<PairingOutliner>;

public:
  explicit PairingOutliner(ModuleOp module) : Base(module) {}

  /// Returns true during function body generation, used to prevent recursive
  /// outlining — the body should expand inline, not emit another func.call.
  bool isGeneratingBody() const { return generatingBody_; }

  /// Emit a call to outlined CyclotomicSquare: (Fp12) -> Fp12.
  Value emitCyclotomicSquareCall(ImplicitLocOpBuilder &b, Value input);

  /// Emit a call to outlined Fp12 multiply: (Fp12, Fp12) -> Fp12.
  /// Used by CyclotomicPow to avoid CRTP-expanding Fp12 Karatsuba in loop body.
  Value emitFp12MulCall(ImplicitLocOpBuilder &b, Value lhs, Value rhs);

  /// Emit a call to outlined MulBy034: (Fp12, Fp2, Fp2, Fp2) -> Fp12.
  Value emitMulBy034Call(ImplicitLocOpBuilder &b, Value fp12, Value c0,
                         Value c3, Value c4);

  /// Emit a call to outlined MulBy014: (Fp12, Fp2, Fp2, Fp2) -> Fp12.
  Value emitMulBy014Call(ImplicitLocOpBuilder &b, Value fp12, Value c0,
                         Value c1, Value c4);

private:
  std::string mangleName(StringRef baseName, field::ExtensionFieldType type);

  bool generatingBody_ = false;

  /// RAII guard that sets generatingBody_ = true during function body emission.
  class BodyGenGuard {
  public:
    explicit BodyGenGuard(PairingOutliner &outliner) : outliner_(outliner) {
      outliner_.generatingBody_ = true;
    }
    ~BodyGenGuard() { outliner_.generatingBody_ = false; }

    BodyGenGuard(const BodyGenGuard &) = delete;
    BodyGenGuard &operator=(const BodyGenGuard &) = delete;

  private:
    PairingOutliner &outliner_;
  };
};

} // namespace mlir::prime_ir::elliptic_curve

// NOLINTNEXTLINE(whitespace/line_length)
#endif // PRIME_IR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_PAIRINGOPERATIONS_PAIRINGOUTLINER_H_
