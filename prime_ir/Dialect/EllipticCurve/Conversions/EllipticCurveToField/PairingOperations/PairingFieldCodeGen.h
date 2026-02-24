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

#ifndef PRIME_IR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_PAIRINGOPERATIONS_PAIRINGFIELDCODEGEN_H_
#define PRIME_IR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_PAIRINGOPERATIONS_PAIRINGFIELDCODEGEN_H_

#include <array>

#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Value.h"
#include "prime_ir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/FieldDialectArithmetic.h"
#include "prime_ir/Dialect/Field/IR/FieldOps.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"
#include "prime_ir/Utils/BuilderContext.h"
#include "zk_dtypes/include/elliptic_curve/pairing/pairing_traits_forward.h"
#include "zk_dtypes/include/field/cubic_extension_field_operation.h"
#include "zk_dtypes/include/field/extension_field_operation_traits_forward.h"
#include "zk_dtypes/include/field/quadratic_extension_field_operation.h"

namespace mlir::prime_ir::elliptic_curve {

// Forward declarations.
class PairingFpCodeGen;
class PairingFp2CodeGen;
class PairingFp6CodeGen;
class PairingFp12CodeGen;
class PairingG1AffineCodeGen;
class PairingG2AffineCodeGen;
class PairingOutliner;

// Tag type for CRTP specialization of pairing algorithms.
// Used as the Derived parameter for BNCurve<Config, Derived> to dispatch
// field operations to MLIR IR code generation instead of concrete computation.
struct PairingCodeGenDerived {};

// ==========================================================================
// PairingCodeGenContext: Thread-local storage for MLIR types used by codegen.
//
// Set up before pairing code generation and accessed by the static methods
// (One(), Zero(), TwoInv()) on the codegen types.
// ==========================================================================
class PairingCodeGenContext {
public:
  static PairingCodeGenContext &GetInstance() {
    thread_local static PairingCodeGenContext instance;
    return instance;
  }

  field::PrimeFieldType fpType;
  field::ExtensionFieldType fp2Type;
  field::ExtensionFieldType fp6Type;
  field::ExtensionFieldType fp12Type;
  Value g2CurveB; // G2 curve coefficient b as Fp2 MLIR value.
  PairingOutliner *outliner = nullptr; // Outliner for composite pairing ops.

private:
  PairingCodeGenContext() = default;
};

} // namespace mlir::prime_ir::elliptic_curve

// ==========================================================================
// ExtensionFieldOperationTraits specializations.
// Must be in zk_dtypes namespace and defined before CRTP base instantiation.
// ==========================================================================
namespace zk_dtypes {

template <>
class ExtensionFieldOperationTraits<
    mlir::prime_ir::elliptic_curve::PairingFp2CodeGen> {
public:
  static constexpr size_t kDegree = 2;
  using BaseField = mlir::prime_ir::elliptic_curve::PairingFpCodeGen;
};

template <>
class ExtensionFieldOperationTraits<
    mlir::prime_ir::elliptic_curve::PairingFp6CodeGen> {
public:
  static constexpr size_t kDegree = 3;
  using BaseField = mlir::prime_ir::elliptic_curve::PairingFp2CodeGen;
};

template <>
class ExtensionFieldOperationTraits<
    mlir::prime_ir::elliptic_curve::PairingFp12CodeGen> {
public:
  static constexpr size_t kDegree = 2;
  using BaseField = mlir::prime_ir::elliptic_curve::PairingFp6CodeGen;
};

// ==========================================================================
// PairingTraits specialization for code generation.
// Maps pairing type aliases to MLIR IR builder types.
// ==========================================================================
template <>
struct PairingTraits<mlir::prime_ir::elliptic_curve::PairingCodeGenDerived> {
  using Fp = mlir::prime_ir::elliptic_curve::PairingFpCodeGen;
  using Fp2 = mlir::prime_ir::elliptic_curve::PairingFp2CodeGen;
  using Fp12 = mlir::prime_ir::elliptic_curve::PairingFp12CodeGen;
  using G1AffinePoint = mlir::prime_ir::elliptic_curve::PairingG1AffineCodeGen;
  using G2AffinePoint = mlir::prime_ir::elliptic_curve::PairingG2AffineCodeGen;
  using BoolType = mlir::Value;

  static Fp2 G2CurveB();
  static Fp2 TwistMulByQX();
  static Fp2 TwistMulByQY();

  // NAF bit access for codegen: emits tensor.extract + arith.cmpi ops.
  static mlir::Value GetNafBit(mlir::Value iv);
  static mlir::Value IsNafNonZero(mlir::Value nafBit);
  static mlir::Value IsNafPositive(mlir::Value nafBit);
};

} // namespace zk_dtypes

namespace mlir::prime_ir::elliptic_curve {

// ==========================================================================
// PairingFpCodeGen: Prime field codegen type.
//
// Wraps an mlir::Value of prime field type and emits field.* dialect ops
// for all arithmetic operations via FieldDialectArithmetic CRTP base.
// ==========================================================================
class PairingFpCodeGen : public FieldDialectArithmetic<PairingFpCodeGen> {
public:
  PairingFpCodeGen() = default;
  explicit PairingFpCodeGen(Value value) : value(value) {}

  operator Value() const { return value; }
  Value getValue() const { return value; }

  static constexpr size_t ExtensionDegree() { return 1; }

  // --- Constants ---
  PairingFpCodeGen CreateConst(int64_t constant) const;
  bool IsZero() const { return false; } // Always false for codegen
  static PairingFpCodeGen Zero();
  static PairingFpCodeGen One();
  static PairingFpCodeGen TwoInv();

private:
  Value value;
};

// ==========================================================================
// PairingFp2CodeGen: Quadratic extension field (Fp2 = Fp[u]/(u² + 1)).
//
// Inherits QuadraticExtensionFieldOperation for specialized ops
// (MulBy014/034, CyclotomicSquare, Inverse via CRTP decomposition).
// Inherits FieldDialectArithmetic for basic field.* ops (+, -, negate, Double).
// Overrides complex ops to emit field.* ops directly at this level.
// ==========================================================================
class PairingFp2CodeGen
    : public zk_dtypes::QuadraticExtensionFieldOperation<PairingFp2CodeGen>,
      public FieldDialectArithmetic<PairingFp2CodeGen> {
public:
  using BaseField = PairingFpCodeGen;

  // Resolve base class ambiguity: use FieldDialectArithmetic's field.* ops.
  using FieldDialectArithmetic<PairingFp2CodeGen>::operator+;
  using FieldDialectArithmetic<PairingFp2CodeGen>::operator-;
  using FieldDialectArithmetic<PairingFp2CodeGen>::Double;

  PairingFp2CodeGen() = default;
  explicit PairingFp2CodeGen(Value value);

  operator Value() const { return this->value; }
  Value getValue() const { return this->value; }

  static constexpr size_t ExtensionDegree() { return 2; }

  // --- CRTP interface for zk_dtypes mixins ---
  std::array<PairingFpCodeGen, 2> ToCoeffs() const;
  PairingFp2CodeGen FromCoeffs(const std::array<PairingFpCodeGen, 2> &c) const;
  PairingFpCodeGen NonResidue() const;
  zk_dtypes::ExtensionFieldMulAlgorithm GetMulAlgorithm() const {
    return zk_dtypes::ExtensionFieldMulAlgorithm::kKaratsuba;
  }
  zk_dtypes::ExtensionFieldMulAlgorithm GetSquareAlgorithm() const {
    return zk_dtypes::ExtensionFieldMulAlgorithm::kKaratsuba;
  }

  // --- Override complex ops to emit field.* at this level ---
  PairingFp2CodeGen operator*(const PairingFp2CodeGen &other) const;
  PairingFp2CodeGen operator*(const PairingFpCodeGen &scalar) const;
  PairingFp2CodeGen Square() const;
  PairingFp2CodeGen Inverse() const;

  // --- Constants ---
  PairingFp2CodeGen CreateConst(int64_t constant) const;
  PairingFp2CodeGen CreateRationalConst(int64_t, int64_t) const {
    llvm_unreachable("CreateRationalConst not needed for pairing codegen");
  }
  bool IsZero() const { return false; }
  static PairingFp2CodeGen Zero();
  static PairingFp2CodeGen One();

  // --- Frobenius (manual, not using FrobeniusOperation mixin) ---
  // For Fp2 with non-residue = -1: Frobenius<odd> = conjugation, <even> = id.
  template <size_t E>
  PairingFp2CodeGen Frobenius() const {
    if constexpr (E % 2 == 0) {
      return *this;
    } else {
      auto c = ToCoeffs();
      return FromCoeffs({c[0], -c[1]});
    }
  }

  // Needed by FrobeniusOperation (inherited but unused for Fp2).
  std::array<std::array<PairingFpCodeGen, 1>, 1> GetFrobeniusCoeffs() const {
    llvm_unreachable("GetFrobeniusCoeffs not used for Fp2");
  }
  std::array<std::array<PairingFpCodeGen, 1>, 1>
  GetRelativeFrobeniusCoeffs() const {
    llvm_unreachable("GetRelativeFrobeniusCoeffs not used for Fp2");
  }

  bool HasSimpleNonResidue() const { return true; }

  // Component-wise arith.select for Fp2 values.
  static PairingFp2CodeGen Select(Value condition, const PairingFp2CodeGen &a,
                                  const PairingFp2CodeGen &b);

private:
  Value value;
  Value nonResidue;
};

// ==========================================================================
// PairingFp6CodeGen: Cubic extension field (Fp6 = Fp2[v]/(v³ - ξ)).
//
// Inherits CubicExtensionFieldOperation for MulBy01, MulBy1, and Inverse.
// Inherits FieldDialectArithmetic for basic field.* ops (+, -, negate, Double).
// Frobenius uses inherited FrobeniusOperation with GetFrobeniusCoeffs().
// ==========================================================================
class PairingFp6CodeGen
    : public zk_dtypes::CubicExtensionFieldOperation<PairingFp6CodeGen>,
      public FieldDialectArithmetic<PairingFp6CodeGen> {
public:
  using BaseField = PairingFp2CodeGen;

  // Resolve base class ambiguity: use FieldDialectArithmetic's field.* ops.
  using FieldDialectArithmetic<PairingFp6CodeGen>::operator+;
  using FieldDialectArithmetic<PairingFp6CodeGen>::operator-;
  using FieldDialectArithmetic<PairingFp6CodeGen>::Double;

  PairingFp6CodeGen() = default;
  explicit PairingFp6CodeGen(Value value);

  operator Value() const { return this->value; }
  Value getValue() const { return this->value; }

  static constexpr size_t ExtensionDegree() { return 6; }

  // --- CRTP interface ---
  std::array<PairingFp2CodeGen, 3> ToCoeffs() const;
  PairingFp6CodeGen FromCoeffs(const std::array<PairingFp2CodeGen, 3> &c) const;
  PairingFp2CodeGen NonResidue() const;
  zk_dtypes::ExtensionFieldMulAlgorithm GetMulAlgorithm() const {
    return zk_dtypes::ExtensionFieldMulAlgorithm::kKaratsuba;
  }
  zk_dtypes::ExtensionFieldMulAlgorithm GetSquareAlgorithm() const {
    return zk_dtypes::ExtensionFieldMulAlgorithm::kCustom;
  }

  // --- Override complex ops to emit field.* at this level ---
  PairingFp6CodeGen operator*(const PairingFp6CodeGen &other) const;
  PairingFp6CodeGen operator*(const PairingFp2CodeGen &scalar) const;
  PairingFp6CodeGen Square() const;
  PairingFp6CodeGen Inverse() const;

  // --- Constants ---
  PairingFp6CodeGen CreateConst(int64_t constant) const;
  PairingFp6CodeGen CreateRationalConst(int64_t, int64_t) const {
    llvm_unreachable("CreateRationalConst not needed for pairing codegen");
  }
  bool IsZero() const { return false; }
  static PairingFp6CodeGen Zero();
  static PairingFp6CodeGen One();

  // --- Frobenius ---
  // Delegates to inherited FrobeniusOperation::Frobenius<E>(), which uses
  // GetFrobeniusCoeffs() to obtain precomputed BN254 coefficients.
  std::array<std::array<PairingFp2CodeGen, 2>, 5> GetFrobeniusCoeffs() const;
  std::array<std::array<PairingFp2CodeGen, 2>, 2>
  GetRelativeFrobeniusCoeffs() const {
    llvm_unreachable("GetRelativeFrobeniusCoeffs not used for Fp6");
  }

  bool HasSimpleNonResidue() const { return true; }

  // Component-wise select: decomposes to 3 Fp2 selects.
  static PairingFp6CodeGen Select(Value condition, const PairingFp6CodeGen &a,
                                  const PairingFp6CodeGen &b);

private:
  Value value;
  Value nonResidue;
};

// ==========================================================================
// PairingFp12CodeGen: Quadratic extension (Fp12 = Fp6[w]/(w² - v)).
//
// Inherits QuadraticExtensionFieldOperation for MulBy014/034 and
// CyclotomicOperation for CyclotomicSquare/Inverse/Pow.
// Inherits FieldDialectArithmetic for basic field.* ops (+, -, negate, Double).
// Frobenius uses inherited FrobeniusOperation with GetFrobeniusCoeffs().
// ==========================================================================
class PairingFp12CodeGen
    : public zk_dtypes::QuadraticExtensionFieldOperation<PairingFp12CodeGen>,
      public FieldDialectArithmetic<PairingFp12CodeGen> {
public:
  using BaseField = PairingFp6CodeGen;

  // Resolve base class ambiguity: use FieldDialectArithmetic's field.* ops
  // for basic arithmetic.
  using FieldDialectArithmetic<PairingFp12CodeGen>::operator+;
  using FieldDialectArithmetic<PairingFp12CodeGen>::operator-;
  using FieldDialectArithmetic<PairingFp12CodeGen>::Double;

  // Resolve base class ambiguity for square/inverse: use CRTP decomposition.
  using zk_dtypes::QuadraticExtensionFieldOperation<PairingFp12CodeGen>::Square;
  using zk_dtypes::QuadraticExtensionFieldOperation<
      PairingFp12CodeGen>::Inverse;

  // Fp12 multiply: outlined via PairingOutliner when available to keep LLVM IR
  // compact. Fp12 operator* is only called from the final exponentiation (the
  // Miller loop uses MulBy034/014), so outlining doesn't affect Miller loop
  // constant folding.
  PairingFp12CodeGen operator*(const PairingFp12CodeGen &other) const;

  PairingFp12CodeGen() = default;
  explicit PairingFp12CodeGen(Value value);

  operator Value() const { return this->value; }
  Value getValue() const { return this->value; }

  static constexpr size_t ExtensionDegree() { return 12; }

  // --- CRTP interface ---
  std::array<PairingFp6CodeGen, 2> ToCoeffs() const;
  PairingFp12CodeGen
  FromCoeffs(const std::array<PairingFp6CodeGen, 2> &c) const;
  PairingFp6CodeGen NonResidue() const;
  zk_dtypes::ExtensionFieldMulAlgorithm GetMulAlgorithm() const {
    return zk_dtypes::ExtensionFieldMulAlgorithm::kKaratsuba;
  }
  zk_dtypes::ExtensionFieldMulAlgorithm GetSquareAlgorithm() const {
    return zk_dtypes::ExtensionFieldMulAlgorithm::kKaratsuba;
  }

  // --- Outlined / loop-structured composite operations ---
  // These hide the base class CRTP methods. CyclotomicSquare/MulBy034/MulBy014
  // emit func.call to outlined helpers. CyclotomicPow emits scf.for instead
  // of unrolling the square-and-multiply loop.
  PairingFp12CodeGen CyclotomicPow(const zk_dtypes::BigInt<1> &exponent) const;
  PairingFp12CodeGen CyclotomicSquare() const;
  PairingFp12CodeGen MulBy034(const PairingFp2CodeGen &beta0,
                              const PairingFp2CodeGen &beta3,
                              const PairingFp2CodeGen &beta4) const;
  PairingFp12CodeGen MulBy014(const PairingFp2CodeGen &beta0,
                              const PairingFp2CodeGen &beta1,
                              const PairingFp2CodeGen &beta4) const;

  // --- Constants ---
  PairingFp12CodeGen CreateConst(int64_t constant) const;
  PairingFp12CodeGen CreateRationalConst(int64_t, int64_t) const {
    llvm_unreachable("CreateRationalConst not needed for pairing codegen");
  }
  bool IsZero() const { return false; }
  static PairingFp12CodeGen Zero();
  static PairingFp12CodeGen One();

  // --- Frobenius ---
  // Delegates to inherited FrobeniusOperation::Frobenius<E>(), which uses
  // GetFrobeniusCoeffs() to obtain precomputed BN254 coefficients.
  std::array<std::array<PairingFp6CodeGen, 1>, 11> GetFrobeniusCoeffs() const;
  std::array<std::array<PairingFp6CodeGen, 1>, 1>
  GetRelativeFrobeniusCoeffs() const {
    llvm_unreachable("GetRelativeFrobeniusCoeffs not used for Fp12");
  }

  bool HasSimpleNonResidue() const { return true; }

  // Component-wise select: decomposes to 2 Fp6 selects.
  static PairingFp12CodeGen Select(Value condition, const PairingFp12CodeGen &a,
                                   const PairingFp12CodeGen &b);

private:
  Value value;
  Value nonResidue;
};

// ==========================================================================
// Point codegen types for pairing.
// ==========================================================================

// G1 affine point: wraps (x, y) as PairingFpCodeGen values.
class PairingG1AffineCodeGen {
public:
  PairingG1AffineCodeGen() = default;
  PairingG1AffineCodeGen(PairingFpCodeGen x, PairingFpCodeGen y)
      : xCoord(x), yCoord(y) {}

  const PairingFpCodeGen &x() const { return xCoord; }
  const PairingFpCodeGen &y() const { return yCoord; }
  bool IsZero() const { return false; }

  PairingG1AffineCodeGen operator-() const { return {xCoord, -yCoord}; }

private:
  PairingFpCodeGen xCoord;
  PairingFpCodeGen yCoord;
};

// G2 affine point: wraps (x, y) as PairingFp2CodeGen values.
class PairingG2AffineCodeGen {
public:
  PairingG2AffineCodeGen() = default;
  PairingG2AffineCodeGen(PairingFp2CodeGen x, PairingFp2CodeGen y)
      : xCoord(x), yCoord(y) {}

  const PairingFp2CodeGen &x() const { return xCoord; }
  const PairingFp2CodeGen &y() const { return yCoord; }
  bool IsZero() const { return false; }

  PairingG2AffineCodeGen operator-() const { return {xCoord, -yCoord}; }

  // Component-wise select for G2 affine points.
  static PairingG2AffineCodeGen Select(Value condition,
                                       const PairingG2AffineCodeGen &a,
                                       const PairingG2AffineCodeGen &b) {
    return {PairingFp2CodeGen::Select(condition, a.xCoord, b.xCoord),
            PairingFp2CodeGen::Select(condition, a.yCoord, b.yCoord)};
  }

private:
  PairingFp2CodeGen xCoord;
  PairingFp2CodeGen yCoord;
};

} // namespace mlir::prime_ir::elliptic_curve

// NOLINTNEXTLINE(whitespace/line_length)
#endif // PRIME_IR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_PAIRINGOPERATIONS_PAIRINGFIELDCODEGEN_H_
