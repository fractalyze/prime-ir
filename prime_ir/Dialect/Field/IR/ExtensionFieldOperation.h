// Copyright 2026 The PrimeIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

#ifndef PRIME_IR_DIALECT_FIELD_IR_EXTENSIONFIELDOPERATION_H_
#define PRIME_IR_DIALECT_FIELD_IR_EXTENSIONFIELDOPERATION_H_

#include <array>
#include <cassert>

#include "llvm/ADT/bit.h"
#include "mlir/IR/MLIRContext.h"
#include "prime_ir/Dialect/Field/IR/ExtensionFieldOperationSelector.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"
#include "prime_ir/Dialect/Field/IR/PrimeFieldOperation.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithTypes.h"
#include "prime_ir/Utils/Power.h"
#include "prime_ir/Utils/ZkDtypes.h"

namespace mlir::prime_ir::field {

// Forward declaration with default base field type
template <size_t N, typename BaseFieldT = PrimeFieldOperation>
class ExtensionFieldOperation;

} // namespace mlir::prime_ir::field

namespace zk_dtypes {

// Traits specialization: supports both prime field base and tower extensions
template <size_t N, typename BaseFieldT>
class ExtensionFieldOperationTraits<
    mlir::prime_ir::field::ExtensionFieldOperation<N, BaseFieldT>> {
public:
  using BaseField = BaseFieldT;
  static constexpr size_t kDegree = N;
};

} // namespace zk_dtypes

namespace mlir::prime_ir::field {

// Helper to determine storage bit width for base field (works for both prime
// and extension fields).
namespace detail {
template <typename BaseFieldT>
struct BaseFieldBitWidth;

template <>
struct BaseFieldBitWidth<PrimeFieldOperation> {
  static unsigned get(ExtensionFieldType efType) {
    return cast<PrimeFieldType>(efType.getBaseField()).getStorageBitWidth();
  }
};

template <size_t M, typename InnerBaseT>
struct BaseFieldBitWidth<ExtensionFieldOperation<M, InnerBaseT>> {
  static unsigned get(ExtensionFieldType efType) {
    // For tower extensions, get the bit width from the underlying prime field
    return efType.getBasePrimeField().getStorageBitWidth();
  }
};
// Helper to detect if a zk_dtypes field type is a prime field or extension
// field. Prime fields don't have kDegreeOverBaseField, extension fields do.
template <typename F>
struct IsPrimeField {
private:
  template <typename U>
  // NOLINTNEXTLINE(readability/casting)
  static auto test(int)
      -> decltype(U::Config::kDegreeOverBaseField, std::false_type{});
  template <typename>
  static std::true_type test(...);

public:
  static constexpr bool value = decltype(test<F>(0))::value;
};

// Type trait to map zk_dtypes extension field types to ExtensionFieldOperation.
// Recursively determines the correct ExtensionFieldOperation type for towers.
// Uses partial specialization with bool parameter to avoid eager evaluation.
template <typename ExtF, bool IsPrime>
struct ZkDtypeToExtensionFieldOpImpl;

// Specialization for when BaseField is prime field (non-tower)
template <typename ExtF>
struct ZkDtypeToExtensionFieldOpImpl<ExtF, true> {
  static constexpr size_t N = ExtF::Config::kDegreeOverBaseField;
  using type = ExtensionFieldOperation<N, PrimeFieldOperation>;
};

// Specialization for when BaseField is extension field (tower)
template <typename ExtF>
struct ZkDtypeToExtensionFieldOpImpl<ExtF, false> {
  static constexpr size_t N = ExtF::Config::kDegreeOverBaseField;
  using BaseF = typename ExtF::Config::BaseField;
  using type = ExtensionFieldOperation<
      N,
      typename ZkDtypeToExtensionFieldOpImpl<
          BaseF, IsPrimeField<typename BaseF::Config::BaseField>::value>::type>;
};

// Public interface that automatically determines the IsPrime parameter
template <typename ExtF>
struct ZkDtypeToExtensionFieldOp
    : ZkDtypeToExtensionFieldOpImpl<
          ExtF, IsPrimeField<typename ExtF::Config::BaseField>::value> {};

// Montgomery conversion direction enum and traits for unified fromMont/toMont.
enum class MontDirection { ToMont, FromMont };

template <MontDirection Dir>
struct MontConversionTraits;

template <>
struct MontConversionTraits<MontDirection::FromMont> {
  static constexpr bool kSourceMont = true;
  static constexpr bool kTargetMont = false;

  static PrimeFieldType convertPrimeType(PrimeFieldType pf) {
    return PrimeFieldType::get(pf.getContext(), pf.getModulus(), false);
  }

  template <typename T>
  static auto convertValue(const T &v) {
    return v.fromMont();
  }

  static Attribute convertNonResidue(Attribute nr, IntegerAttr modulus) {
    if (auto intAttr = dyn_cast<IntegerAttr>(nr)) {
      return mod_arith::getAttrAsStandardForm(modulus, intAttr);
    }
    return mod_arith::getAttrAsStandardForm(modulus,
                                            cast<DenseIntElementsAttr>(nr));
  }
};

template <>
struct MontConversionTraits<MontDirection::ToMont> {
  static constexpr bool kSourceMont = false;
  static constexpr bool kTargetMont = true;

  static PrimeFieldType convertPrimeType(PrimeFieldType pf) {
    return PrimeFieldType::get(pf.getContext(), pf.getModulus(), true);
  }

  template <typename T>
  static auto convertValue(const T &v) {
    return v.toMont();
  }

  static Attribute convertNonResidue(Attribute nr, IntegerAttr modulus) {
    if (auto intAttr = dyn_cast<IntegerAttr>(nr)) {
      return mod_arith::getAttrAsMontgomeryForm(modulus, intAttr);
    }
    return mod_arith::getAttrAsMontgomeryForm(modulus,
                                              cast<DenseIntElementsAttr>(nr));
  }
};

// Unified helper to recursively convert an extension field type to/from
// Montgomery form.
template <MontDirection Dir>
ExtensionFieldType convertExtFieldType(ExtensionFieldType ef) {
  using Traits = MontConversionTraits<Dir>;
  Type baseField = ef.getBaseField();
  Type newBaseField;

  if (auto pf = dyn_cast<PrimeFieldType>(baseField)) {
    newBaseField = Traits::convertPrimeType(pf);
  } else {
    // Recursive case: base field is also an extension field
    newBaseField =
        convertExtFieldType<Dir>(cast<ExtensionFieldType>(baseField));
  }

  Attribute newNonResidue = Traits::convertNonResidue(
      ef.getNonResidue(), ef.getBasePrimeField().getModulus());

  return ExtensionFieldType::get(ef.getContext(), ef.getDegree(), newBaseField,
                                 newNonResidue);
}

} // namespace detail

// Extension field operation class for compile-time evaluation.
// Inherits from zk_dtypes CRTP operations for Square/Inverse algorithms.
// Supports tower extensions via the BaseFieldT template parameter.
template <size_t N, typename BaseFieldT>
class ExtensionFieldOperation
    : public ExtensionFieldOperationSelector<N>::template Type<
          ExtensionFieldOperation<N, BaseFieldT>> {
public:
  static constexpr bool kIsTower =
      !std::is_same_v<BaseFieldT, PrimeFieldOperation>;

  ExtensionFieldOperation() = default;

  ExtensionFieldOperation(int64_t coeff, ExtensionFieldType efType)
      : ExtensionFieldOperation(
            APInt(detail::BaseFieldBitWidth<BaseFieldT>::get(efType), coeff),
            efType) {}

  ExtensionFieldOperation(const APInt &coeff, ExtensionFieldType efType)
      : efType(efType) {
    coeffs[0] = createBaseFieldOp(coeff, efType);
    for (size_t i = 1; i < N; ++i) {
      coeffs[i] = createBaseFieldOp(
          APInt::getZero(detail::BaseFieldBitWidth<BaseFieldT>::get(efType)),
          efType);
    }
  }

  // Construct from APInt coefficients (convenient one-liner usage)
  ExtensionFieldOperation(const SmallVector<APInt> &coeffs,
                          ExtensionFieldType efType)
      : efType(efType) {
    assert(coeffs.size() == N);
    for (size_t i = 0; i < N; ++i) {
      this->coeffs[i] = createBaseFieldOp(coeffs[i], efType);
    }
  }

  ExtensionFieldOperation(const std::array<BaseFieldT, N> &coeffs,
                          ExtensionFieldType efType)
      : coeffs(coeffs), efType(efType) {
    assert(coeffs.size() == N);
  }

  static ExtensionFieldOperation fromUnchecked(DenseIntElementsAttr attr,
                                               ExtensionFieldType efType) {
    SmallVector<APInt> coeffs{attr.getValues<APInt>().begin(),
                              attr.getValues<APInt>().end()};
    return fromUnchecked(coeffs, efType);
  }

  static ExtensionFieldOperation fromUnchecked(const APInt &coeff,
                                               ExtensionFieldType efType) {
    unsigned bitWidth = detail::BaseFieldBitWidth<BaseFieldT>::get(efType);
    APInt adjustedCoeff = coeff.zextOrTrunc(bitWidth);
    SmallVector<APInt> coeffs(N, APInt::getZero(bitWidth));
    coeffs[0] = adjustedCoeff;
    return fromUnchecked(coeffs, efType);
  }

  static ExtensionFieldOperation fromUnchecked(const SmallVector<APInt> &coeffs,
                                               ExtensionFieldType efType) {
    // Handle flattened prime field coefficients (e.g., from constant folding)
    // If size matches degreeOverPrime, treat as flattened; otherwise as direct.
    unsigned degreeOverPrime = efType.getDegreeOverPrime();
    if (coeffs.size() == degreeOverPrime && degreeOverPrime != N) {
      // Flattened coefficients - use fromFlatPrimeCoeffs
      return fromFlatPrimeCoeffs(coeffs, efType);
    }

    // Direct coefficients (size == N) - original behavior
    assert(coeffs.size() == N);
    std::array<BaseFieldT, N> newCoeffs;
    for (size_t i = 0; i < N; ++i) {
      newCoeffs[i] = createBaseFieldOpUnchecked(coeffs[i], efType);
    }
    return fromUnchecked(newCoeffs, efType);
  }

  static ExtensionFieldOperation
  fromUnchecked(const std::array<BaseFieldT, N> &coeffs,
                ExtensionFieldType efType) {
    ExtensionFieldOperation ret;
    ret.coeffs = coeffs;
    ret.efType = efType;
    return ret;
  }

  template <typename ExtF>
  static ExtensionFieldType getExtensionFieldType(MLIRContext *context) {
    using BaseF = typename ExtF::Config::BaseField;

    if constexpr (detail::IsPrimeField<BaseF>::value) {
      // Non-tower: base is prime field
      IntegerAttr nonResidue =
          convertToIntegerAttr(context, ExtF::Config::kNonResidue.value());
      auto modulusBits = llvm::bit_ceil(BaseF::Config::kModulusBits);
      IntegerAttr modulus = IntegerAttr::get(
          IntegerType::get(context, modulusBits),
          convertToAPInt(BaseF::Config::kModulus, modulusBits));
      PrimeFieldType pfType =
          PrimeFieldType::get(context, modulus, ExtF::kUseMontgomery);
      return ExtensionFieldType::get(context, N, pfType, nonResidue);
    } else {
      // Tower: base is extension field
      // For tower extensions, non-residue is an element in the base extension
      // field. Store all coefficients as DenseIntElementsAttr.
      constexpr size_t baseDegree = BaseF::Config::kDegreeOverBaseField;
      SmallVector<APInt> nrCoeffs;
      for (size_t i = 0; i < baseDegree; ++i) {
        nrCoeffs.push_back(
            convertToAPInt(ExtF::Config::kNonResidue[i].value()));
      }
      using BaseFieldOp =
          typename detail::ZkDtypeToExtensionFieldOp<BaseF>::type;
      ExtensionFieldType baseEfType =
          BaseFieldOp::template getExtensionFieldType<BaseF>(context);
      auto storageType = baseEfType.getBasePrimeField().getStorageType();
      auto nonResidue = DenseIntElementsAttr::get(
          RankedTensorType::get({static_cast<int64_t>(baseDegree)},
                                storageType),
          nrCoeffs);
      return ExtensionFieldType::get(context, N, baseEfType, nonResidue);
    }
  }

  // Convert a zk_dtypes extension field element to ExtensionFieldOperation.
  // Supports both non-tower and tower extensions.
  template <typename ExtF>
  static ExtensionFieldOperation fromZkDtype(MLIRContext *context,
                                             const ExtF &ef) {
    std::array<BaseFieldT, N> coeffs;
    for (size_t i = 0; i < N; ++i) {
      coeffs[i] = BaseFieldT::fromZkDtype(context, ef[i]);
    }
    return fromUnchecked(coeffs, getExtensionFieldType<ExtF>(context));
  }

  ExtensionFieldOperation getZero() const {
    std::array<BaseFieldT, N> zeroCoeffs;
    for (size_t i = 0; i < N; ++i) {
      zeroCoeffs[i] = coeffs[0].getZero();
    }
    return fromUnchecked(zeroCoeffs, efType);
  }

  ExtensionFieldOperation getOne() const {
    std::array<BaseFieldT, N> oneCoeffs;
    oneCoeffs[0] = coeffs[0].getOne();
    for (size_t i = 1; i < N; ++i) {
      oneCoeffs[i] = coeffs[0].getZero();
    }
    return fromUnchecked(oneCoeffs, efType);
  }

  const std::array<BaseFieldT, N> &getCoeffs() const { return coeffs; }
  Type getType() const { return efType; }

  // Convert coefficients to APInts (convenient for constant folding one-liners)
  // Only available for non-tower extensions.
  template <typename T = BaseFieldT,
            std::enable_if_t<std::is_same_v<T, PrimeFieldOperation>, int> = 0>
  operator SmallVector<APInt>() const {
    SmallVector<APInt> result;
    for (const auto &c : coeffs) {
      result.push_back(static_cast<APInt>(c));
    }
    return result;
  }

  // Construct from flattened prime field coefficients.
  // Works for both non-tower and tower extensions.
  // For Fp4 = (Fp2)^2: [a0, a1, a2, a3] -> [[a0,a1], [a2,a3]]
  static ExtensionFieldOperation
  fromFlatPrimeCoeffs(ArrayRef<APInt> flatCoeffs, ExtensionFieldType efType) {
    unsigned expectedSize = efType.getDegreeOverPrime();
    assert(flatCoeffs.size() == expectedSize &&
           "Coefficient count mismatch for extension field");

    std::array<BaseFieldT, N> newCoeffs;
    if constexpr (kIsTower) {
      // Get base extension field type and its degree over prime
      auto baseEfType = cast<ExtensionFieldType>(efType.getBaseField());
      unsigned baseSize = baseEfType.getDegreeOverPrime();

      // Split flatCoeffs into N groups, each of size baseSize
      for (size_t i = 0; i < N; ++i) {
        ArrayRef<APInt> subCoeffs = flatCoeffs.slice(i * baseSize, baseSize);
        newCoeffs[i] = BaseFieldT::fromFlatPrimeCoeffs(subCoeffs, baseEfType);
      }
    } else {
      // Non-tower: each coefficient is directly an APInt
      for (size_t i = 0; i < N; ++i) {
        auto baseFieldType = cast<PrimeFieldType>(efType.getBaseField());
        newCoeffs[i] =
            PrimeFieldOperation::fromUnchecked(flatCoeffs[i], baseFieldType);
      }
    }
    return fromUnchecked(newCoeffs, efType);
  }

  // Flatten all prime field coefficients to a single SmallVector<APInt>.
  // Works for both non-tower and tower extensions.
  // For Fp4 = (Fp2)^2: [[a0,a1], [a2,a3]] -> [a0, a1, a2, a3]
  SmallVector<APInt> flattenToPrimeCoeffs() const {
    SmallVector<APInt> result;
    for (const auto &c : coeffs) {
      if constexpr (kIsTower) {
        // Recursively flatten base extension field coefficients
        auto subCoeffs = c.flattenToPrimeCoeffs();
        result.append(subCoeffs.begin(), subCoeffs.end());
      } else {
        // Base is prime field - just get the APInt directly
        result.push_back(static_cast<APInt>(c));
      }
    }
    return result;
  }

  // Get DenseIntElementsAttr with all prime field coefficients.
  // Works for both non-tower and tower extensions.
  DenseIntElementsAttr getFlatDenseIntElementsAttr() const {
    auto primeField = efType.getBasePrimeField();
    auto flatCoeffs = flattenToPrimeCoeffs();
    return DenseIntElementsAttr::get(
        RankedTensorType::get({static_cast<int64_t>(flatCoeffs.size())},
                              primeField.getStorageType()),
        flatCoeffs);
  }

  ExtensionFieldOperation &operator+=(const ExtensionFieldOperation &other) {
    return *this = *this + other;
  }
  ExtensionFieldOperation &operator-=(const ExtensionFieldOperation &other) {
    return *this = *this - other;
  }
  ExtensionFieldOperation &operator*=(const ExtensionFieldOperation &other) {
    return *this = *this * other;
  }
  ExtensionFieldOperation &operator/=(const ExtensionFieldOperation &other) {
    return *this = *this / other;
  }

  ExtensionFieldOperation dbl() const { return this->Double(); }

  ExtensionFieldOperation square() const { return this->Square(); }

  ExtensionFieldOperation power(const APInt &exponent) const {
    return prime_ir::power(*this, exponent);
  }

  ExtensionFieldOperation inverse() const { return this->Inverse(); }

  // Unified Montgomery conversion implementation.
  template <detail::MontDirection Dir>
  ExtensionFieldOperation convertMont() const {
    using Traits = detail::MontConversionTraits<Dir>;
    assert(efType.isMontgomery() == Traits::kSourceMont);

    ExtensionFieldType newType;
    if constexpr (kIsTower) {
      // Tower extension: base field is an ExtensionFieldType
      auto baseEfType = cast<ExtensionFieldType>(efType.getBaseField());
      auto newBaseType = detail::convertExtFieldType<Dir>(baseEfType);
      Attribute newNonResidueAttr = Traits::convertNonResidue(
          efType.getNonResidue(), baseEfType.getBasePrimeField().getModulus());
      newType = ExtensionFieldType::get(efType.getContext(), efType.getDegree(),
                                        newBaseType, newNonResidueAttr);
    } else {
      // Non-tower: base field is a PrimeFieldType
      auto baseFieldType = cast<PrimeFieldType>(efType.getBaseField());
      auto newBaseType = Traits::convertPrimeType(baseFieldType);
      auto nonResidueOp = PrimeFieldOperation::fromUnchecked(
          cast<IntegerAttr>(efType.getNonResidue()), baseFieldType);
      auto newNonResidueAttr =
          Traits::convertValue(nonResidueOp).getIntegerAttr();
      newType = ExtensionFieldType::get(efType.getContext(), efType.getDegree(),
                                        newBaseType, newNonResidueAttr);
    }

    std::array<BaseFieldT, N> newCoeffs;
    for (size_t i = 0; i < N; ++i) {
      newCoeffs[i] = Traits::convertValue(coeffs[i]);
    }
    return fromUnchecked(newCoeffs, newType);
  }

  ExtensionFieldOperation fromMont() const {
    return convertMont<detail::MontDirection::FromMont>();
  }

  ExtensionFieldOperation toMont() const {
    return convertMont<detail::MontDirection::ToMont>();
  }

  bool isOne() const {
    for (size_t i = 1; i < N; ++i) {
      if (!coeffs[i].isZero()) {
        return false;
      }
    }
    return coeffs[0].isOne();
  }

  bool isZero() const {
    return llvm::all_of(coeffs, [](const BaseFieldT &c) { return c.isZero(); });
  }

  bool operator==(const ExtensionFieldOperation &other) const {
    assert(efType == other.efType);
    return coeffs == other.coeffs;
  }
  bool operator!=(const ExtensionFieldOperation &other) const {
    assert(efType == other.efType);
    return coeffs != other.coeffs;
  }

private:
  friend class FieldOperation;
  // PascalCase methods (zk_dtypes compatible)
  template <typename>
  friend class zk_dtypes::QuadraticExtensionFieldOperation;
  template <typename>
  friend class zk_dtypes::CubicExtensionFieldOperation;
  template <typename>
  friend class zk_dtypes::QuarticExtensionFieldOperation;
  template <typename>
  friend class zk_dtypes::KaratsubaOperation;
  template <typename>
  friend class zk_dtypes::ToomCookOperation;
  template <typename>
  friend class zk_dtypes::ExtensionFieldOperation;
  template <typename>
  friend class zk_dtypes::FrobeniusOperation;
  template <typename, size_t>
  friend class zk_dtypes::VandermondeMatrix;

  template <size_t N2, typename B2>
  friend raw_ostream &operator<<(raw_ostream &os,
                                 const ExtensionFieldOperation<N2, B2> &op);

  const std::array<BaseFieldT, N> &ToCoeffs() const { return coeffs; }

  ExtensionFieldOperation FromCoeffs(const std::array<BaseFieldT, N> &c) const {
    return ExtensionFieldOperation(c, efType);
  }

  // NonResidue returns an element in the base field
  BaseFieldT NonResidue() const {
    if constexpr (kIsTower) {
      // For towers, non-residue can be stored as:
      // - DenseIntElementsAttr: full coefficients (e.g., from zk_dtypes)
      // - IntegerAttr: scalar that gets embedded as [value, 0, 0, ...]
      auto baseEfType = cast<ExtensionFieldType>(efType.getBaseField());
      Attribute nrAttr = efType.getNonResidue();
      if (auto denseAttr = dyn_cast<DenseIntElementsAttr>(nrAttr)) {
        return BaseFieldT::fromUnchecked(denseAttr, baseEfType);
      }
      // Scalar IntegerAttr: embed as [value, 0, 0, ...] in base extension field
      auto intAttr = cast<IntegerAttr>(nrAttr);
      return BaseFieldT(intAttr.getValue(), baseEfType);
    } else {
      auto baseFieldType = cast<PrimeFieldType>(efType.getBaseField());
      return PrimeFieldOperation::fromUnchecked(
          cast<IntegerAttr>(efType.getNonResidue()), baseFieldType);
    }
  }

  // https://github.com/fractalyze/zk_dtypes/blob/8d5f43c/zk_dtypes/include/field/extension_field.h#L500
  zk_dtypes::ExtensionFieldMulAlgorithm GetSquareAlgorithm() const {
    if constexpr (kIsTower) {
      // NOTE(chokobole): Avoid Toom-Cook algorithm due to caching conflicts.
      // `zk_dtypes::VandermondeMatrix` caches interpolation coefficients.
      // However, field types in `zk_dtypes` are distinct based on their
      // Montgomery property, while `ExtensionFieldOperation` are shared
      // across different domains.
      //
      // If a standard-domain extension field attempts to use a Vandermonde
      // matrix already cached in the Montgomery domain, the arithmetic will be
      // incorrect.
      return zk_dtypes::ExtensionFieldMulAlgorithm::kKaratsuba;
    } else if constexpr (N == 2) {
      auto baseFieldType = cast<PrimeFieldType>(efType.getBaseField());
      // Heuristic: custom squaring when n² > 2n + C
      unsigned limbNums = (baseFieldType.getStorageBitWidth() + 63) / 64;
      if (limbNums * N >= 2) {
        return zk_dtypes::ExtensionFieldMulAlgorithm::kCustom2;
      }
      return zk_dtypes::ExtensionFieldMulAlgorithm::kKaratsuba;
    } else if constexpr (N == 3) {
      auto baseFieldType = cast<PrimeFieldType>(efType.getBaseField());
      // Heuristic: custom squaring when n² > 4n
      unsigned limbNums = (baseFieldType.getStorageBitWidth() + 63) / 64;
      if (limbNums * N >= 4) {
        return zk_dtypes::ExtensionFieldMulAlgorithm::kCustom;
      }
      return zk_dtypes::ExtensionFieldMulAlgorithm::kKaratsuba;
    } else {
      // NOTE(chokobole): Avoid Toom-Cook algorithm due to caching conflicts.
      return zk_dtypes::ExtensionFieldMulAlgorithm::kKaratsuba;
    }
  }

  zk_dtypes::ExtensionFieldMulAlgorithm GetMulAlgorithm() const {
    // NOTE(chokobole): See the comment above why we should not use Toom-Cook
    // algorithm.
    return zk_dtypes::ExtensionFieldMulAlgorithm::kKaratsuba;
  }

  BaseFieldT CreateConstBaseField(int64_t x) const {
    if constexpr (kIsTower) {
      auto baseEfType = cast<ExtensionFieldType>(efType.getBaseField());
      APInt value(efType.getBasePrimeField().getStorageBitWidth(), x);
      return BaseFieldT::fromUnchecked(value, baseEfType);
    } else {
      auto baseFieldType = cast<PrimeFieldType>(efType.getBaseField());
      return PrimeFieldOperation(x, baseFieldType);
    }
  }

  // Create a constant in this extension field (needed for zk_dtypes interface)
  // Returns [x, 0, 0, ...] embedded in the extension field
  ExtensionFieldOperation CreateConst(int64_t x) const {
    std::array<BaseFieldT, N> newCoeffs;
    newCoeffs[0] = CreateConstBaseField(x);
    for (size_t i = 1; i < N; ++i) {
      newCoeffs[i] = CreateConstBaseField(0);
    }
    return fromUnchecked(newCoeffs, efType);
  }

  // Frobenius coefficients: only available for non-tower extensions
  // coeffs[e-1][i-1] = ξ^(i * e * (p - 1) / n)
  template <typename T = BaseFieldT,
            std::enable_if_t<std::is_same_v<T, PrimeFieldOperation>, int> = 0>
  std::array<std::array<PrimeFieldOperation, N - 1>, N - 1>
  GetFrobeniusCoeffs() const {
    auto baseFieldType = cast<PrimeFieldType>(efType.getBaseField());
    APInt p = baseFieldType.getModulus().getValue();
    APInt nr = cast<IntegerAttr>(efType.getNonResidue()).getValue();
    PrimeFieldOperation nrOp(nr, baseFieldType);

    unsigned extBitWidth = p.getBitWidth() * 2;
    APInt pMinus1 = (p - 1).zext(extBitWidth);
    APInt nVal(extBitWidth, N);

    std::array<std::array<PrimeFieldOperation, N - 1>, N - 1> result;
    for (size_t e = 1; e < N; ++e) {
      for (size_t i = 1; i < N; ++i) {
        APInt exp =
            APInt(extBitWidth, i) * APInt(extBitWidth, e) * pMinus1.udiv(nVal);
        result[e - 1][i - 1] = nrOp.power(exp);
      }
    }
    return result;
  }

  // Helper to create a base field operation from an APInt
  static BaseFieldT createBaseFieldOp(const APInt &value,
                                      ExtensionFieldType efType) {
    if constexpr (kIsTower) {
      auto baseEfType = cast<ExtensionFieldType>(efType.getBaseField());
      return BaseFieldT(value, baseEfType);
    } else {
      auto baseFieldType = cast<PrimeFieldType>(efType.getBaseField());
      return PrimeFieldOperation(value, baseFieldType);
    }
  }

  // Helper to create a base field operation without reduction checks
  static BaseFieldT createBaseFieldOpUnchecked(const APInt &value,
                                               ExtensionFieldType efType) {
    if constexpr (kIsTower) {
      auto baseEfType = cast<ExtensionFieldType>(efType.getBaseField());
      return BaseFieldT::fromUnchecked(value, baseEfType);
    } else {
      auto baseFieldType = cast<PrimeFieldType>(efType.getBaseField());
      return PrimeFieldOperation::fromUnchecked(value, baseFieldType);
    }
  }

  std::array<BaseFieldT, N> coeffs;
  ExtensionFieldType efType;
};

template <size_t N, typename BaseFieldT>
raw_ostream &operator<<(raw_ostream &os,
                        const ExtensionFieldOperation<N, BaseFieldT> &op) {
  llvm::interleaveComma(op.coeffs, os, [&](const BaseFieldT &c) { os << c; });
  return os;
}

// Explicit instantiations for non-tower extension fields (base is prime field)
extern template class ExtensionFieldOperation<2, PrimeFieldOperation>;
extern template class ExtensionFieldOperation<3, PrimeFieldOperation>;
extern template class ExtensionFieldOperation<4, PrimeFieldOperation>;

extern template raw_ostream &
operator<<(raw_ostream &os,
           const ExtensionFieldOperation<2, PrimeFieldOperation> &op);
extern template raw_ostream &
operator<<(raw_ostream &os,
           const ExtensionFieldOperation<3, PrimeFieldOperation> &op);
extern template raw_ostream &
operator<<(raw_ostream &os,
           const ExtensionFieldOperation<4, PrimeFieldOperation> &op);

// Type aliases for non-tower extension fields (backward compatible)
using QuadraticExtensionFieldOperation =
    ExtensionFieldOperation<2, PrimeFieldOperation>;
using CubicExtensionFieldOperation =
    ExtensionFieldOperation<3, PrimeFieldOperation>;
using QuarticExtensionFieldOperation =
    ExtensionFieldOperation<4, PrimeFieldOperation>;

// Type aliases for tower extension fields
// Tower over quadratic: e.g., Fp4 = (Fp2)^2, Fp6 = (Fp2)^3
using TowerQuadraticOverQuadraticOp =
    ExtensionFieldOperation<2, QuadraticExtensionFieldOperation>;
using TowerCubicOverQuadraticOp =
    ExtensionFieldOperation<3, QuadraticExtensionFieldOperation>;
using TowerQuarticOverQuadraticOp =
    ExtensionFieldOperation<4, QuadraticExtensionFieldOperation>;

// Tower over cubic: e.g., Fp6 = (Fp3)^2
using TowerQuadraticOverCubicOp =
    ExtensionFieldOperation<2, CubicExtensionFieldOperation>;

// Depth-2 towers (tower over tower)
// Fp12 = ((Fp2)^3)^2 = (Fp6)^2
using TowerQuadraticOverCubicOverQuadraticOp =
    ExtensionFieldOperation<2, TowerCubicOverQuadraticOp>;
// Fp12 = ((Fp2)^2)^3 = (Fp4)^3 (alternative construction)
using TowerCubicOverQuadraticOverQuadraticOp =
    ExtensionFieldOperation<3, TowerQuadraticOverQuadraticOp>;
// Fp8 = ((Fp2)^2)^2 = (Fp4)^2
using TowerQuadraticOverQuadraticOverQuadraticOp =
    ExtensionFieldOperation<2, TowerQuadraticOverQuadraticOp>;
// Fp12 = ((Fp3)^2)^2 = (Fp6)^2 (alternative)
using TowerQuadraticOverQuadraticOverCubicOp =
    ExtensionFieldOperation<2, TowerQuadraticOverCubicOp>;

// Depth-3 towers (for Fp24 = (Fp12)^2 = (((Fp2)^3)^2)^2)
using TowerQuadraticOverQuadraticOverCubicOverQuadraticOp =
    ExtensionFieldOperation<2, TowerQuadraticOverCubicOverQuadraticOp>;

} // namespace mlir::prime_ir::field

#endif // PRIME_IR_DIALECT_FIELD_IR_EXTENSIONFIELDOPERATION_H_
