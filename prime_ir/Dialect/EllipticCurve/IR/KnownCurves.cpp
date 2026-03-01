/* Copyright 2025 The PrimeIR Authors.

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

#include "prime_ir/Dialect/EllipticCurve/IR/KnownCurves.h"

#include "prime_ir/Dialect/Field/IR/ExtensionFieldOperation.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"
#include "prime_ir/Utils/KnownModulus.h"
#include "prime_ir/Utils/ZkDtypes.h"
#include "zk_dtypes/include/elliptic_curve/bn/bn254/fqx6.h"
#include "zk_dtypes/include/elliptic_curve/bn/bn254/g1.h"
#include "zk_dtypes/include/elliptic_curve/bn/bn254/g2.h"

namespace mlir::prime_ir::elliptic_curve {

namespace {

template <typename Curve>
bool isKnownCurve(ShortWeierstrassAttr attr) {
  using BaseField = typename Curve::BaseField;
  if constexpr (BaseField::ExtensionDegree() == 1) {
    auto a = cast<IntegerAttr>(attr.getA());
    auto b = cast<IntegerAttr>(attr.getB());
    auto gX = cast<IntegerAttr>(attr.getGx());
    auto gY = cast<IntegerAttr>(attr.getGy());

    unsigned bitWidth = a.getValue().getBitWidth();
    auto eq = [bitWidth](IntegerAttr lhs, const BaseField &rhs) {
      return lhs.getValue() == convertToAPInt(rhs.value(), bitWidth);
    };

    if (eq(a, Curve::Config::kA) && eq(b, Curve::Config::kB) &&
        eq(gX, Curve::Config::kX) && eq(gY, Curve::Config::kY)) {
      return true;
    }
    return false;
  } else if constexpr (BaseField::ExtensionDegree() == 2) {
    auto a = cast<DenseIntElementsAttr>(attr.getA()).getValues<APInt>();
    auto b = cast<DenseIntElementsAttr>(attr.getB()).getValues<APInt>();
    auto gX = cast<DenseIntElementsAttr>(attr.getGx()).getValues<APInt>();
    auto gY = cast<DenseIntElementsAttr>(attr.getGy()).getValues<APInt>();

    unsigned bitWidth = a[0].getBitWidth();
    auto eq = [bitWidth](auto lhs, const BaseField &rhs) {
      for (size_t i = 0; i < BaseField::ExtensionDegree(); ++i) {
        if (lhs[i] != convertToAPInt(rhs[i].value(), bitWidth))
          return false;
      }
      return true;
    };

    if (eq(a, Curve::Config::kA) && eq(b, Curve::Config::kB) &&
        eq(gX, Curve::Config::kX) && eq(gY, Curve::Config::kY)) {
      return true;
    }
    return false;
  }
}

} // namespace

// TODO(chokobole): Use zk_dtypes macros to automate type registration.
std::optional<std::string> getKnownCurveAlias(ShortWeierstrassAttr attr) {
  if (auto pfType = dyn_cast<field::PrimeFieldType>(attr.getBaseField())) {
    auto modulus = pfType.getModulus().getValue();
    auto alias = getKnownModulusAlias(modulus);
    if (!alias.has_value())
      return std::nullopt;
    if (alias == "bn254_bf") {
      if (pfType.isMontgomery()) {
        if (isKnownCurve<zk_dtypes::bn254::G1CurveMont>(attr)) {
          return "bn254_g1";
        }
      } else {
        if (isKnownCurve<zk_dtypes::bn254::G1Curve>(attr)) {
          return "bn254_g1";
        }
      }
    }
  } else if (auto efType =
                 dyn_cast<field::ExtensionFieldType>(attr.getBaseField())) {
    if (auto pfType = dyn_cast<field::PrimeFieldType>(efType.getBaseField())) {
      auto modulus = pfType.getModulus().getValue();
      auto alias = getKnownModulusAlias(modulus);
      if (!alias.has_value())
        return std::nullopt;
      if (alias == "bn254_bf") {
        if (pfType.isMontgomery()) {
          if (isKnownCurve<zk_dtypes::bn254::G2CurveMont>(attr)) {
            return "bn254_g2";
          }
        } else {
          if (isKnownCurve<zk_dtypes::bn254::G2Curve>(attr)) {
            return "bn254_g2";
          }
        }
      }
    }
  }
  return std::nullopt;
}

std::optional<PairingCurveFamily>
getKnownPairingCurveFamily(ShortWeierstrassAttr g1Attr,
                           ShortWeierstrassAttr g2Attr) {
  auto g1Alias = getKnownCurveAlias(g1Attr);
  auto g2Alias = getKnownCurveAlias(g2Attr);
  if (!g1Alias || !g2Alias)
    return std::nullopt;

  if (*g1Alias == "bn254_g1" && *g2Alias == "bn254_g2")
    return PairingCurveFamily::kBN254;

  return std::nullopt;
}

namespace {

// Builds an MLIR ExtensionFieldType from a zk_dtypes extension field type.
// NOTE: Only works for non-tower and 1-level tower extensions (base field's
// non-residue components must be prime field elements).
template <typename ExtF>
field::ExtensionFieldType buildExtFieldType(MLIRContext *ctx) {
  using Op = typename field::detail::ZkDtypeToExtensionFieldOp<ExtF>::type;
  return Op::template getExtensionFieldType<ExtF>(ctx);
}

} // namespace

field::ExtensionFieldType buildBN254Fp2Type(MLIRContext *ctx,
                                            bool isMontgomery) {
  if (isMontgomery)
    return buildExtFieldType<zk_dtypes::bn254::FqX2Mont>(ctx);
  return buildExtFieldType<zk_dtypes::bn254::FqX2>(ctx);
}

field::ExtensionFieldType buildBN254Fp6Type(MLIRContext *ctx,
                                            bool isMontgomery) {
  if (isMontgomery)
    return buildExtFieldType<zk_dtypes::bn254::FqX6Mont>(ctx);
  return buildExtFieldType<zk_dtypes::bn254::FqX6>(ctx);
}

field::ExtensionFieldType buildBN254Fp12Type(MLIRContext *ctx,
                                             bool isMontgomery) {
  // Fp12 = Fp6[w] / (w² - v) is a 2-level tower extension.
  // The getExtensionFieldType template doesn't handle 2-level towers
  // (non-residue components are extension field elements), so we construct
  // the type manually.
  auto fp6Type = buildBN254Fp6Type(ctx, isMontgomery);
  auto storageType = fp6Type.getBasePrimeField().getStorageType();
  unsigned bitWidth = fp6Type.getBasePrimeField().getStorageBitWidth();

  // Non-residue: v = (0, 1, 0) in Fp6 = (c₀, c₁, c₂) where cᵢ ∈ Fp2.
  // Flattened to 6 prime field values: [c₀₀, c₀₁, c₁₀, c₁₁, c₂₀, c₂₁]
  //   c₀ = Fp2(0) = {0, 0}
  //   c₁ = Fp2(1) = {1, 0}
  //   c₂ = Fp2(0) = {0, 0}
  APInt zero(bitWidth, 0);
  APInt one;
  if (isMontgomery) {
    one = convertToAPInt(zk_dtypes::bn254::FqMont::One().value(), bitWidth);
  } else {
    one = APInt(bitWidth, 1);
  }

  SmallVector<APInt, 6> nrCoeffs = {zero, zero, one, zero, zero, zero};
  auto nonResidue = DenseIntElementsAttr::get(
      RankedTensorType::get({6}, storageType), nrCoeffs);
  return field::ExtensionFieldType::get(ctx, 2, fp6Type, nonResidue);
}

} // namespace mlir::prime_ir::elliptic_curve
