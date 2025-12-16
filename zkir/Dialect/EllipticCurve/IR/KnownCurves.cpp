/* Copyright 2025 The ZKIR Authors.

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

#include "zkir/Dialect/EllipticCurve/IR/KnownCurves.h"

#include "zk_dtypes/include/elliptic_curve/bn/bn254/g1.h"
#include "zk_dtypes/include/elliptic_curve/bn/bn254/g2.h"
#include "zkir/Dialect/Field/IR/FieldTypes.h"
#include "zkir/Utils/KnownModulus.h"
#include "zkir/Utils/ZkDtypes.h"

namespace mlir::zkir::elliptic_curve {

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
        if (isKnownCurve<zk_dtypes::bn254::G1Curve>(attr)) {
          return "bn254_g1";
        }
      } else {
        if (isKnownCurve<zk_dtypes::bn254::G1CurveStd>(attr)) {
          return "bn254_g1";
        }
      }
    }
  } else if (auto efType = dyn_cast<field::ExtensionFieldTypeInterface>(
                 attr.getBaseField())) {
    if (auto pfType =
            dyn_cast<field::PrimeFieldType>(efType.getBaseFieldType())) {
      auto modulus = pfType.getModulus().getValue();
      auto alias = getKnownModulusAlias(modulus);
      if (!alias.has_value())
        return std::nullopt;
      if (alias == "bn254_bf") {
        if (pfType.isMontgomery()) {
          if (isKnownCurve<zk_dtypes::bn254::G2Curve>(attr)) {
            return "bn254_g2";
          }
        } else {
          if (isKnownCurve<zk_dtypes::bn254::G2CurveStd>(attr)) {
            return "bn254_g2";
          }
        }
      }
    }
  }
  return std::nullopt;
}

} // namespace mlir::zkir::elliptic_curve
