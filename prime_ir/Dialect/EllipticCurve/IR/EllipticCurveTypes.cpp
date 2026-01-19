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

#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveTypes.h"

namespace mlir::prime_ir::elliptic_curve {

#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveTypesInterfaces.cpp.inc"

#define DEFINE_ELLIPTIC_CURVE_TYPE_INTERFACE_METHODS(TYPE, N)                  \
  unsigned TYPE##Type::getNumCoords() const { return N; }                      \
  Type TYPE##Type::getBaseFieldType() const {                                  \
    return getCurve().getBaseField();                                          \
  }                                                                            \
  Attribute TYPE##Type::getCurveAttr() const { return getCurve(); }            \
  PointKind TYPE##Type::getPointKind() const { return PointKind::k##TYPE; }

DEFINE_ELLIPTIC_CURVE_TYPE_INTERFACE_METHODS(Affine, 2);
DEFINE_ELLIPTIC_CURVE_TYPE_INTERFACE_METHODS(Jacobian, 3);
DEFINE_ELLIPTIC_CURVE_TYPE_INTERFACE_METHODS(XYZZ, 4);

#undef DEFINE_ELLIPTIC_CURVE_TYPE_INTERFACE_METHODS

} // namespace mlir::prime_ir::elliptic_curve
