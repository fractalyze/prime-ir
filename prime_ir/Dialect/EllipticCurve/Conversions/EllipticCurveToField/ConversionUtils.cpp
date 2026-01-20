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

#include "prime_ir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/ConversionUtils.h"

#include "mlir/Support/LLVM.h"
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveOps.h"
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveTypes.h"

namespace mlir::prime_ir::elliptic_curve {
namespace {

SmallVector<Type> coordsTypeRange(Type type) {
  PointTypeInterface pointType = cast<PointTypeInterface>(type);
  return SmallVector<Type>(pointType.getNumCoords(),
                           pointType.getBaseFieldType());
}

} // namespace

Operation::result_range toCoords(ImplicitLocOpBuilder &b, Value point) {
  return b.create<ExtToCoordsOp>(coordsTypeRange(point.getType()), point)
      .getResults();
}

Value fromCoords(ImplicitLocOpBuilder &b, Type type, ValueRange coords) {
  return b.create<ExtFromCoordOp>(type, coords);
}

} // namespace mlir::prime_ir::elliptic_curve
