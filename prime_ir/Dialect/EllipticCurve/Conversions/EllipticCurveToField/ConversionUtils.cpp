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
  if (auto affineType = dyn_cast<AffineType>(type)) {
    return SmallVector<Type>(2, affineType.getCurve().getBaseField());
  } else if (auto jacobianType = dyn_cast<JacobianType>(type)) {
    return SmallVector<Type>(3, jacobianType.getCurve().getBaseField());
  } else if (auto xyzzType = dyn_cast<XYZZType>(type)) {
    return SmallVector<Type>(4, xyzzType.getCurve().getBaseField());
  } else {
    llvm_unreachable("Unsupported point-like type for coords type range");
    return SmallVector<Type>();
  }
}

} // namespace

Operation::result_range toCoords(ImplicitLocOpBuilder &b, Value point) {
  return b.create<ExtractOp>(coordsTypeRange(point.getType()), point)
      .getResults();
}

Value fromCoords(ImplicitLocOpBuilder &b, Type type, ValueRange coords) {
  return b.create<PointOp>(type, coords);
}

} // namespace mlir::prime_ir::elliptic_curve
