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

#ifndef PRIME_IR_DIALECT_ELLIPTICCURVE_IR_POINTOPERATIONSELECTOR_H_
#define PRIME_IR_DIALECT_ELLIPTICCURVE_IR_POINTOPERATIONSELECTOR_H_

#include <cstddef>

#include "prime_ir/Dialect/EllipticCurve/IR/PointKind.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/affine_point_operation.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/jacobian_point_operation.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/point_xyzz_operation.h"

namespace mlir::prime_ir::elliptic_curve {

// Selects the appropriate zk_dtypes extension elliptic curve point operation
// based on kind.
template <PointKind Kind>
struct PointOperationSelector;

template <>
struct PointOperationSelector<PointKind::kAffine> {
  template <typename Derived>
  using Type = zk_dtypes::AffinePointOperation<Derived>;
};

template <>
struct PointOperationSelector<PointKind::kJacobian> {
  template <typename Derived>
  using Type = zk_dtypes::JacobianPointOperation<Derived>;
};

template <>
struct PointOperationSelector<PointKind::kXYZZ> {
  template <typename Derived>
  using Type = zk_dtypes::PointXyzzOperation<Derived>;
};

} // namespace mlir::prime_ir::elliptic_curve

#endif // PRIME_IR_DIALECT_ELLIPTICCURVE_IR_POINTOPERATIONSELECTOR_H_
