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

#ifndef ZKIR_DIALECT_ELLIPTICCURVE_IR_ELLIPTICCURVEOPS_H_
#define ZKIR_DIALECT_ELLIPTICCURVE_IR_ELLIPTICCURVEOPS_H_

#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Types.h"

// IWYU pragma: begin_keep
// Headers needed for EllipticCurveOps.h.inc
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "zkir/Dialect/Field/IR/FieldTypes.h"
// IWYU pragma: end_keep

#define GET_OP_CLASSES
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveOps.h.inc"

namespace mlir::zkir::elliptic_curve {

// WARNING: Assumes Jacobian or XYZZ point types
Value createZeroPoint(ImplicitLocOpBuilder &b, Type pointType);

} // namespace mlir::zkir::elliptic_curve

#endif // ZKIR_DIALECT_ELLIPTICCURVE_IR_ELLIPTICCURVEOPS_H_
