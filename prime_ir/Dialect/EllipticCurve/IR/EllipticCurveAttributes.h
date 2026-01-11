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

#ifndef PRIME_IR_DIALECT_ELLIPTICCURVE_IR_ELLIPTICCURVEATTRIBUTES_H_
#define PRIME_IR_DIALECT_ELLIPTICCURVE_IR_ELLIPTICCURVEATTRIBUTES_H_

// IWYU pragma: begin_keep
// Headers needed for EllipticCurveAttributes.h.inc
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
// IWYU pragma: end_keep

#define GET_ATTRDEF_CLASSES
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveAttributes.h.inc"

namespace mlir::prime_ir::elliptic_curve {

ShortWeierstrassAttr getCurveFromPointLike(Type pointLike);

} // namespace mlir::prime_ir::elliptic_curve

#endif // PRIME_IR_DIALECT_ELLIPTICCURVE_IR_ELLIPTICCURVEATTRIBUTES_H_
