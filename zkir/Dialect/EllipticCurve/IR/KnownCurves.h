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

#ifndef ZKIR_DIALECT_ELLIPTICCURVE_IR_KNOWNCURVES_H_
#define ZKIR_DIALECT_ELLIPTICCURVE_IR_KNOWNCURVES_H_

#include <optional>
#include <string>

#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveAttributes.h"

namespace mlir::zkir::elliptic_curve {

std::optional<std::string> getKnownCurveAlias(ShortWeierstrassAttr attr);

} // namespace mlir::zkir::elliptic_curve

#endif // ZKIR_DIALECT_ELLIPTICCURVE_IR_KNOWNCURVES_H_
