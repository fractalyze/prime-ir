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

#ifndef PRIME_IR_IR_ATTRIBUTES_H_
#define PRIME_IR_IR_ATTRIBUTES_H_

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "prime_ir/IR/DenseElementBytes.h"

namespace mlir::prime_ir {

ShapedType maybeConvertPrimeIRToBuiltinType(ShapedType type);

// Attr-level dual of maybeConvertPrimeIRToBuiltinType: retype a native-typed
// dense value (EF or EC point) as its storage-int form. Returns attr unchanged
// when the type already maps to itself (PF/BF/ModArith and builtin ints).
DenseElementsAttr maybeConvertPrimeIRToBuiltinAttr(DenseElementsAttr attr);

} // namespace mlir::prime_ir

#endif // PRIME_IR_IR_ATTRIBUTES_H_
