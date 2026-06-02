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

// Retype a field-typed DenseElementsAttr<tensor<...x!PF/!EF>> as the
// equivalent storage-int-typed DenseElementsAttr (the form upstream fold
// paths expect). Pass-through for DenseIntElementsAttr and non-field element
// types. Splat caveat: tensor<Nx!EF> with degree>1 stores one EF element's
// bytes; expanded to a full prime-coeff row so the storage-int view's
// splat-vs-full size check passes.
DenseElementsAttr maybeDemoteFieldDenseToStorageInt(DenseElementsAttr attr);

} // namespace mlir::prime_ir

#endif // PRIME_IR_IR_ATTRIBUTES_H_
