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

#include "prime_ir/Dialect/Field/IR/FieldAttributes.h"

#include "llvm/ADT/SmallString.h"
#include "mlir/Support/LLVM.h"
#include "prime_ir/Dialect/Field/IR/FieldAttributesInterfaces.cpp.inc"
#include "prime_ir/Dialect/Field/IR/FieldOperation.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"

namespace mlir::prime_ir::field {

// static
Attribute RootOfUnityAttr::parse(AsmParser &parser, Type odsType) {
  IntegerAttr root, degree;
  PrimeFieldType type;
  if (failed(parser.parseLess()) || failed(parser.parseAttribute(root)) ||
      failed(parser.parseComma()) || failed(parser.parseAttribute(degree)) ||
      failed(parser.parseGreater()) || failed(parser.parseColonType(type)))
    return nullptr;

  root = cast<IntegerAttr>(maybeToMontgomery(type, root));

  return RootOfUnityAttr::get(parser.getContext(), type, root, degree);
}

void RootOfUnityAttr::print(AsmPrinter &printer) const {
  Attribute root = cast<IntegerAttr>(maybeToStandard(getType(), getRoot()));
  printer << '<' << root << ',' << getDegree() << "> : " << getType();
}

// static
LogicalResult
RootOfUnityAttr::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                        PrimeFieldType type, IntegerAttr root,
                        IntegerAttr degree) {
  auto rootOp = PrimeFieldOperation::fromUnchecked(root, type);
  APInt degreeValue = degree.getValue();
  if (rootOp.power(degreeValue).isOne()) {
    return success();
  }

  SmallString<40> degreeValueStr;
  degreeValue.toString(degreeValueStr, 10, false);
  emitError() << rootOp.toString() << " is not a root of unity of degree "
              << degreeValueStr;
  return failure();
}

} // namespace mlir::prime_ir::field
