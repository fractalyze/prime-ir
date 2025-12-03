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

#include "zkir/Dialect/Field/C/FieldAttributes.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "zkir/Dialect/Field/IR/FieldAttributes.h"

using namespace mlir;
using namespace mlir::zkir::field;

//===----------------------------------------------------------------------===//
// PrimeField attributes.
//===----------------------------------------------------------------------===//

MlirTypeID zkirPrimeFieldAttrGetTypeID() {
  return wrap(PrimeFieldAttr::getTypeID());
}

bool zkirAttrIsAPrimeField(MlirAttribute attr) {
  return llvm::isa<PrimeFieldAttr>(unwrap(attr));
}

MlirAttribute zkirPrimeFieldAttrGet(MlirType type, MlirAttribute value) {
  auto pfType = llvm::cast<PrimeFieldType>(unwrap(type));
  auto intAttr = llvm::cast<IntegerAttr>(unwrap(value));
  return wrap(PrimeFieldAttr::get(pfType.getContext(), pfType, intAttr));
}

MlirAttribute zkirPrimeFieldAttrGetValue(MlirAttribute attr) {
  return wrap(llvm::cast<PrimeFieldAttr>(unwrap(attr)).getValue());
}
