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

#include "zkir/Dialect/Field/IR/FieldAttributes.h"

#include "llvm/ADT/SmallString.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "zkir/Dialect/Field/IR/FieldAttributesInterfaces.cpp.inc"
#include "zkir/Utils/APIntUtils.h"

namespace mlir::zkir::field {

// static
LogicalResult
RootOfUnityAttr::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                        PrimeFieldType type, IntegerAttr root,
                        IntegerAttr degree) {
  if (type.isMontgomery()) {
    // NOTE(batzor): Montgomery form is not supported for root of unity because
    // verification logic assumes standard form. Also, `PrimitiveRootAttr` in
    // the `Poly` dialect should also handle it if we want to allow this in the
    // future.
    emitError() << "root of unity must be in standard form";
    return failure();
  }
  APInt modulus = type.getModulus().getValue();
  APInt rootOfUnity = root.getValue();
  APInt degreeValue = degree.getValue();

  if (!expMod(rootOfUnity, degreeValue, modulus).isOne()) {
    SmallString<40> rootOfUnityStr;
    rootOfUnity.toString(rootOfUnityStr, 10, false);
    SmallString<40> degreeValueStr;
    degreeValue.toString(degreeValueStr, 10, false);
    emitError() << rootOfUnityStr << " is not a root of unity of degree "
                << degreeValueStr;
    return failure();
  }

  return success();
}

} // namespace mlir::zkir::field
