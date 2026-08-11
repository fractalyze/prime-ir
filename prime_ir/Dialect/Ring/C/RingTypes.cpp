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

#include "prime_ir/Dialect/Ring/IR/RingTypes.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "prime_ir/Dialect/Ring/C/RingTypes.h"

using namespace mlir;
using namespace mlir::prime_ir::ring;

MlirTypeID primeIRRqTypeGetTypeID() { return wrap(RqType::getTypeID()); }

bool primeIRTypeIsARq(MlirType type) { return llvm::isa<RqType>(unwrap(type)); }

MlirType primeIRRqTypeGet(MlirContext ctx, intptr_t nModuli,
                          const int64_t *moduli, MlirAttribute ringDegree,
                          PrimeIRRingDomain domain) {
  MLIRContext *context = unwrap(ctx);
  return wrap(RqType::get(
      context,
      DenseI64ArrayAttr::get(context, llvm::ArrayRef<int64_t>(moduli, nModuli)),
      llvm::cast<IntegerAttr>(unwrap(ringDegree)),
      static_cast<Domain>(domain)));
}

intptr_t primeIRRqTypeGetNumModuli(MlirType type) {
  return llvm::cast<RqType>(unwrap(type)).getModuli().size();
}

int64_t primeIRRqTypeGetModulus(MlirType type, intptr_t pos) {
  return llvm::cast<RqType>(unwrap(type)).getModuli().asArrayRef()[pos];
}

MlirAttribute primeIRRqTypeGetRingDegree(MlirType type) {
  return wrap(llvm::cast<RqType>(unwrap(type)).getRingDegree());
}

PrimeIRRingDomain primeIRRqTypeGetDomain(MlirType type) {
  return static_cast<PrimeIRRingDomain>(
      llvm::cast<RqType>(unwrap(type)).getDomain());
}
