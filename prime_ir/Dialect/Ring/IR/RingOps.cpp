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

#include "prime_ir/Dialect/Ring/IR/RingOps.h"

#include "mlir/IR/BuiltinAttributes.h"

namespace mlir::prime_ir::ring {

LogicalResult BaseConvertOp::verify() {
  RqType in = llvm::cast<RqType>(getInput().getType());
  RqType out = llvm::cast<RqType>(getOutput().getType());
  if (in.getRingDegree() != out.getRingDegree()) {
    return emitOpError("input and output rings must share the degree N (")
           << in.getRingDegree() << " vs " << out.getRingDegree() << ")";
  }
  return success();
}

LogicalResult RescaleOp::verify() {
  RqType in = llvm::cast<RqType>(getInput().getType());
  RqType out = llvm::cast<RqType>(getOutput().getType());
  if (in.getRingDegree() != out.getRingDegree()) {
    return emitOpError("input and output must share the degree N");
  }
  ArrayRef<int64_t> inM = in.getModuli().asArrayRef();
  ArrayRef<int64_t> outM = out.getModuli().asArrayRef();
  if (outM.size() + 1 != inM.size()) {
    return emitOpError("rescale drops exactly one modulus (output has ")
           << outM.size() << " limbs, input has " << inM.size() << ")";
  }
  if (inM.take_front(outM.size()) != outM) {
    return emitOpError(
        "output basis must be the input basis without its last modulus");
  }
  return success();
}

LogicalResult GadgetProductOp::verify() {
  if (getBaseBits() <= 0) {
    return emitOpError("baseBits must be positive");
  }
  if (getKeys().empty()) {
    return emitOpError("gadget_product needs at least one key");
  }
  return success();
}

LogicalResult GadgetDecomposeOp::verify() {
  int64_t levels = getLevels();
  if (getBaseBits() <= 0 || levels <= 0) {
    return emitOpError("baseBits and levels must be positive");
  }
  if ((int64_t)getDigits().size() != levels) {
    return emitOpError("expected ") << levels << " digit results, got "
                                    << getDigits().size();
  }
  return success();
}

LogicalResult AutomorphismOp::verify() {
  int64_t g = getExponent();
  if (g <= 0 || (g & 1) == 0) {
    return emitOpError("exponent must be a positive odd integer (coprime to 2N "
                       "for a power-of-two N), got ")
           << g;
  }
  return success();
}

} // namespace mlir::prime_ir::ring
