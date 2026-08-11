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

#include <cstdint>

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir::prime_ir::ring {

// Ops that are defined only on the coefficient basis share this check; the two
// bases are indistinguishable in storage, so nothing but the type catches the
// mistake.
static LogicalResult requireCoeff(Operation *op, RqType ty,
                                  llvm::StringRef what, llvm::StringRef why) {
  if (ty.isCoeff()) {
    return success();
  }
  return op->emitOpError(what) << " must be in the coeff basis (" << why << ")";
}

LogicalResult BaseConvertOp::verify() {
  RqType in = llvm::cast<RqType>(getInput().getType());
  RqType out = llvm::cast<RqType>(getOutput().getType());
  if (in.getRingDegree() != out.getRingDegree()) {
    return emitOpError("input and output rings must share the degree N (")
           << in.getRingDegree() << " vs " << out.getRingDegree() << ")";
  }
  if (in.getDomain() != out.getDomain()) {
    return emitOpError("base conversion does not change basis, but input is ")
           << stringifyDomain(in.getDomain()) << " and output is "
           << stringifyDomain(out.getDomain());
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
  if (in.getDomain() != out.getDomain()) {
    return emitOpError("rescale does not change basis, but input is ")
           << stringifyDomain(in.getDomain()) << " and output is "
           << stringifyDomain(out.getDomain());
  }
  return success();
}

LogicalResult MulOp::verify() {
  RqType ty = llvm::cast<RqType>(getOutput().getType());
  if (ty.isEval()) {
    return success();
  }
  return emitOpError(
      "operands must be in the eval basis; the coefficient-basis "
      "product is a negacyclic convolution, which needs the "
      "transform that lives above this dialect");
}

LogicalResult GadgetDecomposeOp::verify() {
  int64_t levels = getLevels();
  if (getBaseBits() <= 0 || levels <= 0) {
    return emitOpError("baseBits and levels must be positive");
  }
  if (static_cast<int64_t>(getDigits().size()) != levels) {
    return emitOpError("expected ")
           << levels << " digit results, got " << getDigits().size();
  }
  RqType in = llvm::cast<RqType>(getInput().getType());
  if (failed(
          requireCoeff(*this, in, "input",
                       "digit extraction does not commute with the CRT map"))) {
    return failure();
  }
  for (Value digit : getDigits()) {
    if (llvm::cast<RqType>(digit.getType()).getDomain() != Domain::Coeff) {
      return emitOpError("digits must be in the coeff basis");
    }
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
