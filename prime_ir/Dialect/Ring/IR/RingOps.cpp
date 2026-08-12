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

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"

namespace mlir::prime_ir::ring {

// Shared by from_limbs and to_limbs: each limb is the residue vector of one
// modulus, so it must be a field tensor over that q_i with the ring's N
// entries. Checking the modulus against the limb's own type is the reason this
// bridge exists -- an attribute restating the moduli could drift from them.
static LogicalResult verifyLimbTypes(Operation *op, RqType ring,
                                     TypeRange limbs) {
  if (limbs.size() != ring.getModuli().size()) {
    return op->emitOpError("expects ")
           << ring.getModuli().size() << " limb"
           << (ring.getModuli().size() == 1 ? "" : "s") << ", but got "
           << limbs.size();
  }
  for (auto [i, limbType] : llvm::enumerate(limbs)) {
    auto tensorType = llvm::dyn_cast<RankedTensorType>(limbType);
    auto fieldType =
        tensorType
            ? llvm::dyn_cast<field::PrimeFieldType>(tensorType.getElementType())
            : nullptr;
    if (!fieldType) {
      return op->emitOpError("limb ")
             << i << " must be a prime field tensor, but got " << limbType;
    }
    uint64_t modulus = fieldType.getModulus().getValue().getZExtValue();
    uint64_t expected = ring.getModuli()[i];
    if (modulus != expected) {
      return op->emitOpError("limb ") << i << " has modulus " << modulus
                                      << ", but the ring's is " << expected;
    }
    // The bridge reinterprets the limb into the ring's residue rows, so a
    // width mismatch would have to be a widening copy instead.
    unsigned limbWidth =
        fieldType.getModulus().getType().getIntOrFloatBitWidth();
    if (limbWidth != ring.getStorageWidth()) {
      return op->emitOpError("limb ")
             << i << " is stored in i" << limbWidth
             << ", but the ring's residues are i" << ring.getStorageWidth();
    }
    int64_t n = ring.getRingDegree().getValue().getSExtValue();
    if (tensorType.getRank() != 1 || tensorType.getDimSize(0) != n) {
      return op->emitOpError("limb ")
             << i << " must have " << n << " elements, but got "
             << (tensorType.getRank() == 1 ? tensorType.getDimSize(0)
                                           : tensorType.getRank());
    }
  }
  return success();
}

LogicalResult FromLimbsOp::verify() {
  return verifyLimbTypes(*this, llvm::cast<RqType>(getOutput().getType()),
                         getLimbs().getTypes());
}

LogicalResult ToLimbsOp::verify() {
  return verifyLimbTypes(*this, llvm::cast<RqType>(getInput().getType()),
                         getLimbs().getTypes());
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
