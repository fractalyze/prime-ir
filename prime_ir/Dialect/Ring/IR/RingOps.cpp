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

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringExtras.h"
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
  int64_t n = ring.getRingDegree().getValue().getSExtValue();
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
    // A field modulus is an arbitrary-width APInt -- i256 is the ordinary
    // spelling for curve fields -- so it is compared without being narrowed to
    // a word first.
    APInt modulus = fieldType.getModulus().getValue();
    auto expected = static_cast<uint64_t>(ring.getModuli()[i]);
    if (modulus.getActiveBits() > 64 || modulus.getZExtValue() != expected) {
      return op->emitOpError("limb ")
             << i << " has modulus " << llvm::toString(modulus, 10, false)
             << ", but the ring's is " << expected;
    }
    // Residues are stored canonically throughout the lowering. A Montgomery
    // limb carries the extra factor of R, which every ring op would then
    // silently compound; converting it is the caller's job.
    if (fieldType.isMontgomery()) {
      return op->emitOpError("limb ")
             << i
             << " is in Montgomery form, but the ring's residues are "
                "canonical";
    }
    // The bridge reinterprets the limb into the ring's residue rows, so a
    // width mismatch would have to be a widening copy instead.
    unsigned limbWidth = fieldType.getTypeSizeInBits();
    if (limbWidth != ring.getStorageWidth()) {
      return op->emitOpError("limb ")
             << i << " is stored in i" << limbWidth
             << ", but the ring's residues are i" << ring.getStorageWidth();
    }
    if (tensorType.getRank() != 1) {
      return op->emitOpError("limb ")
             << i << " must be 1-D, but got " << tensorType;
    }
    if (tensorType.getDimSize(0) != n) {
      return op->emitOpError("limb ")
             << i << " must have " << n << " elements, but got "
             << tensorType.getDimSize(0);
    }
  }
  return success();
}

// The raw bridge: the tensor IS the lowered representation, so it has to be
// exactly that -- limb-major [L, N] in the ring's residue word. Nothing
// downstream re-checks it, and the patterns slice rows out by those extents.
static LogicalResult verifyResidueTensor(Operation *op, RqType ring,
                                         Type tensorType) {
  auto shaped = llvm::cast<RankedTensorType>(tensorType);
  auto l = static_cast<int64_t>(ring.getModuli().size());
  int64_t n = ring.getRingDegree().getValue().getSExtValue();
  if (shaped.getRank() != 2 || shaped.getDimSize(0) != l ||
      shaped.getDimSize(1) != n) {
    return op->emitOpError("residue tensor must be ")
           << l << "x" << n << ", but got " << shaped;
  }
  if (shaped.getElementType() != ring.getStorageType()) {
    return op->emitOpError("residue tensor element must be the ring's storage "
                           "type i")
           << ring.getStorageWidth() << ", but got " << shaped.getElementType();
  }
  return success();
}

LogicalResult FromTensorOp::verify() {
  return verifyResidueTensor(*this, llvm::cast<RqType>(getOutput().getType()),
                             getInput().getType());
}

LogicalResult ToTensorOp::verify() {
  return verifyResidueTensor(*this, llvm::cast<RqType>(getInput().getType()),
                             getOutput().getType());
}

LogicalResult FromLimbsOp::verify() {
  return verifyLimbTypes(*this, llvm::cast<RqType>(getOutput().getType()),
                         getLimbs().getTypes());
}

LogicalResult ToLimbsOp::verify() {
  return verifyLimbTypes(*this, llvm::cast<RqType>(getInput().getType()),
                         getLimbs().getTypes());
}

// rescale and base_convert both combine limb i and limb j at the same position
// k, which reads that position of every limb as the residues of one common
// integer coefficient -- what the coefficient basis means. In the evaluation
// basis position k holds an evaluation at a root of unity chosen per modulus,
// so the values there are residues of nothing in common and there is no CRT
// combination to perform; reaching the coefficient basis first is the caller's
// job, as it is in production RNS libraries. Both also write the result with
// the input's residue word, so the two rings must agree on it.
static LogicalResult verifyCrossLimbBases(Operation *op, RqType in, RqType out,
                                          StringRef what) {
  if (in.getRingDegree() != out.getRingDegree()) {
    return op->emitOpError("input and output rings must share the degree N (")
           << in.getRingDegree() << " vs " << out.getRingDegree() << ")";
  }
  if (in.getStorageType() != out.getStorageType()) {
    return op->emitOpError("input and output rings must share the residue "
                           "storage type, but got i")
           << in.getStorageWidth() << " and i" << out.getStorageWidth();
  }
  if (!in.isCoeff() || !out.isCoeff()) {
    return op->emitOpError(what)
           << " combines residues across limbs at a fixed position, which "
              "only the coefficient basis relates; got "
           << stringifyDomain(in.getDomain()) << " to "
           << stringifyDomain(out.getDomain());
  }
  return success();
}

LogicalResult BaseConvertOp::verify() {
  return verifyCrossLimbBases(*this, llvm::cast<RqType>(getInput().getType()),
                              llvm::cast<RqType>(getOutput().getType()),
                              "base conversion");
}

LogicalResult RescaleOp::verify() {
  RqType in = llvm::cast<RqType>(getInput().getType());
  RqType out = llvm::cast<RqType>(getOutput().getType());
  if (failed(verifyCrossLimbBases(*this, in, out, "rescale"))) {
    return failure();
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
