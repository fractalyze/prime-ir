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

// Raw words carry no basis, so a bridge into the ring asserts one -- there is
// nothing to check when the words come from outside the module. But a value
// that just came back out of the opposite bridge still carries the basis of the
// ring it left, and naming a different one is then a relabel with nothing
// between. The transform that does change the basis is the NTT, which lives
// above this dialect.
static LogicalResult verifyNoRelabel(Operation *op, RqType producer,
                                     RqType named) {
  if (producer == named || producer.withDomain(named.getDomain()) != named) {
    return success();
  }
  return op->emitOpError("cannot relabel the basis of a value that came "
                         "straight back from the matching bridge: the "
                         "residues are ")
         << stringifyDomain(producer.getDomain()) << " and this names them "
         << stringifyDomain(named.getDomain());
}

// The limb bridges round-trip only when every limb of this op is the matching
// result of one `to_limbs`, in order; anything else is a genuine regrouping.
static ToLimbsOp matchingLimbProducer(OperandRange limbs) {
  if (limbs.empty()) {
    return {};
  }
  auto producer = limbs[0].getDefiningOp<ToLimbsOp>();
  if (!producer || producer.getLimbs().size() != limbs.size()) {
    return {};
  }
  for (auto [i, limb] : llvm::enumerate(limbs)) {
    if (limb != producer.getLimbs()[i]) {
      return {};
    }
  }
  return producer;
}

LogicalResult FromTensorOp::verify() {
  auto ring = llvm::cast<RqType>(getOutput().getType());
  if (failed(verifyResidueTensor(*this, ring, getInput().getType()))) {
    return failure();
  }
  if (auto producer = getInput().getDefiningOp<ToTensorOp>()) {
    return verifyNoRelabel(
        *this, llvm::cast<RqType>(producer.getInput().getType()), ring);
  }
  return success();
}

OpFoldResult FromTensorOp::fold(FoldAdaptor adaptor) {
  auto producer = getInput().getDefiningOp<ToTensorOp>();
  if (producer && producer.getInput().getType() == getOutput().getType()) {
    return producer.getInput();
  }
  return {};
}

LogicalResult ToTensorOp::verify() {
  return verifyResidueTensor(*this, llvm::cast<RqType>(getInput().getType()),
                             getOutput().getType());
}

OpFoldResult ToTensorOp::fold(FoldAdaptor adaptor) {
  auto producer = getInput().getDefiningOp<FromTensorOp>();
  if (producer && producer.getInput().getType() == getOutput().getType()) {
    return producer.getInput();
  }
  return {};
}

LogicalResult FromLimbsOp::verify() {
  auto ring = llvm::cast<RqType>(getOutput().getType());
  if (failed(verifyLimbTypes(*this, ring, getLimbs().getTypes()))) {
    return failure();
  }
  if (ToLimbsOp producer = matchingLimbProducer(getLimbs())) {
    return verifyNoRelabel(
        *this, llvm::cast<RqType>(producer.getInput().getType()), ring);
  }
  return success();
}

OpFoldResult FromLimbsOp::fold(FoldAdaptor adaptor) {
  ToLimbsOp producer = matchingLimbProducer(getLimbs());
  if (producer && producer.getInput().getType() == getOutput().getType()) {
    return producer.getInput();
  }
  return {};
}

LogicalResult ToLimbsOp::verify() {
  return verifyLimbTypes(*this, llvm::cast<RqType>(getInput().getType()),
                         getLimbs().getTypes());
}

LogicalResult ToLimbsOp::fold(FoldAdaptor adaptor,
                              SmallVectorImpl<OpFoldResult> &results) {
  auto producer = getInput().getDefiningOp<FromLimbsOp>();
  if (!producer ||
      !llvm::equal(producer.getLimbs().getTypes(), getLimbs().getTypes())) {
    return failure();
  }
  llvm::append_range(results, producer.getLimbs());
  return success();
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
