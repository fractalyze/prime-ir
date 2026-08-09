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

#include "prime_ir/Dialect/Poly/IR/PolyOps.h"

#include "mlir/include/mlir/IR/PatternMatch.h"
#include "prime_ir/Dialect/Field/IR/FieldOperation.h"

namespace mlir::prime_ir::poly {

namespace {
#include "prime_ir/Dialect/Poly/IR/PolyCanonicalization.cpp.inc"
} // namespace

void FromTensorOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.add<ToFromTensor>(context);
}

void ToTensorOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.add<FromToTensor>(context);
}

void NTTOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<INTTAfterNTT>(context);
}

LogicalResult NTTOp::verify() {
  if (!getNegacyclic())
    return success();

  // The twist is `psi^j` on the *natural* coefficient index. Without the
  // bit-reversal the caller owns the ordering, so slot `j` need not hold
  // coefficient `j` and the factor would land on the wrong one. The round
  // trip still closes but the pointwise product stops being a negacyclic
  // convolution, which is the whole reason for the transform.
  if (!getBitReverse())
    return emitOpError("negacyclic requires `bit_reverse`");

  // Supplied twiddles cannot carry the twist: the butterfly indexes them as
  // `roots[indexJ * rootExp]`, with no block index, so one table entry is
  // shared across every block of a stage — a per-coefficient diagonal is not
  // expressible. Accepting the pair would silently lower to a cyclic
  // transform.
  if (getTwiddles())
    return emitOpError("negacyclic is not supported with `twiddles`");

  auto rootOfUnity = getRoot();
  if (!rootOfUnity)
    return emitOpError("negacyclic requires `root`");

  auto tensorType = cast<RankedTensorType>(getOutput().getType());
  if (tensorType.isDynamicDim(0))
    return emitOpError("negacyclic requires a static length");
  int64_t degree = tensorType.getShape()[0];

  // `psi^n == -1` rather than a check on the *stated* degree. `RootOfUnityAttr`
  // verifies only `root^degree == 1`, which is divisibility and not order, so
  // an n-th root relabelled as a 2n-th one passes it, and would then drive
  // the core with `psi^2` of half the needed order, silently computing a
  // transform over the wrong ring. This is the defining property instead of a
  // proxy for it, and it is also what fails when `q - 1` lacks the 2-adicity
  // for a 2n-th root at all.
  auto coeffType = cast<field::PrimeFieldType>(tensorType.getElementType());
  auto psi = field::PrimeFieldOperation::fromUnchecked(rootOfUnity->getRoot(),
                                                       coeffType);
  unsigned bits = coeffType.getModulus().getValue().getBitWidth();
  if (!(psi.power(APInt(bits, degree)) + psi.getOne()).isZero())
    return emitOpError("negacyclic over ")
           << degree << " coefficients needs `root^" << degree
           << " == -1`, i.e. a primitive root of unity of degree "
           << 2 * degree;

  return success();
}

} // namespace mlir::prime_ir::poly
