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

#include "prime_ir/Dialect/ModArith/Conversions/ModArithToArith/Reducer/BarrettReducer.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithTypes.h" // IWYU pragma: keep

namespace mlir::prime_ir::mod_arith {

namespace {

// Create a splat constant that works for both static and dynamic tensor
// shapes. Duplicates the helper in MontReducer.cpp; consolidate once a third
// reducer needs it.
Value createSplatConst(ImplicitLocOpBuilder &b, TypedAttr scalarAttr,
                       ShapedType shapedType, Value shapeRef) {
  if (shapedType.hasStaticShape()) {
    return arith::ConstantOp::create(
        b, SplatElementsAttr::get(shapedType, scalarAttr));
  }
  assert(shapeRef &&
         "A shape reference value must be provided for dynamic shapes.");
  Value scalar = arith::ConstantOp::create(b, scalarAttr);
  SmallVector<Value> dynamicDims;
  for (int64_t i = 0; i < shapedType.getRank(); ++i) {
    if (shapedType.isDynamicDim(i)) {
      auto idx = arith::ConstantIndexOp::create(b, i);
      dynamicDims.push_back(tensor::DimOp::create(b, shapeRef, idx));
    }
  }
  Value empty = tensor::EmptyOp::create(b, shapedType, dynamicDims);
  return linalg::FillOp::create(b, scalar, empty).getResult(0);
}

} // namespace

BarrettReducer::BarrettReducer(ImplicitLocOpBuilder &b,
                               ModArithType modArithType)
    : b(b), modAttr(modArithType.getModulus()) {
  BarrettAttr barrettAttr = modArithType.getBarrettAttr();
  muAttr = barrettAttr.getMu();
  extBitWidth = barrettAttr.getExtBitWidth();
}

Value BarrettReducer::createExtModulusConst(Type inputType, Value inputValue) {
  // Build a 2k-bit-typed copy of the modulus, then splat to shaped if needed.
  IntegerType extScalar = IntegerType::get(b.getContext(), extBitWidth);
  IntegerAttr extModAttr =
      IntegerAttr::get(extScalar, modAttr.getValue().zext(extBitWidth));
  if (auto shapedType = dyn_cast<ShapedType>(inputType)) {
    auto shapedExt =
        cast<ShapedType>(shapedType.cloneWith(std::nullopt, extScalar));
    return createSplatConst(b, extModAttr, shapedExt, inputValue);
  }
  return arith::ConstantOp::create(b, extModAttr);
}

Value BarrettReducer::createExtMuConst(Type inputType, Value inputValue) {
  // muAttr is already typed at extBitWidth; splat if needed.
  if (auto shapedType = dyn_cast<ShapedType>(inputType)) {
    auto shapedExt =
        cast<ShapedType>(shapedType.cloneWith(std::nullopt, muAttr.getType()));
    return createSplatConst(b, muAttr, shapedExt, inputValue);
  }
  return arith::ConstantOp::create(b, muAttr);
}

Value BarrettReducer::reduce(Value lhs, Value rhs) {
  // Compute the 2k-bit storage type from the per-element bit width of the
  // input. For a shaped input, reshape extScalar onto the input's shape.
  unsigned k = cast<IntegerType>(getElementTypeOrSelf(lhs)).getWidth();
  assert(2 * k == extBitWidth &&
         "BarrettReducer: input bit width must match modulus storage width");
  (void)k;

  IntegerType extScalar = IntegerType::get(b.getContext(), extBitWidth);
  Type extType = extScalar;
  if (auto shapedType = dyn_cast<ShapedType>(lhs.getType()))
    extType = shapedType.cloneWith(std::nullopt, extScalar);

  // Lift inputs to 2k bits and multiply: prod = a * b < p² ≤ 2^(2k).
  Value lhsExt = arith::ExtUIOp::create(b, extType, lhs);
  Value rhsExt = arith::ExtUIOp::create(b, extType, rhs);
  Value prod = arith::MulIOp::create(b, lhsExt, rhsExt);

  // q' = high_half(prod * mu), i.e. (prod * mu) >> 2k.
  // Using arith.mului_extended avoids materializing a 4k-bit intermediate;
  // we keep all arithmetic at 2k bits.
  Value muConst = createExtMuConst(lhs.getType(), lhs);
  auto prodMu = arith::MulUIExtendedOp::create(b, prod, muConst);
  Value qPrime = prodMu.getHigh();

  // r = prod - q' * p. q' ≤ floor(prod/p), so q' * p < 2^(2k); the narrow
  // (low-half) multiply suffices. r is in [0, 2p).
  Value pExt = createExtModulusConst(lhs.getType(), lhs);
  Value qp = arith::MulIOp::create(b, qPrime, pExt);
  Value r = arith::SubIOp::create(b, prod, qp);

  // Single conditional subtraction: if (r >= p) r -= p.
  Value cmp = arith::CmpIOp::create(b, arith::CmpIPredicate::uge, r, pExt);
  Value rSub = arith::SubIOp::create(b, r, pExt);
  Value rCanon = arith::SelectOp::create(b, cmp, rSub, r);

  // Truncate back to k bits — rCanon is in [0, p) ⊂ [0, 2^k).
  return arith::TruncIOp::create(b, lhs.getType(), rCanon);
}

} // namespace mlir::prime_ir::mod_arith
