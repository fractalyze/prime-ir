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

#include "prime_ir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/PairingOperations/PairingCodeGen.h"

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "prime_ir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/ConversionUtils.h"
#include "prime_ir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/PairingOperations/PairingOutliner.h"
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveOps.h"
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveTypes.h"
#include "prime_ir/Dialect/EllipticCurve/IR/KnownCurves.h"
#include "prime_ir/Dialect/Field/IR/FieldOps.h"
#include "prime_ir/Utils/BuilderContext.h"
#include "prime_ir/Utils/ControlFlowOperation.h"
#include "zk_dtypes/include/elliptic_curve/bn/bn254/bn254_curve.h"

namespace mlir::prime_ir::elliptic_curve {

using BN254Config = zk_dtypes::bn254::BN254CurveConfig;
using BN254PairingCurve =
    zk_dtypes::BNCurve<BN254Config, PairingCodeGenDerived>;

namespace {

// ==========================================================================
// Extract G1/G2 points from tensor values.
// ==========================================================================

SmallVector<PairingG1AffineCodeGen> extractG1Points(ImplicitLocOpBuilder &b,
                                                    Value g1Tensor) {
  auto tensorType = cast<RankedTensorType>(g1Tensor.getType());
  int64_t numPoints = tensorType.getDimSize(0);

  SmallVector<PairingG1AffineCodeGen> points;
  for (int64_t i = 0; i < numPoints; ++i) {
    Value idx = b.create<arith::ConstantIndexOp>(i);
    Value point = b.create<tensor::ExtractOp>(g1Tensor, ValueRange{idx});
    // Decompose point into (x, y) coordinates.
    Operation::result_range coords = toCoords(b, point);
    points.push_back(
        {PairingFpCodeGen(coords[0]), PairingFpCodeGen(coords[1])});
  }
  return points;
}

SmallVector<PairingG2AffineCodeGen> extractG2Points(ImplicitLocOpBuilder &b,
                                                    Value g2Tensor) {
  auto tensorType = cast<RankedTensorType>(g2Tensor.getType());
  int64_t numPoints = tensorType.getDimSize(0);

  SmallVector<PairingG2AffineCodeGen> points;
  for (int64_t i = 0; i < numPoints; ++i) {
    Value idx = b.create<arith::ConstantIndexOp>(i);
    Value point = b.create<tensor::ExtractOp>(g2Tensor, ValueRange{idx});
    Operation::result_range coords = toCoords(b, point);
    points.push_back(
        {PairingFp2CodeGen(coords[0]), PairingFp2CodeGen(coords[1])});
  }
  return points;
}

// Get the G2 curve coefficient b as an Fp2 constant.
PairingFp2CodeGen getG2CurveB(ShortWeierstrassAttr g2Attr) {
  auto &ctx = PairingCodeGenContext::GetInstance();
  auto *b = BuilderContext::GetInstance().Top();

  // G2 curve b coefficient is stored as a DenseIntElementsAttr in the curve.
  auto bAttr = cast<DenseIntElementsAttr>(g2Attr.getB());
  auto bVals = bAttr.getValues<APInt>();
  auto pfType = cast<field::PrimeFieldType>(ctx.fp2Type.getBaseField());
  Value b0 = b->create<field::ConstantOp>(
      pfType, IntegerAttr::get(pfType.getStorageType(), bVals[0]));
  Value b1 = b->create<field::ConstantOp>(
      pfType, IntegerAttr::get(pfType.getStorageType(), bVals[1]));
  return PairingFp2CodeGen(
      b->create<field::ExtFromCoeffsOp>(ctx.fp2Type, ValueRange{b0, b1}));
}

// Dispatch FusedMultiMillerLoop by number of pairs (known at compile time).
template <int NumPairs>
PairingFp12CodeGen
fusedMillerLoop(const SmallVector<PairingG1AffineCodeGen> &g1Pts,
                const SmallVector<PairingG2AffineCodeGen> &g2Pts) {
  std::array<PairingG1AffineCodeGen, NumPairs> g1;
  std::array<PairingG2AffineCodeGen, NumPairs> g2;
  for (int i = 0; i < NumPairs; ++i) {
    g1[i] = g1Pts[i];
    g2[i] = g2Pts[i];
  }
  return BN254PairingCurve::FusedMultiMillerLoop<NumPairs>(g1, g2);
}

} // namespace

// ==========================================================================
// Top-level pairing check emission.
//
// Uses CRTP BNCurve::FusedMultiMillerLoop for the fused Miller loop (scf.for
// over NAF bits) and BNCurve::FinalExponentiation. Heavy composite ops
// (MulBy034, CyclotomicSquare) are outlined as func.call via PairingOutliner.
// ==========================================================================
Value emitBN254PairingCheck(ImplicitLocOpBuilder &builder,
                            PairingCurveFamily family, Value g1Points,
                            Value g2Points, bool isMontgomery) {
  ScopedBuilderContext scope(&builder);

  // Set up the type context.
  auto &ctx = PairingCodeGenContext::GetInstance();
  auto g1TensorType = cast<RankedTensorType>(g1Points.getType());
  auto g2TensorType = cast<RankedTensorType>(g2Points.getType());
  auto g1PointType = cast<PointTypeInterface>(g1TensorType.getElementType());
  auto g2PointType = cast<PointTypeInterface>(g2TensorType.getElementType());

  ctx.fpType = cast<field::PrimeFieldType>(g1PointType.getBaseFieldType());
  ctx.fp2Type = cast<field::ExtensionFieldType>(g2PointType.getBaseFieldType());
  ctx.fp6Type = buildBN254Fp6Type(builder.getContext(), isMontgomery);
  ctx.fp12Type = buildBN254Fp12Type(builder.getContext(), isMontgomery);

  // Set up outliner for pairing composite ops (CyclotomicSquare, MulBy034/014).
  auto moduleOp =
      builder.getInsertionBlock()->getParentOp()->getParentOfType<ModuleOp>();
  PairingOutliner outliner(moduleOp);
  ctx.outliner = &outliner;

  // Extract points from tensors.
  auto g1Pts = extractG1Points(builder, g1Points);
  auto g2Pts = extractG2Points(builder, g2Points);
  int numPairs = g1Pts.size();

  // Get G2 curve b coefficient and store in context for PairingTraits.
  auto g2CurveAttr = cast<ShortWeierstrassAttr>(g2PointType.getCurveAttr());
  ctx.g2CurveB = getG2CurveB(g2CurveAttr);

  // Fused Miller loop + Frobenius corrections via CRTP.
  //
  // Always use 2-pair batches to avoid an MLIR Canonicalizer bug that
  // incorrectly removes loop block arguments when >2 G2 projective pairs
  // flow through scf.if yield → cf.cond_br merge in the LLVM dialect.
  // The Miller function is multiplicative: f₁₂₃₄ = f₁₂ · f₃₄.
  PairingFp12CodeGen millerResult;
  int offset = 0;
  while (offset + 2 <= numPairs) {
    SmallVector<PairingG1AffineCodeGen> g1Batch(g1Pts.begin() + offset,
                                                g1Pts.begin() + offset + 2);
    SmallVector<PairingG2AffineCodeGen> g2Batch(g2Pts.begin() + offset,
                                                g2Pts.begin() + offset + 2);
    auto batchResult = fusedMillerLoop<2>(g1Batch, g2Batch);
    millerResult = (offset == 0) ? batchResult : millerResult * batchResult;
    offset += 2;
  }
  if (offset < numPairs) {
    SmallVector<PairingG1AffineCodeGen> g1Batch(g1Pts.begin() + offset,
                                                g1Pts.end());
    SmallVector<PairingG2AffineCodeGen> g2Batch(g2Pts.begin() + offset,
                                                g2Pts.end());
    auto batchResult = fusedMillerLoop<1>(g1Batch, g2Batch);
    millerResult = (offset == 0) ? batchResult : millerResult * batchResult;
  }

  // Final exponentiation via CRTP.
  auto pairingResult = BN254PairingCurve::FinalExponentiation(millerResult);

  ctx.outliner = nullptr;

  // Compare to Fp12 identity (1).
  PairingFp12CodeGen one = PairingFp12CodeGen::One();
  Value result = builder.create<field::CmpOp>(arith::CmpIPredicate::eq,
                                              (Value)pairingResult, (Value)one);
  return result;
}

} // namespace mlir::prime_ir::elliptic_curve
