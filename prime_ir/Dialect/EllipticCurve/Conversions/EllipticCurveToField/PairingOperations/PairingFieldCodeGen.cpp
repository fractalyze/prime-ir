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

#include "prime_ir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/PairingOperations/PairingFieldCodeGen.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "prime_ir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/PairingOperations/PairingOutliner.h"
#include "prime_ir/Dialect/Field/IR/FieldOps.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"
#include "prime_ir/Dialect/Field/IR/PrimeFieldOperation.h"
#include "prime_ir/Utils/BitSerialAlgorithm.h"
#include "prime_ir/Utils/BuilderContext.h"
#include "prime_ir/Utils/ZkDtypes.h"
#include "zk_dtypes/include/elliptic_curve/bn/bn254/bn254_curve.h"

namespace mlir::prime_ir::elliptic_curve {

namespace {

ImplicitLocOpBuilder *getBuilder() {
  return BuilderContext::GetInstance().Top();
}

} // namespace

// ==========================================================================
// PairingFpCodeGen implementation
// ==========================================================================

PairingFpCodeGen PairingFpCodeGen::CreateConst(int64_t constant) const {
  return PairingFpCodeGen(
      field::createFieldConstant(value.getType(), *getBuilder(), constant));
}

PairingFpCodeGen PairingFpCodeGen::Zero() {
  auto &ctx = PairingCodeGenContext::GetInstance();
  return PairingFpCodeGen(field::createFieldZero(ctx.fpType, *getBuilder()));
}

PairingFpCodeGen PairingFpCodeGen::One() {
  auto &ctx = PairingCodeGenContext::GetInstance();
  return PairingFpCodeGen(field::createFieldOne(ctx.fpType, *getBuilder()));
}

PairingFpCodeGen PairingFpCodeGen::TwoInv() {
  auto &ctx = PairingCodeGenContext::GetInstance();
  auto twoInv = field::PrimeFieldOperation(int64_t{2}, ctx.fpType).inverse();
  return PairingFpCodeGen(
      field::createFieldConstant(ctx.fpType, *getBuilder(), APInt(twoInv)));
}

// ==========================================================================
// PairingFp2CodeGen implementation
// ==========================================================================

PairingFp2CodeGen::PairingFp2CodeGen(Value value) : value(value) {
  auto efType = cast<field::ExtensionFieldType>(value.getType());
  auto pfType = cast<field::PrimeFieldType>(efType.getBaseField());
  // Extract non-residue from the type.
  auto nrAttr = cast<IntegerAttr>(efType.getNonResidue());
  nonResidue =
      field::createFieldConstant(pfType, *getBuilder(), nrAttr.getValue());
}

std::array<PairingFpCodeGen, 2> PairingFp2CodeGen::ToCoeffs() const {
  auto coeffs = field::toCoeffs(*getBuilder(), value);
  return {PairingFpCodeGen(coeffs[0]), PairingFpCodeGen(coeffs[1])};
}

PairingFp2CodeGen
PairingFp2CodeGen::FromCoeffs(const std::array<PairingFpCodeGen, 2> &c) const {
  return PairingFp2CodeGen(field::fromCoeffs(*getBuilder(), value.getType(),
                                             {(Value)c[0], (Value)c[1]}));
}

PairingFpCodeGen PairingFp2CodeGen::NonResidue() const {
  return PairingFpCodeGen(nonResidue);
}

PairingFp2CodeGen
PairingFp2CodeGen::operator*(const PairingFp2CodeGen &o) const {
  return PairingFp2CodeGen(
      getBuilder()->create<field::MulOp>(value, o.value).getOutput());
}

PairingFp2CodeGen
PairingFp2CodeGen::operator*(const PairingFpCodeGen &scalar) const {
  // Scalar multiply: (a0, a1) * s = (a0 * s, a1 * s)
  auto c = ToCoeffs();
  return FromCoeffs({c[0] * scalar, c[1] * scalar});
}

PairingFp2CodeGen PairingFp2CodeGen::Square() const {
  return PairingFp2CodeGen(
      getBuilder()->create<field::SquareOp>(value).getOutput());
}

PairingFp2CodeGen PairingFp2CodeGen::Inverse() const {
  return PairingFp2CodeGen(
      getBuilder()->create<field::InverseOp>(value).getOutput());
}

PairingFp2CodeGen PairingFp2CodeGen::CreateConst(int64_t constant) const {
  auto *b = getBuilder();
  return PairingFp2CodeGen(
      field::createFieldConstant(value.getType(), *b, constant));
}

PairingFp2CodeGen PairingFp2CodeGen::Zero() {
  auto &ctx = PairingCodeGenContext::GetInstance();
  return PairingFp2CodeGen(field::createFieldZero(ctx.fp2Type, *getBuilder()));
}

PairingFp2CodeGen PairingFp2CodeGen::One() {
  auto &ctx = PairingCodeGenContext::GetInstance();
  return PairingFp2CodeGen(field::createFieldOne(ctx.fp2Type, *getBuilder()));
}

PairingFp2CodeGen PairingFp2CodeGen::Select(Value condition,
                                            const PairingFp2CodeGen &a,
                                            const PairingFp2CodeGen &b) {
  auto aC = a.ToCoeffs();
  auto bC = b.ToCoeffs();
  auto *builder = getBuilder();
  Value s0 = builder->create<mlir::arith::SelectOp>(condition, (Value)aC[0],
                                                    (Value)bC[0]);
  Value s1 = builder->create<mlir::arith::SelectOp>(condition, (Value)aC[1],
                                                    (Value)bC[1]);
  auto &ctx = PairingCodeGenContext::GetInstance();
  return PairingFp2CodeGen(field::fromCoeffs(*builder, ctx.fp2Type, {s0, s1}));
}

PairingFp6CodeGen PairingFp6CodeGen::Select(Value condition,
                                            const PairingFp6CodeGen &a,
                                            const PairingFp6CodeGen &b) {
  auto aC = a.ToCoeffs();
  auto bC = b.ToCoeffs();
  auto s0 = PairingFp2CodeGen::Select(condition, aC[0], bC[0]);
  auto s1 = PairingFp2CodeGen::Select(condition, aC[1], bC[1]);
  auto s2 = PairingFp2CodeGen::Select(condition, aC[2], bC[2]);
  auto &ctx = PairingCodeGenContext::GetInstance();
  return PairingFp6CodeGen(field::fromCoeffs(
      *getBuilder(), ctx.fp6Type, {(Value)s0, (Value)s1, (Value)s2}));
}

PairingFp12CodeGen PairingFp12CodeGen::Select(Value condition,
                                              const PairingFp12CodeGen &a,
                                              const PairingFp12CodeGen &b) {
  auto aC = a.ToCoeffs();
  auto bC = b.ToCoeffs();
  auto s0 = PairingFp6CodeGen::Select(condition, aC[0], bC[0]);
  auto s1 = PairingFp6CodeGen::Select(condition, aC[1], bC[1]);
  auto &ctx = PairingCodeGenContext::GetInstance();
  return PairingFp12CodeGen(
      field::fromCoeffs(*getBuilder(), ctx.fp12Type, {(Value)s0, (Value)s1}));
}

// ==========================================================================
// PairingFp6CodeGen implementation
// ==========================================================================

PairingFp6CodeGen::PairingFp6CodeGen(Value value) : value(value) {
  auto efType = cast<field::ExtensionFieldType>(value.getType());
  auto fp2Type = cast<field::ExtensionFieldType>(efType.getBaseField());
  // The non-residue is a DenseIntElementsAttr with 2 prime field values
  // (since the non-residue is in Fp2).
  auto nrAttr = cast<DenseIntElementsAttr>(efType.getNonResidue());
  auto vals = nrAttr.getValues<APInt>();
  auto pfType = cast<field::PrimeFieldType>(fp2Type.getBaseField());
  auto *b = getBuilder();
  Value nr0 = field::createFieldConstant(pfType, *b, vals[0]);
  Value nr1 = field::createFieldConstant(pfType, *b, vals[1]);
  nonResidue = field::fromCoeffs(*b, fp2Type, {nr0, nr1});
}

std::array<PairingFp2CodeGen, 3> PairingFp6CodeGen::ToCoeffs() const {
  auto coeffs = field::toCoeffs(*getBuilder(), value);
  return {PairingFp2CodeGen(coeffs[0]), PairingFp2CodeGen(coeffs[1]),
          PairingFp2CodeGen(coeffs[2])};
}

PairingFp6CodeGen
PairingFp6CodeGen::FromCoeffs(const std::array<PairingFp2CodeGen, 3> &c) const {
  return PairingFp6CodeGen(field::fromCoeffs(
      *getBuilder(), value.getType(), {(Value)c[0], (Value)c[1], (Value)c[2]}));
}

PairingFp2CodeGen PairingFp6CodeGen::NonResidue() const {
  return PairingFp2CodeGen(nonResidue);
}

PairingFp6CodeGen
PairingFp6CodeGen::operator*(const PairingFp6CodeGen &o) const {
  return PairingFp6CodeGen(
      getBuilder()->create<field::MulOp>(value, o.value).getOutput());
}

PairingFp6CodeGen
PairingFp6CodeGen::operator*(const PairingFp2CodeGen &scalar) const {
  // Scalar multiply: (c0, c1, c2) * s = (c0 * s, c1 * s, c2 * s)
  auto c = ToCoeffs();
  return FromCoeffs({c[0] * scalar, c[1] * scalar, c[2] * scalar});
}

PairingFp6CodeGen PairingFp6CodeGen::Square() const {
  return PairingFp6CodeGen(
      getBuilder()->create<field::SquareOp>(value).getOutput());
}

PairingFp6CodeGen PairingFp6CodeGen::Inverse() const {
  return PairingFp6CodeGen(
      getBuilder()->create<field::InverseOp>(value).getOutput());
}

PairingFp6CodeGen PairingFp6CodeGen::CreateConst(int64_t constant) const {
  auto *b = getBuilder();
  return PairingFp6CodeGen(
      field::createFieldConstant(value.getType(), *b, constant));
}

PairingFp6CodeGen PairingFp6CodeGen::Zero() {
  auto &ctx = PairingCodeGenContext::GetInstance();
  return PairingFp6CodeGen(field::createFieldZero(ctx.fp6Type, *getBuilder()));
}

PairingFp6CodeGen PairingFp6CodeGen::One() {
  auto &ctx = PairingCodeGenContext::GetInstance();
  return PairingFp6CodeGen(field::createFieldOne(ctx.fp6Type, *getBuilder()));
}

// ==========================================================================
// PairingFp12CodeGen implementation
// ==========================================================================

PairingFp12CodeGen::PairingFp12CodeGen(Value value) : value(value) {
  auto efType = cast<field::ExtensionFieldType>(value.getType());
  auto fp6Type = cast<field::ExtensionFieldType>(efType.getBaseField());

  // The Fp12 non-residue is v = (0, 1, 0) in Fp6, stored as a
  // DenseIntElementsAttr with 6 flattened prime field values.
  // We reconstruct it as an Fp6 value.
  auto nrAttr = cast<DenseIntElementsAttr>(efType.getNonResidue());
  auto nrVals = nrAttr.getValues<APInt>();
  auto fp2Type = cast<field::ExtensionFieldType>(fp6Type.getBaseField());
  auto pfType = cast<field::PrimeFieldType>(fp2Type.getBaseField());

  // Build 3 Fp2 values from 6 prime field values.
  auto *b = getBuilder();
  SmallVector<Value, 3> fp2Vals;
  for (int i = 0; i < 3; ++i) {
    Value v0 = field::createFieldConstant(pfType, *b, nrVals[i * 2]);
    Value v1 = field::createFieldConstant(pfType, *b, nrVals[i * 2 + 1]);
    fp2Vals.push_back(field::fromCoeffs(*b, fp2Type, {v0, v1}));
  }
  nonResidue = field::fromCoeffs(*b, fp6Type, fp2Vals);
}

std::array<PairingFp6CodeGen, 2> PairingFp12CodeGen::ToCoeffs() const {
  auto coeffs = field::toCoeffs(*getBuilder(), value);
  return {PairingFp6CodeGen(coeffs[0]), PairingFp6CodeGen(coeffs[1])};
}

PairingFp12CodeGen PairingFp12CodeGen::FromCoeffs(
    const std::array<PairingFp6CodeGen, 2> &c) const {
  return PairingFp12CodeGen(field::fromCoeffs(*getBuilder(), value.getType(),
                                              {(Value)c[0], (Value)c[1]}));
}

PairingFp6CodeGen PairingFp12CodeGen::NonResidue() const {
  return PairingFp6CodeGen(nonResidue);
}

PairingFp12CodeGen PairingFp12CodeGen::CreateConst(int64_t constant) const {
  auto *b = getBuilder();
  return PairingFp12CodeGen(
      field::createFieldConstant(value.getType(), *b, constant));
}

PairingFp12CodeGen PairingFp12CodeGen::Zero() {
  auto &ctx = PairingCodeGenContext::GetInstance();
  return PairingFp12CodeGen(
      field::createFieldZero(ctx.fp12Type, *getBuilder()));
}

PairingFp12CodeGen PairingFp12CodeGen::One() {
  auto &ctx = PairingCodeGenContext::GetInstance();
  return PairingFp12CodeGen(field::createFieldOne(ctx.fp12Type, *getBuilder()));
}

// --- Outlined Fp12 multiply ---

PairingFp12CodeGen
PairingFp12CodeGen::operator*(const PairingFp12CodeGen &other) const {
  auto &ctx = PairingCodeGenContext::GetInstance();
  if (ctx.outliner && !ctx.outliner->isGeneratingBody()) {
    return PairingFp12CodeGen(
        ctx.outliner->emitFp12MulCall(*getBuilder(), value, other.value));
  }
  // Fall back to CRTP Karatsuba for function body generation.
  return zk_dtypes::QuadraticExtensionFieldOperation<
      PairingFp12CodeGen>::operator*(other);
}

// --- Loop-structured / outlined composite operations ---

PairingFp12CodeGen
PairingFp12CodeGen::CyclotomicPow(const zk_dtypes::BigInt<1> &exponent) const {
  auto *b = BuilderContext::GetInstance().Top();
  auto &ctx = PairingCodeGenContext::GetInstance();

  uint64_t limb = exponent[0];
  assert(limb != 0 && "CyclotomicPow with zero exponent");

  Value expConst =
      b->create<arith::ConstantOp>(b->getIntegerAttr(b->getI64Type(), limb));
  Value identity = (Value)PairingFp12CodeGen::One();

  Value result = generateBitSerialLoop(
      *b, expConst, value, identity,
      [&ctx](ImplicitLocOpBuilder &ib, Value v) -> Value {
        return ctx.outliner->emitCyclotomicSquareCall(ib, v);
      },
      [&ctx](ImplicitLocOpBuilder &ib, Value acc, Value v) -> Value {
        return ctx.outliner->emitFp12MulCall(ib, acc, v);
      });

  return PairingFp12CodeGen(result);
}

PairingFp12CodeGen PairingFp12CodeGen::CyclotomicSquare() const {
  auto &ctx = PairingCodeGenContext::GetInstance();
  if (ctx.outliner && !ctx.outliner->isGeneratingBody()) {
    return PairingFp12CodeGen(
        ctx.outliner->emitCyclotomicSquareCall(*getBuilder(), value));
  }
  // Fall back to base CRTP implementation for function body generation.
  return zk_dtypes::CyclotomicOperation<PairingFp12CodeGen>::CyclotomicSquare();
}

PairingFp12CodeGen
PairingFp12CodeGen::MulBy034(const PairingFp2CodeGen &beta0,
                             const PairingFp2CodeGen &beta3,
                             const PairingFp2CodeGen &beta4) const {
  auto &ctx = PairingCodeGenContext::GetInstance();
  if (ctx.outliner && !ctx.outliner->isGeneratingBody()) {
    return PairingFp12CodeGen(
        ctx.outliner->emitMulBy034Call(*getBuilder(), value, beta0.getValue(),
                                       beta3.getValue(), beta4.getValue()));
  }
  return zk_dtypes::QuadraticExtensionFieldOperation<
      PairingFp12CodeGen>::MulBy034(beta0, beta3, beta4);
}

PairingFp12CodeGen
PairingFp12CodeGen::MulBy014(const PairingFp2CodeGen &beta0,
                             const PairingFp2CodeGen &beta1,
                             const PairingFp2CodeGen &beta4) const {
  auto &ctx = PairingCodeGenContext::GetInstance();
  if (ctx.outliner && !ctx.outliner->isGeneratingBody()) {
    return PairingFp12CodeGen(
        ctx.outliner->emitMulBy014Call(*getBuilder(), value, beta0.getValue(),
                                       beta1.getValue(), beta4.getValue()));
  }
  return zk_dtypes::QuadraticExtensionFieldOperation<
      PairingFp12CodeGen>::MulBy014(beta0, beta1, beta4);
}

} // namespace mlir::prime_ir::elliptic_curve

// ==========================================================================
// PairingTraits<PairingCodeGenDerived> implementation and materialize helpers.
// ==========================================================================
namespace zk_dtypes {

using BN254Config = bn254::BN254CurveConfig;
using mlir::prime_ir::elliptic_curve::PairingCodeGenContext;
using mlir::prime_ir::elliptic_curve::PairingCodeGenDerived;
using mlir::prime_ir::elliptic_curve::PairingFp2CodeGen;
using mlir::prime_ir::elliptic_curve::PairingFp6CodeGen;
using Traits = PairingTraits<PairingCodeGenDerived>;

// Helper: materialize a concrete BN254 Fp2 value as MLIR field constants.
PairingFp2CodeGen materializeBN254Fp2Const(const BN254Config::Fp2 &val) {
  auto &ctx = PairingCodeGenContext::GetInstance();
  mlir::APInt modulus = ctx.fpType.getModulus().getValue();
  unsigned bw = modulus.getBitWidth();

  auto coeffs = val.ToCoeffs();
  mlir::APInt c0 = mlir::prime_ir::convertToAPInt(coeffs[0].value(), bw);
  mlir::APInt c1 = mlir::prime_ir::convertToAPInt(coeffs[1].value(), bw);

  auto pfType = mlir::cast<mlir::prime_ir::field::PrimeFieldType>(
      ctx.fp2Type.getBaseField());
  auto *b = mlir::prime_ir::BuilderContext::GetInstance().Top();
  mlir::Value v0 = mlir::prime_ir::field::createFieldConstant(pfType, *b, c0);
  mlir::Value v1 = mlir::prime_ir::field::createFieldConstant(pfType, *b, c1);
  mlir::Value result =
      mlir::prime_ir::field::fromCoeffs(*b, ctx.fp2Type, {v0, v1});
  return PairingFp2CodeGen(result);
}

// Helper: materialize a concrete BN254 Fp6 value as MLIR field constants.
PairingFp6CodeGen materializeBN254Fp6Const(const BN254Config::Fp6 &val) {
  auto &ctx = PairingCodeGenContext::GetInstance();
  auto coeffs = val.ToCoeffs();
  PairingFp2CodeGen c0 = materializeBN254Fp2Const(coeffs[0]);
  PairingFp2CodeGen c1 = materializeBN254Fp2Const(coeffs[1]);
  PairingFp2CodeGen c2 = materializeBN254Fp2Const(coeffs[2]);
  auto *b = mlir::prime_ir::BuilderContext::GetInstance().Top();
  mlir::Value result = mlir::prime_ir::field::fromCoeffs(
      *b, ctx.fp6Type, {(mlir::Value)c0, (mlir::Value)c1, (mlir::Value)c2});
  return PairingFp6CodeGen(result);
}

Traits::Fp2 Traits::G2CurveB() {
  return PairingFp2CodeGen(PairingCodeGenContext::GetInstance().g2CurveB);
}

Traits::Fp2 Traits::TwistMulByQX() {
  return materializeBN254Fp2Const(BN254Config::kTwistMulByQX);
}

Traits::Fp2 Traits::TwistMulByQY() {
  return materializeBN254Fp2Const(BN254Config::kTwistMulByQY);
}

// --- NAF bit accessor for codegen ---

mlir::Value Traits::GetNafBit(mlir::Value iv) {
  auto *b = mlir::prime_ir::BuilderContext::GetInstance().Top();
  constexpr int64_t kLoopSize = std::size(BN254Config::kAteLoopCount);

  // Create NAF tensor constant (LICM will hoist out of loop).
  llvm::SmallVector<int8_t> nafValues(std::begin(BN254Config::kAteLoopCount),
                                      std::end(BN254Config::kAteLoopCount));
  auto nafTensorType = mlir::RankedTensorType::get({kLoopSize}, b->getI8Type());
  auto nafAttr = mlir::DenseIntElementsAttr::get(nafTensorType, nafValues);
  mlir::Value nafTensor = b->create<mlir::arith::ConstantOp>(nafAttr);
  mlir::Value sizeMin2 = b->create<mlir::arith::ConstantIndexOp>(kLoopSize - 2);

  // NAF bit at index (kLoopSize - 2 - iv).
  mlir::Value bitIdx = b->create<mlir::arith::SubIOp>(sizeMin2, iv);
  return b->create<mlir::tensor::ExtractOp>(nafTensor,
                                            mlir::ValueRange{bitIdx});
}

mlir::Value Traits::IsNafNonZero(mlir::Value nafBit) {
  auto *b = mlir::prime_ir::BuilderContext::GetInstance().Top();
  mlir::Value zeroI8 =
      b->create<mlir::arith::ConstantOp>(b->getI8IntegerAttr(0));
  return b->create<mlir::arith::CmpIOp>(mlir::arith::CmpIPredicate::ne, nafBit,
                                        zeroI8);
}

mlir::Value Traits::IsNafPositive(mlir::Value nafBit) {
  auto *b = mlir::prime_ir::BuilderContext::GetInstance().Top();
  mlir::Value zeroI8 =
      b->create<mlir::arith::ConstantOp>(b->getI8IntegerAttr(0));
  return b->create<mlir::arith::CmpIOp>(mlir::arith::CmpIPredicate::sgt, nafBit,
                                        zeroI8);
}

} // namespace zk_dtypes

// ==========================================================================
// Frobenius coefficient implementations.
// Defined after materialize helpers so they can reference them.
// ==========================================================================
namespace mlir::prime_ir::elliptic_curve {

// Fp6 Frobenius coefficients: delegates to BN254 FqX6::GetFrobeniusCoeffs()
// and materializes the concrete values as MLIR field constants.
// Dimensions: (D - 1) × (n - 1) = (6 - 1) × (3 - 1) = 5 × 2.
std::array<std::array<PairingFp2CodeGen, 2>, 5>
PairingFp6CodeGen::GetFrobeniusCoeffs() const {
  using FqX6 = zk_dtypes::bn254::FqX6;
  const auto &concreteCoeffs = FqX6::GetFrobeniusCoeffs();
  std::array<std::array<PairingFp2CodeGen, 2>, 5> result;
  for (size_t e = 0; e < 5; ++e) {
    for (size_t i = 0; i < 2; ++i) {
      result[e][i] = zk_dtypes::materializeBN254Fp2Const(concreteCoeffs[e][i]);
    }
  }
  return result;
}

// Fp12 Frobenius coefficients: delegates to BN254 FqX12::GetFrobeniusCoeffs()
// and materializes the concrete values as MLIR field constants.
// Dimensions: (D - 1) × (n - 1) = (12 - 1) × (2 - 1) = 11 × 1.
std::array<std::array<PairingFp6CodeGen, 1>, 11>
PairingFp12CodeGen::GetFrobeniusCoeffs() const {
  using FqX12 = zk_dtypes::bn254::FqX12;
  const auto &concreteCoeffs = FqX12::GetFrobeniusCoeffs();
  std::array<std::array<PairingFp6CodeGen, 1>, 11> result;
  for (size_t e = 0; e < 11; ++e) {
    result[e][0] = zk_dtypes::materializeBN254Fp6Const(concreteCoeffs[e][0]);
  }
  return result;
}

} // namespace mlir::prime_ir::elliptic_curve
