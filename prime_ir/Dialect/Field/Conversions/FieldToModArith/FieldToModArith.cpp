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

#include "prime_ir/Dialect/Field/Conversions/FieldToModArith/FieldToModArith.h"

#include <utility>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveOps.h"
#include "prime_ir/Dialect/Field/Conversions/FieldToModArith/ConversionUtils.h"
#include "prime_ir/Dialect/Field/Conversions/FieldToModArith/FieldCodeGen.h"
#include "prime_ir/Dialect/Field/IR/FieldDialect.h"
#include "prime_ir/Dialect/Field/IR/FieldOps.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithOps.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithTypes.h"
#include "prime_ir/Dialect/TensorExt/IR/TensorExtOps.h"
#include "prime_ir/Utils/BitSerialAlgorithm.h"
#include "prime_ir/Utils/BuilderContext.h"
#include "prime_ir/Utils/ConversionUtils.h"
#include "prime_ir/Utils/KnownModulus.h"
#include "prime_ir/Utils/LoweringMode.h"
#include "prime_ir/Utils/ShapedTypeConverter.h"

namespace mlir::prime_ir::field {

#define GEN_PASS_DEF_FIELDTOMODARITH
#include "prime_ir/Dialect/Field/Conversions/FieldToModArith/FieldToModArith.h.inc"

namespace {

// ---- AOT Runtime helpers ----

// Check if AOTRuntime should be used for this extension field operation.
// AOT is used for extension fields that are computationally expensive:
//   degree >= 4 (quartic+, always expensive regardless of prime size)
//   OR degree >= 2 AND prime > 64-bit (large-prime quadratic/cubic)
static bool shouldUseFieldAOTRuntime(Operation *op, Type fieldType,
                                     LoweringMode mode, bool inlineConstOps) {
  if (mode == LoweringMode::Inline)
    return false;
  if (inlineConstOps && hasConstantOperand(op))
    return false;
  auto efType = dyn_cast<ExtensionFieldType>(fieldType);
  if (!efType)
    return false;
  unsigned degree = efType.getDegreeOverPrime();
  unsigned primeBits = efType.getBasePrimeField().getStorageBitWidth();
  if (degree < 2)
    return false;
  bool expensive = degree >= 4 || primeBits > 64;
  if (!expensive)
    return false;
  return mode == LoweringMode::AOTRuntime || mode == LoweringMode::Auto;
}

// Build the zk_dtypes-style extension suffix by walking the tower.
// ef<2x ef<3x pf>> → "x3x2",  ef<2x pf> → "x2",  ef<4x pf> → "x4"
static std::string getExtensionTowerSuffix(ExtensionFieldType efType) {
  std::string suffix;
  Type cur = efType;
  while (auto ef = dyn_cast<ExtensionFieldType>(cur)) {
    suffix = "x" + std::to_string(ef.getDegree()) + suffix;
    cur = ef.getBaseField();
  }
  return suffix;
}

// Build AOT runtime function name for extension field operations.
// Pattern: "ef_<op>_<prime_alias><tower_suffix>[_mont]"
// Follows zk_dtypes tower naming: bn254_bfx2, babybearx4, mersenne31x2x2.
// Example: "ef_mul_bn254_bfx2_mont", "ef_inverse_mersenne31x2x2"
static std::optional<std::string> getFieldAOTFuncName(llvm::StringRef op,
                                                      Type fieldType) {
  auto efType = dyn_cast<ExtensionFieldType>(fieldType);
  if (!efType)
    return std::nullopt;

  // Tower extensions (base field is itself an extension, e.g.,
  // !field.ef<3x !field.ef<2x !PF, ...>, ...>) are not yet AOT-compiled.
  // Only direct extensions over prime fields (e.g., !field.ef<2x !PF, ...>).
  if (efType.isTower())
    return std::nullopt;

  auto baseAlias =
      getKnownModulusAlias(efType.getBasePrimeField().getModulus().getValue());
  if (!baseAlias)
    return std::nullopt;

  std::string towerSuffix = getExtensionTowerSuffix(efType);
  std::string montSuffix =
      isMontgomery(efType.getBasePrimeField()) ? "_mont" : "";
  return ("ef_" + op + "_" + *baseAlias + towerSuffix + montSuffix).str();
}

class FieldToModArithTypeConverter : public ShapedTypeConverter {
public:
  explicit FieldToModArithTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion([](PrimeFieldType type) -> Type {
      return convertPrimeFieldType(type);
    });
    addConversion([](ShapedType type) -> Type {
      if (auto primeFieldType =
              dyn_cast<PrimeFieldType>(type.getElementType())) {
        return convertShapedType(type, type.getShape(),
                                 convertPrimeFieldType(primeFieldType));
      } else if (auto vectorType =
                     dyn_cast<VectorType>(type.getElementType())) {
        if (auto primeFieldType =
                dyn_cast<PrimeFieldType>(vectorType.getElementType())) {
          return convertShapedType(
              type, type.getShape(),
              vectorType.cloneWith(vectorType.getShape(),
                                   convertPrimeFieldType(primeFieldType)));
        }
      }
      return type;
    });
  }
};

struct ConvertConstant : public OpConversionPattern<ConstantOp> {
  explicit ConvertConstant(MLIRContext *context)
      : OpConversionPattern<ConstantOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Skip binary field operations - they are handled by BinaryFieldToArith
    if (isa<BinaryFieldType>(getElementTypeOrSelf(op.getType()))) {
      return failure();
    }

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // Case 1: Prime field constants (scalar or tensor)
    if (auto pfType =
            dyn_cast<PrimeFieldType>(getElementTypeOrSelf(op.getType()))) {
      auto cval = b.create<mod_arith::ConstantOp>(
          typeConverter->convertType(op.getType()), op.getValueAttr());
      rewriter.replaceOp(op, cval);
      return success();
    }

    // Case 2: Tensor of extension field constants
    // Use efficient approach: create prime field tensor constant + bitcast
    if (auto shapedType = dyn_cast<ShapedType>(op.getType())) {
      auto efType = dyn_cast<ExtensionFieldType>(shapedType.getElementType());
      if (!efType) {
        op.emitOpError(
            "unsupported shaped type with non-extension field element type: ")
            << shapedType.getElementType();
        return failure();
      }

      // For tower extensions, use the underlying prime field
      auto modType = cast<mod_arith::ModArithType>(
          typeConverter->convertType(efType.getBasePrimeField()));
      unsigned degree = efType.getDegreeOverPrime();

      auto denseAttr = cast<DenseIntElementsAttr>(op.getValueAttr());
      auto allValues = denseAttr.getValues<APInt>();

      // Create a flattened prime field tensor constant
      // tensor<K x !EF> with degree N becomes tensor<K*N x !ModArith>
      // tensor<!EF> (rank-0) becomes tensor<N x !ModArith>
      SmallVector<int64_t> flatShape(shapedType.getShape());
      if (flatShape.empty()) {
        flatShape.push_back(degree);
      } else {
        flatShape.back() *= degree;
      }

      auto flatTensorType = RankedTensorType::get(flatShape, modType);
      SmallVector<APInt> flatCoeffs(allValues.begin(), allValues.end());
      auto flatDenseAttr = DenseIntElementsAttr::get(
          flatTensorType.clone(modType.getStorageType()), flatCoeffs);
      auto flatConstant =
          b.create<mod_arith::ConstantOp>(flatTensorType, flatDenseAttr);

      // Bitcast the flattened prime field tensor to extension field tensor
      auto efTensorType = RankedTensorType::get(shapedType.getShape(), efType);
      auto bitcast = b.create<BitcastOp>(efTensorType, flatConstant);
      rewriter.replaceOp(op, bitcast);
      return success();
    }

    // Case 3: Scalar extension field constant
    auto efType = dyn_cast<ExtensionFieldType>(op.getType());
    if (!efType) {
      op.emitOpError("unsupported output type");
      return failure();
    }

    // For tower extensions, use the underlying prime field
    auto modType = cast<mod_arith::ModArithType>(
        typeConverter->convertType(efType.getBasePrimeField()));

    auto denseAttr = cast<DenseIntElementsAttr>(op.getValueAttr());
    SmallVector<Value> primeCoeffs;
    for (auto coeff : denseAttr.getValues<APInt>()) {
      auto coeffAttr = IntegerAttr::get(modType.getStorageType(), coeff);
      primeCoeffs.push_back(
          b.create<mod_arith::ConstantOp>(modType, coeffAttr));
    }
    // Use fromPrimeCoeffs to properly handle tower extension fields
    rewriter.replaceOp(op, fromPrimeCoeffs(b, efType, primeCoeffs));
    return success();
  }
};

struct ConvertBitcast : public OpConversionPattern<BitcastOp> {
  explicit ConvertBitcast(MLIRContext *context)
      : OpConversionPattern<BitcastOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BitcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Type inputType = op.getInput().getType();
    Type outputType = op.getType();

    // Check if this is a tensor reinterpret bitcast (extension field <-> prime
    // field tensors)
    if (isTensorReinterpretBitcast(inputType, outputType)) {
      auto outputShaped = cast<ShapedType>(outputType);
      return convertTensorBitcast(op, adaptor, rewriter, outputShaped);
    }

    // Scalar bitcast: just convert to mod_arith.bitcast
    auto bitcast = b.create<mod_arith::BitcastOp>(
        typeConverter->convertType(op.getType()), adaptor.getInput());
    rewriter.replaceOp(op, bitcast);
    return success();
  }

private:
  LogicalResult convertTensorBitcast(BitcastOp op, OpAdaptor adaptor,
                                     ConversionPatternRewriter &rewriter,
                                     ShapedType outputTensorType) const {
    // Get the converted output type
    Type convertedOutputType = typeConverter->convertType(outputTensorType);
    if (!convertedOutputType) {
      return op.emitOpError("failed to convert output type");
    }

    // Keep the field.bitcast op but with updated types.
    // This preserves zero-copy semantics by deferring the actual memory
    // reinterpretation to the bufferization stage, where it will be
    // converted to a memref-level bitcast, and then to LLVM pointer casts.
    //
    // NOTE: We intentionally do NOT extract/reconstruct tensor elements here
    // as that would cause memory copies.
    rewriter.replaceOpWithNewOp<BitcastOp>(op, convertedOutputType,
                                           adaptor.getInput());
    return success();
  }
};

struct ConvertToMont : public OpConversionPattern<ToMontOp> {
  explicit ConvertToMont(MLIRContext *context)
      : OpConversionPattern<ToMontOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ToMontOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Type fieldType = getElementTypeOrSelf(op.getOutput());
    if (isa<PrimeFieldType>(fieldType)) {
      Type modArithType = typeConverter->convertType(op.getType());
      auto extracted =
          b.create<mod_arith::ToMontOp>(modArithType, adaptor.getInput());
      rewriter.replaceOp(op, extracted);
      return success();
    }
    if (auto efType = dyn_cast<ExtensionFieldType>(fieldType)) {
      // Use getBasePrimeField() to handle both direct and tower extensions
      auto basePrimeField = efType.getBasePrimeField();
      Type baseModArithType = typeConverter->convertType(basePrimeField);
      auto coeffs = toModArithCoeffs(b, adaptor.getInput());

      SmallVector<Value> montCoeffs;
      for (auto coeff : coeffs) {
        montCoeffs.push_back(
            b.create<mod_arith::ToMontOp>(baseModArithType, coeff));
      }
      rewriter.replaceOp(op, fromCoeffs(b, fieldType, montCoeffs));
      return success();
    }
    return failure();
  }
};

struct ConvertFromMont : public OpConversionPattern<FromMontOp> {
  explicit ConvertFromMont(MLIRContext *context)
      : OpConversionPattern<FromMontOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FromMontOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Type fieldType = getElementTypeOrSelf(op.getOutput());
    if (isa<PrimeFieldType>(fieldType)) {
      Type resultType = typeConverter->convertType(op.getType());
      auto extracted =
          b.create<mod_arith::FromMontOp>(resultType, adaptor.getInput());
      rewriter.replaceOp(op, extracted);
      return success();
    }
    if (auto efType = dyn_cast<ExtensionFieldType>(fieldType)) {
      // Use getBasePrimeField() to handle both direct and tower extensions
      auto basePrimeField = efType.getBasePrimeField();
      Type baseModArithType = typeConverter->convertType(basePrimeField);
      auto coeffs = toModArithCoeffs(b, adaptor.getInput());

      SmallVector<Value> stdCoeffs;
      for (auto coeff : coeffs) {
        stdCoeffs.push_back(
            b.create<mod_arith::FromMontOp>(baseModArithType, coeff));
      }
      rewriter.replaceOp(op, fromCoeffs(b, fieldType, stdCoeffs));
      return success();
    }
    return failure();
  }
};

template <typename OpT, typename Derived>
struct ConvertFieldOpBase : public OpConversionPattern<OpT> {
  using Base = ConvertFieldOpBase;
  using OpAdaptor = typename OpConversionPattern<OpT>::OpAdaptor;

  ConvertFieldOpBase(const TypeConverter &converter, MLIRContext *context,
                     AOTConfig aotConfig = {})
      : OpConversionPattern<OpT>(converter, context), aotConfig(aotConfig) {
    this->setHasBoundedRewriteRecursion(true);
  }

  LogicalResult
  matchAndRewrite(OpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isa<ShapedType>(op.getOutput().getType()) &&
        isa<ExtensionFieldType>(getElementTypeOrSelf(op.getOutput())))
      return failure();

    // AOT runtime path: emit func.call to pre-compiled function.
    Type fieldType = getElementTypeOrSelf(op.getOutput());
    if (shouldUseFieldAOTRuntime(op, fieldType, aotConfig.mode,
                                 aotConfig.inlineConstOps)) {
      auto funcName =
          static_cast<const Derived *>(this)->getAOTFuncName(fieldType);
      if (funcName) {
        rewriter.replaceOp(op, emitAOTFuncCall(op, *funcName,
                                               op.getOutput().getType(),
                                               op->getOperands(), rewriter));
        return success();
      }
    }

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    ScopedBuilderContext scopedBuilderContext(&b);

    rewriter.replaceOp(
        op,
        {static_cast<const Derived *>(this)->emitInlineCodeGen(op, adaptor)});
    return success();
  }

protected:
  AOTConfig aotConfig;
};

/// Lowers field.inverse for both scalar and tensor types.
///
/// Scalar: delegates to ConvertFieldOpBase (AOT or inline codegen).
/// Tensor EF: Montgomery's batch inversion trick.
///   Given a₀, a₁, ..., aₙ₋₁:
///     Forward:  prefix[0] = a₀; prefix[i] = prefix[i-1] * a[i]
///     Inverse:  inv = prefix[n-1]⁻¹
///     Backward: result[i] = inv * prefix[i-1]; inv = inv * a[i]  (i = n-1..1)
///               result[0] = inv
///   Uses O(3(n-1)) field multiplications + 1 field inversion instead of n
///   independent inversions.
struct ConvertInverse : ConvertFieldOpBase<InverseOp, ConvertInverse> {
  using Base::Base;
  std::optional<std::string> getAOTFuncName(Type fieldType) const {
    return getFieldAOTFuncName("inverse", fieldType);
  }
  Value emitInlineCodeGen(InverseOp op, OpAdaptor adaptor) const {
    Type fieldType = getElementTypeOrSelf(op.getOutput());
    return FieldCodeGen(fieldType, adaptor.getInput(), this->typeConverter)
        .inverse();
  }

  LogicalResult
  matchAndRewrite(InverseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tensorType = dyn_cast<RankedTensorType>(op.getOutput().getType());
    if (!tensorType)
      return Base::matchAndRewrite(op, adaptor, rewriter);

    return emitBatchInverse(op, rewriter, tensorType);
  }

private:
  LogicalResult emitBatchInverse(InverseOp op,
                                 ConversionPatternRewriter &rewriter,
                                 RankedTensorType tensorType) const {
    if (!tensorType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "dynamic shape not yet supported");
    int64_t n = tensorType.getNumElements();
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    Value input = op.getInput();
    Type elemType = tensorType.getElementType();

    if (n == 0) {
      rewriter.replaceOp(op, input);
      return success();
    }

    // Scalar: extract, invert, wrap back.
    if (tensorType.getRank() == 0) {
      Value elem = b.create<tensor::ExtractOp>(input, ValueRange{});
      Value inv = b.create<InverseOp>(elem);
      Value result = b.create<tensor::FromElementsOp>(tensorType, inv);
      rewriter.replaceOp(op, result);
      return success();
    }

    // Collapse multi-dimensional tensors to 1-D for linear indexing.
    auto flatType = RankedTensorType::get({n}, elemType);
    bool needsReshape = tensorType.getRank() != 1;
    if (needsReshape) {
      SmallVector<ReassociationIndices> reassoc = {
          llvm::to_vector(llvm::seq<int64_t>(0, tensorType.getRank()))};
      input = b.create<tensor::CollapseShapeOp>(flatType, input, reassoc);
    }

    Value c0 = b.create<arith::ConstantIndexOp>(0);
    Value result;

    if (n == 1) {
      Value elem = b.create<tensor::ExtractOp>(input, ValueRange{c0});
      Value inv = b.create<InverseOp>(elem);
      result = b.create<tensor::EmptyOp>(flatType.getShape(), elemType);
      result = b.create<tensor::InsertOp>(inv, result, ValueRange{c0});
    } else {
      // n >= 2: Montgomery's batch inversion.
      Value c1 = b.create<arith::ConstantIndexOp>(1);
      Value cN = b.create<arith::ConstantIndexOp>(n);
      Value cNm1 = b.create<arith::ConstantIndexOp>(n - 1);

      // Forward pass: build prefix products.
      // prefix[0] = a[0]; prefix[i] = prefix[i-1] * a[i]
      Value a0 = b.create<tensor::ExtractOp>(input, ValueRange{c0});
      Value prefixInit =
          b.create<tensor::EmptyOp>(flatType.getShape(), elemType);
      prefixInit = b.create<tensor::InsertOp>(a0, prefixInit, ValueRange{c0});

      auto fwdLoop = b.create<scf::ForOp>(
          c1, cN, c1, ValueRange{prefixInit, a0},
          [&](OpBuilder &nb, Location loc, Value iv, ValueRange iterArgs) {
            ImplicitLocOpBuilder lb(loc, nb);
            Value prefixTensor = iterArgs[0];
            Value runningProduct = iterArgs[1];
            Value elem = lb.create<tensor::ExtractOp>(input, ValueRange{iv});
            Value product = lb.create<MulOp>(runningProduct, elem);
            Value updated = lb.create<tensor::InsertOp>(product, prefixTensor,
                                                        ValueRange{iv});
            lb.create<scf::YieldOp>(ValueRange{updated, product});
          });
      Value prefixTensor = fwdLoop.getResult(0);
      Value totalProduct = fwdLoop.getResult(1);

      // Single scalar inverse.
      Value inv = b.create<InverseOp>(totalProduct);

      // Backward pass: recover individual inverses.
      // for i in [n-1, n-2, ..., 1]:
      //   result[i] = inv * prefix[i-1]; inv = inv * a[i]
      Value resultInit =
          b.create<tensor::EmptyOp>(flatType.getShape(), elemType);

      auto bwdLoop = b.create<scf::ForOp>(
          c0, cNm1, c1, ValueRange{resultInit, inv},
          [&](OpBuilder &nb, Location loc, Value iv, ValueRange iterArgs) {
            ImplicitLocOpBuilder lb(loc, nb);
            Value resultTensor = iterArgs[0];
            Value curInv = iterArgs[1];
            // Map forward index iv to reverse index: currIdx = n-1-iv
            Value currIdx = lb.create<arith::SubIOp>(cNm1, iv);
            Value prevIdx = lb.create<arith::SubIOp>(currIdx, c1);
            Value prevPrefix =
                lb.create<tensor::ExtractOp>(prefixTensor, ValueRange{prevIdx});
            Value elemInv = lb.create<MulOp>(curInv, prevPrefix);
            Value updated = lb.create<tensor::InsertOp>(elemInv, resultTensor,
                                                        ValueRange{currIdx});
            Value origElem =
                lb.create<tensor::ExtractOp>(input, ValueRange{currIdx});
            Value newInv = lb.create<MulOp>(curInv, origElem);
            lb.create<scf::YieldOp>(ValueRange{updated, newInv});
          });

      // result[0] = final inv (= a₀⁻¹).
      result = bwdLoop.getResult(0);
      inv = bwdLoop.getResult(1);
      result = b.create<tensor::InsertOp>(inv, result, ValueRange{c0});
    }

    if (needsReshape) {
      SmallVector<ReassociationIndices> reassoc = {
          llvm::to_vector(llvm::seq<int64_t>(0, tensorType.getRank()))};
      result = b.create<tensor::ExpandShapeOp>(tensorType, result, reassoc);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ConvertNegate : ConvertFieldOpBase<NegateOp, ConvertNegate> {
  using Base::Base;
  std::optional<std::string> getAOTFuncName(Type fieldType) const {
    return std::nullopt; // cheap op, always inline
  }
  Value emitInlineCodeGen(NegateOp op, OpAdaptor adaptor) const {
    Type fieldType = getElementTypeOrSelf(op.getOutput());
    return -FieldCodeGen(fieldType, adaptor.getInput(), this->typeConverter);
  }
};

struct ConvertAdd : ConvertFieldOpBase<AddOp, ConvertAdd> {
  using Base::Base;
  std::optional<std::string> getAOTFuncName(Type fieldType) const {
    return std::nullopt; // cheap op, always inline
  }
  Value emitInlineCodeGen(AddOp op, OpAdaptor adaptor) const {
    Type lhsType = getElementTypeOrSelf(op.getLhs().getType());
    Type rhsType = getElementTypeOrSelf(op.getRhs().getType());
    FieldCodeGen lhs(lhsType, adaptor.getLhs(), this->typeConverter);
    FieldCodeGen rhs(rhsType, adaptor.getRhs(), this->typeConverter);
    return lhs + rhs;
  }
};

struct ConvertDouble : ConvertFieldOpBase<DoubleOp, ConvertDouble> {
  using Base::Base;
  std::optional<std::string> getAOTFuncName(Type) const { return std::nullopt; }
  Value emitInlineCodeGen(DoubleOp op, OpAdaptor adaptor) const {
    Type fieldType = getElementTypeOrSelf(op.getOutput());
    return FieldCodeGen(fieldType, adaptor.getInput(), this->typeConverter)
        .dbl();
  }
};

struct ConvertSub : ConvertFieldOpBase<SubOp, ConvertSub> {
  using Base::Base;
  std::optional<std::string> getAOTFuncName(Type fieldType) const {
    return std::nullopt; // cheap op, always inline
  }
  Value emitInlineCodeGen(SubOp op, OpAdaptor adaptor) const {
    Type lhsType = getElementTypeOrSelf(op.getLhs().getType());
    Type rhsType = getElementTypeOrSelf(op.getRhs().getType());
    FieldCodeGen lhs(lhsType, adaptor.getLhs(), this->typeConverter);
    FieldCodeGen rhs(rhsType, adaptor.getRhs(), this->typeConverter);
    return lhs - rhs;
  }
};

struct ConvertMul : ConvertFieldOpBase<MulOp, ConvertMul> {
  using Base::Base;
  std::optional<std::string> getAOTFuncName(Type fieldType) const {
    return getFieldAOTFuncName("mul", fieldType);
  }
  Value emitInlineCodeGen(MulOp op, OpAdaptor adaptor) const {
    Type lhsType = getElementTypeOrSelf(op.getLhs().getType());
    Type rhsType = getElementTypeOrSelf(op.getRhs().getType());
    FieldCodeGen lhs(lhsType, adaptor.getLhs(), this->typeConverter);
    FieldCodeGen rhs(rhsType, adaptor.getRhs(), this->typeConverter);
    return lhs * rhs;
  }
};

struct ConvertSquare : ConvertFieldOpBase<SquareOp, ConvertSquare> {
  using Base::Base;
  std::optional<std::string> getAOTFuncName(Type fieldType) const {
    return getFieldAOTFuncName("square", fieldType);
  }
  Value emitInlineCodeGen(SquareOp op, OpAdaptor adaptor) const {
    Type fieldType = getElementTypeOrSelf(op.getOutput());
    return FieldCodeGen(fieldType, adaptor.getInput(), this->typeConverter)
        .square();
  }
};

struct ConvertPowUI : public OpConversionPattern<PowUIOp> {
  explicit ConvertPowUI(MLIRContext *context)
      : OpConversionPattern<PowUIOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(PowUIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto base = op.getBase();
    auto exp = op.getExp();
    auto fieldType = getElementTypeOrSelf(base);

    APInt modulus;
    Value init;
    if (auto pfType = dyn_cast<PrimeFieldType>(fieldType)) {
      modulus = pfType.getModulus().getValue();
      Type intType = pfType.getStorageType();
      Type stdType = getStandardFormType(pfType);
      Type montType = getMontgomeryFormType(pfType);
      if (auto vecType = dyn_cast<VectorType>(base.getType())) {
        intType = vecType.cloneWith(vecType.getShape(), intType);
        stdType = vecType.cloneWith(vecType.getShape(), stdType);
        montType = vecType.cloneWith(vecType.getShape(), montType);
      }
      init = createScalarOrSplatConstant(b, b.getLoc(), intType, 1);
      init = b.create<BitcastOp>(stdType, init);
      if (pfType.isMontgomery()) {
        init = b.create<ToMontOp>(montType, init);
      }
    } else if (auto efType = dyn_cast<ExtensionFieldType>(fieldType)) {
      modulus = efType.getBasePrimeField().getModulus().getValue();
      init = field::createFieldOne(efType, b);
    } else {
      op.emitOpError("unsupported output type");
      return failure();
    }

    unsigned expBitWidth = cast<IntegerType>(exp.getType()).getWidth();
    unsigned modBitWidth = modulus.getBitWidth();
    if (modBitWidth > expBitWidth) {
      exp = b.create<arith::ExtUIOp>(
          IntegerType::get(exp.getContext(), modBitWidth), exp);
    } else {
      modulus = modulus.zext(expBitWidth);
      modBitWidth = expBitWidth;
    }
    IntegerType intType = cast<IntegerType>(exp.getType());

    auto emitBitSerialLoop = [&](Value exp) {
      return generateBitSerialLoop(
          b, exp, base, init,
          [](ImplicitLocOpBuilder &b, Value v) {
            return b.create<SquareOp>(v);
          },
          [](ImplicitLocOpBuilder &b, Value acc, Value v) {
            return b.create<MulOp>(acc, v);
          });
    };

    // Reduce exponent using Fermat's little theorem:
    // x^(p-1) ≡ 1 (prime field), x^(pᵈ-1) ≡ 1 (extension field)
    APInt order;
    if (auto efType = dyn_cast<ExtensionFieldType>(fieldType)) {
      unsigned degreeOverPrime = efType.getDegreeOverPrime();
      modulus = modulus.zext(modBitWidth * degreeOverPrime);
      order = modulus;
      for (unsigned i = 1; i < degreeOverPrime; ++i)
        order = order * modulus;
      order = order - 1;
    } else {
      order = modulus - 1;
    }

    if (auto expConstOp = exp.getDefiningOp<arith::ConstantOp>()) {
      APInt cExp = cast<IntegerAttr>(expConstOp.getValue()).getValue();
      cExp = cExp.zext(order.getBitWidth());
      cExp = cExp.urem(order);
      intType = IntegerType::get(exp.getContext(), order.getBitWidth());
      exp = b.create<arith::ConstantIntOp>(intType, cExp);
    } else {
      if (order.getBitWidth() > intType.getWidth()) {
        intType = IntegerType::get(exp.getContext(), order.getBitWidth());
        exp = b.create<arith::ExtUIOp>(intType, exp);
      }
      exp = b.create<arith::RemUIOp>(
          exp, b.create<arith::ConstantIntOp>(intType, order));
    }

    rewriter.replaceOp(op, emitBitSerialLoop(exp));
    return success();
  }
};

struct ConvertCmp : public OpConversionPattern<CmpOp> {
  explicit ConvertCmp(MLIRContext *context)
      : OpConversionPattern<CmpOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CmpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Type fieldType = getElementTypeOrSelf(op.getLhs());
    arith::CmpIPredicate predicate = op.getPredicate();
    if (isa<PrimeFieldType>(fieldType)) {
      rewriter.replaceOp(op, compareOnStdDomain(b, fieldType, predicate,
                                                adaptor.getLhs(),
                                                adaptor.getRhs()));
      return success();
    } else if (auto efType = dyn_cast<ExtensionFieldType>(fieldType)) {
      // Recursively flatten tower extensions to prime-level coefficients.
      auto lhsPrimeCoeffs = flattenToPrimeCoeffs(b, adaptor.getLhs());
      auto rhsPrimeCoeffs = flattenToPrimeCoeffs(b, adaptor.getRhs());
      unsigned n = efType.getDegreeOverPrime();
      assert(lhsPrimeCoeffs.size() == n && rhsPrimeCoeffs.size() == n);
      PrimeFieldType basePF = efType.getBasePrimeField();
      SmallVector<Value> cmpResults;
      for (unsigned i = 0; i < n; ++i) {
        cmpResults.push_back(compareOnStdDomain(
            b, basePF, predicate, lhsPrimeCoeffs[i], rhsPrimeCoeffs[i]));
      }
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
      Value result = cmpResults.front();
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
      if (predicate == arith::CmpIPredicate::eq) {
        for (unsigned i = 1; i < n; ++i) {
          result = b.create<arith::AndIOp>(result, cmpResults[i]);
        }
      } else if (predicate == arith::CmpIPredicate::ne) {
        for (unsigned i = 1; i < n; ++i) {
          result = b.create<arith::OrIOp>(result, cmpResults[i]);
        }
      } else {
        llvm_unreachable(
            "Unsupported comparison predicate for extension field type");
      }
      rewriter.replaceOp(op, result);
      return success();
    }
    return failure();
  }

  // Recursively flatten a (possibly tower) extension field value to all its
  // prime-level coefficients (mod_arith values).
  SmallVector<Value> flattenToPrimeCoeffs(ImplicitLocOpBuilder &b,
                                          Value val) const {
    if (isa<mod_arith::ModArithType>(val.getType())) {
      return {val};
    }
    auto coeffs = toModArithCoeffs(b, val);
    SmallVector<Value> result;
    for (Value c : coeffs) {
      auto sub = flattenToPrimeCoeffs(b, c);
      result.append(sub.begin(), sub.end());
    }
    return result;
  }

  Value compareOnStdDomain(ImplicitLocOpBuilder &b, Type fieldType,
                           arith::CmpIPredicate predicate, Value lhs,
                           Value rhs) const {
    if (isMontgomery(fieldType)) {
      auto modArithLhsType = cast<mod_arith::ModArithType>(lhs.getType());
      auto stdModArithType = mod_arith::ModArithType::get(
          modArithLhsType.getContext(), modArithLhsType.getModulus(),
          /*isMontgomery=*/false);

      Value standardLhs = b.create<mod_arith::FromMontOp>(stdModArithType, lhs);
      Value standardRhs = b.create<mod_arith::FromMontOp>(stdModArithType, rhs);

      return b.create<mod_arith::CmpOp>(predicate, standardLhs, standardRhs);
    } else {
      return b.create<mod_arith::CmpOp>(predicate, lhs, rhs);
    }
  }
};

namespace rewrites {
// In an inner namespace to avoid conflicts with canonicalization patterns
#include "prime_ir/Dialect/Field/Conversions/FieldToModArith/FieldToModArith.cpp.inc"
} // namespace rewrites

// Check if a type contains BinaryFieldType
bool containsBinaryFieldType(Type type) {
  Type elemType = getElementTypeOrSelf(type);
  return isa<BinaryFieldType>(elemType);
}

// Check if any type in the operation contains BinaryFieldType
bool operationContainsBinaryFieldType(Operation *op) {
  // Check result types
  for (Type type : op->getResultTypes()) {
    if (containsBinaryFieldType(type))
      return true;
  }
  // Check operand types
  for (Value operand : op->getOperands()) {
    if (containsBinaryFieldType(operand.getType()))
      return true;
  }
  return false;
}

} // namespace

struct FieldToModArith : impl::FieldToModArithBase<FieldToModArith> {
  using FieldToModArithBase::FieldToModArithBase;

  void runOnOperation() override;
};

void FieldToModArith::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();
  FieldToModArithTypeConverter typeConverter(context);

  LoweringMode mode = mlir::prime_ir::parseLoweringMode(loweringMode);

  ConversionTarget target(*context);

  // Mark field operations as dynamically legal if they contain BinaryFieldType
  // (those will be handled by BinaryFieldToArith pass instead)
  target.addDynamicallyLegalOp<ConstantOp, CmpOp, PowUIOp>(
      [](Operation *op) { return operationContainsBinaryFieldType(op); });

  // ConvertFieldOpBase patterns cannot inline-codegen shaped extension field
  // types. ElementwiseMappable ops (add, sub, ...) are scalarized by
  // convert-elementwise-to-linalg. Mark them as legal here so
  // field-to-mod-arith passes through these ops without failing.
  // Note: InverseOp is NOT listed — ConvertInverse handles all tensor inverses
  // via Montgomery's batch inversion trick.
  target
      .addDynamicallyLegalOp<AddOp, SubOp, MulOp, NegateOp, DoubleOp, SquareOp>(
          [](Operation *op) {
            if (operationContainsBinaryFieldType(op))
              return true;
            return llvm::any_of(op->getResultTypes(), [](Type t) {
              return isa<ShapedType>(t) &&
                     isa<ExtensionFieldType>(getElementTypeOrSelf(t));
            });
          });

  // Mark remaining field dialect ops as illegal (prime/extension field ops)
  target.addIllegalDialect<FieldDialect>();
  target.addLegalDialect<mod_arith::ModArithDialect>();
  target.addLegalDialect<func::FuncDialect>();
  target.addLegalOp<func::FuncOp, func::CallOp, func::ReturnOp>();

  RewritePatternSet patterns(context);
  rewrites::populateWithGenerated(patterns);

  patterns.add<
      // clang-format off
      ConvertAdd,
      ConvertDouble,
      ConvertMul,
      ConvertInverse,
      ConvertNegate,
      ConvertSquare,
      ConvertSub
      // clang-format on
      >(typeConverter, context, AOTConfig{mode, inlineConstantOps});

  patterns.add<
      // clang-format off
      ConvertBitcast,
      ConvertConstant,
      ConvertCmp,
      ConvertFromMont,
      ConvertPowUI,
      ConvertToMont
      // clang-format on
      >(typeConverter, context);

  // Catch-all: converts any op whose operands/results carry field types.
  // Op-specific patterns above have root-op-name priority in the applicator.
  patterns.add<ConvertAny<void>>(typeConverter, context);

  addStructuralConversionPatterns(typeConverter, patterns, target);

  // Any op not explicitly registered is dynamically legal iff its types are
  // already converted.  This covers ops from downstream or unrelated dialects.
  target.markUnknownOpDynamicallyLegal(
      [&](Operation *op) { return typeConverter.isLegal(op); });

  // Field dialect ops that stay as field ops after type conversion (they keep
  // the same op name but with mod_arith element types) need an explicit
  // override because addIllegalDialect<FieldDialect> would otherwise reject
  // them even after successful conversion.
  target.addDynamicallyLegalOp<BitcastOp, ExtFromCoeffsOp, ExtToCoeffsOp>(
      [&](auto op) { return typeConverter.isLegal(op); });

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

} // namespace mlir::prime_ir::field
