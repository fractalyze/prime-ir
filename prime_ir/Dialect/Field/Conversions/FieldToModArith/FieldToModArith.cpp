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
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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
#include "prime_ir/Dialect/Field/Conversions/FieldToModArith/IntrinsicFunctionGenerator.h"
#include "prime_ir/Dialect/Field/IR/FieldDialect.h"
#include "prime_ir/Dialect/Field/IR/FieldOps.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithOps.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithTypes.h"
#include "prime_ir/Dialect/TensorExt/IR/TensorExtOps.h"
#include "prime_ir/Utils/BitSerialAlgorithm.h"
#include "prime_ir/Utils/BuilderContext.h"
#include "prime_ir/Utils/ConversionUtils.h"
#include "prime_ir/Utils/ShapedTypeConverter.h"

namespace mlir::prime_ir::field {

#define GEN_PASS_DEF_FIELDTOMODARITH
#include "prime_ir/Dialect/Field/Conversions/FieldToModArith/FieldToModArith.h.inc"

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

      auto modType = cast<mod_arith::ModArithType>(
          typeConverter->convertType(efType.getBaseField()));
      unsigned degree = efType.getDegreeOverPrime();

      auto denseAttr = cast<DenseIntElementsAttr>(op.getValueAttr());
      auto allValues = denseAttr.getValues<APInt>();

      // Create a flattened prime field tensor constant
      // tensor<K x !EF> with degree N becomes tensor<K*N x !ModArith>
      SmallVector<int64_t> flatShape(shapedType.getShape());
      flatShape.back() *= degree;

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

    auto modType = cast<mod_arith::ModArithType>(
        typeConverter->convertType(efType.getBaseField()));

    auto denseAttr = cast<DenseIntElementsAttr>(op.getValueAttr());
    SmallVector<Value> coeffs;
    for (auto coeff : denseAttr.getValues<APInt>()) {
      auto coeffAttr = IntegerAttr::get(modType.getStorageType(), coeff);
      coeffs.push_back(b.create<mod_arith::ConstantOp>(modType, coeffAttr));
    }
    rewriter.replaceOp(op, fromCoeffs(b, op.getType(), coeffs));
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
      auto baseField = cast<PrimeFieldType>(efType.getBaseField());
      Type baseModArithType = typeConverter->convertType(baseField);
      auto coeffs = toCoeffs(b, adaptor.getInput());

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
      auto baseField = cast<PrimeFieldType>(efType.getBaseField());
      Type baseModArithType = typeConverter->convertType(baseField);
      auto coeffs = toCoeffs(b, adaptor.getInput());

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

struct ConvertInverse : public OpConversionPattern<InverseOp> {
  ConvertInverse(const TypeConverter &converter, MLIRContext *context,
                 IntrinsicFunctionGenerator *generator = nullptr,
                 LoweringMode mode = LoweringMode::Inline)
      : OpConversionPattern<InverseOp>(converter, context),
        generator(generator), mode(mode) {}

  LogicalResult
  matchAndRewrite(InverseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    ScopedBuilderContext scopedBuilderContext(&b);

    Type fieldType = getElementTypeOrSelf(op.getOutput());

    // Use intrinsic for quartic extension fields if enabled
    if (generator &&
        IntrinsicFunctionGenerator::shouldUseIntrinsic(fieldType, mode)) {
      auto efType = cast<ExtensionFieldType>(fieldType);
      Value result = generator->emitQuarticInverseCall(
          rewriter, op.getLoc(), efType, adaptor.getInput());
      rewriter.replaceOp(op, result);
      return success();
    }

    FieldCodeGen codeGen(fieldType, adaptor.getInput(), typeConverter);
    rewriter.replaceOp(op, {codeGen.inverse()});
    return success();
  }

private:
  IntrinsicFunctionGenerator *generator;
  LoweringMode mode;
};

struct ConvertNegate : public OpConversionPattern<NegateOp> {
  explicit ConvertNegate(MLIRContext *context)
      : OpConversionPattern<NegateOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(NegateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    ScopedBuilderContext scopedBuilderContext(&b);

    Type fieldType = getElementTypeOrSelf(op.getOutput());
    FieldCodeGen codeGen(fieldType, adaptor.getInput(), typeConverter);
    rewriter.replaceOp(op, {-codeGen});
    return success();
  }
};

struct ConvertAdd : public OpConversionPattern<AddOp> {
  explicit ConvertAdd(MLIRContext *context)
      : OpConversionPattern<AddOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    ScopedBuilderContext scopedBuilderContext(&b);

    Type fieldType = getElementTypeOrSelf(op.getOutput());
    FieldCodeGen lhsCodeGen(fieldType, adaptor.getLhs(), typeConverter);
    FieldCodeGen rhsCodeGen(fieldType, adaptor.getRhs(), typeConverter);
    rewriter.replaceOp(op, {lhsCodeGen + rhsCodeGen});
    return success();
  }
};

struct ConvertDouble : public OpConversionPattern<DoubleOp> {
  explicit ConvertDouble(MLIRContext *context)
      : OpConversionPattern<DoubleOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DoubleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    ScopedBuilderContext scopedBuilderContext(&b);

    Type fieldType = getElementTypeOrSelf(op.getOutput());
    FieldCodeGen codeGen(fieldType, adaptor.getInput(), typeConverter);
    rewriter.replaceOp(op, {codeGen.dbl()});
    return success();
  }
};

struct ConvertSub : public OpConversionPattern<SubOp> {
  explicit ConvertSub(MLIRContext *context)
      : OpConversionPattern<SubOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SubOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    ScopedBuilderContext scopedBuilderContext(&b);

    Type fieldType = getElementTypeOrSelf(op.getOutput());
    FieldCodeGen lhsCodeGen(fieldType, adaptor.getLhs(), typeConverter);
    FieldCodeGen rhsCodeGen(fieldType, adaptor.getRhs(), typeConverter);
    rewriter.replaceOp(op, {lhsCodeGen - rhsCodeGen});
    return success();
  }
};

struct ConvertMul : public OpConversionPattern<MulOp> {
  ConvertMul(const TypeConverter &converter, MLIRContext *context,
             IntrinsicFunctionGenerator *generator = nullptr,
             LoweringMode mode = LoweringMode::Inline)
      : OpConversionPattern<MulOp>(converter, context), generator(generator),
        mode(mode) {}

  LogicalResult
  matchAndRewrite(MulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    ScopedBuilderContext scopedBuilderContext(&b);

    Type fieldType = getElementTypeOrSelf(op.getOutput());

    // Use intrinsic for quartic extension fields if enabled
    if (generator &&
        IntrinsicFunctionGenerator::shouldUseIntrinsic(fieldType, mode)) {
      auto efType = cast<ExtensionFieldType>(fieldType);
      Value result = generator->emitQuarticMulCall(
          rewriter, op.getLoc(), efType, adaptor.getLhs(), adaptor.getRhs());
      rewriter.replaceOp(op, result);
      return success();
    }

    FieldCodeGen lhsCodeGen(fieldType, adaptor.getLhs(), typeConverter);
    FieldCodeGen rhsCodeGen(fieldType, adaptor.getRhs(), typeConverter);
    rewriter.replaceOp(op, {lhsCodeGen * rhsCodeGen});
    return success();
  }

private:
  IntrinsicFunctionGenerator *generator;
  LoweringMode mode;
};

struct ConvertSquare : public OpConversionPattern<SquareOp> {
  ConvertSquare(const TypeConverter &converter, MLIRContext *context,
                IntrinsicFunctionGenerator *generator = nullptr,
                LoweringMode mode = LoweringMode::Inline)
      : OpConversionPattern<SquareOp>(converter, context), generator(generator),
        mode(mode) {}

  LogicalResult
  matchAndRewrite(SquareOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    ScopedBuilderContext scopedBuilderContext(&b);

    Type fieldType = getElementTypeOrSelf(op.getOutput());

    // Use intrinsic for quartic extension fields if enabled
    if (generator &&
        IntrinsicFunctionGenerator::shouldUseIntrinsic(fieldType, mode)) {
      auto efType = cast<ExtensionFieldType>(fieldType);
      Value result = generator->emitQuarticSquareCall(
          rewriter, op.getLoc(), efType, adaptor.getInput());
      rewriter.replaceOp(op, result);
      return success();
    }

    FieldCodeGen codeGen(fieldType, adaptor.getInput(), typeConverter);
    rewriter.replaceOp(op, {codeGen.square()});
    return success();
  }

private:
  IntrinsicFunctionGenerator *generator;
  LoweringMode mode;
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
      // TODO(chokobole): Support towers of extension field.
      modulus =
          cast<PrimeFieldType>(efType.getBaseField()).getModulus().getValue();
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

    // If `exp` is a constant, unroll the while loop.
    if (auto expConstOp = exp.getDefiningOp<arith::ConstantOp>()) {
      APInt cExp = cast<IntegerAttr>(expConstOp.getValue()).getValue();
      cExp = cExp.urem(modulus - 1);
      APInt cZero = APInt::getZero(cExp.getBitWidth());
      APInt cOne = cZero + 1;

      // Depending on the type, we need to perform the loop.
      Value result = init;
      Value factor = base;
      APInt currExp = cExp;
      SmallVector<Value> factors;
      while (!currExp.isZero()) {
        if ((currExp & cOne).getBoolValue()) {
          result = b.create<MulOp>(result, factor);
        }
        factor = b.create<SquareOp>(factor);
        currExp = currExp.lshr(1);
      }
      rewriter.replaceOp(op, result);
      return success();
    }

    // For prime field, x^(p-1) ≡ 1 mod p, so x^n ≡ x^(n mod (p-1)) mod p
    // For extension field of degree d, x^(pᵈ-1) ≡ 1 mod pᵈ, so
    // x^n ≡ x^(n mod (pᵈ-1)) mod pᵈ
    if (isa<PrimeFieldType>(fieldType)) {
      exp = b.create<arith::RemUIOp>(
          exp, b.create<arith::ConstantIntOp>(intType, modulus - 1));
    } else if (auto efType = dyn_cast<ExtensionFieldType>(fieldType)) {
      unsigned degree = efType.getDegree();
      modulus = modulus.zext(modBitWidth * degree);
      APInt order = modulus;
      for (unsigned i = 1; i < degree; ++i) {
        order = order * modulus;
      }
      order = order - 1;
      exp = b.create<arith::ExtUIOp>(
          IntegerType::get(exp.getContext(), order.getBitWidth()), exp);
      intType = IntegerType::get(exp.getContext(), order.getBitWidth());
      exp = b.create<arith::RemUIOp>(
          exp, b.create<arith::ConstantIntOp>(intType, order));
    }

    Value result = generateBitSerialLoop(
        b, exp, base, init,
        [](ImplicitLocOpBuilder &b, Value v) { return b.create<SquareOp>(v); },
        [](ImplicitLocOpBuilder &b, Value acc, Value v) {
          return b.create<MulOp>(acc, v);
        });
    rewriter.replaceOp(op, result);
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
      // For extension fields, we compare each coefficient separately.
      auto lhsCoeffs = toCoeffs(b, adaptor.getLhs());
      auto rhsCoeffs = toCoeffs(b, adaptor.getRhs());
      unsigned n = efType.getDegreeOverPrime();
      SmallVector<Value> cmpResults;
      for (unsigned i = 0; i < n; ++i) {
        cmpResults.push_back(compareOnStdDomain(b, fieldType, predicate,
                                                lhsCoeffs[i], rhsCoeffs[i]));
      }
      Value result = cmpResults[0];
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

struct FieldToModArith : impl::FieldToModArithBase<FieldToModArith> {
  using FieldToModArithBase::FieldToModArithBase;

  void runOnOperation() override;
};

void FieldToModArith::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();
  FieldToModArithTypeConverter typeConverter(context);

  // Parse the lowering mode option
  LoweringMode mode = mlir::prime_ir::parseLoweringMode(loweringMode);

  // Create intrinsic function generator if needed
  std::unique_ptr<IntrinsicFunctionGenerator> intrinsicGenerator;
  if (mode != LoweringMode::Inline) {
    intrinsicGenerator =
        std::make_unique<IntrinsicFunctionGenerator>(module, &typeConverter);
  }

  ConversionTarget target(*context);
  target.addIllegalDialect<FieldDialect>();
  target.addLegalDialect<mod_arith::ModArithDialect>();
  target.addLegalDialect<func::FuncDialect>();
  target.addLegalOp<func::FuncOp, func::CallOp, func::ReturnOp>();

  RewritePatternSet patterns(context);
  rewrites::populateWithGenerated(patterns);

  // Register patterns that support intrinsic mode
  patterns.add<ConvertInverse, ConvertMul, ConvertSquare>(
      typeConverter, context, intrinsicGenerator.get(), mode);

  patterns.add<
      // clang-format off
      ConvertAdd,
      ConvertBitcast,
      ConvertConstant,
      ConvertCmp,
      ConvertDouble,
      ConvertFromMont,
      ConvertNegate,
      ConvertPowUI,
      ConvertSub,
      ConvertToMont,
      ConvertAny<ExtFromCoeffsOp>,
      ConvertAny<ExtToCoeffsOp>,
      ConvertAny<affine::AffineForOp>,
      ConvertAny<affine::AffineLoadOp>,
      ConvertAny<affine::AffineParallelOp>,
      ConvertAny<affine::AffineStoreOp>,
      ConvertAny<affine::AffineYieldOp>,
      ConvertAny<arith::SelectOp>,
      ConvertAny<bufferization::AllocTensorOp>,
      ConvertAny<bufferization::MaterializeInDestinationOp>,
      ConvertAny<bufferization::ToBufferOp>,
      ConvertAny<bufferization::ToTensorOp>,
      ConvertAny<elliptic_curve::FromCoordsOp>,
      ConvertAny<elliptic_curve::ToCoordsOp>,
      ConvertAny<linalg::BroadcastOp>,
      ConvertAny<linalg::GenericOp>,
      ConvertAny<linalg::MapOp>,
      ConvertAny<linalg::ReduceOp>,
      ConvertAny<linalg::TransposeOp>,
      ConvertAny<linalg::YieldOp>,
      ConvertAny<memref::AllocOp>,
      ConvertAny<memref::AllocaOp>,
      ConvertAny<memref::CastOp>,
      ConvertAny<memref::CopyOp>,
      ConvertAny<memref::DimOp>,
      ConvertAny<memref::LoadOp>,
      ConvertAny<memref::StoreOp>,
      ConvertAny<memref::SubViewOp>,
      ConvertAny<memref::ViewOp>,
      ConvertAny<sparse_tensor::AssembleOp>,
      ConvertAny<tensor::CastOp>,
      ConvertAny<tensor::ConcatOp>,
      ConvertAny<tensor::DimOp>,
      ConvertAny<tensor::EmptyOp>,
      ConvertAny<tensor::ExtractOp>,
      ConvertAny<tensor::ExtractSliceOp>,
      ConvertAny<tensor::FromElementsOp>,
      ConvertAny<tensor::InsertOp>,
      ConvertAny<tensor::InsertSliceOp>,
      ConvertAny<tensor::PadOp>,
      ConvertAny<tensor::ReshapeOp>,
      ConvertAny<tensor::YieldOp>,
      ConvertAny<tensor_ext::BitReverseOp>,
      ConvertAny<ub::PoisonOp>,
      ConvertAny<vector::BroadcastOp>,
      ConvertAny<vector::SplatOp>,
      ConvertAny<vector::TransferReadOp>,
      ConvertAny<vector::TransferWriteOp>
      // clang-format on
      >(typeConverter, context);

  addStructuralConversionPatterns(typeConverter, patterns, target);

  target.addDynamicallyLegalOp<
      // clang-format off
      BitcastOp,
      ExtFromCoeffsOp,
      ExtToCoeffsOp,
      affine::AffineForOp,
      affine::AffineLoadOp,
      affine::AffineParallelOp,
      affine::AffineStoreOp,
      affine::AffineYieldOp,
      arith::SelectOp,
      bufferization::AllocTensorOp,
      bufferization::MaterializeInDestinationOp,
      bufferization::ToBufferOp,
      bufferization::ToTensorOp,
      elliptic_curve::FromCoordsOp,
      elliptic_curve::ToCoordsOp,
      linalg::BroadcastOp,
      linalg::GenericOp,
      linalg::MapOp,
      linalg::ReduceOp,
      linalg::TransposeOp,
      linalg::YieldOp,
      memref::AllocOp,
      memref::AllocaOp,
      memref::CastOp,
      memref::CopyOp,
      memref::DimOp,
      memref::LoadOp,
      memref::StoreOp,
      memref::SubViewOp,
      memref::ViewOp,
      sparse_tensor::AssembleOp,
      tensor::CastOp,
      tensor::ConcatOp,
      tensor::DimOp,
      tensor::EmptyOp,
      tensor::ExtractOp,
      tensor::ExtractSliceOp,
      tensor::FromElementsOp,
      tensor::InsertOp,
      tensor::InsertSliceOp,
      tensor::PadOp,
      tensor::ReshapeOp,
      tensor::YieldOp,
      tensor_ext::BitReverseOp,
      ub::PoisonOp,
      vector::BroadcastOp,
      vector::SplatOp,
      vector::TransferReadOp,
      vector::TransferWriteOp
      // clang-format on
      >([&](auto op) { return typeConverter.isLegal(op); });

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

} // namespace mlir::prime_ir::field
