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
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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
                     IntrinsicFunctionGenerator *generator = nullptr,
                     LoweringMode mode = LoweringMode::Inline)
      : OpConversionPattern<OpT>(converter, context), generator(generator),
        mode(mode) {
    this->setHasBoundedRewriteRecursion(true);
  }

  LogicalResult
  matchAndRewrite(OpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    ScopedBuilderContext scopedBuilderContext(&b);
    Type fieldType = getElementTypeOrSelf(op.getOutput());

    if (generator &&
        IntrinsicFunctionGenerator::shouldUseIntrinsic(op, fieldType, mode)) {
      auto efType = cast<ExtensionFieldType>(fieldType);
      rewriter.replaceOp(op,
                         static_cast<const Derived *>(this)->emitIntrinsicCall(
                             rewriter, op.getLoc(), efType, adaptor));
      return success();
    }

    rewriter.replaceOp(
        op, {static_cast<const Derived *>(this)->emitInlineCodeGen(fieldType,
                                                                   adaptor)});
    return success();
  }

protected:
  IntrinsicFunctionGenerator *generator;
  LoweringMode mode;
};

struct ConvertInverse : ConvertFieldOpBase<InverseOp, ConvertInverse> {
  using Base::Base;
  Value emitIntrinsicCall(OpBuilder &b, Location loc, ExtensionFieldType efType,
                          OpAdaptor adaptor) const {
    return generator->emitQuarticInverseCall(b, loc, efType,
                                             adaptor.getInput());
  }
  Value emitInlineCodeGen(Type fieldType, OpAdaptor adaptor) const {
    return FieldCodeGen(fieldType, adaptor.getInput(), this->typeConverter)
        .inverse();
  }
};

struct ConvertNegate : ConvertFieldOpBase<NegateOp, ConvertNegate> {
  using Base::Base;
  Value emitIntrinsicCall(OpBuilder &b, Location loc, ExtensionFieldType efType,
                          OpAdaptor adaptor) const {
    return generator->emitNegateCall(b, loc, efType, adaptor.getInput());
  }
  Value emitInlineCodeGen(Type fieldType, OpAdaptor adaptor) const {
    return -FieldCodeGen(fieldType, adaptor.getInput(), this->typeConverter);
  }
};

struct ConvertAdd : ConvertFieldOpBase<AddOp, ConvertAdd> {
  using Base::Base;
  Value emitIntrinsicCall(OpBuilder &b, Location loc, ExtensionFieldType efType,
                          OpAdaptor adaptor) const {
    return generator->emitAddCall(b, loc, efType, adaptor.getLhs(),
                                  adaptor.getRhs());
  }
  Value emitInlineCodeGen(Type fieldType, OpAdaptor adaptor) const {
    FieldCodeGen lhs(fieldType, adaptor.getLhs(), this->typeConverter);
    FieldCodeGen rhs(fieldType, adaptor.getRhs(), this->typeConverter);
    return lhs + rhs;
  }
};

struct ConvertDouble : ConvertFieldOpBase<DoubleOp, ConvertDouble> {
  using Base::Base;
  Value emitIntrinsicCall(OpBuilder &b, Location loc, ExtensionFieldType efType,
                          OpAdaptor adaptor) const {
    return generator->emitDoubleCall(b, loc, efType, adaptor.getInput());
  }
  Value emitInlineCodeGen(Type fieldType, OpAdaptor adaptor) const {
    return FieldCodeGen(fieldType, adaptor.getInput(), this->typeConverter)
        .dbl();
  }
};

struct ConvertSub : ConvertFieldOpBase<SubOp, ConvertSub> {
  using Base::Base;
  Value emitIntrinsicCall(OpBuilder &b, Location loc, ExtensionFieldType efType,
                          OpAdaptor adaptor) const {
    return generator->emitSubCall(b, loc, efType, adaptor.getLhs(),
                                  adaptor.getRhs());
  }
  Value emitInlineCodeGen(Type fieldType, OpAdaptor adaptor) const {
    FieldCodeGen lhs(fieldType, adaptor.getLhs(), this->typeConverter);
    FieldCodeGen rhs(fieldType, adaptor.getRhs(), this->typeConverter);
    return lhs - rhs;
  }
};

struct ConvertMul : ConvertFieldOpBase<MulOp, ConvertMul> {
  using Base::Base;
  Value emitIntrinsicCall(OpBuilder &b, Location loc, ExtensionFieldType efType,
                          OpAdaptor adaptor) const {
    return generator->emitQuarticMulCall(b, loc, efType, adaptor.getLhs(),
                                         adaptor.getRhs());
  }
  Value emitInlineCodeGen(Type fieldType, OpAdaptor adaptor) const {
    FieldCodeGen lhs(fieldType, adaptor.getLhs(), this->typeConverter);
    FieldCodeGen rhs(fieldType, adaptor.getRhs(), this->typeConverter);
    return lhs * rhs;
  }
};

struct ConvertSquare : ConvertFieldOpBase<SquareOp, ConvertSquare> {
  using Base::Base;
  Value emitIntrinsicCall(OpBuilder &b, Location loc, ExtensionFieldType efType,
                          OpAdaptor adaptor) const {
    return generator->emitQuarticSquareCall(b, loc, efType, adaptor.getInput());
  }
  Value emitInlineCodeGen(Type fieldType, OpAdaptor adaptor) const {
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

namespace {

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

/// Outline high-degree extension field ops into shared intrinsic functions.
///
/// This pre-processing step runs before applyPartialConversion to work around
/// a fundamental limitation: intrinsic functions created during dialect
/// conversion (via raw OpBuilder) are invisible to the conversion framework,
/// and applyPartialConversion rolls back all changes on failure.
///
/// Multi-level intrinsic strategy:
/// 1. Pre-create intrinsic functions for ALL tower levels (Fp12, Fp6, Fp2)
///    so the generator finds existing functions during conversion
/// 2. Replace user function ops with calls to top-level intrinsics
/// 3. During conversion, intrinsic function bodies expand one level,
///    with sub-ops redirected to lower-level intrinsics via the generator
static void outlineExtensionFieldOps(ModuleOp module, LoweringMode mode,
                                     IntrinsicFunctionGenerator &generator) {
  // Phase 1: Discover all extension field types and pre-create intrinsics
  // for the full tower hierarchy. This ensures the generator's
  // getOrCreateFunction always finds existing functions (via SymbolTable
  // lookup) during conversion, avoiding raw OpBuilder function creation.
  DenseSet<Type> seenTypes;
  module.walk([&](Operation *op) {
    if (!isa<AddOp, SubOp, NegateOp, DoubleOp, MulOp, SquareOp, InverseOp>(op))
      return;
    Type fieldType = getElementTypeOrSelf(op->getResult(0).getType());
    if (auto efType = dyn_cast<ExtensionFieldType>(fieldType)) {
      if (efType.getDegreeOverPrime() >= 2 &&
          efType.getBasePrimeField().getStorageBitWidth() > 64 &&
          !seenTypes.contains(efType)) {
        seenTypes.insert(efType);
        generator.preCreateIntrinsicsForTower(efType);
      }
    }
  });

  // Phase 2: Replace qualifying ops in user functions with intrinsic calls
  SmallVector<Operation *> opsToOutline;
  module.walk([&](Operation *op) {
    if (isInsideIntrinsicFunction(op))
      return;
    if (!isa<AddOp, SubOp, NegateOp, DoubleOp, MulOp, SquareOp, InverseOp>(op))
      return;
    Type fieldType = getElementTypeOrSelf(op->getResult(0).getType());
    if (IntrinsicFunctionGenerator::shouldUseIntrinsic(op, fieldType, mode))
      opsToOutline.push_back(op);
  });

  for (Operation *op : opsToOutline) {
    OpBuilder builder(op);
    auto efType = cast<ExtensionFieldType>(
        getElementTypeOrSelf(op->getResult(0).getType()));

    Value result;
    if (auto addOp = dyn_cast<AddOp>(op)) {
      result = generator.emitAddCall(builder, op->getLoc(), efType,
                                     addOp.getLhs(), addOp.getRhs());
    } else if (auto subOp = dyn_cast<SubOp>(op)) {
      result = generator.emitSubCall(builder, op->getLoc(), efType,
                                     subOp.getLhs(), subOp.getRhs());
    } else if (auto negateOp = dyn_cast<NegateOp>(op)) {
      result = generator.emitNegateCall(builder, op->getLoc(), efType,
                                        negateOp.getInput());
    } else if (auto doubleOp = dyn_cast<DoubleOp>(op)) {
      result = generator.emitDoubleCall(builder, op->getLoc(), efType,
                                        doubleOp.getInput());
    } else if (auto mulOp = dyn_cast<MulOp>(op)) {
      result = generator.emitQuarticMulCall(builder, op->getLoc(), efType,
                                            mulOp.getLhs(), mulOp.getRhs());
    } else if (auto squareOp = dyn_cast<SquareOp>(op)) {
      result = generator.emitQuarticSquareCall(builder, op->getLoc(), efType,
                                               squareOp.getInput());
    } else if (auto inverseOp = dyn_cast<InverseOp>(op)) {
      result = generator.emitQuarticInverseCall(builder, op->getLoc(), efType,
                                                inverseOp.getInput());
    }

    op->getResult(0).replaceAllUsesWith(result);
    op->erase();
  }
}

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

  // Generator must outlive both outlining and conversion to maintain the
  // SymbolTable cache — conversion patterns look up existing functions.
  std::optional<IntrinsicFunctionGenerator> generator;
  if (mode != LoweringMode::Inline) {
    generator.emplace(module);

    // Pre-pass: outline high-degree extension field ops into shared functions.
    // Also pre-creates intrinsic functions for all tower levels so the
    // generator always finds existing functions during conversion.
    outlineExtensionFieldOps(module, mode, *generator);

    // Set LLVM internal linkage on intrinsic functions to avoid duplicate
    // symbol errors when multiple HLO computations are compiled as separate
    // LLVM modules and linked together in the JIT.
    // Only set when the LLVM dialect is loaded (e.g., in the ZKX JIT pipeline).
    if (context->getLoadedDialect<LLVM::LLVMDialect>()) {
      auto internalLinkage =
          LLVM::LinkageAttr::get(context, LLVM::Linkage::Internal);
      module.walk([&](func::FuncOp funcOp) {
        if (funcOp.getName().starts_with("__prime_ir_"))
          funcOp->setAttr("llvm.linkage", internalLinkage);
      });
    }
  }

  ConversionTarget target(*context);

  // Mark field operations as dynamically legal if they contain BinaryFieldType
  // (those will be handled by BinaryFieldToArith pass instead)
  target.addDynamicallyLegalOp<ConstantOp, AddOp, SubOp, MulOp, NegateOp,
                               DoubleOp, SquareOp, InverseOp, CmpOp, PowUIOp>(
      [](Operation *op) { return operationContainsBinaryFieldType(op); });

  // Mark remaining field dialect ops as illegal (prime/extension field ops)
  target.addIllegalDialect<FieldDialect>();
  target.addLegalDialect<mod_arith::ModArithDialect>();
  target.addLegalDialect<func::FuncDialect>();
  target.addLegalOp<func::FuncOp, func::CallOp, func::ReturnOp>();

  RewritePatternSet patterns(context);
  rewrites::populateWithGenerated(patterns);

  // Pass the generator to conversion patterns for multi-level intrinsic calls:
  // when converting Fp12 intrinsic bodies, the Fp6 sub-ops should call pre-
  // created Fp6 intrinsics instead of being expanded inline.
  IntrinsicFunctionGenerator *generatorPtr =
      generator.has_value() ? &*generator : nullptr;
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
      >(typeConverter, context, generatorPtr, mode);

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
