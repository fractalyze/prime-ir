/* Copyright 2025 The ZKIR Authors.

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

#include "zkir/Dialect/Field/Conversions/FieldToModArith/FieldToModArith.h"

#include <utility>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
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
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveOps.h"
#include "zkir/Dialect/Field/Conversions/FieldToModArith/Extension/CubicExtensionField.h"
#include "zkir/Dialect/Field/IR/FieldDialect.h"
#include "zkir/Dialect/Field/IR/FieldOps.h"
#include "zkir/Dialect/Field/IR/FieldTypes.h"
#include "zkir/Dialect/ModArith/IR/ModArithDialect.h"
#include "zkir/Dialect/ModArith/IR/ModArithOps.h"
#include "zkir/Dialect/ModArith/IR/ModArithTypes.h"
#include "zkir/Dialect/TensorExt/IR/TensorExtOps.h"
#include "zkir/Utils/ConversionUtils.h"
#include "zkir/Utils/ShapedTypeConverter.h"

namespace mlir::zkir::field {

#define GEN_PASS_DEF_FIELDTOMODARITH
#include "zkir/Dialect/Field/Conversions/FieldToModArith/FieldToModArith.h.inc"

namespace {
mod_arith::ModArithType convertPrimeFieldType(PrimeFieldType type) {
  IntegerAttr modulus = type.getModulus();
  bool isMontgomery = type.isMontgomery();
  return mod_arith::ModArithType::get(type.getContext(), modulus, isMontgomery);
}

SmallVector<Type> coeffsTypeRange(Type type) {
  auto extField = cast<ExtensionFieldTypeInterface>(type);
  auto baseField = cast<PrimeFieldType>(extField.getBaseFieldType());
  return SmallVector<Type>(extField.getDegreeOverBase(),
                           convertPrimeFieldType(baseField));
}

Operation::result_range extractCoeffs(ImplicitLocOpBuilder &b,
                                      Value extFieldElement) {
  return b
      .create<ExtToCoeffsOp>(coeffsTypeRange(extFieldElement.getType()),
                             extFieldElement)
      .getResults();
}
} // namespace

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

    mod_arith::ModArithType modType;
    if (auto pfType = dyn_cast<PrimeFieldType>(op.getType())) {
      modType =
          cast<mod_arith::ModArithType>(typeConverter->convertType(pfType));
    } else if (auto efType =
                   dyn_cast<ExtensionFieldTypeInterface>(op.getType())) {
      modType = cast<mod_arith::ModArithType>(
          typeConverter->convertType(efType.getBaseFieldType()));
    } else {
      op.emitOpError("unsupported output type");
      return failure();
    }

    if (auto pfAttr = dyn_cast<PrimeFieldAttr>(op.getValueAttr())) {
      auto cval = b.create<mod_arith::ConstantOp>(modType, pfAttr.getValue());
      rewriter.replaceOp(op, cval);
      return success();
    }

    auto efAttr = cast<ExtensionFieldAttrInterface>(op.getValueAttr());
    auto efType = cast<ExtensionFieldTypeInterface>(op.getType());
    unsigned n = efType.getDegreeOverBase();

    SmallVector<Value> coeffs;
    for (unsigned i = 0; i < n; ++i) {
      auto coeff = cast<PrimeFieldAttr>(efAttr.getCoeff(i));
      coeffs.push_back(
          b.create<mod_arith::ConstantOp>(modType, coeff.getValue()));
    }
    auto ext = b.create<ExtFromCoeffsOp>(TypeRange{op.getType()}, coeffs);
    rewriter.replaceOp(op, ext);
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

    auto bitcast = b.create<mod_arith::BitcastOp>(
        typeConverter->convertType(op.getType()), adaptor.getInput());
    rewriter.replaceOp(op, bitcast);
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
    if (auto efType = dyn_cast<ExtensionFieldTypeInterface>(fieldType)) {
      auto baseField = cast<PrimeFieldType>(efType.getBaseFieldType());
      Type baseModArithType = typeConverter->convertType(baseField);
      auto coeffs = extractCoeffs(b, adaptor.getInput());

      SmallVector<Value> montCoeffs;
      for (auto coeff : coeffs) {
        montCoeffs.push_back(
            b.create<mod_arith::ToMontOp>(baseModArithType, coeff));
      }
      auto ext = b.create<ExtFromCoeffsOp>(TypeRange{fieldType}, montCoeffs);
      rewriter.replaceOp(op, ext);
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
    if (auto efType = dyn_cast<ExtensionFieldTypeInterface>(fieldType)) {
      auto baseField = cast<PrimeFieldType>(efType.getBaseFieldType());
      Type baseModArithType = typeConverter->convertType(baseField);
      auto coeffs = extractCoeffs(b, adaptor.getInput());

      SmallVector<Value> stdCoeffs;
      for (auto coeff : coeffs) {
        stdCoeffs.push_back(
            b.create<mod_arith::FromMontOp>(baseModArithType, coeff));
      }
      auto ext = b.create<ExtFromCoeffsOp>(TypeRange{fieldType}, stdCoeffs);
      rewriter.replaceOp(op, ext);
      return success();
    }
    return failure();
  }
};

struct ConvertInverse : public OpConversionPattern<InverseOp> {
  explicit ConvertInverse(MLIRContext *context)
      : OpConversionPattern<InverseOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(InverseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Type fieldType = getElementTypeOrSelf(op.getOutput());
    if (isa<PrimeFieldType>(fieldType)) {
      auto inv = b.create<mod_arith::InverseOp>(adaptor.getInput());
      rewriter.replaceOp(op, inv);
      return success();
    }
    if (auto f2Type = dyn_cast<QuadraticExtFieldType>(fieldType)) {
      // construct beta as a mod arith constant
      auto beta = b.create<mod_arith::ConstantOp>(
          typeConverter->convertType(f2Type.getBaseField()),
          f2Type.getNonResidue().getValue());

      // denominator = a₀² - a₁²β
      auto coeffs = extractCoeffs(b, adaptor.getInput());
      auto lowSquared = b.create<mod_arith::SquareOp>(coeffs[0]);
      auto highSquared = b.create<mod_arith::SquareOp>(coeffs[1]);
      auto betaTimesHighSquared = b.create<mod_arith::MulOp>(beta, highSquared);
      auto denominator =
          b.create<mod_arith::SubOp>(lowSquared, betaTimesHighSquared);
      auto denominatorInv = b.create<mod_arith::InverseOp>(denominator);

      // c₀ = a₀ / denominator
      auto c0 = b.create<mod_arith::MulOp>(coeffs[0], denominatorInv);
      // c₁ = -a₁ / denominator
      auto highNegated = b.create<mod_arith::NegateOp>(coeffs[1]);
      auto c1 = b.create<mod_arith::MulOp>(highNegated, denominatorInv);
      auto f2 =
          b.create<ExtFromCoeffsOp>(TypeRange{f2Type}, ValueRange{c0, c1});
      rewriter.replaceOp(op, f2);
      return success();
    }
    if (auto f3Type = dyn_cast<CubicExtFieldType>(fieldType)) {
      auto xi = b.create<mod_arith::ConstantOp>(
          typeConverter->convertType(f3Type.getBaseField()),
          f3Type.getNonResidue().getValue());

      auto coeffs = extractCoeffs(b, adaptor.getInput());
      CubicExtensionField f3Field(b, xi);
      auto result = f3Field.inverse(coeffs[0], coeffs[1], coeffs[2]);

      auto f3 = b.create<ExtFromCoeffsOp>(
          TypeRange{f3Type}, ValueRange{result[0], result[1], result[2]});
      rewriter.replaceOp(op, f3);
      return success();
    }
    return failure();
  }
};

struct ConvertNegate : public OpConversionPattern<NegateOp> {
  explicit ConvertNegate(MLIRContext *context)
      : OpConversionPattern<NegateOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(NegateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Type fieldType = getElementTypeOrSelf(op.getOutput());
    if (isa<PrimeFieldType>(fieldType)) {
      auto neg = b.create<mod_arith::NegateOp>(adaptor.getInput());
      rewriter.replaceOp(op, neg);
      return success();
    }
    if (auto efType = dyn_cast<ExtensionFieldTypeInterface>(fieldType)) {
      auto coeffs = extractCoeffs(b, adaptor.getInput());
      SmallVector<Value> negCoeffs;
      for (unsigned i = 0; i < efType.getDegreeOverPrime(); ++i) {
        negCoeffs.push_back(b.create<mod_arith::NegateOp>(coeffs[i]));
      }
      auto ext = b.create<ExtFromCoeffsOp>(TypeRange{fieldType}, negCoeffs);
      rewriter.replaceOp(op, ext);
      return success();
    }
    return failure();
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

    Type fieldType = getElementTypeOrSelf(op.getOutput());
    if (isa<PrimeFieldType>(fieldType)) {
      auto add = b.create<mod_arith::AddOp>(adaptor.getLhs(), adaptor.getRhs());
      rewriter.replaceOp(op, add);
      return success();
    }
    if (auto efType = dyn_cast<ExtensionFieldTypeInterface>(fieldType)) {
      auto lhsCoeffs = extractCoeffs(b, adaptor.getLhs());
      auto rhsCoeffs = extractCoeffs(b, adaptor.getRhs());
      SmallVector<Value> resultCoeffs;
      for (unsigned i = 0; i < efType.getDegreeOverPrime(); ++i) {
        resultCoeffs.push_back(
            b.create<mod_arith::AddOp>(lhsCoeffs[i], rhsCoeffs[i]));
      }
      auto ext = b.create<ExtFromCoeffsOp>(TypeRange{fieldType}, resultCoeffs);
      rewriter.replaceOp(op, ext);
      return success();
    }
    return failure();
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

    Type fieldType = getElementTypeOrSelf(op.getOutput());
    if (isa<PrimeFieldType>(fieldType)) {
      auto doubled = b.create<mod_arith::DoubleOp>(adaptor.getInput());
      rewriter.replaceOp(op, doubled);
      return success();
    }
    if (auto efType = dyn_cast<ExtensionFieldTypeInterface>(fieldType)) {
      auto coeffs = extractCoeffs(b, adaptor.getInput());
      SmallVector<Value> resultCoeffs;
      for (unsigned i = 0; i < efType.getDegreeOverPrime(); ++i) {
        resultCoeffs.push_back(b.create<mod_arith::DoubleOp>(coeffs[i]));
      }
      auto ext = b.create<ExtFromCoeffsOp>(TypeRange{fieldType}, resultCoeffs);
      rewriter.replaceOp(op, ext);
      return success();
    }
    return failure();
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

    Type fieldType = getElementTypeOrSelf(op.getOutput());
    if (isa<PrimeFieldType>(fieldType)) {
      auto sub = b.create<mod_arith::SubOp>(adaptor.getLhs(), adaptor.getRhs());
      rewriter.replaceOp(op, sub);
      return success();
    }
    if (auto efType = dyn_cast<ExtensionFieldTypeInterface>(fieldType)) {
      auto lhsCoeffs = extractCoeffs(b, adaptor.getLhs());
      auto rhsCoeffs = extractCoeffs(b, adaptor.getRhs());
      SmallVector<Value> resultCoeffs;
      for (unsigned i = 0; i < efType.getDegreeOverPrime(); ++i) {
        resultCoeffs.push_back(
            b.create<mod_arith::SubOp>(lhsCoeffs[i], rhsCoeffs[i]));
      }
      auto ext = b.create<ExtFromCoeffsOp>(TypeRange{fieldType}, resultCoeffs);
      rewriter.replaceOp(op, ext);
      return success();
    }
    return failure();
  }
};

struct ConvertMul : public OpConversionPattern<MulOp> {
  explicit ConvertMul(MLIRContext *context)
      : OpConversionPattern<MulOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Type fieldType = getElementTypeOrSelf(op.getOutput());
    if (isa<PrimeFieldType>(fieldType)) {
      auto mul = b.create<mod_arith::MulOp>(adaptor.getLhs(), adaptor.getRhs());
      rewriter.replaceOp(op, mul);
      return success();
    }
    if (auto f2Type = dyn_cast<QuadraticExtFieldType>(fieldType)) {
      // construct beta as a mod arith constant
      auto beta = b.create<mod_arith::ConstantOp>(
          typeConverter->convertType(f2Type.getBaseField()),
          f2Type.getNonResidue().getValue());

      Operation::result_range lhsCoeffs = extractCoeffs(b, adaptor.getLhs());
      Operation::result_range rhsCoeffs = extractCoeffs(b, adaptor.getRhs());
      // v₀ = a₀ * b₀
      // v₁ = a₁ * b₁
      auto v0 = b.create<mod_arith::MulOp>(lhsCoeffs[0], rhsCoeffs[0]);
      auto v1 = b.create<mod_arith::MulOp>(lhsCoeffs[1], rhsCoeffs[1]);

      // c₀ = v₀ + βv₁
      auto betaTimesV1 = b.create<mod_arith::MulOp>(beta, v1);
      auto c0 = b.create<mod_arith::AddOp>(v0, betaTimesV1);

      // c₁ = (a₀ + a₁)(b₀ + b₁) - v₀ - v₁
      auto sumLhs = b.create<mod_arith::AddOp>(lhsCoeffs[0], lhsCoeffs[1]);
      auto sumRhs = b.create<mod_arith::AddOp>(rhsCoeffs[0], rhsCoeffs[1]);
      auto sumProduct = b.create<mod_arith::MulOp>(sumLhs, sumRhs);
      Value c1 = b.create<mod_arith::SubOp>(sumProduct, v0);
      c1 = b.create<mod_arith::SubOp>(c1, v1);

      auto f2 =
          b.create<ExtFromCoeffsOp>(TypeRange{f2Type}, ValueRange{c0, c1});
      rewriter.replaceOp(op, f2);
      return success();
    }
    if (auto f3Type = dyn_cast<CubicExtFieldType>(fieldType)) {
      auto xi = b.create<mod_arith::ConstantOp>(
          typeConverter->convertType(f3Type.getBaseField()),
          f3Type.getNonResidue().getValue());

      auto lhsCoeffs = extractCoeffs(b, adaptor.getLhs());
      auto rhsCoeffs = extractCoeffs(b, adaptor.getRhs());

      CubicExtensionField f3Field(b, xi);
      auto result = f3Field.mul(lhsCoeffs[0], lhsCoeffs[1], lhsCoeffs[2],
                                rhsCoeffs[0], rhsCoeffs[1], rhsCoeffs[2]);

      auto f3 = b.create<ExtFromCoeffsOp>(
          TypeRange{f3Type}, ValueRange{result[0], result[1], result[2]});
      rewriter.replaceOp(op, f3);
      return success();
    }
    return failure();
  }
};

struct ConvertSquare : public OpConversionPattern<SquareOp> {
  explicit ConvertSquare(MLIRContext *context)
      : OpConversionPattern<SquareOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SquareOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Type fieldType = getElementTypeOrSelf(op.getOutput());
    if (isa<PrimeFieldType>(fieldType)) {
      auto square = b.create<mod_arith::SquareOp>(adaptor.getInput());
      rewriter.replaceOp(op, square);
      return success();
    }
    if (auto f2Type = dyn_cast<QuadraticExtFieldType>(fieldType)) {
      // construct beta as a mod arith constant
      auto beta = b.create<mod_arith::ConstantOp>(
          typeConverter->convertType(f2Type.getBaseField()),
          f2Type.getNonResidue().getValue());

      auto coeffs = extractCoeffs(b, adaptor.getInput());

      // v₀ = a₀ - a₁
      Value v0 = b.create<mod_arith::SubOp>(coeffs[0], coeffs[1]);

      // v₁ = a₀ - βa₁
      auto betaA1 = b.create<mod_arith::MulOp>(beta, coeffs[1]);
      auto v1 = b.create<mod_arith::SubOp>(coeffs[0], betaA1);

      // v₂ = a₀ * a₁
      auto v2 = b.create<mod_arith::MulOp>(coeffs[0], coeffs[1]);

      // v₀ = v₀ * v₁ + v₂
      auto v0TimesV1 = b.create<mod_arith::MulOp>(v0, v1);
      v0 = b.create<mod_arith::AddOp>(v0TimesV1, v2);

      // c₁ = v₂ + v₂
      auto c1 = b.create<mod_arith::DoubleOp>(v2);
      // c₀ = v₀ + βv₂
      auto betaV2 = b.create<mod_arith::MulOp>(beta, v2);
      auto c0 = b.create<mod_arith::AddOp>(v0, betaV2);
      auto f2 =
          b.create<ExtFromCoeffsOp>(TypeRange{f2Type}, ValueRange{c0, c1});
      rewriter.replaceOp(op, f2);
      return success();
    }
    if (auto f3Type = dyn_cast<CubicExtFieldType>(fieldType)) {
      auto xi = b.create<mod_arith::ConstantOp>(
          typeConverter->convertType(f3Type.getBaseField()),
          f3Type.getNonResidue().getValue());

      auto coeffs = extractCoeffs(b, adaptor.getInput());
      CubicExtensionField f3Field(b, xi);
      auto result = f3Field.square(coeffs[0], coeffs[1], coeffs[2]);

      auto f3 = b.create<ExtFromCoeffsOp>(
          TypeRange{f3Type}, ValueRange{result[0], result[1], result[2]});
      rewriter.replaceOp(op, f3);
      return success();
    }
    return failure();
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
    } else if (auto f2Type = dyn_cast<QuadraticExtFieldType>(fieldType)) {
      modulus = f2Type.getBaseField().getModulus().getValue();
      init = f2Type.isMontgomery()
                 ? b.create<ToMontOp>(
                        f2Type,
                        b.create<ConstantOp>(cast<QuadraticExtFieldType>(
                                                 getStandardFormType(f2Type)),
                                             1, 0))
                       .getResult()
                 : b.create<ConstantOp>(f2Type, 1, 0);
    } else if (auto f3Type = dyn_cast<CubicExtFieldType>(fieldType)) {
      modulus = f3Type.getBaseField().getModulus().getValue();
      init =
          f3Type.isMontgomery()
              ? b.create<ToMontOp>(
                     f3Type,
                     b.create<F3ConstantOp>(
                         cast<CubicExtFieldType>(getStandardFormType(f3Type)),
                         b.create<ConstantOp>(f3Type.getBaseField(), 1),
                         b.create<ConstantOp>(f3Type.getBaseField(), 0),
                         b.create<ConstantOp>(f3Type.getBaseField(), 0)))
                    .getResult()
              : b.create<F3ConstantOp>(
                    f3Type, b.create<ConstantOp>(f3Type.getBaseField(), 1),
                    b.create<ConstantOp>(f3Type.getBaseField(), 0),
                    b.create<ConstantOp>(f3Type.getBaseField(), 0));
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
    // For quadratic extension field, x^(p²-1) ≡ 1 mod p², so
    // x^n ≡ x^(n mod (p²-1)) mod p²
    // For cubic extension field, x^(p³-1) ≡ 1 mod p³, so
    // x^n ≡ x^(n mod (p³-1)) mod p³
    if (isa<PrimeFieldType>(fieldType)) {
      exp = b.create<arith::RemUIOp>(
          exp, b.create<arith::ConstantIntOp>(intType, modulus - 1));
    } else if (isa<QuadraticExtFieldType>(fieldType)) {
      modulus = modulus.zext(modBitWidth * 2);
      modulus = modulus * modulus - 1;
      exp = b.create<arith::ExtUIOp>(
          IntegerType::get(exp.getContext(), modulus.getBitWidth()), exp);
      intType = IntegerType::get(exp.getContext(), modulus.getBitWidth());
      exp = b.create<arith::RemUIOp>(
          exp, b.create<arith::ConstantIntOp>(intType, modulus));
    } else if (isa<CubicExtFieldType>(fieldType)) {
      modulus = modulus.zext(modBitWidth * 3);
      modulus = modulus * modulus * modulus - 1;
      exp = b.create<arith::ExtUIOp>(
          IntegerType::get(exp.getContext(), modulus.getBitWidth()), exp);
      intType = IntegerType::get(exp.getContext(), modulus.getBitWidth());
      exp = b.create<arith::RemUIOp>(
          exp, b.create<arith::ConstantIntOp>(intType, modulus));
    }

    Value zero = b.create<arith::ConstantIntOp>(intType, 0);
    Value one = b.create<arith::ConstantIntOp>(intType, 1);
    Value powerOfP = base;
    auto ifOp = b.create<scf::IfOp>(
        b.create<arith::CmpIOp>(arith::CmpIPredicate::ne,
                                b.create<arith::AndIOp>(exp, one), zero),
        [&](OpBuilder &builder, Location loc) {
          ImplicitLocOpBuilder b(loc, builder);
          auto newResult = b.create<MulOp>(init, powerOfP);
          b.create<scf::YieldOp>(ValueRange{newResult});
        },
        [&](OpBuilder &builder, Location loc) {
          ImplicitLocOpBuilder b(loc, builder);
          b.create<scf::YieldOp>(ValueRange{init});
        });
    exp = b.create<arith::ShRUIOp>(exp, one);
    init = ifOp.getResult(0);
    auto whileOp = b.create<scf::WhileOp>(
        TypeRange{intType, fieldType, fieldType},
        ValueRange{exp, powerOfP, init},
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          ImplicitLocOpBuilder b(loc, builder);
          auto cond =
              b.create<arith::CmpIOp>(arith::CmpIPredicate::ugt, args[0], zero);
          b.create<scf::ConditionOp>(cond, args);
        },
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          ImplicitLocOpBuilder b(loc, builder);
          auto currExp = args[0];
          auto currPowerOfP = args[1];
          auto currResult = args[2];

          auto newPowerOfP = b.create<SquareOp>(currPowerOfP);
          auto masked = b.create<arith::AndIOp>(currExp, one);
          auto isOdd =
              b.create<arith::CmpIOp>(arith::CmpIPredicate::ne, masked, zero);
          auto ifOp = b.create<scf::IfOp>(
              isOdd,
              [&](OpBuilder &builder, Location loc) {
                ImplicitLocOpBuilder b(loc, builder);
                auto newResult = b.create<MulOp>(currResult, newPowerOfP);
                b.create<scf::YieldOp>(ValueRange{newResult});
              },
              [&](OpBuilder &builder, Location loc) {
                ImplicitLocOpBuilder b(loc, builder);
                b.create<scf::YieldOp>(ValueRange{currResult});
              });
          auto shifted = b.create<arith::ShRUIOp>(currExp, one);
          b.create<scf::YieldOp>(
              ValueRange{shifted, newPowerOfP, ifOp.getResult(0)});
        });
    rewriter.replaceOp(op, whileOp.getResult(2));
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
    } else if (auto efType = dyn_cast<ExtensionFieldTypeInterface>(fieldType)) {
      // For extension fields, we compare each coefficient separately.
      auto lhsCoeffs = extractCoeffs(b, adaptor.getLhs());
      auto rhsCoeffs = extractCoeffs(b, adaptor.getRhs());
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

struct ConvertF2Constant : public OpConversionPattern<F2ConstantOp> {
  explicit ConvertF2Constant(MLIRContext *context)
      : OpConversionPattern<F2ConstantOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(F2ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto f2 = b.create<ExtFromCoeffsOp>(
        op.getType(), ValueRange{adaptor.getC0(), adaptor.getC1()});
    rewriter.replaceOp(op, f2);
    return success();
  }
};

struct ConvertF3Constant : public OpConversionPattern<F3ConstantOp> {
  explicit ConvertF3Constant(MLIRContext *context)
      : OpConversionPattern<F3ConstantOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(F3ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto f3 = b.create<ExtFromCoeffsOp>(
        op.getType(),
        ValueRange{adaptor.getC0(), adaptor.getC1(), adaptor.getC2()});
    rewriter.replaceOp(op, f3);
    return success();
  }
};

namespace rewrites {
// In an inner namespace to avoid conflicts with canonicalization patterns
#include "zkir/Dialect/Field/Conversions/FieldToModArith/FieldToModArith.cpp.inc"
} // namespace rewrites

struct FieldToModArith : impl::FieldToModArithBase<FieldToModArith> {
  using FieldToModArithBase::FieldToModArithBase;

  void runOnOperation() override;
};

void FieldToModArith::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();
  FieldToModArithTypeConverter typeConverter(context);

  ConversionTarget target(*context);
  target.addIllegalDialect<FieldDialect>();
  target.addLegalDialect<mod_arith::ModArithDialect>();

  RewritePatternSet patterns(context);
  rewrites::populateWithGenerated(patterns);
  patterns.add<
      // clang-format off
      ConvertAdd,
      ConvertBitcast,
      ConvertConstant,
      ConvertCmp,
      ConvertDouble,
      ConvertF2Constant,
      ConvertF3Constant,
      ConvertFromMont,
      ConvertInverse,
      ConvertMul,
      ConvertNegate,
      ConvertPowUI,
      ConvertSquare,
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
      ConvertAny<elliptic_curve::ExtractOp>,
      ConvertAny<elliptic_curve::PointOp>,
      ConvertAny<linalg::BroadcastOp>,
      ConvertAny<linalg::GenericOp>,
      ConvertAny<linalg::MapOp>,
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
      ConvertAny<vector::SplatOp>
      // clang-format on
      >(typeConverter, context);

  addStructuralConversionPatterns(typeConverter, patterns, target);

  target.addDynamicallyLegalOp<
      // clang-format off
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
      elliptic_curve::ExtractOp,
      elliptic_curve::PointOp,
      linalg::BroadcastOp,
      linalg::GenericOp,
      linalg::MapOp,
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
      vector::SplatOp
      // clang-format on
      >([&](auto op) { return typeConverter.isLegal(op); });

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

} // namespace mlir::zkir::field
