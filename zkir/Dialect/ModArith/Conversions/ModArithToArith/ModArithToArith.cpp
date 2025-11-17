#include "zkir/Dialect/ModArith/Conversions/ModArithToArith/ModArithToArith.h"

#include <utility>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
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
#include "zkir/Dialect/Field/IR/FieldOps.h"
#include "zkir/Dialect/ModArith/Conversions/ModArithToArith/Inverter/BYInverter.h"
#include "zkir/Dialect/ModArith/Conversions/ModArithToArith/Reducer/MontReducer.h"
#include "zkir/Dialect/ModArith/IR/ModArithDialect.h"
#include "zkir/Dialect/ModArith/IR/ModArithOps.h"
#include "zkir/Dialect/ModArith/IR/ModArithTypes.h"
#include "zkir/Dialect/TensorExt/IR/TensorExtOps.h"
#include "zkir/Utils/APIntUtils.h"
#include "zkir/Utils/ConversionUtils.h"
#include "zkir/Utils/ShapedTypeConverter.h"

// IWYU pragma: begin_keep
// Headers needed for ModArithToArith.h.inc
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
// IWYU pragma: end_keep

namespace mlir::zkir::mod_arith {

#define GEN_PASS_DEF_MODARITHTOARITH
#include "zkir/Dialect/ModArith/Conversions/ModArithToArith/ModArithToArith.h.inc"

class ModArithToArithTypeConverter : public ShapedTypeConverter {
public:
  explicit ModArithToArithTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion(
        [](ModArithType type) -> Type { return convertModArithType(type); });
    addConversion([](ShapedType type) -> Type {
      if (auto modArithType = dyn_cast<ModArithType>(type.getElementType())) {
        return convertShapedType(type, type.getShape(),
                                 convertModArithType(modArithType));
      }
      if (auto vectorType = dyn_cast<VectorType>(type.getElementType())) {
        if (auto modArithType =
                dyn_cast<ModArithType>(vectorType.getElementType())) {
          return convertShapedType(
              type, type.getShape(),
              vectorType.cloneWith(vectorType.getShape(),
                                   convertModArithType(modArithType)));
        }
      }
      return type;
    });
  }

private:
  static IntegerType convertModArithType(ModArithType type) {
    APInt modulus = type.getModulus().getValue();
    return IntegerType::get(type.getContext(), modulus.getBitWidth());
  }
};

namespace {
Value getSignedFormFromCanonical(Value input, TypedAttr modAttr) {
  auto minOp = input.getDefiningOp<arith::MinUIOp>();
  if (!minOp) {
    return {};
  }
  auto addOpLhs = minOp.getLhs().getDefiningOp<arith::AddIOp>();
  auto subOpLhs = minOp.getLhs().getDefiningOp<arith::SubIOp>();
  auto addOpRhs = minOp.getRhs().getDefiningOp<arith::AddIOp>();
  auto subOpRhs = minOp.getRhs().getDefiningOp<arith::SubIOp>();

  if (!(addOpLhs && subOpRhs) && !(subOpLhs && addOpRhs)) {
    return {};
  }

  arith::AddIOp addOp = addOpLhs ? addOpLhs : addOpRhs;
  arith::SubIOp subOp = subOpLhs ? subOpLhs : subOpRhs;

  // min(a, a + cmod) -> a
  // min(a + cmod, a) -> a
  if (addOp.getLhs() == subOp.getResult()) {
    if (auto addedConst = addOp.getRhs().getDefiningOp<arith::ConstantOp>()) {
      if (addedConst.getValue() == modAttr) {
        return subOp.getResult();
      } else if (auto splatAttr =
                     dyn_cast<SplatElementsAttr>(addedConst.getValue())) {
        if (splatAttr.getSplatValue<IntegerAttr>() == modAttr) {
          return subOp.getResult();
        }
      }
    }
  }

  // min(a, a - cmod) -> a - cmod
  // min(a - cmod, a) -> a - cmod
  if (addOp.getResult() == subOp.getLhs()) {
    if (auto subtractedConst =
            subOp.getRhs().getDefiningOp<arith::ConstantOp>()) {
      if (subtractedConst.getValue() == modAttr) {
        return subOp.getResult();
      } else if (auto splatAttr =
                     dyn_cast<SplatElementsAttr>(subtractedConst.getValue())) {
        if (splatAttr.getSplatValue<IntegerAttr>() == modAttr) {
          return subOp.getResult();
        }
      }
    }
  }

  // min(a - cmod, a) -> a - cmod
  if (subOpLhs && addOpRhs && subOpLhs.getLhs() == addOpRhs.getResult()) {
    if (auto subtractedConst =
            subOpLhs.getRhs().getDefiningOp<arith::ConstantOp>()) {
      if (subtractedConst.getValue() == modAttr) {
        return subOpLhs.getResult();
      } else if (auto splatAttr =
                     dyn_cast<SplatElementsAttr>(subtractedConst.getValue())) {
        if (splatAttr.getSplatValue<IntegerAttr>() == modAttr) {
          return subOpLhs.getResult();
        }
      }
    }
  }
  return {};
}
} // namespace
// A helper function to generate the attribute or type
// needed to represent the result of mod_arith op as an integer
// before applying a remainder operation
template <typename Op>
static TypedAttr modulusAttr(Op op, bool extended = false) {
  auto type = op.getType();
  auto modArithType = getResultModArithType(op);
  APInt modulus = modArithType.getModulus().getValue();

  auto width = modulus.getBitWidth();
  if (extended) {
    width *= 2;
  }

  auto intType = IntegerType::get(op.getContext(), width);
  auto truncmod = modulus.zextOrTrunc(width);

  if (auto st = dyn_cast<ShapedType>(type)) {
    auto containerType = st.cloneWith(st.getShape(), intType);
    return DenseElementsAttr::get(containerType, truncmod);
  }
  return IntegerAttr::get(intType, truncmod);
}

// used for extui/trunci
template <typename Op>
static inline Type modulusType(Op op, bool extended = false) {
  return modulusAttr(op, extended).getType();
}

struct ConvertBitcast : public OpConversionPattern<BitcastOp> {
  explicit ConvertBitcast(MLIRContext *context)
      : OpConversionPattern<BitcastOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BitcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
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

    auto cval = b.create<arith::ConstantOp>(op.getLoc(), adaptor.getValue());
    rewriter.replaceOp(op, cval);
    return success();
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

    Type intType = modulusType(op);
    Value zero;
    if (isa<ShapedType>(intType)) {
      zero = b.create<arith::ConstantOp>(SplatElementsAttr::get(
          cast<ShapedType>(intType),
          IntegerAttr::get(getElementTypeOrSelf(intType), 0)));
    } else {
      zero = b.create<arith::ConstantIntOp>(intType, 0);
    }
    auto cmp = b.create<arith::CmpIOp>(arith::CmpIPredicate::eq,
                                       adaptor.getInput(), zero);
    auto cmod = b.create<arith::ConstantOp>(modulusAttr(op));
    auto sub = b.create<arith::SubIOp>(cmod, adaptor.getInput());
    auto result = b.create<arith::SelectOp>(cmp, adaptor.getInput(), sub);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ConvertMontReduce : public OpConversionPattern<MontReduceOp> {
  explicit ConvertMontReduce(MLIRContext *context)
      : OpConversionPattern<MontReduceOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MontReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // `T` is the operand (e.g. the result of a multiplication, twice the
    // bitwidth of modulus).
    Value tLow = adaptor.getLow();
    Value tHigh = adaptor.getHigh();

    // Perform Montgomery reduction using MontReducer helper class.
    MontReducer reducer(b, getResultModArithType(op));
    Value result = reducer.reduce(tLow, tHigh);
    rewriter.replaceOp(op, result);
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

    TypedAttr modAttr = modulusAttr(op);
    ModArithType modType = getResultModArithType(op);
    MontgomeryAttr montAttr = modType.getMontgomeryAttr();
    APInt rSquaredInt = montAttr.getRSquared().getValue();
    Value rSquaredConst = createScalarOrSplatConstant(
        b, b.getLoc(), modAttr.getType(), rSquaredInt);

    // x * R = REDC(x * rSquared)
    Value rSquared = b.create<BitcastOp>(op.getType(), rSquaredConst);
    Value bitcast = b.create<BitcastOp>(op.getType(), adaptor.getInput());
    auto product = b.create<MontMulOp>(op.getType(), bitcast, rSquared);
    rewriter.replaceOp(op, product);
    return success();
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

    // x * R⁻¹ = REDC(x)
    Value zeroHighConst = createScalarOrSplatConstant(
        b, b.getLoc(), modulusAttr(op).getType(), 0);
    auto reduced =
        b.create<MontReduceOp>(op.getType(), op.getInput(), zeroHighConst);

    rewriter.replaceOp(op, reduced);
    return success();
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

    ModArithType modType = getResultModArithType(op);
    if (modType.isMontgomery()) {
      auto result = b.create<MontInverseOp>(op.getType(), op.getInput());
      rewriter.replaceOp(op, result);
      return success();
    }

    BYInverter inverter(b, op.getInput().getType());
    if (auto shapedType = dyn_cast<ShapedType>(op.getInput().getType())) {
      Value result =
          inverter.BatchGenerate(adaptor.getInput(), false, shapedType);
      rewriter.replaceOp(op, result);
      return success();
    }
    Value result = inverter.Generate(adaptor.getInput(), false);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ConvertMontInverse : public OpConversionPattern<MontInverseOp> {
  explicit ConvertMontInverse(MLIRContext *context)
      : OpConversionPattern<MontInverseOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MontInverseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    ModArithType modType = getResultModArithType(op);
    if (!modType.isMontgomery()) {
      return op->emitError(
          "MontInverseOp with non-Montgomery type is not supported in "
          "ModArithToArith conversion");
    }

    BYInverter inverter(b, op.getInput().getType());
    if (auto shapedType = dyn_cast<ShapedType>(op.getInput().getType())) {
      Value result =
          inverter.BatchGenerate(adaptor.getInput(), true, shapedType);
      rewriter.replaceOp(op, result);
      return success();
    }
    Value result = inverter.Generate(adaptor.getInput(), true);
    rewriter.replaceOp(op, result);
    return success();
  }
};

// It is assumed inputs are canonical representatives
// ModArithType ensures add/sub result can not overflow
struct ConvertAdd : public OpConversionPattern<AddOp> {
  explicit ConvertAdd(MLIRContext *context)
      : OpConversionPattern<AddOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    ModArithType modArithType = getResultModArithType(op);
    APInt modulus = modArithType.getModulus().getValue();
    unsigned storageWidth = modArithType.getStorageBitWidth();
    unsigned modWidth = modulus.getActiveBits();

    Value result;
    if (modWidth == storageWidth) {
      auto add =
          b.create<arith::AddUIExtendedOp>(adaptor.getLhs(), adaptor.getRhs());
      MontReducer montReducer(b, getResultModArithType(op));
      result =
          montReducer.getCanonicalFromExtended(add.getSum(), add.getOverflow());
    } else {
      auto noOverflow = arith::IntegerOverflowFlagsAttr::get(
          b.getContext(),
          arith::IntegerOverflowFlags::nuw | arith::IntegerOverflowFlags::nsw);
      auto add = b.create<arith::AddIOp>(adaptor.getLhs(), adaptor.getRhs(),
                                         noOverflow);
      MontReducer montReducer(b, getResultModArithType(op));
      result = montReducer.getCanonicalFromExtended(add);
    }
    rewriter.replaceOp(op, result);
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

    ModArithType modArithType = getResultModArithType(op);
    APInt modulus = modArithType.getModulus().getValue();
    unsigned storageWidth = modArithType.getStorageBitWidth();
    unsigned modWidth = modulus.getActiveBits();

    Value result;
    if (modWidth == storageWidth) {
      result = b.create<AddOp>(op.getInput(), op.getInput());
    } else {
      TypedAttr modAttr = modulusAttr(op);
      Value one =
          createScalarOrSplatConstant(b, b.getLoc(), modAttr.getType(), 1);
      auto noOverflow = arith::IntegerOverflowFlagsAttr::get(
          b.getContext(),
          arith::IntegerOverflowFlags::nuw | arith::IntegerOverflowFlags::nsw);
      auto shifted =
          b.create<arith::ShLIOp>(adaptor.getInput(), one, noOverflow);
      MontReducer montReducer(b, getResultModArithType(op));
      result = montReducer.getCanonicalFromExtended(shifted);
    }
    rewriter.replaceOp(op, result);
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
    MontReducer montReducer(b, getResultModArithType(op));
    auto result =
        montReducer.getCanonicalDiff(adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(op, result);
    return success();
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

    ModArithType modType = getResultModArithType(op);
    TypedAttr modAttr = modulusAttr(op);
    APInt modulus = modType.getModulus().getValue();
    MontgomeryAttr montAttr = modType.getMontgomeryAttr();

    Value zero =
        createScalarOrSplatConstant(b, b.getLoc(), modAttr.getType(), 0);
    Value one =
        createScalarOrSplatConstant(b, b.getLoc(), modAttr.getType(), 1);
    Value four =
        createScalarOrSplatConstant(b, b.getLoc(), modAttr.getType(), 4);
    Value cmod = b.create<arith::ConstantOp>(modAttr);
    if (auto constRhs = op.getRhs().getDefiningOp<ConstantOp>()) {
      IntegerAttr rhsInt =
          dyn_cast_if_present<IntegerAttr>(constRhs.getValue());
      if (auto denseIntAttr =
              dyn_cast_if_present<SplatElementsAttr>(constRhs.getValue())) {
        rhsInt = denseIntAttr.getSplatValue<IntegerAttr>();
      }
      if (rhsInt) {
        IntegerAttr rhsStd, negRhsStd;
        if (modType.isMontgomery()) {
          IntegerAttr rInv = montAttr.getRInv();
          rhsStd = IntegerAttr::get(
              rhsInt.getType(),
              mulMod(rhsInt.getValue(), rInv.getValue(), modulus));
          negRhsStd =
              IntegerAttr::get(rhsInt.getType(), modulus - rhsStd.getValue());
        } else {
          rhsStd = rhsInt;
          negRhsStd =
              IntegerAttr::get(rhsInt.getType(), modulus - rhsInt.getValue());
        }

        // modulus = k * 2^twoAdicity + 1
        size_t twoAdicity = (modulus - 1).countTrailingZeros();
        APInt k = modulus.lshr(twoAdicity);
        Value kConst =
            createScalarOrSplatConstant(b, b.getLoc(), modAttr.getType(), k);

        for (size_t i = 0; i < montAttr.getInvTwoPowers().size(); i++) {
          if (rhsStd == montAttr.getInvTwoPowers()[i] ||
              negRhsStd == montAttr.getInvTwoPowers()[i]) {
            bool isNegated = negRhsStd == montAttr.getInvTwoPowers()[i];
            if (i == 0) {
              // Efficient halve: if odd, add modulus, then shift right by 1
              Value lhs = adaptor.getLhs();
              auto lhsIsOdd = b.create<arith::AndIOp>(lhs, one);
              auto needsAdd = b.create<arith::CmpIOp>(arith::CmpIPredicate::ne,
                                                      lhsIsOdd, zero);
              auto halvedInput = b.create<arith::SelectOp>(
                  needsAdd, b.create<arith::AddIOp>(lhs, cmod), lhs);
              auto halved = b.create<arith::ShRUIOp>(halvedInput, one);
              auto negatedHalved = b.create<arith::SubIOp>(cmod, halved);
              rewriter.replaceOp(op, isNegated ? negatedHalved : halved);
              return success();
            } else {
              size_t invDegree = i + 1;
              Value invDegreeConst = createScalarOrSplatConstant(
                  b, b.getLoc(), modAttr.getType(), invDegree);
              size_t degreeDelta = twoAdicity - invDegree;
              Value degreeDeltaConst = createScalarOrSplatConstant(
                  b, b.getLoc(), modAttr.getType(), degreeDelta);

              // Create mask for low invDegree bits
              APInt maskVal =
                  APInt::getLowBitsSet(modulus.getBitWidth(), invDegree);
              Value mask = createScalarOrSplatConstant(
                  b, b.getLoc(), modAttr.getType(), maskVal);

              // hi = lhs >> invDegree
              auto hi =
                  b.create<arith::ShRUIOp>(adaptor.getLhs(), invDegreeConst);

              // lo = last invDegree bits of lhs
              auto lo = b.create<arith::AndIOp>(adaptor.getLhs(), mask);

              // loTimesK = lo * k
              Value loTimesK;
              // TODO(batzor): this is temporary optimization for BabyBear. We
              // need to replace this with a more general solution.
              if (k == 15) {
                auto loTimes16 = b.create<arith::ShLIOp>(lo, four);
                loTimesK = b.create<arith::SubIOp>(loTimes16, lo);
              } else {
                loTimesK = b.create<arith::MulIOp>(lo, kConst);
              }

              // loShifted = loTimesK << degreeDelta
              auto loShifted =
                  b.create<arith::ShLIOp>(loTimesK, degreeDeltaConst);

              // loIsNotZero = (lo != 0)
              auto loIsNotZero =
                  b.create<arith::CmpIOp>(arith::CmpIPredicate::ne, lo, zero);

              // loCorrected = loIsNotZero ? loShifted : cmod
              auto loCorrected =
                  b.create<arith::SelectOp>(loIsNotZero, loShifted, cmod);

              // result = loCorrected - hi
              auto result = b.create<arith::SubIOp>(loCorrected, hi);
              auto negatedResult = b.create<arith::SubIOp>(cmod, result);

              // NOTE(batzor): This inverted negation is as intended.
              // WARN(batzor): The output can be modulus when LHS is 0. This is
              // generally safe since all other operations are safe under this
              // range but zero check would fail. This can be fixed after we
              // introduce proper range analysis.
              rewriter.replaceOp(op, isNegated ? result : negatedResult);
              return success();
            }
          }
        }
      }
    }

    if (modType.isMontgomery()) {
      auto result = b.create<MontMulOp>(op.getType(), op.getLhs(), op.getRhs());
      rewriter.replaceOp(op, result);
      return success();
    }

    // Use standard multiplication and reduction
    auto cmodExt =
        b.create<arith::ConstantOp>(modulusAttr(op, /*extended=*/true));
    auto lhs = b.create<arith::ExtUIOp>(modulusType(op, /*extended=*/true),
                                        adaptor.getLhs());
    auto rhs = b.create<arith::ExtUIOp>(modulusType(op, /*extended=*/true),
                                        adaptor.getRhs());
    auto mul = b.create<arith::MulIOp>(lhs, rhs);
    auto remu = b.create<arith::RemUIOp>(mul, cmodExt);
    auto trunc = b.create<arith::TruncIOp>(modulusType(op), remu);

    rewriter.replaceOp(op, trunc);
    return success();
  }
};

struct ConvertMac : public OpConversionPattern<MacOp> {
  explicit ConvertMac(MLIRContext *context)
      : OpConversionPattern<MacOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MacOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    ModArithType modType = getResultModArithType(op);
    if (modType.isMontgomery()) {
      auto mul =
          b.create<arith::MulUIExtendedOp>(adaptor.getLhs(), adaptor.getRhs());
      auto sum =
          b.create<arith::AddUIExtendedOp>(mul.getLow(), adaptor.getAcc());
      auto high = b.create<arith::AddIOp>(mul.getHigh(), sum.getOverflow());
      auto reduced = b.create<MontReduceOp>(op.getType(), sum.getSum(), high);
      rewriter.replaceOp(op, reduced);
      return success();
    } else {
      auto noOverflow = arith::IntegerOverflowFlagsAttr::get(
          b.getContext(),
          arith::IntegerOverflowFlags::nuw | arith::IntegerOverflowFlags::nsw);

      auto cmodExt =
          b.create<arith::ConstantOp>(modulusAttr(op, /*extended=*/true));
      auto x = b.create<arith::ExtUIOp>(cmodExt.getType(), adaptor.getLhs());
      auto y = b.create<arith::ExtUIOp>(cmodExt.getType(), adaptor.getRhs());
      auto acc = b.create<arith::ExtUIOp>(cmodExt.getType(), adaptor.getAcc());
      auto mul = b.create<arith::MulIOp>(x, y);
      auto add = b.create<arith::AddIOp>(mul, acc, noOverflow);
      auto remu = b.create<arith::RemUIOp>(add, cmodExt);
      auto trunc = b.create<arith::TruncIOp>(modulusType(op), remu);

      rewriter.replaceOp(op, trunc);
      return success();
    }
  }
};

struct ConvertMontMul : public OpConversionPattern<MontMulOp> {
  explicit ConvertMontMul(MLIRContext *context)
      : OpConversionPattern<MontMulOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MontMulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    TypedAttr modAttr = modulusAttr(op);
    ModArithType modType = getResultModArithType(op);
    if (!modType.isMontgomery()) {
      return op->emitError(
          "MontMulOp with non-Montgomery type is not supported in "
          "ModArithToArith conversion");
    }
    Value signedLhs = getSignedFormFromCanonical(adaptor.getLhs(), modAttr);
    Value signedRhs = getSignedFormFromCanonical(adaptor.getRhs(), modAttr);
    if (signedLhs && signedRhs) {
      auto mul = b.create<arith::MulSIExtendedOp>(signedLhs, signedRhs);
      auto reduced =
          b.create<MontReduceOp>(op.getType(), mul.getLow(), mul.getHigh());
      rewriter.replaceOp(op, reduced);
      return success();
    } else {
      auto mul =
          b.create<arith::MulUIExtendedOp>(adaptor.getLhs(), adaptor.getRhs());
      auto reduced =
          b.create<MontReduceOp>(op.getType(), mul.getLow(), mul.getHigh());

      rewriter.replaceOp(op, reduced);
      return success();
    }
    return failure();
  }
};

namespace {

struct MulExtendedResult {
  Value lo;
  Value hi;
};

template <typename Op>
MulExtendedResult squareExtended(ImplicitLocOpBuilder &b, Op op, Value input) {
  auto noOverflow = arith::IntegerOverflowFlagsAttr::get(
      b.getContext(),
      arith::IntegerOverflowFlags::nuw | arith::IntegerOverflowFlags::nsw);

  ModArithType modType = getResultModArithType(op);
  IntegerType intType = modType.getStorageType();
  IntegerType intExtType = intType.scaleElementBitwidth(2);

  const unsigned modBitWidth = intType.getWidth();
  const unsigned limbWidth = modBitWidth > APInt::APINT_BITS_PER_WORD
                                 ? APInt::APINT_BITS_PER_WORD
                                 : modBitWidth;
  const unsigned numLimbs = (modBitWidth + limbWidth - 1) / limbWidth;

  MontReducer montReducer(b, modType);
  if (numLimbs == 1) {
    // When squaring, we can just use signed multiplication since the sign will
    // cancel out.
    auto signedInput = getSignedFormFromCanonical(input, modulusAttr(op));
    auto results =
        signedInput
            ? b.create<arith::MulSIExtendedOp>(signedInput, signedInput)
                  .getResults()
            : b.create<arith::MulUIExtendedOp>(input, input).getResults();
    return {results[0], results[1]};
  }

  Type limbType = IntegerType::get(b.getContext(), limbWidth);
  Value zeroLimb = b.create<arith::ConstantIntOp>(limbType, 0);

  auto decomposeToLimbs = [&b, limbType, limbWidth,
                           numLimbs](SmallVector<Value> &limbs, Value input,
                                     Type type) {
    if (numLimbs == 1 && type == limbType) {
      limbs[0] = input;
      return limbs;
    }
    limbs[0] = b.create<arith::TruncIOp>(limbType, input);
    Value remaining = input;
    Value shift = b.create<arith::ConstantIntOp>(type, limbWidth);
    for (unsigned i = 1; i < limbs.size(); ++i) {
      remaining = b.create<arith::ShRUIOp>(remaining, shift);
      limbs[i] = b.create<arith::TruncIOp>(limbType, remaining);
    }
    return limbs;
  };
  SmallVector<Value> limbs(numLimbs);
  decomposeToLimbs(limbs, input, intType);
  SmallVector<Value> resultVec(2 * numLimbs, zeroLimb);
  Value carry = zeroLimb;

  // Calculate x + y * z + carry
  auto mulAddWithCarry = [&b, limbType](Value x, Value y, Value z,
                                        Value carry) {
    auto yz = b.create<arith::MulUIExtendedOp>(y, z);
    Value hi = yz.getHigh();
    Value lo = yz.getLow();
    auto addResult = b.create<arith::AddUIExtendedOp>(x, lo);
    Value carry1 = addResult.getOverflow();
    auto addResult2 =
        b.create<arith::AddUIExtendedOp>(addResult.getSum(), carry);
    Value carry2 = addResult2.getOverflow();
    MulExtendedResult mulResult;
    mulResult.lo = addResult2.getSum();
    mulResult.hi =
        b.create<arith::AddIOp>(hi, b.create<arith::ExtUIOp>(limbType, carry1));
    mulResult.hi = b.create<arith::AddIOp>(
        mulResult.hi, b.create<arith::ExtUIOp>(limbType, carry2));
    return mulResult;
  };

  // Add off-diagonal entries to result buffer
  for (unsigned i = 0; i < numLimbs; ++i) {
    for (unsigned j = i + 1; j < numLimbs; ++j) {
      // (carry, sum) = r[i+j] + a[i] * a[j] + carry
      MulExtendedResult mulResult =
          mulAddWithCarry(resultVec[i + j], limbs[i], limbs[j], carry);
      resultVec[i + j] = mulResult.lo;
      carry = mulResult.hi;
    }
    resultVec[i + numLimbs] = carry;
    carry = zeroLimb;
  }

  // Reconstruct a single integer value by combining all limbs
  Value result = b.create<arith::ConstantIntOp>(intExtType, 0);
  for (unsigned i = 0; i < 2 * numLimbs; ++i) {
    Value rAtI = b.create<arith::ExtUIOp>(intExtType, resultVec[i]);
    Value shifted = b.create<arith::ShLIOp>(
        rAtI, b.create<arith::ConstantIntOp>(intExtType, i * limbWidth));
    result = b.create<arith::OrIOp>(result, shifted);
  }

  // Multiply result by 2. It's safe to assume no overflow
  result = b.create<arith::ShLIOp>(
      result, b.create<arith::ConstantIntOp>(intExtType, 1), noOverflow);

  decomposeToLimbs(resultVec, result, intExtType);

  // Add diagonal entries to result buffer
  for (unsigned i = 0; i < numLimbs; ++i) {
    // (carry, r[2*i]) = r[2*i] + a[i] * a[i] + carry
    MulExtendedResult mulResult =
        mulAddWithCarry(resultVec[2 * i], limbs[i], limbs[i], carry);
    resultVec[2 * i] = mulResult.lo;
    carry = mulResult.hi;

    // (carry, r[2*i+1]) = r[2*i+1] + carry
    auto addResult =
        b.create<arith::AddUIExtendedOp>(resultVec[2 * i + 1], carry);
    resultVec[2 * i + 1] = addResult.getSum();
    carry = b.create<arith::ExtUIOp>(limbType, addResult.getOverflow());
  }

  // Reconstruct `lo` and `hi` values by composing individual limbs
  Value zero = b.create<arith::ConstantIntOp>(intType, 0);
  Value resultLow = zero;
  Value resultHigh = zero;
  for (unsigned i = 0; i < 2 * numLimbs; ++i) {
    Value rAtI = numLimbs == 1
                     ? resultVec[i]
                     : b.create<arith::ExtUIOp>(intType, resultVec[i]);
    if (i < numLimbs) {
      auto shifted = b.create<arith::ShLIOp>(
          rAtI, b.create<arith::ConstantIntOp>(intType, i * limbWidth));
      resultLow = b.create<arith::OrIOp>(resultLow, shifted);
    } else {
      auto shifted = b.create<arith::ShLIOp>(
          rAtI,
          b.create<arith::ConstantIntOp>(intType, (i - numLimbs) * limbWidth));
      resultHigh = b.create<arith::OrIOp>(resultHigh, shifted);
    }
  }
  return MulExtendedResult{resultLow, resultHigh};
}

} // namespace

struct ConvertSquare : public OpConversionPattern<SquareOp> {
  explicit ConvertSquare(MLIRContext *context)
      : OpConversionPattern<SquareOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SquareOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    ModArithType modType = getResultModArithType(op);
    if (modType.isMontgomery()) {
      auto result = b.create<MontSquareOp>(op.getType(), op.getInput());
      rewriter.replaceOp(op, result);
      return success();
    }

    Type intExtType = modulusType(op, /*extended=*/true);
    MulExtendedResult result = squareExtended(b, op, adaptor.getInput());
    Value lowExt = b.create<arith::ExtUIOp>(intExtType, result.lo);
    Value highExt = b.create<arith::ExtUIOp>(intExtType, result.hi);
    Value shift = b.create<arith::ConstantIntOp>(intExtType,
                                                 modType.getStorageBitWidth());
    highExt = b.create<arith::ShLIOp>(highExt, shift);
    Value squared = b.create<arith::OrIOp>(lowExt, highExt);

    Value cmod =
        b.create<arith::ConstantOp>(modulusAttr(op, /*extended=*/true));
    Value remu = b.create<arith::RemUIOp>(squared, cmod);
    Value trunc =
        b.create<arith::TruncIOp>(modulusType(op, /*extended=*/false), remu);
    rewriter.replaceOp(op, trunc);
    return success();
  };
};

struct ConvertMontSquare : public OpConversionPattern<MontSquareOp> {
  explicit ConvertMontSquare(MLIRContext *context)
      : OpConversionPattern<MontSquareOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MontSquareOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    ModArithType modType = getResultModArithType(op);
    if (!modType.isMontgomery()) {
      return op->emitError(
          "MontSquareOp with non-Montgomery type is not supported in "
          "ModArithToArith conversion");
    }
    auto result = squareExtended(b, op, adaptor.getInput());
    auto reduced = b.create<MontReduceOp>(op.getType(), result.lo, result.hi);

    rewriter.replaceOp(op, reduced);
    return success();
  }
};

// TODO(ashjeong): Account for Montgomery domain inputs. Currently only accounts
// for base domain inputs.
struct ConvertCmp : public OpConversionPattern<CmpOp> {
  explicit ConvertCmp(MLIRContext *context)
      : OpConversionPattern<CmpOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CmpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    unsigned outputBitWidth = dyn_cast<ModArithType>(op.getLhs().getType())
                                  .getModulus()
                                  .getValue()
                                  .getBitWidth();
    auto signlessIntType =
        IntegerType::get(b.getContext(), outputBitWidth, IntegerType::Signless);
    auto extractedLHS = b.create<BitcastOp>(signlessIntType, op.getLhs());
    auto extractedRHS = b.create<BitcastOp>(signlessIntType, op.getRhs());

    auto cmpOp =
        b.create<arith::CmpIOp>(op.getPredicate(), extractedLHS, extractedRHS);
    rewriter.replaceOp(op, cmpOp);
    return success();
  }
};

namespace rewrites {
// In an inner namespace to avoid conflicts with canonicalization patterns
#include "zkir/Dialect/ModArith/Conversions/ModArithToArith/ModArithToArith.cpp.inc"
} // namespace rewrites

struct ModArithToArith : impl::ModArithToArithBase<ModArithToArith> {
  using ModArithToArithBase::ModArithToArithBase;

  void runOnOperation() override;
};

void ModArithToArith::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();
  ModArithToArithTypeConverter typeConverter(context);

  ConversionTarget target(*context);
  target.addIllegalDialect<ModArithDialect>();
  target.addLegalDialect<arith::ArithDialect>();

  RewritePatternSet patterns(context);
  rewrites::populateWithGenerated(patterns);
  patterns.add<
      // clang-format off
      ConvertAdd,
      ConvertBitcast,
      ConvertCmp,
      ConvertConstant,
      ConvertDouble,
      ConvertFromMont,
      ConvertInverse,
      ConvertMac,
      ConvertMontInverse,
      ConvertMontMul,
      ConvertMontReduce,
      ConvertMontSquare,
      ConvertMul,
      ConvertNegate,
      ConvertSquare,
      ConvertSub,
      ConvertToMont,
      ConvertAny<affine::AffineApplyOp>,
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
      ConvertAny<field::ExtFromCoeffsOp>,
      ConvertAny<field::ExtToCoeffsOp>,
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
      affine::AffineApplyOp,
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
      field::ExtFromCoeffsOp,
      field::ExtToCoeffsOp,
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

} // namespace mlir::zkir::mod_arith
