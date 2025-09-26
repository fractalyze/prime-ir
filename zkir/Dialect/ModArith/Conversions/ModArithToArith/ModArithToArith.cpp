#include "zkir/Dialect/ModArith/Conversions/ModArithToArith/ModArithToArith.h"

#include <utility>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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
#include "zkir/Dialect/ModArith/IR/ModArithDialect.h"
#include "zkir/Dialect/ModArith/IR/ModArithOps.h"
#include "zkir/Dialect/ModArith/IR/ModArithTypes.h"
#include "zkir/Dialect/TensorExt/IR/TensorExtOps.h"
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

// A helper function to generate the attribute or type
// needed to represent the result of mod_arith op as an integer
// before applying a remainder operation
template <typename Op>
static TypedAttr modulusAttr(Op op, bool mul = false) {
  auto type = op.getType();
  auto modArithType = getResultModArithType(op);
  APInt modulus = modArithType.getModulus().getValue();

  auto width = modulus.getBitWidth();
  if (mul) {
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
static inline Type modulusType(Op op, bool mul = false) {
  return modulusAttr(op, mul).getType();
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
    TypedAttr modAttr = modulusAttr(op);

    // `T` is the operand (e.g. the result of a multiplication, twice the
    // bitwidth of modulus).
    Value tLow = adaptor.getLow();
    Value tHigh = adaptor.getHigh();

    // Extract Montgomery constants:
    // `nPrime` = -n⁻¹ mod R, where R is the base and n is the modulus
    // `bInv` = b⁻¹ mod n = (2ʷ)⁻¹ mod n, where w is the word size (e.g. 64)
    MontgomeryAttr montAttr = getResultModArithType(op).getMontgomeryAttr();
    TypedAttr nPrimeAttr = montAttr.getNPrime();
    TypedAttr bInvAttr = montAttr.getBInv();

    // Retrieve the modulus bitwidth.
    const unsigned modBitWidth =
        cast<IntegerType>(getElementTypeOrSelf(modAttr.getType())).getWidth();

    // Compute number of limbs.
    const unsigned limbWidth = nPrimeAttr.getType().getIntOrFloatBitWidth();
    const unsigned numLimbs = (modBitWidth + limbWidth - 1) / limbWidth;

    // Prepare constants for limb operations.
    Type limbType = nPrimeAttr.getType();
    TypedAttr limbWidthAttr =
        b.getIntegerAttr(getElementTypeOrSelf(tLow), limbWidth);
    TypedAttr limbMaskAttr =
        b.getIntegerAttr(getElementTypeOrSelf(tLow),
                         APInt::getAllOnes(limbWidth).zext(modBitWidth));
    TypedAttr limbShiftAttr = b.getIntegerAttr(getElementTypeOrSelf(tLow),
                                               (numLimbs - 1) * limbWidth);
    TypedAttr oneAttr = b.getIntegerAttr(getElementTypeOrSelf(tLow), 1);

    arith::IntegerOverflowFlags overflowFlag(arith::IntegerOverflowFlags::nuw &
                                             arith::IntegerOverflowFlags::nsw);
    auto noOverflow =
        arith::IntegerOverflowFlagsAttr::get(b.getContext(), overflowFlag);

    // Splat the attributes to match the shape of `tLow`.
    if (auto shapedType = dyn_cast<ShapedType>(tLow.getType())) {
      limbType = shapedType.cloneWith(std::nullopt, limbType);
      nPrimeAttr =
          SplatElementsAttr::get(cast<ShapedType>(limbType), nPrimeAttr);
      limbWidthAttr = SplatElementsAttr::get(shapedType, limbWidthAttr);
      limbMaskAttr = SplatElementsAttr::get(shapedType, limbMaskAttr);
      limbShiftAttr = SplatElementsAttr::get(shapedType, limbShiftAttr);
      modAttr = isa<VectorType>(modAttr.getType())
                    ? modAttr
                    : SplatElementsAttr::get(shapedType, modAttr);
      bInvAttr = SplatElementsAttr::get(shapedType, bInvAttr);
      oneAttr = SplatElementsAttr::get(shapedType, oneAttr);
    }

    // Create constants for the Montgomery reduction.
    auto nPrimeConst = b.create<arith::ConstantOp>(nPrimeAttr);
    auto limbWidthConst = b.create<arith::ConstantOp>(limbWidthAttr);
    auto limbMaskConst = b.create<arith::ConstantOp>(limbMaskAttr);
    auto limbShiftConst = b.create<arith::ConstantOp>(limbShiftAttr);
    auto modConst = b.create<arith::ConstantOp>(modAttr);
    auto bInvConst = b.create<arith::ConstantOp>(bInvAttr);
    auto oneConst = b.create<arith::ConstantOp>(oneAttr);

    // If the number of limbs is 1, the 2ʷ is larger than the modulus, so we
    // can use `nInv` instead of `nPrime` and avoid carry check.
    if (numLimbs == 1) {
      TypedAttr nInvAttr = montAttr.getNInv();
      if (isa<ShapedType>(tLow.getType())) {
        nInvAttr = SplatElementsAttr::get(cast<ShapedType>(limbType), nInvAttr);
      }
      auto nInvConst = b.create<arith::ConstantOp>(nInvAttr);
      // Compute `m` = `tLow` * `nInv` (mod `base`)
      auto m = b.create<arith::MulIOp>(tLow, nInvConst);
      // Compute `m` * `n`
      auto mN = b.create<arith::MulUIExtendedOp>(m, modConst);

      // Calculate `T` - `mN`, which should result in zeroed low limb since it
      // should be divisible by `base`. The low part subtraction cannot
      // underflow since if `tLow` < `mN.getLow()`, then `tLow` -
      // `mN.getLow()` cannot result in zero low limb. This means, `tLow` is
      // always equal to `mN.getLow()` so we can skip the low subtractions.
      // The reduction result will be just `tHigh` - `mN.getHigh()` mod n.
      auto underflow = b.create<arith::CmpIOp>(arith::CmpIPredicate::ult, tHigh,
                                               mN.getHigh());
      auto sub = b.create<arith::SubIOp>(tHigh, mN.getHigh());
      auto addIfUnderflow = b.create<arith::AddIOp>(sub, modConst);
      auto result = b.create<arith::SelectOp>(underflow, addIfUnderflow, sub);
      rewriter.replaceOp(op, result);
      return success();
    }

    // Because the number of limbs (`numLimbs`) is known at compile time, we can
    // unroll the loop as a straight-line chain of operations.
    // The result of the `i`th iteration is `T` * b⁻¹ mod n.
    for (unsigned i = 0; i < numLimbs - 1; ++i) {
      // Shift `T` right by `limbWidth`.
      Value freeCoeff = b.create<arith::AndIOp>(tLow, limbMaskConst);
      tLow = b.create<arith::ShRUIOp>(tLow, limbWidthConst);
      Value tHighLowerLimb = b.create<arith::ShLIOp>(tHigh, limbShiftConst);
      tLow = b.create<arith::OrIOp>(tLow, tHighLowerLimb);
      tHigh = b.create<arith::ShRUIOp>(tHigh, limbWidthConst);

      // Compute `m` = `freeCoeff` * (b⁻¹ mod n) and add to `T`.
      auto m = b.create<arith::MulUIExtendedOp>(freeCoeff, bInvConst);
      tLow = b.create<arith::AddIOp>(tLow, m.getLow());
      Value carry =
          b.create<arith::CmpIOp>(arith::CmpIPredicate::ult, tLow, m.getLow());
      tHigh = b.create<arith::AddIOp>(tHigh, m.getHigh(), noOverflow);
      auto tHighPlusOne = b.create<arith::AddIOp>(tHigh, oneConst, noOverflow);
      tHigh = b.create<arith::SelectOp>(carry, tHighPlusOne, tHigh);
    }
    // The last iteration is the same as normal Montgomery reduction.
    Value freeCoeff = b.create<arith::TruncIOp>(limbType, tLow);
    // Compute `m` = `freeCoeff` * `nPrime` (mod `base`)
    auto m = b.create<arith::MulIOp>(freeCoeff, nPrimeConst);
    // Compute `m` * `n`
    Value mExt = b.create<arith::ExtUIOp>(tLow.getType(), m);
    auto mN = b.create<arith::MulUIExtendedOp>(modConst, mExt);
    // Add the product to `T`.
    tLow = b.create<arith::AddIOp>(tLow, mN.getLow());
    auto carry =
        b.create<arith::CmpIOp>(arith::CmpIPredicate::ult, tLow, mN.getLow());
    tHigh = b.create<arith::AddIOp>(tHigh, mN.getHigh(), noOverflow);
    auto tHighPlusOne = b.create<arith::AddIOp>(tHigh, oneConst, noOverflow);
    tHigh = b.create<arith::SelectOp>(carry, tHighPlusOne, tHigh);
    // Shift right `T` by `limbWidth` to discard the zeroed limb.
    tLow = b.create<arith::ShRUIOp>(tLow, limbWidthConst);
    tHigh = b.create<arith::ShLIOp>(tHigh, limbShiftConst);
    tLow = b.create<arith::OrIOp>(tLow, tHigh);

    // Final conditional subtraction: if (`tLow` >= `modulus`) then subtract
    // `modulus`.
    auto cmp =
        b.create<arith::CmpIOp>(arith::CmpIPredicate::uge, tLow, modConst);
    auto sub = b.create<arith::SubIOp>(tLow, modConst, noOverflow);
    auto result = b.create<arith::SelectOp>(cmp, sub, tLow);

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

    ModArithType resultType = getResultModArithType(op);
    MontgomeryAttr montAttr = resultType.getMontgomeryAttr();
    TypedAttr rSquaredAttr = montAttr.getRSquared();
    if (auto shapedType = dyn_cast<ShapedType>(op.getType())) {
      auto intShapedType = shapedType.cloneWith(
          std::nullopt, typeConverter->convertType(resultType));
      rSquaredAttr = SplatElementsAttr::get(intShapedType, rSquaredAttr);
    }
    // x * R = REDC(x * rSquared)
    auto rSquared = b.create<arith::ConstantOp>(rSquaredAttr);
    auto product =
        b.create<arith::MulUIExtendedOp>(adaptor.getInput(), rSquared);
    auto reduced =
        b.create<MontReduceOp>(resultType, product.getLow(), product.getHigh());
    rewriter.replaceOp(op, reduced);
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

    ModArithType resultType = getResultModArithType(op);
    TypedAttr zeroAttr =
        b.getIntegerAttr(typeConverter->convertType(resultType), 0);
    if (auto shapedType = dyn_cast<ShapedType>(op.getType())) {
      auto intShapedType = shapedType.cloneWith(
          std::nullopt, typeConverter->convertType(resultType));
      zeroAttr = SplatElementsAttr::get(intShapedType, zeroAttr);
    }

    // x * R⁻¹ = REDC(x)
    auto zeroHighConst = b.create<arith::ConstantOp>(zeroAttr);
    auto reduced =
        b.create<MontReduceOp>(resultType, adaptor.getInput(), zeroHighConst);

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

    ModArithType resultType = getResultModArithType(op);
    if (resultType.isMontgomery()) {
      auto result = b.create<MontInverseOp>(resultType, op.getInput());
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

    ModArithType resultType = getResultModArithType(op);
    if (!resultType.isMontgomery()) {
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

    arith::IntegerOverflowFlags overflowFlag(arith::IntegerOverflowFlags::nuw &
                                             arith::IntegerOverflowFlags::nsw);
    auto noOverflow =
        arith::IntegerOverflowFlagsAttr::get(b.getContext(), overflowFlag);

    auto cmod = b.create<arith::ConstantOp>(modulusAttr(op));
    auto add =
        b.create<arith::AddIOp>(adaptor.getLhs(), adaptor.getRhs(), noOverflow);
    auto ifge = b.create<arith::CmpIOp>(arith::CmpIPredicate::uge, add, cmod);
    auto sub = b.create<arith::SubIOp>(add, cmod, noOverflow);
    auto select = b.create<arith::SelectOp>(ifge, sub, add);

    rewriter.replaceOp(op, select);
    return success();
  }
};

struct ConvertDouble : public OpConversionPattern<DoubleOp> {
  explicit ConvertDouble(mlir::MLIRContext *context)
      : OpConversionPattern<DoubleOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DoubleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    ModArithType modType = getResultModArithType(op);
    auto intType = modType.getStorageType();

    Value cmod = b.create<arith::ConstantOp>(modulusAttr(op));
    Value one = b.create<arith::ConstantIntOp>(intType, 1);
    auto shifted = b.create<arith::ShLIOp>(adaptor.getInput(), one);
    auto ifge =
        b.create<arith::CmpIOp>(arith::CmpIPredicate::uge, shifted, cmod);
    auto sub = b.create<arith::SubIOp>(shifted, cmod);
    auto select = b.create<arith::SelectOp>(ifge, sub, shifted);

    rewriter.replaceOp(op, select);
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
    arith::IntegerOverflowFlags overflowFlag(arith::IntegerOverflowFlags::nuw &
                                             arith::IntegerOverflowFlags::nsw);
    auto noOverflow =
        arith::IntegerOverflowFlagsAttr::get(b.getContext(), overflowFlag);

    auto cmod = b.create<arith::ConstantOp>(modulusAttr(op));
    auto sub =
        b.create<arith::SubIOp>(adaptor.getLhs(), adaptor.getRhs(), noOverflow);
    auto add = b.create<arith::AddIOp>(sub, cmod, noOverflow);
    auto ifge = b.create<arith::CmpIOp>(arith::CmpIPredicate::uge,
                                        adaptor.getLhs(), adaptor.getRhs());
    auto select = b.create<arith::SelectOp>(ifge, sub, add);

    rewriter.replaceOp(op, select);
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

    ModArithType resultType = getResultModArithType(op);
    if (resultType.isMontgomery()) {
      auto result = b.create<MontMulOp>(resultType, op.getLhs(), op.getRhs());
      rewriter.replaceOp(op, result);
      return success();
    }
    auto cmod = b.create<arith::ConstantOp>(modulusAttr(op, true));
    auto lhs =
        b.create<arith::ExtUIOp>(modulusType(op, true), adaptor.getLhs());
    auto rhs =
        b.create<arith::ExtUIOp>(modulusType(op, true), adaptor.getRhs());
    auto mul = b.create<arith::MulIOp>(lhs, rhs);
    auto remu = b.create<arith::RemUIOp>(mul, cmod);
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
    ModArithType resultType = getResultModArithType(op);
    if (resultType.isMontgomery()) {
      auto mul =
          b.create<arith::MulUIExtendedOp>(adaptor.getLhs(), adaptor.getRhs());
      auto sum =
          b.create<arith::AddUIExtendedOp>(mul.getLow(), adaptor.getAcc());
      auto high = b.create<arith::AddIOp>(mul.getHigh(), sum.getOverflow());
      auto reduced =
          b.create<mod_arith::MontReduceOp>(resultType, sum.getSum(), high);
      rewriter.replaceOp(op, reduced);
      return success();
    } else {
      arith::IntegerOverflowFlags overflowFlag(
          arith::IntegerOverflowFlags::nuw & arith::IntegerOverflowFlags::nsw);
      auto noOverflow =
          arith::IntegerOverflowFlagsAttr::get(b.getContext(), overflowFlag);

      auto cmod = b.create<arith::ConstantOp>(modulusAttr(op, true));
      auto x =
          b.create<arith::ExtUIOp>(modulusType(op, true), adaptor.getLhs());
      auto y =
          b.create<arith::ExtUIOp>(modulusType(op, true), adaptor.getRhs());
      auto acc =
          b.create<arith::ExtUIOp>(modulusType(op, true), adaptor.getAcc());
      auto mul = b.create<arith::MulIOp>(x, y);
      auto add = b.create<arith::AddIOp>(mul, acc, noOverflow);
      auto remu = b.create<arith::RemUIOp>(add, cmod);
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

    ModArithType resultType = getResultModArithType(op);
    if (!resultType.isMontgomery()) {
      return op->emitError(
          "MontMulOp with non-Montgomery type is not supported in "
          "ModArithToArith conversion");
    }
    auto mul =
        b.create<arith::MulUIExtendedOp>(adaptor.getLhs(), adaptor.getRhs());
    auto reduced = b.create<mod_arith::MontReduceOp>(
        getResultModArithType(op), mul.getLow(), mul.getHigh());

    rewriter.replaceOp(op, reduced);
    return success();
  }
};

namespace {

struct MulExtendedResult {
  Value lo;
  Value hi;
};

template <typename Op>
MulExtendedResult squareExtended(ImplicitLocOpBuilder &b, Op op, Value input) {
  arith::IntegerOverflowFlags overflowFlag(arith::IntegerOverflowFlags::nuw &
                                           arith::IntegerOverflowFlags::nsw);
  auto noOverflow =
      arith::IntegerOverflowFlagsAttr::get(b.getContext(), overflowFlag);

  Type intType = modulusType(op, /*mul=*/false);
  Type resultType = modulusType(op, /*mul=*/true);

  const unsigned modBitWidth = cast<IntegerType>(intType).getWidth();
  const unsigned limbWidth = modBitWidth > APInt::APINT_BITS_PER_WORD
                                 ? APInt::APINT_BITS_PER_WORD
                                 : modBitWidth;
  const unsigned numLimbs = (modBitWidth + limbWidth - 1) / limbWidth;

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
  auto mulAddWithCarry = [&b, limbType](mlir::Value x, mlir::Value y,
                                        mlir::Value z, mlir::Value carry) {
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
  Value result = b.create<arith::ConstantIntOp>(resultType, 0);
  for (unsigned i = 0; i < 2 * numLimbs; ++i) {
    Value rAtI = b.create<arith::ExtUIOp>(resultType, resultVec[i]);
    Value shifted = b.create<arith::ShLIOp>(
        rAtI, b.create<arith::ConstantIntOp>(resultType, i * limbWidth));
    result = b.create<arith::OrIOp>(result, shifted);
  }

  // Multiply result by 2. It's safe to assume no overflow
  result = b.create<arith::ShLIOp>(
      result, b.create<arith::ConstantIntOp>(resultType, 1), noOverflow);

  decomposeToLimbs(resultVec, result, resultType);

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
  explicit ConvertSquare(mlir::MLIRContext *context)
      : OpConversionPattern<SquareOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SquareOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    ModArithType resultType = getResultModArithType(op);
    if (resultType.isMontgomery()) {
      auto result =
          b.create<mod_arith::MontSquareOp>(resultType, op.getInput());
      rewriter.replaceOp(op, result);
      return success();
    }

    Type mulResultType = modulusType(op, /*mul=*/true);
    MulExtendedResult result = squareExtended(b, op, adaptor.getInput());
    Value lowExt = b.create<arith::ExtUIOp>(mulResultType, result.lo);
    Value highExt = b.create<arith::ExtUIOp>(mulResultType, result.hi);
    Value shift = b.create<arith::ConstantIntOp>(
        mulResultType, resultType.getStorageBitWidth());
    highExt = b.create<arith::ShLIOp>(highExt, shift);
    Value squared = b.create<arith::OrIOp>(lowExt, highExt);

    Value cmod = b.create<arith::ConstantOp>(modulusAttr(op, /*mul=*/true));
    Value remu = b.create<arith::RemUIOp>(squared, cmod);
    Value trunc =
        b.create<arith::TruncIOp>(modulusType(op, /*mul=*/false), remu);
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

    ModArithType resultType = getResultModArithType(op);
    if (!resultType.isMontgomery()) {
      return op->emitError(
          "MontSquareOp with non-Montgomery type is not supported in "
          "ModArithToArith conversion");
    }
    auto result = squareExtended(b, op, adaptor.getInput());
    auto reduced = b.create<mod_arith::MontReduceOp>(getResultModArithType(op),
                                                     result.lo, result.hi);

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
    auto extractedLHS =
        b.create<mod_arith::BitcastOp>(signlessIntType, op.getLhs());
    auto extractedRHS =
        b.create<mod_arith::BitcastOp>(signlessIntType, op.getRhs());

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
      ConvertAny<tensor::ReshapeOp>,
      ConvertAny<tensor_ext::BitReverseOp>
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
      tensor::ReshapeOp,
      tensor_ext::BitReverseOp
      // clang-format on
      >([&](auto op) { return typeConverter.isLegal(op); });

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

} // namespace mlir::zkir::mod_arith
