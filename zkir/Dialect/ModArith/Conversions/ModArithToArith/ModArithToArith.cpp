#include "zkir/Dialect/ModArith/Conversions/ModArithToArith/ModArithToArith.h"

#include <utility>

#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
#include "zkir/Dialect/ModArith/IR/ModArithDialect.h"
#include "zkir/Dialect/ModArith/IR/ModArithOps.h"
#include "zkir/Dialect/ModArith/IR/ModArithTypes.h"
#include "zkir/Utils/ConversionUtils.h"

namespace mlir::zkir::mod_arith {

#define GEN_PASS_DEF_MODARITHTOARITH
#include "zkir/Dialect/ModArith/Conversions/ModArithToArith/ModArithToArith.h.inc"

static IntegerType convertModArithType(ModArithType type) {
  APInt modulus = type.getModulus().getValue();
  return IntegerType::get(type.getContext(), modulus.getBitWidth());
}

static Type convertModArithLikeType(ShapedType type) {
  if (auto modArithType = dyn_cast<ModArithType>(type.getElementType())) {
    return type.cloneWith(type.getShape(), convertModArithType(modArithType));
  }
  return type;
}

class ModArithToArithTypeConverter : public TypeConverter {
 public:
  explicit ModArithToArithTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion(
        [](ModArithType type) -> Type { return convertModArithType(type); });
    addConversion(
        [](ShapedType type) -> Type { return convertModArithLikeType(type); });
  }
};

// A helper function to generate the attribute or type
// needed to represent the result of mod_arith op as an integer
// before applying a remainder operation
template <typename Op>
static TypedAttr modulusAttr(Op op, bool mul = false) {
  auto type = op.getResult().getType();
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

struct ConvertEncapsulate : public OpConversionPattern<EncapsulateOp> {
  explicit ConvertEncapsulate(MLIRContext *context)
      : OpConversionPattern<EncapsulateOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      EncapsulateOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getOperands()[0]);
    return success();
  }
};

struct ConvertExtract : public OpConversionPattern<ExtractOp> {
  explicit ConvertExtract(MLIRContext *context)
      : OpConversionPattern<ExtractOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ExtractOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getOperands()[0]);
    return success();
  }
};

struct ConvertConstant : public OpConversionPattern<ConstantOp> {
  explicit ConvertConstant(MLIRContext *context)
      : OpConversionPattern<ConstantOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ConstantOp op, OpAdaptor adaptor,
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

  LogicalResult matchAndRewrite(
      NegateOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto cmod = b.create<arith::ConstantOp>(modulusAttr(op));
    auto sub = b.create<arith::SubIOp>(cmod, adaptor.getOperands()[0]);
    rewriter.replaceOp(op, sub);
    return success();
  }
};

struct ConvertReduce : public OpConversionPattern<ReduceOp> {
  explicit ConvertReduce(MLIRContext *context)
      : OpConversionPattern<ReduceOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ReduceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    arith::IntegerOverflowFlags overflowFlag(arith::IntegerOverflowFlags::nuw &
                                             arith::IntegerOverflowFlags::nsw);
    auto noOverflow =
        arith::IntegerOverflowFlagsAttr::get(b.getContext(), overflowFlag);

    auto cmod = b.create<arith::ConstantOp>(modulusAttr(op));
    // ModArithType ensures cmod can be correctly interpreted as a signed number
    auto rems = b.create<arith::RemSIOp>(adaptor.getOperands()[0], cmod);
    auto add = b.create<arith::AddIOp>(rems, cmod, noOverflow);
    // TODO(google/heir #710): better with a subifge
    auto remu = b.create<arith::RemUIOp>(add, cmod);
    rewriter.replaceOp(op, remu);
    return success();
  }
};

struct ConvertMontReduce : public OpConversionPattern<MontReduceOp> {
  explicit ConvertMontReduce(MLIRContext *context)
      : OpConversionPattern<MontReduceOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      MontReduceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // `T` is the operand (e.g. the result of a multiplication, twice the
    // bitwidth of modulus).
    Value tLow = adaptor.getOperands()[0];
    Value tHigh = adaptor.getOperands()[1];

    // Extract Montgomery constants: `nPrime` and `modulus`.
    TypedAttr nPrimeAttr = op.getMontgomeryAttr().getNPrime();
    TypedAttr modAttr = modulusAttr(op);

    // Retrieve the modulus bitwidth.
    unsigned modBitWidth = cast<IntegerType>(modAttr.getType()).getWidth();

    // Compute number of limbs.
    const unsigned limbWidth = APInt::APINT_BITS_PER_WORD;
    unsigned numLimbs = (modBitWidth + limbWidth - 1) / limbWidth;

    // Prepare constants for limb operations.
    Type limbType = nPrimeAttr.getType();
    TypedAttr limbWidthAttr =
        b.getIntegerAttr(getElementTypeOrSelf(tLow), limbWidth);
    TypedAttr limbMaskAttr =
        b.getIntegerAttr(getElementTypeOrSelf(tLow),
                         APInt::getAllOnes(limbWidth).zext(modBitWidth));
    TypedAttr limbShiftAttr = b.getIntegerAttr(getElementTypeOrSelf(tLow),
                                               (numLimbs - 1) * limbWidth);

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
      modAttr = SplatElementsAttr::get(shapedType, modAttr);
    }

    // Create constants for the Montgomery reduction.
    auto nPrimeConst = b.create<arith::ConstantOp>(nPrimeAttr);
    auto limbWidthConst = b.create<arith::ConstantOp>(limbWidthAttr);
    auto limbMaskConst = b.create<arith::ConstantOp>(limbMaskAttr);
    auto limbShiftConst = b.create<arith::ConstantOp>(limbShiftAttr);
    auto modConst = b.create<arith::ConstantOp>(modAttr);

    // Because the number of limbs (`numLimbs`) is known at compile time, we can
    // unroll the loop as a straight-line chain of operations.
    for (unsigned i = 0; i < numLimbs; ++i) {
      // Extract the current lowest limb: `tLow` (mod `base`)
      auto lowerLimb = b.create<arith::TruncIOp>(limbType, tLow);
      // Compute `m` = `lowerLimb` * `nPrime` (mod `base`)
      auto m = b.create<arith::MulIOp>(lowerLimb, nPrimeConst);
      // Compute `m` * `N` , where `N` is modulus
      auto mExt = b.create<arith::ExtUIOp>(tLow.getType(), m);
      auto mN = b.create<arith::MulUIExtendedOp>(modConst, mExt);
      // Add the product to `T`.
      auto sum = b.create<arith::AddUIExtendedOp>(tLow, mN.getLow());
      tLow = sum.getSum();
      tHigh = b.create<arith::AddIOp>(tHigh, mN.getHigh(), noOverflow);
      // Add carry from the `sum` to `tHigh`.
      auto carryExt =
          b.create<arith::ExtUIOp>(tHigh.getType(), sum.getOverflow());
      tHigh = b.create<arith::AddIOp>(tHigh, carryExt, noOverflow);
      // Shift right by `limbWidth` to discard the zeroed limb.
      tLow = b.create<arith::ShRUIOp>(tLow, limbWidthConst);
      // copy the lowest limb of `tHigh` to the highest limb of `tLow`
      Value tHighLimb = b.create<arith::AndIOp>(tHigh, limbMaskConst);
      tHighLimb = b.create<arith::ShLIOp>(tHighLimb, limbShiftConst);
      tLow = b.create<arith::OrIOp>(tLow, tHighLimb);
      // Shift right `tHigh` by `limbWidth`.
      tHigh = b.create<arith::ShRUIOp>(tHigh, limbWidthConst);
    }

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

  LogicalResult matchAndRewrite(
      ToMontOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // x * R = REDC(x * rSquared)
    auto rSquared =
        b.create<arith::ConstantOp>(op.getMontgomery().getRSquared());
    auto product =
        b.create<arith::MulUIExtendedOp>(adaptor.getOperands()[0], rSquared);
    auto reduced =
        b.create<MontReduceOp>(op.getResult().getType(), product.getLow(),
                               product.getHigh(), op.getMontgomery());
    rewriter.replaceOp(op, reduced);
    return success();
  }
};

struct ConvertFromMont : public OpConversionPattern<FromMontOp> {
  explicit ConvertFromMont(MLIRContext *context)
      : OpConversionPattern<FromMontOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      FromMontOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // x * R⁻¹ = REDC(x)
    auto zeroHighConst = b.create<arith::ConstantOp>(
        IntegerAttr::get(op.getMontgomery().getRSquared().getType(), 0));
    auto reduced = b.create<MontReduceOp>(op.getResult().getType(),
                                          adaptor.getOperands()[0],
                                          zeroHighConst, op.getMontgomery());
    rewriter.replaceOp(op, reduced);
    return success();
  }
};

struct ConvertInverse : public OpConversionPattern<InverseOp> {
  explicit ConvertInverse(MLIRContext *context)
      : OpConversionPattern<InverseOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      InverseOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // TODO(batzor): Support tensor input.
    if (isa<ShapedType>(op.getInput().getType())) {
      return op->emitError("tensor input not supported");
    }

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto operand = adaptor.getInput();
    auto modArithType = getResultModArithType(op);
    auto modulus = modArithType.getModulus();
    auto resultType = modulus.getType();

    Value zero = b.create<arith::ConstantOp>(IntegerAttr::get(resultType, 0));
    Value one = b.create<arith::ConstantOp>(IntegerAttr::get(resultType, 1));
    Value r0 = b.create<arith::ConstantOp>(
        IntegerAttr::get(resultType, modulus.getValue()));
    Value r1 = operand;

    // Prepare the initial values vector.
    SmallVector<Value, 4> initValues = {r0, r1, zero, one};
    // Create a vector of types corresponding to the initial values.
    SmallVector<Type, 4> resultTypes;
    for (Value v : initValues) resultTypes.push_back(v.getType());

    auto whileOp = b.create<scf::WhileOp>(
        resultTypes, initValues,
        /*beforeBuilder =*/
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          ImplicitLocOpBuilder beforeBuilder(loc, builder);

          // Condition: continue while r1 != 0
          Value cond = beforeBuilder.create<arith::CmpIOp>(
              arith::CmpIPredicate::ne, args[1], zero);
          beforeBuilder.create<scf::ConditionOp>(cond, args);
        },
        /*afterBuilder =*/
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          ImplicitLocOpBuilder builderAfter(loc, builder);

          // Extract current values: r0, r1, t0, t1.
          Value currR0 = args[0];
          Value currR1 = args[1];
          Value currT0 = args[2];
          Value currT1 = args[3];
          // Compute quotient: q = r0 / r1.
          Value q = builderAfter.create<arith::DivUIOp>(currR0, currR1);
          // Compute new remainder: newR = r0 % r1.
          Value newR = builderAfter.create<arith::RemUIOp>(currR0, currR1);
          // Compute new coefficient: newT = t0 - (t1 * q).
          Value prod = builderAfter.create<arith::MulIOp>(currT1, q);
          Value newT = builderAfter.create<arith::SubIOp>(currT0, prod);
          // Yield updated loop-carried values: (r1, newR, t1, newT).
          builderAfter.create<scf::YieldOp>(
              ValueRange({currR1, newR, currT1, newT}));
        });

    // After the loop, final values are:
    //   finalR0 = gcd(x, mod) and finalT = candidate inverse.
    Value finalR0 = whileOp.getResult(0);
    Value finalT = whileOp.getResult(2);

    // Check if the gcd is 1 (i.e. the inverse exists).
    Value gcdIsOne =
        b.create<arith::CmpIOp>(arith::CmpIPredicate::eq, finalR0, one);
    auto ifOp = b.create<scf::IfOp>(resultType, gcdIsOne, /*withElse=*/true);

    // Then branch: inverse exists.
    {
      Block *thenBlock = &ifOp.getThenRegion().front();
      ImplicitLocOpBuilder bThen(op.getLoc(), rewriter);
      bThen.setInsertionPointToEnd(thenBlock);
      // If finalT is negative, adjust it by adding the modulus.
      Value isNeg =
          bThen.create<arith::CmpIOp>(arith::CmpIPredicate::slt, finalT, zero);
      auto innerIf =
          bThen.create<scf::IfOp>(resultType, isNeg, /*withElse=*/true);
      {
        Block *innerThen = &innerIf.getThenRegion().front();
        ImplicitLocOpBuilder bInnerThen(op.getLoc(), rewriter);
        bInnerThen.setInsertionPointToEnd(innerThen);
        Value posInv = bInnerThen.create<arith::AddIOp>(
            op.getLoc(), finalT, bInnerThen.create<arith::ConstantOp>(modulus));
        bInnerThen.create<scf::YieldOp>(posInv);
      }
      {
        Block *innerElse = &innerIf.getElseRegion().front();
        ImplicitLocOpBuilder bInnerElse(op.getLoc(), rewriter);
        bInnerElse.setInsertionPointToEnd(innerElse);
        bInnerElse.create<scf::YieldOp>(finalT);
      }
      bThen.create<scf::YieldOp>(innerIf.getResult(0));
    }
    // Else branch: inverse does not exist, so return 0.
    {
      Block *elseBlock = &ifOp.getElseRegion().front();
      ImplicitLocOpBuilder bElse(op.getLoc(), rewriter);
      bElse.setInsertionPointToEnd(elseBlock);
      bElse.create<scf::YieldOp>(zero);
    }

    rewriter.replaceOp(op, ifOp.getResult(0));
    return success();
  }
};

// It is assumed inputs are canonical representatives
// ModArithType ensures add/sub result can not overflow
struct ConvertAdd : public OpConversionPattern<AddOp> {
  explicit ConvertAdd(MLIRContext *context)
      : OpConversionPattern<AddOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      AddOp op, OpAdaptor adaptor,
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

struct ConvertSub : public OpConversionPattern<SubOp> {
  explicit ConvertSub(MLIRContext *context)
      : OpConversionPattern<SubOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SubOp op, OpAdaptor adaptor,
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

  LogicalResult matchAndRewrite(
      MulOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

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

  LogicalResult matchAndRewrite(
      MacOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    arith::IntegerOverflowFlags overflowFlag(arith::IntegerOverflowFlags::nuw &
                                             arith::IntegerOverflowFlags::nsw);
    auto noOverflow =
        arith::IntegerOverflowFlagsAttr::get(b.getContext(), overflowFlag);

    auto cmod = b.create<arith::ConstantOp>(modulusAttr(op, true));
    auto x = b.create<arith::ExtUIOp>(modulusType(op, true),
                                      adaptor.getOperands()[0]);
    auto y = b.create<arith::ExtUIOp>(modulusType(op, true),
                                      adaptor.getOperands()[1]);
    auto acc = b.create<arith::ExtUIOp>(modulusType(op, true),
                                        adaptor.getOperands()[2]);
    auto mul = b.create<arith::MulIOp>(x, y);
    auto add = b.create<arith::AddIOp>(mul, acc, noOverflow);
    auto remu = b.create<arith::RemUIOp>(add, cmod);
    auto trunc = b.create<arith::TruncIOp>(modulusType(op), remu);

    rewriter.replaceOp(op, trunc);
    return success();
  }
};

struct ConvertMontMul : public OpConversionPattern<MontMulOp> {
  explicit ConvertMontMul(MLIRContext *context)
      : OpConversionPattern<MontMulOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      MontMulOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto mul =
        b.create<arith::MulUIExtendedOp>(adaptor.getLhs(), adaptor.getRhs());
    auto reduced = b.create<mod_arith::MontReduceOp>(
        getResultModArithType(op), mul.getLow(), mul.getHigh(),
        op.getMontgomery());

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

  LogicalResult matchAndRewrite(
      CmpOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    unsigned outputBitWidth = dyn_cast<ModArithType>(op.getLhs().getType())
                                  .getModulus()
                                  .getValue()
                                  .getBitWidth();
    auto signlessIntType =
        IntegerType::get(b.getContext(), outputBitWidth, IntegerType::Signless);
    auto extractedLHS =
        b.create<mod_arith::ExtractOp>(signlessIntType, op.getLhs());
    auto extractedRHS =
        b.create<mod_arith::ExtractOp>(signlessIntType, op.getRhs());

    auto cmpOp =
        b.create<arith::CmpIOp>(op.getPredicate(), extractedLHS, extractedRHS);
    rewriter.replaceOp(op, cmpOp);
    return success();
  }
};

namespace rewrites {
// In an inner namespace to avoid conflicts with canonicalization patterns
#include "zkir/Dialect/ModArith/Conversions/ModArithToArith/ModArithToArith.cpp.inc"
}  // namespace rewrites

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
  patterns
      .add<ConvertNegate, ConvertEncapsulate, ConvertExtract, ConvertReduce,
           ConvertMontReduce, ConvertToMont, ConvertFromMont, ConvertAdd,
           ConvertSub, ConvertMul, ConvertMontMul, ConvertMac, ConvertCmp,
           ConvertConstant, ConvertInverse, ConvertAny<affine::AffineForOp>,
           ConvertAny<affine::AffineParallelOp>,
           ConvertAny<affine::AffineLoadOp>, ConvertAny<affine::AffineApplyOp>,
           ConvertAny<affine::AffineStoreOp>, ConvertAny<affine::AffineYieldOp>,
           ConvertAny<bufferization::MaterializeInDestinationOp>,
           ConvertAny<bufferization::ToMemrefOp>,
           ConvertAny<bufferization::ToTensorOp>, ConvertAny<linalg::GenericOp>,
           ConvertAny<linalg::MapOp>, ConvertAny<linalg::YieldOp>,
           ConvertAny<memref::LoadOp>, ConvertAny<memref::StoreOp>,
           ConvertAny<tensor::CastOp>, ConvertAny<tensor::ExtractOp>,
           ConvertAny<tensor::ExtractSliceOp>,
           ConvertAny<tensor::InsertSliceOp>, ConvertAny<tensor::EmptyOp>,
           ConvertAny<tensor::FromElementsOp>, ConvertAny<tensor::ConcatOp>,
           ConvertAny<tensor::ReshapeOp>, ConvertAny<tensor::InsertOp>>(
          typeConverter, context);

  addStructuralConversionPatterns(typeConverter, patterns, target);

  target.addDynamicallyLegalOp<
      affine::AffineForOp, affine::AffineParallelOp, affine::AffineLoadOp,
      affine::AffineApplyOp, affine::AffineStoreOp, affine::AffineYieldOp,
      bufferization::MaterializeInDestinationOp, bufferization::ToMemrefOp,
      bufferization::ToTensorOp, linalg::GenericOp, linalg::MapOp,
      linalg::YieldOp, memref::LoadOp, memref::StoreOp, tensor::CastOp,
      tensor::ExtractOp, tensor::ExtractSliceOp, tensor::InsertSliceOp,
      tensor::EmptyOp, tensor::FromElementsOp, tensor::ConcatOp,
      tensor::ReshapeOp, tensor::InsertOp>(
      [&](auto op) { return typeConverter.isLegal(op); });

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // namespace mlir::zkir::mod_arith
