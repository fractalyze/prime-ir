#include "zkir/Dialect/ModArith/Conversions/ModArithToArith/ModArithToArith.h"

#include <utility>

#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
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

IntegerType convertModArithType(ModArithType type) {
  APInt modulus = type.getModulus().getValue();
  return IntegerType::get(type.getContext(), modulus.getBitWidth());
}

Type convertModArithLikeType(ShapedType type) {
  if (auto modArithType = llvm::dyn_cast<ModArithType>(type.getElementType())) {
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
TypedAttr modulusAttr(Op op, bool mul = false) {
  auto type = op.getResult().getType();
  auto modArithType = getResultModArithType(op);
  APInt modulus = modArithType.getModulus().getValue();

  auto width = modulus.getBitWidth();
  if (mul) {
    width *= 2;
  }

  auto intType = IntegerType::get(op.getContext(), width);
  auto truncmod = modulus.zextOrTrunc(width);

  if (auto st = mlir::dyn_cast<ShapedType>(type)) {
    auto containerType = st.cloneWith(st.getShape(), intType);
    return DenseElementsAttr::get(containerType, truncmod);
  }
  return IntegerAttr::get(intType, truncmod);
}

// used for extui/trunci
template <typename Op>
inline Type modulusType(Op op, bool mul = false) {
  return modulusAttr(op, mul).getType();
}

struct ConvertEncapsulate : public OpConversionPattern<EncapsulateOp> {
  explicit ConvertEncapsulate(mlir::MLIRContext *context)
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
  explicit ConvertExtract(mlir::MLIRContext *context)
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
  explicit ConvertConstant(mlir::MLIRContext *context)
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
  explicit ConvertNegate(mlir::MLIRContext *context)
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
  explicit ConvertReduce(mlir::MLIRContext *context)
      : OpConversionPattern<ReduceOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ReduceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto cmod = b.create<arith::ConstantOp>(modulusAttr(op));
    // ModArithType ensures cmod can be correctly interpreted as a signed number
    auto rems = b.create<arith::RemSIOp>(adaptor.getOperands()[0], cmod);
    auto add = b.create<arith::AddIOp>(rems, cmod);
    // TODO(google/heir #710): better with a subifge
    auto remu = b.create<arith::RemUIOp>(add, cmod);
    rewriter.replaceOp(op, remu);
    return success();
  }
};

struct ConvertMontReduce : public OpConversionPattern<MontReduceOp> {
  explicit ConvertMontReduce(mlir::MLIRContext *context)
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
      tHigh = b.create<arith::AddIOp>(tHigh, mN.getHigh());
      // Add carry from the `sum` to `tHigh`.
      auto carryExt =
          b.create<arith::ExtUIOp>(tHigh.getType(), sum.getOverflow());
      tHigh = b.create<arith::AddIOp>(tHigh, carryExt);
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
    auto sub = b.create<arith::SubIOp>(tLow, modConst);
    auto result = b.create<arith::SelectOp>(cmp, sub, tLow);

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ConvertToMont : public OpConversionPattern<ToMontOp> {
  explicit ConvertToMont(mlir::MLIRContext *context)
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
  explicit ConvertFromMont(mlir::MLIRContext *context)
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
  explicit ConvertInverse(mlir::MLIRContext *context)
      : OpConversionPattern<InverseOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      InverseOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // TODO(batzor): Support tensor input.
    if (mlir::isa<ShapedType>(op.getInput().getType())) {
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
  explicit ConvertAdd(mlir::MLIRContext *context)
      : OpConversionPattern<AddOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      AddOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto cmod = b.create<arith::ConstantOp>(modulusAttr(op));
    auto add = b.create<arith::AddIOp>(adaptor.getLhs(), adaptor.getRhs());
    auto ifge = b.create<arith::CmpIOp>(arith::CmpIPredicate::uge, add, cmod);
    auto sub = b.create<arith::SubIOp>(add, cmod);
    auto select = b.create<arith::SelectOp>(ifge, sub, add);

    rewriter.replaceOp(op, select);
    return success();
  }
};

struct ConvertSub : public OpConversionPattern<SubOp> {
  explicit ConvertSub(mlir::MLIRContext *context)
      : OpConversionPattern<SubOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SubOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto cmod = b.create<arith::ConstantOp>(modulusAttr(op));
    auto sub = b.create<arith::SubIOp>(adaptor.getLhs(), adaptor.getRhs());
    auto add = b.create<arith::AddIOp>(sub, cmod);
    auto ifge = b.create<arith::CmpIOp>(arith::CmpIPredicate::uge,
                                        adaptor.getLhs(), adaptor.getRhs());
    auto select = b.create<arith::SelectOp>(ifge, sub, add);

    rewriter.replaceOp(op, select);
    return success();
  }
};

struct ConvertMul : public OpConversionPattern<MulOp> {
  explicit ConvertMul(mlir::MLIRContext *context)
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
  explicit ConvertMac(mlir::MLIRContext *context)
      : OpConversionPattern<MacOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      MacOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto cmod = b.create<arith::ConstantOp>(modulusAttr(op, true));
    auto x = b.create<arith::ExtUIOp>(modulusType(op, true),
                                      adaptor.getOperands()[0]);
    auto y = b.create<arith::ExtUIOp>(modulusType(op, true),
                                      adaptor.getOperands()[1]);
    auto acc = b.create<arith::ExtUIOp>(modulusType(op, true),
                                        adaptor.getOperands()[2]);
    auto mul = b.create<arith::MulIOp>(x, y);
    auto add = b.create<arith::AddIOp>(mul, acc);
    auto remu = b.create<arith::RemUIOp>(add, cmod);
    auto trunc = b.create<arith::TruncIOp>(modulusType(op), remu);

    rewriter.replaceOp(op, trunc);
    return success();
  }
};

struct ConvertMontMul : public OpConversionPattern<MontMulOp> {
  explicit ConvertMontMul(mlir::MLIRContext *context)
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

struct ConvertSquare : public OpConversionPattern<SquareOp> {
  explicit ConvertSquare(mlir::MLIRContext *context)
      : OpConversionPattern<SquareOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SquareOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    const unsigned limbWidth = APInt::APINT_BITS_PER_WORD;

    Value input = adaptor.getInput();
    Type intType = modulusType(op, false);
    Type resultType = modulusType(op, true);
    Type limbType = IntegerType::get(op.getContext(), limbWidth);
    Type mulResultLimbType = IntegerType::get(op.getContext(), 2 * limbWidth);

    const unsigned modBitWidth = cast<IntegerType>(intType).getWidth();
    const unsigned mulResultBitWidth = cast<IntegerType>(resultType).getWidth();
    const unsigned numLimbs = (modBitWidth + limbWidth - 1) / limbWidth;

    Value zeroLimb = b.create<arith::ConstantOp>(IntegerAttr::get(limbType, 0));

    // Divide input into vector of size numLimbs / type limbType
    SmallVector<Value> limbs(numLimbs);
    for (unsigned i = 0; i < numLimbs; ++i) {
      Value limb = b.create<arith::ShRUIOp>(
          input, b.create<arith::ConstantOp>(
                     IntegerAttr::get(intType, i * limbWidth)));
      limbs[i] = b.create<arith::TruncIOp>(limbType, limb);
    }
    // Result buffer of size 2 * numLimbs / type limbType
    SmallVector<Value> resultVec(2 * numLimbs, zeroLimb);
    Value temp = zeroLimb;

    // add offdiagonal entries to result buffer
    for (unsigned i = 0; i < numLimbs; ++i) {
      // get a[i]
      Value a_i = limbs[i];
      for (unsigned j = i + 1; j < numLimbs; ++j) {
        // get r[i+j]
        Value r_ij = resultVec[i + j];
        // get a[j]
        Value a_j = limbs[j];

        // (temp, sum) = MulAddWithCarry r[i+j] + a[i] * a[j] + temp
        auto ai_aj = b.create<arith::MulUIExtendedOp>(a_i, a_j);
        Value hi = ai_aj.getHigh();
        Value lo = ai_aj.getLow();
        auto addResult = b.create<arith::AddUIExtendedOp>(r_ij, lo);
        Value carry1 = addResult.getOverflow();
        auto addResult2 =
            b.create<arith::AddUIExtendedOp>(addResult.getSum(), temp);
        Value carry2 = addResult2.getOverflow();
        Value sum = addResult2.getSum();
        temp = b.create<arith::AddIOp>(
            hi, b.create<arith::ExtUIOp>(limbType, carry1));
        temp = b.create<arith::AddIOp>(
            temp, b.create<arith::ExtUIOp>(limbType, carry2));

        // set r[i+j] to sum
        resultVec[i + j] = sum;
      }
      // set r[i+N] to temp
      resultVec[i + numLimbs] = temp;
      // Reset temp
      temp = zeroLimb;
    }

    // Build the result from the buffer vector
    Value result = b.create<arith::ConstantOp>(IntegerAttr::get(resultType, 0));
    for (unsigned i = 0; i < 2 * numLimbs; ++i) {
      // get r[i]
      Value r_i = b.create<arith::ExtUIOp>(resultType, resultVec[i]);
      // shift r[i] to the correct position
      Value shifted = b.create<arith::ShLIOp>(
          r_i, b.create<arith::ConstantOp>(
                   IntegerAttr::get(resultType, i * limbWidth)));
      // add it to the result
      result = b.create<arith::OrIOp>(result, shifted);
    }

    // Multiply result by 2. It's safe to assume no overflow
    result = b.create<arith::ShLIOp>(
        result, b.create<arith::ConstantOp>(IntegerAttr::get(resultType, 1)));

    // Store result in the result buffer again
    for (unsigned i = 0; i < 2 * numLimbs; ++i) {
      Value shifted = b.create<arith::ShRUIOp>(
          result, b.create<arith::ConstantOp>(
                      IntegerAttr::get(resultType, i * limbWidth)));
      resultVec[i] = b.create<arith::TruncIOp>(limbType, shifted);
    }

    // add diagonal entries to result buffer
    for (unsigned i = 0; i < numLimbs; ++i) {
      // get r[2*i]
      Value r_2i = resultVec[2 * i];

      // (temp, r[2*i]) = r[2*i] + a[i] * a[i] + temp
      Value a_i = limbs[i];
      auto aiSquared = b.create<arith::MulUIExtendedOp>(a_i, a_i);
      Value hi = aiSquared.getHigh();
      Value lo = aiSquared.getLow();
      auto addResult = b.create<arith::AddUIExtendedOp>(r_2i, lo);
      Value carry1 = addResult.getOverflow();
      auto addResult2 =
          b.create<arith::AddUIExtendedOp>(addResult.getSum(), temp);
      Value carry2 = addResult2.getOverflow();
      Value sum = addResult2.getSum();
      temp = b.create<arith::AddIOp>(
          hi, b.create<arith::ExtUIOp>(limbType, carry1));
      temp = b.create<arith::AddIOp>(
          temp, b.create<arith::ExtUIOp>(limbType, carry2));

      // set r[2*i] to sum
      resultVec[2 * i] = sum;

      // get r[2*i+1]
      Value r_2i1 = resultVec[2 * i + 1];

      // AddWithCarry r[2*i+1] + temp
      addResult = b.create<arith::AddUIExtendedOp>(r_2i1, temp);
      sum = addResult.getSum();

      // set r[2*i+1] to sum
      resultVec[2 * i + 1] = sum;

      // set temp to the carry
      temp = b.create<arith::ExtUIOp>(limbType, addResult.getOverflow());
    }

    // build the resultLow and resultHigh from the limbs
    Value resultLow = b.create<arith::ConstantOp>(IntegerAttr::get(intType, 0));
    Value resultHigh =
        b.create<arith::ConstantOp>(IntegerAttr::get(intType, 0));
    for (unsigned i = 0; i < 2 * numLimbs; ++i) {
      Value r_i = b.create<arith::ExtUIOp>(intType, resultVec[i]);
      if (i < numLimbs) {
        auto shifted = b.create<arith::ShLIOp>(
            r_i, b.create<arith::ConstantOp>(
                     IntegerAttr::get(intType, i * limbWidth)));
        resultLow = b.create<arith::OrIOp>(resultLow, shifted);
      } else {
        auto shifted = b.create<arith::ShLIOp>(
            r_i, b.create<arith::ConstantOp>(
                     IntegerAttr::get(intType, (i - numLimbs) * limbWidth)));
        resultHigh = b.create<arith::OrIOp>(resultHigh, shifted);
      }
    }

    // Montgomery reduction
    auto reduced = b.create<mod_arith::MontReduceOp>(
        getResultModArithType(op), resultLow, resultHigh, op.getMontgomery());

    rewriter.replaceOp(op, reduced);
    return success();
  };
};

struct ConvertDouble : public OpConversionPattern<DoubleOp> {
  explicit ConvertDouble(mlir::MLIRContext *context)
      : OpConversionPattern<DoubleOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      DoubleOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    ModArithType modType = getResultModArithType(op);
    auto intType = modType.getModulus().getType();

    Value cmod = b.create<arith::ConstantOp>(modulusAttr(op));
    Value one = b.create<arith::ConstantOp>(IntegerAttr::get(intType, 1));
    adaptor.getInput();
    auto shifted = b.create<arith::ShLIOp>(adaptor.getInput(), one);
    auto ifge =
        b.create<arith::CmpIOp>(arith::CmpIPredicate::uge, shifted, cmod);
    auto sub = b.create<arith::SubIOp>(shifted, cmod);
    auto select = b.create<arith::SelectOp>(ifge, sub, shifted);

    rewriter.replaceOp(op, select);
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
           ConvertSub, ConvertMul, ConvertMontMul, ConvertMac, ConvertConstant,
           ConvertInverse, ConvertSquare, ConvertDouble, affine::AffineForOp>,
      ConvertAny<affine::AffineParallelOp>, ConvertAny<affine::AffineLoadOp>,
      ConvertAny<affine::AffineApplyOp>, ConvertAny<affine::AffineStoreOp>,
      ConvertAny<affine::AffineYieldOp>, ConvertAny<bufferization::ToMemrefOp>,
      ConvertAny<bufferization::ToTensorOp>, ConvertAny<linalg::GenericOp>,
      ConvertAny<linalg::YieldOp>, ConvertAny<tensor::CastOp>,
      ConvertAny<tensor::ExtractOp>, ConvertAny<tensor::FromElementsOp>,
      ConvertAny < tensor::InsertOp >> (typeConverter, context);
  addStructuralConversionPatterns(typeConverter, patterns, target);

  target.addDynamicallyLegalOp<
      affine::AffineForOp, affine::AffineParallelOp, affine::AffineLoadOp,
      affine::AffineApplyOp, affine::AffineStoreOp, affine::AffineYieldOp,
      bufferization::ToMemrefOp, bufferization::ToTensorOp, linalg::GenericOp,
      linalg::YieldOp, tensor::CastOp, tensor::ExtractOp,
      tensor::FromElementsOp, tensor::InsertOp>(
      [&](auto op) { return typeConverter.isLegal(op); });

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // namespace mlir::zkir::mod_arith
