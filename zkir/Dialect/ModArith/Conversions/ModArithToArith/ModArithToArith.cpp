#include "zkir/Dialect/ModArith/Conversions/ModArithToArith/ModArithToArith.h"

#include <utility>

#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
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
    Value T = adaptor.getOperands()[0];

    // Extract Montgomery constants: `nPrime` and `modulus`.
    IntegerAttr nPrimeAttr = op.getMontgomeryAttr().getNPrime();
    Value nPrime = b.create<arith::ConstantOp>(nPrimeAttr);
    TypedAttr modAttr = modulusAttr(op);
    Value mod = b.create<arith::ConstantOp>(modAttr);

    // Retrieve the modulus bitwidth.
    unsigned modBitWidth = cast<IntegerType>(modAttr.getType()).getWidth();

    // Compute number of limbs.
    const unsigned limbWidth = APInt::APINT_BITS_PER_WORD;
    unsigned numLimbs = (modBitWidth + limbWidth - 1) / limbWidth;

    // Arith operations require the operands to be of same bit width
    Value modExt = b.create<arith::ExtUIOp>(T.getType(), mod);

    // Prepare constants for limb operations.
    Value limbWidthConst =
        b.create<arith::ConstantOp>(b.getIntegerAttr(T.getType(), limbWidth));

    // Because the number of limbs (numLimbs) is known at compile time, we can
    // unroll the loop as a straight-line chain of operations. Let `u` be the
    // current working value, initially `T`.
    Value u = T;
    for (unsigned i = 0; i < numLimbs; ++i) {
      // Extract the current lowest limb: `u` (mod `base`)
      Value lowerLimb = b.create<arith::TruncIOp>(nPrimeAttr.getType(), u);
      // Compute `m` = `lowerLimb` * `nPrime` (mod `base`)
      Value m = b.create<arith::MulIOp>(lowerLimb, nPrime);
      // Compute `m` * `N` , where `N` is modulus
      Value mExt = b.create<arith::ExtUIOp>(T.getType(), m);
      Value mN = b.create<arith::MulIOp>(modExt, mExt);
      // Add the product to `u`.
      Value sum = b.create<arith::AddIOp>(u, mN);
      // Shift right by `limbWidth` to discard the zeroed limb.
      u = b.create<arith::ShRUIOp>(sum, limbWidthConst);
    }

    // Final conditional subtraction: if (`u_final` >= modulus) then subtract
    // modulus.
    Value cmp = b.create<arith::CmpIOp>(arith::CmpIPredicate::uge, u, modExt);
    Value sub = b.create<arith::SubIOp>(u, modExt);
    Value result = b.create<arith::SelectOp>(cmp, sub, u);

    // Truncate the result to the bitwidth of the modulus.
    Value truncated = b.create<arith::TruncIOp>(mod.getType(), result);

    rewriter.replaceOp(op, truncated);
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
    IntegerAttr rSquaredAttr = op.getMontgomery().getRSquared();

    // x * R = REDC(x * rSquared)
    auto rSquared =
        b.create<arith::ConstantOp>(op.getMontgomery().getRSquared());
    auto extended = b.create<arith::ExtUIOp>(rSquaredAttr.getType(),
                                             adaptor.getOperands()[0]);

    // TODO(batzor): Use extended multiplication to avoid full length
    // multiplication. Now we extend both operands to 2x the bitwidth of the
    // modulus to avoid the truncation in multiplication.
    auto product = b.create<arith::MulIOp>(extended, rSquared);
    auto reduced = b.create<MontReduceOp>(op.getResult().getType(), product,
                                          op.getMontgomery());
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
    auto extended = b.create<arith::ExtUIOp>(
        op.getMontgomery().getRSquared().getType(), adaptor.getOperands()[0]);
    auto reduced = b.create<MontReduceOp>(op.getResult().getType(), extended,
                                          op.getMontgomery());
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

    auto lhs =
        b.create<arith::ExtUIOp>(modulusType(op, true), adaptor.getLhs());
    auto rhs =
        b.create<arith::ExtUIOp>(modulusType(op, true), adaptor.getRhs());
    auto mul = b.create<arith::MulIOp>(lhs, rhs);
    auto reduced = b.create<mod_arith::MontReduceOp>(
        getResultModArithType(op), mul.getResult(), op.getMontgomery());

    rewriter.replaceOp(op, reduced);
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
      .add<ConvertEncapsulate, ConvertExtract, ConvertReduce, ConvertMontReduce,
           ConvertToMont, ConvertFromMont, ConvertAdd, ConvertSub, ConvertMul,
           ConvertMontMul, ConvertMac, ConvertConstant, ConvertInverse,
           ConvertAny<affine::AffineForOp>, ConvertAny<affine::AffineYieldOp>,
           ConvertAny<linalg::GenericOp>, ConvertAny<linalg::YieldOp>,
           ConvertAny<tensor::CastOp>, ConvertAny<tensor::ExtractOp>,
           ConvertAny<tensor::FromElementsOp>, ConvertAny<tensor::InsertOp>>(
          typeConverter, context);

  addStructuralConversionPatterns(typeConverter, patterns, target);

  target.addDynamicallyLegalOp<affine::AffineForOp, affine::AffineYieldOp,
                               linalg::GenericOp, linalg::YieldOp,
                               tensor::CastOp, tensor::ExtractOp,
                               tensor::FromElementsOp, tensor::InsertOp>(
      [&](auto op) { return typeConverter.isLegal(op); });

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // namespace mlir::zkir::mod_arith
