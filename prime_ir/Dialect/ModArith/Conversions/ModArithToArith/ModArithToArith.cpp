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

#include "prime_ir/Dialect/ModArith/Conversions/ModArithToArith/ModArithToArith.h"

#include <utility>

#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveOps.h"
#include "prime_ir/Dialect/Field/IR/FieldOps.h"
#include "prime_ir/Dialect/ModArith/Conversions/ModArithToArith/Inverter/BYInverter.h"
#include "prime_ir/Dialect/ModArith/Conversions/ModArithToArith/Reducer/MontReducer.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithDialect.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithOperation.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithOps.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithTypes.h"
#include "prime_ir/Dialect/TensorExt/IR/TensorExtOps.h"
#include "prime_ir/Utils/ConversionUtils.h"
#include "prime_ir/Utils/ShapedTypeConverter.h"

namespace mlir::prime_ir::mod_arith {

#define GEN_PASS_DEF_MODARITHTOARITH
#include "prime_ir/Dialect/ModArith/Conversions/ModArithToArith/ModArithToArith.h.inc"

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
    return IntegerType::get(type.getContext(), type.getStorageBitWidth());
  }
};

//===----------------------------------------------------------------------===//
// Lazy Reduction: IRA + BoundMap
//===----------------------------------------------------------------------===//

namespace {

using BoundMap = DenseMap<Value, uint64_t>;

// Base class for conversion patterns that carry a BoundMap for lazy reduction.
template <typename SourceOp>
struct BoundMapPattern : public OpConversionPattern<SourceOp> {
  BoundMapPattern(const TypeConverter &tc, MLIRContext *context,
                  const BoundMap *boundMap, PatternBenefit benefit = 1)
      : OpConversionPattern<SourceOp>(tc, context, benefit),
        boundMap(boundMap) {}

protected:
  // Lookup bound for a value. Returns 1 (canonical) if no map or not found.
  uint64_t lookupBound(Value v) const {
    if (!boundMap)
      return 1;
    auto it = boundMap->find(v);
    return it != boundMap->end() ? it->second : 1;
  }

  const BoundMap *boundMap;
};

// Returns storage bit width for a type, handling ModArithType.
unsigned getModArithStorageBitwidth(Type type) {
  if (auto modType = dyn_cast<ModArithType>(getElementTypeOrSelf(type)))
    return modType.getStorageBitWidth();
  return ConstantIntRanges::getStorageBitwidth(type);
}

// IRA subclass that initializes ModArithType function args to [0, p).
class ModArithRangeAnalysis : public dataflow::IntegerRangeAnalysis {
public:
  using IntegerRangeAnalysis::IntegerRangeAnalysis;

  void setToEntryState(dataflow::IntegerValueRangeLattice *lattice) override {
    Value anchor = lattice->getAnchor();
    if (!isa<ModArithType>(getElementTypeOrSelf(anchor.getType()))) {
      IntegerRangeAnalysis::setToEntryState(lattice);
      return;
    }
    propagateIfChanged(lattice, lattice->join(IntegerValueRange{
                                    getCanonicalRange(anchor.getType())}));
  }

  LogicalResult visitOperation(
      Operation *op,
      ArrayRef<const dataflow::IntegerValueRangeLattice *> operands,
      ArrayRef<dataflow::IntegerValueRangeLattice *> results) override {
    auto inferable = dyn_cast<InferIntRangeInterface>(op);
    if (!inferable) {
      setAllToEntryStates(results);
      return success();
    }

    auto argRanges = llvm::map_to_vector(
        operands, [](const dataflow::IntegerValueRangeLattice *lattice) {
          return lattice->getValue();
        });

    auto joinCallback = [&](Value v, const IntegerValueRange &attrs) {
      auto result = dyn_cast<OpResult>(v);
      if (!result)
        return;

      dataflow::IntegerValueRangeLattice *lattice =
          results[result.getResultNumber()];
      IntegerValueRange oldRange = lattice->getValue();
      ChangeResult changed = lattice->join(attrs);

      // Loop-variant detection: widen to max range to prevent infinite
      // iteration.
      bool isYieldedResult = llvm::any_of(v.getUsers(), [](Operation *op) {
        return op->hasTrait<OpTrait::IsTerminator>();
      });
      if (isYieldedResult && !oldRange.isUninitialized() &&
          !(lattice->getValue() == oldRange)) {
        unsigned w = getModArithStorageBitwidth(v.getType());
        if (w > 0) {
          changed |=
              lattice->join(IntegerValueRange{ConstantIntRanges::maxRange(w)});
        }
      }
      propagateIfChanged(lattice, changed);
    };

    inferable.inferResultRangesFromOptional(argRanges, joinCallback);
    return success();
  }
};

// Compute the optimal bound for a value from its IRA range.
// bound = smallest k such that umax < k * p.
uint64_t computeBound(Value v, DataFlowSolver &solver) {
  auto modType = dyn_cast<ModArithType>(getElementTypeOrSelf(v.getType()));
  if (!modType)
    return 1;

  auto *lattice = solver.lookupState<dataflow::IntegerValueRangeLattice>(v);
  if (!lattice || lattice->getValue().isUninitialized())
    return 1;

  APInt umax = lattice->getValue().getValue().umax();
  unsigned w = modType.getStorageBitWidth();
  APInt p = modType.getModulus().getValue().zextOrTrunc(w);
  if (p.isZero())
    return 1;

  // bound = ceil((umax + 1) / p)
  APInt umaxExt = umax.zext(w + 1) + 1;
  APInt pExt = p.zext(w + 1);
  APInt bound = (umaxExt + pExt - 1).udiv(pExt);
  uint64_t boundVal = bound.getLimitedValue(modType.getMaxBound() + 1);
  return std::max(uint64_t{1}, boundVal);
}

// Build the boundMap for a function by running IRA.
void buildBoundMap(func::FuncOp funcOp, BoundMap &boundMap) {
  DataFlowSolver solver;
  solver.load<dataflow::DeadCodeAnalysis>();
  solver.load<ModArithRangeAnalysis>();
  if (failed(solver.initializeAndRun(funcOp))) {
    return; // Conservative: no bounds recorded.
  }

  funcOp.walk([&](Operation *op) {
    for (auto result : op->getResults()) {
      if (!isa<ModArithType>(getElementTypeOrSelf(result.getType())))
        continue;
      uint64_t bound = computeBound(result, solver);
      if (bound > 1)
        boundMap[result] = bound;
    }
  });
}

} // namespace

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

struct ConvertNegate : public BoundMapPattern<NegateOp> {
  using BoundMapPattern::BoundMapPattern;

  LogicalResult
  matchAndRewrite(NegateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Value input = adaptor.getInput();
    uint64_t inputBound = lookupBound(op.getInput());
    if (inputBound > 1) {
      MontReducer reducer(b, getResultModArithType(op));
      input = reducer.getCanonicalFromExtended(input, inputBound);
    }

    Type intType = modulusType(op);
    Value zero;
    if (isa<ShapedType>(intType)) {
      zero = b.create<arith::ConstantOp>(SplatElementsAttr::get(
          cast<ShapedType>(intType),
          IntegerAttr::get(getElementTypeOrSelf(intType), 0)));
    } else {
      zero = b.create<arith::ConstantIntOp>(intType, 0);
    }
    auto cmp = b.create<arith::CmpIOp>(arith::CmpIPredicate::eq, input, zero);
    auto cmod = b.create<arith::ConstantOp>(modulusAttr(op));
    auto sub = b.create<arith::SubIOp>(cmod, input);
    auto result = b.create<arith::SelectOp>(cmp, input, sub);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ConvertMontReduce : public BoundMapPattern<MontReduceOp> {
  using BoundMapPattern::BoundMapPattern;

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
    uint64_t resBound = lookupBound(op.getResult());
    Value result = (resBound >= 2) ? reducer.reduceLazy(tLow, tHigh)
                                   : reducer.reduce(tLow, tHigh);
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

struct ConvertInverse : public BoundMapPattern<InverseOp> {
  using BoundMapPattern::BoundMapPattern;

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

    // Reduce lazy input before inverting.
    Value input = adaptor.getInput();
    uint64_t inputBound = lookupBound(op.getInput());
    if (inputBound > 1) {
      MontReducer reducer(b, modType);
      input = reducer.getCanonicalFromExtended(input, inputBound);
    }

    BYInverter inverter(b, op.getInput().getType());
    if (auto shapedType = dyn_cast<ShapedType>(op.getInput().getType())) {
      auto convertedType = cast<ShapedType>(input.getType());
      int64_t rank = shapedType.getRank();

      if (rank == 0) {
        // Rank-0: extract scalar, invert, wrap back.
        Value scalar = b.create<tensor::ExtractOp>(input);
        Value inverted = inverter.Generate(scalar, false);
        Value result = b.create<tensor::FromElementsOp>(convertedType,
                                                        ValueRange{inverted});
        rewriter.replaceOp(op, result);
        return success();
      }

      // Rank >= 2: flatten to rank-1 via collapse_shape.
      if (rank > 1) {
        auto flatType = RankedTensorType::get({shapedType.getNumElements()},
                                              convertedType.getElementType());
        SmallVector<ReassociationIndices> reassoc = {
            llvm::to_vector(llvm::seq<int64_t>(0, rank))};
        input = b.create<tensor::CollapseShapeOp>(flatType, input, reassoc);
      }

      auto flatShaped = cast<ShapedType>(input.getType());
      Value result = inverter.BatchGenerate(input, false, flatShaped);

      // Rank >= 2: restore original shape via expand_shape.
      if (rank > 1) {
        SmallVector<ReassociationIndices> reassoc = {
            llvm::to_vector(llvm::seq<int64_t>(0, rank))};
        result = b.create<tensor::ExpandShapeOp>(shapedType, result, reassoc);
      }

      rewriter.replaceOp(op, result);
      return success();
    }
    Value result = inverter.Generate(input, false);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ConvertMontInverse : public BoundMapPattern<MontInverseOp> {
  using BoundMapPattern::BoundMapPattern;

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

    // Reduce lazy input before inverting.
    Value input = adaptor.getInput();
    uint64_t inputBound = lookupBound(op.getInput());
    if (inputBound > 1) {
      MontReducer reducer(b, modType);
      input = reducer.getCanonicalFromExtended(input, inputBound);
    }

    BYInverter inverter(b, op.getInput().getType());
    if (auto shapedType = dyn_cast<ShapedType>(op.getInput().getType())) {
      auto convertedType = cast<ShapedType>(input.getType());
      int64_t rank = shapedType.getRank();

      if (rank == 0) {
        Value scalar = b.create<tensor::ExtractOp>(input);
        Value inverted = inverter.Generate(scalar, true);
        Value result = b.create<tensor::FromElementsOp>(convertedType,
                                                        ValueRange{inverted});
        rewriter.replaceOp(op, result);
        return success();
      }

      if (rank > 1) {
        auto flatType = RankedTensorType::get({shapedType.getNumElements()},
                                              convertedType.getElementType());
        SmallVector<ReassociationIndices> reassoc = {
            llvm::to_vector(llvm::seq<int64_t>(0, rank))};
        input = b.create<tensor::CollapseShapeOp>(flatType, input, reassoc);
      }

      auto flatShaped = cast<ShapedType>(input.getType());
      Value result = inverter.BatchGenerate(input, true, flatShaped);

      if (rank > 1) {
        SmallVector<ReassociationIndices> reassoc = {
            llvm::to_vector(llvm::seq<int64_t>(0, rank))};
        result = b.create<tensor::ExpandShapeOp>(shapedType, result, reassoc);
      }

      rewriter.replaceOp(op, result);
      return success();
    }
    Value result = inverter.Generate(input, true);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ConvertAdd : public BoundMapPattern<AddOp> {
  using BoundMapPattern::BoundMapPattern;

  LogicalResult
  matchAndRewrite(AddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    ModArithType modArithType = getResultModArithType(op);
    APInt modulus = modArithType.getModulus().getValue();
    unsigned storageWidth = modArithType.getStorageBitWidth();
    unsigned modWidth = modulus.getActiveBits();
    uint64_t resBound = lookupBound(op.getResult());

    // nuw is valid iff lhsBound + rhsBound <= getMaxBound(), i.e., the sum
    // fits in storageWidth bits without overflow. Pre-reduce any lazy inputs
    // that would violate this before emitting the addition.
    uint64_t lhsBound = lookupBound(op.getLhs());
    uint64_t rhsBound = lookupBound(op.getRhs());
    uint64_t maxBound = modArithType.getMaxBound();
    MontReducer montReducer(b, modArithType);
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    if (lhsBound + rhsBound > maxBound) {
      if (lhsBound > 1)
        lhs = montReducer.getCanonicalFromExtended(lhs, lhsBound);
      if (rhsBound > 1)
        rhs = montReducer.getCanonicalFromExtended(rhs, rhsBound);
    }

    Value result;
    if (modWidth == storageWidth) {
      auto add = b.create<arith::AddUIExtendedOp>(lhs, rhs);
      // In the full-width case (modWidth == storageWidth), A + B can exceed the
      // storage type range, producing an overflow bit, so lazy reduction is not
      // possible.
      result =
          montReducer.getCanonicalFromExtended(add.getSum(), add.getOverflow());
    } else {
      // nuw (no unsigned wrap) is safe when storageWidth - modWidth >= 1.
      // nsw (no signed wrap) is only safe when storageWidth - modWidth >= 2,
      // because the sum of two (modWidth)-bit values can be up to
      // 2^(modWidth+1) - 2, which overflows signed if modWidth + 1 >=
      // storageWidth.
      auto flags = arith::IntegerOverflowFlags::nuw;
      if (storageWidth - modWidth >= 2) {
        flags = flags | arith::IntegerOverflowFlags::nsw;
      }
      auto noOverflow =
          arith::IntegerOverflowFlagsAttr::get(b.getContext(), flags);
      auto add = b.create<arith::AddIOp>(lhs, rhs, noOverflow);
      if (resBound > 1) {
        result = add.getResult();
      } else {
        result = montReducer.getCanonicalFromExtended(add);
      }
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ConvertDouble : public BoundMapPattern<DoubleOp> {
  using BoundMapPattern::BoundMapPattern;

  LogicalResult
  matchAndRewrite(DoubleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    ModArithType modArithType = getResultModArithType(op);
    APInt modulus = modArithType.getModulus().getValue();
    unsigned storageWidth = modArithType.getStorageBitWidth();
    unsigned modWidth = modulus.getActiveBits();
    uint64_t resBound = lookupBound(op.getResult());

    // double(x) = 2x. nuw requires 2 * inputBound <= maxBound.
    // Pre-reduce lazy input that would overflow storage.
    uint64_t inputBound = lookupBound(op.getInput());
    uint64_t maxBound = modArithType.getMaxBound();
    MontReducer montReducer(b, modArithType);
    Value input = adaptor.getInput();
    if (2 * inputBound > maxBound) {
      if (inputBound > 1)
        input = montReducer.getCanonicalFromExtended(input, inputBound);
    }

    Value result;
    if (modWidth == storageWidth) {
      result = b.create<AddOp>(op.getInput(), op.getInput());
    } else {
      TypedAttr modAttr = modulusAttr(op);
      Value one =
          createScalarOrSplatConstant(b, b.getLoc(), modAttr.getType(), 1);
      // nuw (no unsigned wrap) is safe when storageWidth - modWidth >= 1.
      // nsw (no signed wrap) is only safe when storageWidth - modWidth >= 2.
      auto flags = arith::IntegerOverflowFlags::nuw;
      if (storageWidth - modWidth >= 2) {
        flags = flags | arith::IntegerOverflowFlags::nsw;
      }
      auto noOverflow =
          arith::IntegerOverflowFlagsAttr::get(b.getContext(), flags);
      auto shifted = b.create<arith::ShLIOp>(input, one, noOverflow);
      if (resBound > 1) {
        result = shifted.getResult();
      } else {
        result = montReducer.getCanonicalFromExtended(shifted);
      }
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ConvertSub : public BoundMapPattern<SubOp> {
  using BoundMapPattern::BoundMapPattern;

  LogicalResult
  matchAndRewrite(SubOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    ModArithType modType = getResultModArithType(op);
    uint64_t resBound = lookupBound(op.getResult());

    // Lazy sub computes lhs + correction - rhs where correction = rhsBound * p.
    // Result max = (lhsBound + rhsBound) * p - 1, must fit in w bits.
    // Pre-reduce lazy inputs if lhsBound + rhsBound > maxBound.
    uint64_t lhsBound = lookupBound(op.getLhs());
    uint64_t rhsBound = lookupBound(op.getRhs());
    uint64_t maxBound = modType.getMaxBound();
    MontReducer montReducer(b, modType);
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    if (lhsBound + rhsBound > maxBound) {
      if (lhsBound > 1)
        lhs = montReducer.getCanonicalFromExtended(lhs, lhsBound);
      if (rhsBound > 1)
        rhs = montReducer.getCanonicalFromExtended(rhs, rhsBound);
      lhsBound = 1;
      rhsBound = 1;
    }

    if (resBound > 1) {
      // Lazy sub: (lhs - rhs + correction) without final conditional sub.
      // correction = rhsBound * p ensures lhs + correction - rhs >= 0.
      APInt modulus = modType.getModulus().getValue();
      APInt correction = modulus * APInt(modulus.getBitWidth(), rhsBound);
      Value corrConst =
          createScalarOrSplatConstant(b, b.getLoc(), lhs.getType(), correction);
      auto lhsPlusCorrected = b.create<arith::AddIOp>(lhs, corrConst);
      auto result = b.create<arith::SubIOp>(lhsPlusCorrected, rhs);
      rewriter.replaceOp(op, result);
    } else {
      auto result = montReducer.getCanonicalDiff(lhs, rhs);
      rewriter.replaceOp(op, result);
    }
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
        {
          auto rhsStdOp =
              ModArithOperation::fromUnchecked(rhsInt.getValue(), modType);
          if (modType.isMontgomery()) {
            rhsStdOp = rhsStdOp.fromMont();
          }
          rhsStd = rhsStdOp.getIntegerAttr();
          negRhsStd = (-rhsStdOp).getIntegerAttr();
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

    auto cmodExt =
        b.create<arith::ConstantOp>(modulusAttr(op, /*extended=*/true));
    Type wideType = modulusType(op, /*extended=*/true);
    Value lhs = b.create<arith::ExtUIOp>(wideType, adaptor.getLhs());
    Value rhs = b.create<arith::ExtUIOp>(wideType, adaptor.getRhs());
    Value mul = b.create<arith::MulIOp>(lhs, rhs);

    if (APInt modulusPlusOne = modulus + 1; modulusPlusOne.isPowerOf2()) {
      unsigned k = modulusPlusOne.countTrailingZeros(); // e.g., 31

      // === Mersenne Reduction Strategy ===
      // Logic: A * B = H * 2ᵏ + L == H + L (mod 2ᵏ - 1)

      // 1. Reduce
      Value kConst =
          b.create<arith::ConstantOp>(wideType, b.getIntegerAttr(wideType, k));

      // hi = mul >> k
      Value hi = b.create<arith::ShRUIOp>(mul, kConst);
      // lo = mul & (2ᵏ - 1) (which is modulus)
      Value lo = b.create<arith::AndIOp>(mul, cmodExt);

      // sum = hi + lo
      Value sum = b.create<arith::AddIOp>(hi, lo);

      // 2. Final Correction: if (sum >= p) sum -= p
      // Note: A single subtraction is sufficient because max(H+L) < 2*p
      Value cmp =
          b.create<arith::CmpIOp>(arith::CmpIPredicate::uge, sum, cmodExt);
      Value sub = b.create<arith::SubIOp>(sum, cmodExt);
      Value reduced = b.create<arith::SelectOp>(cmp, sub, sum);

      rewriter.replaceOp(op,
                         b.create<arith::TruncIOp>(modulusType(op), reduced));
      return success();
    }

    // Use standard multiplication and reduction
    auto remu = b.create<arith::RemUIOp>(mul, cmodExt);
    auto trunc = b.create<arith::TruncIOp>(modulusType(op), remu);

    rewriter.replaceOp(op, trunc);
    return success();
  }
};

struct ConvertMontMul : public BoundMapPattern<MontMulOp> {
  using BoundMapPattern::BoundMapPattern;

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
    uint64_t resBound = lookupBound(op.getResult());

    // REDC precondition: T < p * 2ʷ. With lazy inputs of bounds k_a, k_b,
    // T < k_a * k_b * p², so we need k_a * k_b * p < 2ʷ, i.e.,
    // k_a * k_b <= getMaxBound(). Pre-reduce inputs that violate this.
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    uint64_t lhsBound = lookupBound(op.getLhs());
    uint64_t rhsBound = lookupBound(op.getRhs());
    uint64_t maxBound = modType.getMaxBound();
    MontReducer reducer(b, modType);
    if (lhsBound * rhsBound > maxBound) {
      if (lhsBound > 1)
        lhs = reducer.getCanonicalFromExtended(lhs, lhsBound);
      if (rhsBound > 1)
        rhs = reducer.getCanonicalFromExtended(rhs, rhsBound);
    }

    Value signedLhs = getSignedFormFromCanonical(lhs, modAttr);
    Value signedRhs = getSignedFormFromCanonical(rhs, modAttr);

    Value lo, hi;
    if (signedLhs && signedRhs) {
      auto mul = b.create<arith::MulSIExtendedOp>(signedLhs, signedRhs);
      lo = mul.getLow();
      hi = mul.getHigh();
    } else {
      auto mul = b.create<arith::MulUIExtendedOp>(lhs, rhs);
      lo = mul.getLow();
      hi = mul.getHigh();
    }

    Value result =
        (resBound >= 2) ? reducer.reduceLazy(lo, hi) : reducer.reduce(lo, hi);
    rewriter.replaceOp(op, result);
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

struct ConvertMontSquare : public BoundMapPattern<MontSquareOp> {
  using BoundMapPattern::BoundMapPattern;

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
    // REDC precondition: T < p * 2^w. With lazy input of bound k,
    // T < k² * p², so we need k² * p < 2^w, i.e., k² <= getMaxBound().
    // Pre-reduce input if this is violated.
    Value input = adaptor.getInput();
    uint64_t inputBound = lookupBound(op.getInput());
    uint64_t maxBound = modType.getMaxBound();
    MontReducer reducer(b, modType);
    if (inputBound * inputBound > maxBound) {
      if (inputBound > 1)
        input = reducer.getCanonicalFromExtended(input, inputBound);
    }

    auto sqResult = squareExtended(b, op, input);
    uint64_t resBound = lookupBound(op.getResult());

    Value result = (resBound >= 2)
                       ? reducer.reduceLazy(sqResult.lo, sqResult.hi)
                       : reducer.reduce(sqResult.lo, sqResult.hi);
    rewriter.replaceOp(op, result);
    return success();
  }
};

// TODO(ashjeong): Account for Montgomery domain inputs. Currently only accounts
// for base domain inputs.
struct ConvertCmp : public BoundMapPattern<CmpOp> {
  using BoundMapPattern::BoundMapPattern;

  LogicalResult
  matchAndRewrite(CmpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto modArithType = dyn_cast<ModArithType>(op.getLhs().getType());

    // Reduce lazy operands before comparison.
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    MontReducer reducer(b, modArithType);
    uint64_t lhsBound = lookupBound(op.getLhs());
    if (lhsBound > 1)
      lhs = reducer.getCanonicalFromExtended(lhs, lhsBound);
    uint64_t rhsBound = lookupBound(op.getRhs());
    if (rhsBound > 1)
      rhs = reducer.getCanonicalFromExtended(rhs, rhsBound);

    auto cmpOp = b.create<arith::CmpIOp>(op.getPredicate(), lhs, rhs);
    rewriter.replaceOp(op, cmpOp);
    return success();
  }
};

namespace rewrites {
// In an inner namespace to avoid conflicts with canonicalization patterns
#include "prime_ir/Dialect/ModArith/Conversions/ModArithToArith/ModArithToArith.cpp.inc"
} // namespace rewrites

//===----------------------------------------------------------------------===//
// Boundary reduction patterns (only active with lazy-reduction)
//===----------------------------------------------------------------------===//

// Reduce lazy values at func.return boundaries.
struct ConvertReturnWithReduction : public BoundMapPattern<func::ReturnOp> {
  ConvertReturnWithReduction(const TypeConverter &tc, MLIRContext *context,
                             const BoundMap *boundMap)
      : BoundMapPattern(tc, context, boundMap, /*benefit=*/10) {}

  LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    SmallVector<Value> newOperands;
    bool changed = false;

    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      Value origOperand = op.getOperand(i);
      Value convertedOperand = adaptor.getOperands()[i];
      auto modType =
          dyn_cast<ModArithType>(getElementTypeOrSelf(origOperand.getType()));
      uint64_t bound = lookupBound(origOperand);
      if (modType && bound > 1) {
        MontReducer reducer(b, modType);
        newOperands.push_back(
            reducer.getCanonicalFromExtended(convertedOperand, bound));
        changed = true;
      } else {
        newOperands.push_back(convertedOperand);
      }
    }

    if (!changed)
      return failure();

    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, newOperands);
    return success();
  }
};

// Reduce lazy values at memref.store boundaries.
struct ConvertStoreWithReduction : public BoundMapPattern<memref::StoreOp> {
  ConvertStoreWithReduction(const TypeConverter &tc, MLIRContext *context,
                            const BoundMap *boundMap)
      : BoundMapPattern(tc, context, boundMap, /*benefit=*/10) {}

  LogicalResult
  matchAndRewrite(memref::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    Value storedValue = op.getValue();
    auto modType =
        dyn_cast<ModArithType>(getElementTypeOrSelf(storedValue.getType()));
    uint64_t bound = lookupBound(storedValue);
    if (!modType || bound <= 1)
      return failure();

    MontReducer reducer(b, modType);
    Value reduced = reducer.getCanonicalFromExtended(adaptor.getValue(), bound);
    rewriter.replaceOpWithNewOp<memref::StoreOp>(
        op, reduced, adaptor.getMemref(), adaptor.getIndices());
    return success();
  }
};

// Reduce lazy values at scf.yield boundaries (for scf.for iter_args).
struct ConvertYieldWithReduction : public BoundMapPattern<scf::YieldOp> {
  ConvertYieldWithReduction(const TypeConverter &tc, MLIRContext *context,
                            const BoundMap *boundMap)
      : BoundMapPattern(tc, context, boundMap, /*benefit=*/10) {}

  LogicalResult
  matchAndRewrite(scf::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isa<scf::ForOp>(op->getParentOp()))
      return failure();

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    SmallVector<Value> newOperands;
    bool changed = false;

    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      Value origOperand = op.getOperand(i);
      Value convertedOperand = adaptor.getOperands()[i];
      auto modType =
          dyn_cast<ModArithType>(getElementTypeOrSelf(origOperand.getType()));
      uint64_t bound = lookupBound(origOperand);
      if (modType && bound > 1) {
        MontReducer reducer(b, modType);
        newOperands.push_back(
            reducer.getCanonicalFromExtended(convertedOperand, bound));
        changed = true;
      } else {
        newOperands.push_back(convertedOperand);
      }
    }

    if (!changed)
      return failure();

    rewriter.replaceOpWithNewOp<scf::YieldOp>(op, newOperands);
    return success();
  }
};

// Reduce lazy values at bufferization.materialize_in_destination boundaries.
struct ConvertMaterializeWithReduction
    : public BoundMapPattern<bufferization::MaterializeInDestinationOp> {
  ConvertMaterializeWithReduction(const TypeConverter &tc, MLIRContext *context,
                                  const BoundMap *boundMap)
      : BoundMapPattern(tc, context, boundMap, /*benefit=*/10) {}

  LogicalResult
  matchAndRewrite(bufferization::MaterializeInDestinationOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    Value source = op.getSource();
    auto modType =
        dyn_cast<ModArithType>(getElementTypeOrSelf(source.getType()));
    uint64_t bound = lookupBound(source);
    if (!modType || bound <= 1)
      return failure();

    MontReducer reducer(b, modType);
    Value reduced =
        reducer.getCanonicalFromExtended(adaptor.getSource(), bound);
    rewriter.replaceOpWithNewOp<bufferization::MaterializeInDestinationOp>(
        op, reduced, adaptor.getDest());
    return success();
  }
};

struct ModArithToArith : impl::ModArithToArithBase<ModArithToArith> {
  using ModArithToArithBase::ModArithToArithBase;

  void runOnOperation() override;
};

namespace {

// Check if a type contains mod_arith types.
bool containsModArithType(Type type) {
  if (isa<ModArithType>(type)) {
    return true;
  }
  if (auto shapedType = dyn_cast<ShapedType>(type)) {
    return containsModArithType(shapedType.getElementType());
  }
  return false;
}

// Check if a linalg op has mod_arith types.
bool hasModArithTypes(linalg::LinalgOp op) {
  return llvm::any_of(op->getOperandTypes(), containsModArithType) ||
         llvm::any_of(op->getResultTypes(), containsModArithType);
}

} // namespace

void ModArithToArith::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();
  ModArithToArithTypeConverter typeConverter(context);

  // Pre-processing: Generalize named linalg ops with mod_arith types.
  // Named linalg ops (matvec, matmul, dot) have verifiers that expect simple
  // add/mul ops in their body. When we convert mod_arith to arith, the body
  // becomes complex (Montgomery reduction, etc.), violating this constraint.
  // Generalizing to linalg.generic first avoids this issue.
  IRRewriter rewriter(context);
  module.walk([&](linalg::LinalgOp op) {
    if (!isa<linalg::MatvecOp, linalg::MatmulOp, linalg::DotOp>(op)) {
      return WalkResult::advance();
    }
    if (!hasModArithTypes(op)) {
      return WalkResult::advance();
    }
    rewriter.setInsertionPoint(op);
    if (failed(linalg::generalizeNamedOp(rewriter, op))) {
      op.emitWarning("failed to generalize named linalg op");
    }
    return WalkResult::advance();
  });

  // Build per-function bound maps for lazy reduction.
  BoundMap moduleBoundMap;
  if (lazyReduction) {
    DenseMap<func::FuncOp, BoundMap> funcBoundMaps;
    module.walk([&](func::FuncOp funcOp) {
      buildBoundMap(funcOp, funcBoundMaps[funcOp]);
    });
    // Merge into a single module-level bound map (values are unique across
    // functions so there is no collision).
    for (auto &[_, fm] : funcBoundMaps) {
      for (auto &[v, b] : fm) {
        moduleBoundMap[v] = b;
      }
    }
  }
  const BoundMap *bm = lazyReduction ? &moduleBoundMap : nullptr;

  ConversionTarget target(*context);
  target.addIllegalDialect<ModArithDialect>();
  target.addLegalDialect<arith::ArithDialect>();

  RewritePatternSet patterns(context);
  rewrites::populateWithGenerated(patterns);
  // Patterns that use BoundMap for lazy reduction analysis.
  patterns.add<
      // clang-format off
      ConvertAdd,
      ConvertCmp,
      ConvertDouble,
      ConvertInverse,
      ConvertMontInverse,
      ConvertMontMul,
      ConvertMontReduce,
      ConvertMontSquare,
      ConvertNegate,
      ConvertSub
      // clang-format on
      >(typeConverter, context, bm);
  // Patterns that don't use BoundMap.
  patterns.add<
      // clang-format off
      ConvertBitcast,
      ConvertConstant,
      ConvertFromMont,
      ConvertMul,
      ConvertSquare,
      ConvertToMont
      // clang-format on
      >(typeConverter, context);
  // Boundary patterns: reduce lazy values that escape via return/store/yield.
  if (lazyReduction) {
    patterns.add<
        // clang-format off
        ConvertMaterializeWithReduction,
        ConvertReturnWithReduction,
        ConvertStoreWithReduction,
        ConvertYieldWithReduction
        // clang-format on
        >(typeConverter, context, bm);
  }

  // Catch-all: converts any op whose operands/results carry mod_arith types.
  patterns.add<ConvertAny<void>>(typeConverter, context);

  addStructuralConversionPatterns(typeConverter, patterns, target);

  target.markUnknownOpDynamicallyLegal(
      [&](Operation *op) { return typeConverter.isLegal(op); });

  // arith::SelectOp is in the legal ArithDialect but may carry mod_arith
  // types (e.g., arith.select i1, !mod_arith.int, !mod_arith.int).
  // Override the dialect-level legal status so it gets converted.
  target.addDynamicallyLegalOp<arith::SelectOp>(
      [&](auto op) { return typeConverter.isLegal(op); });

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

} // namespace mlir::prime_ir::mod_arith
