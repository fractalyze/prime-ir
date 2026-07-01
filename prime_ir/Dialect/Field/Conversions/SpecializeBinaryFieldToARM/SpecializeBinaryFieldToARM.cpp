/* Copyright 2026 The PrimeIR Authors.

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

#include "prime_ir/Dialect/Field/Conversions/SpecializeBinaryFieldToARM/SpecializeBinaryFieldToARM.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "prime_ir/Dialect/Field/Conversions/BinaryFieldToArith/BinaryFieldTables.h"
#include "prime_ir/Dialect/Field/IR/FieldDialect.h"
#include "prime_ir/Dialect/Field/IR/FieldOps.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"

namespace mlir::prime_ir::field {

#define GEN_PASS_DEF_SPECIALIZEBINARYFIELDTOARM
#include "prime_ir/Dialect/Field/Conversions/SpecializeBinaryFieldToARM/SpecializeBinaryFieldToARM.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Target Feature Helper
//===----------------------------------------------------------------------===//

// Add ARM crypto target features to the parent function.
// PMULL instructions require the AES/crypto extension.
void addARMCryptoTargetFeatures(Operation *op) {
  auto funcOp = op->getParentOfType<func::FuncOp>();
  if (!funcOp)
    return;

  // Check if target features are already set
  if (funcOp->hasAttr("llvm.target_features"))
    return;

  // Add target features for ARM crypto extensions (PMULL).
  // "+neon,+crypto" enables the PMULL instruction. This must be a
  // TargetFeaturesAttr (not a plain StringAttr): convert-func-to-llvm moves
  // the `llvm.target_features` attribute onto the llvm.func `target_features`
  // property, whose conversion rejects a raw string.
  funcOp->setAttr(
      "llvm.target_features",
      LLVM::TargetFeaturesAttr::get(funcOp.getContext(), "+neon,+crypto"));
}

//===----------------------------------------------------------------------===//
// ARM PMULL Intrinsic Helpers
//===----------------------------------------------------------------------===//

// Emit pmull instruction for 64-bit carryless multiplication.
// pmull Vd.1q, Vn.1d, Vm.1d - multiply low 64-bit halves, produce 128-bit
// result The result is stored in a vector<2xi64>.
Value emitPMULL(ImplicitLocOpBuilder &b, Value lhs, Value rhs) {
  auto vecType = VectorType::get(2, b.getI64Type());
  // ARM GCC-style inline assembly
  // Input: two 64-bit values packed in 128-bit registers
  // Output: 128-bit result from polynomial multiplication
  return LLVM::InlineAsmOp::create(b, vecType, ValueRange{lhs, rhs},
                                   "pmull $0.1q, $1.1d, $2.1d", "=w,w,w",
                                   /*has_side_effects=*/false,
                                   /*is_align_stack=*/false,
                                   LLVM::TailCallKind::None,
                                   LLVM::AsmDialectAttr{},
                                   /*operand_attrs=*/ArrayAttr())
      .getResult(0);
}

// Emit pmull2 instruction for high 64-bit halves.
// pmull2 Vd.1q, Vn.2d, Vm.2d - multiply high 64-bit halves, produce 128-bit
// result
Value emitPMULL2(ImplicitLocOpBuilder &b, Value lhs, Value rhs) {
  auto vecType = VectorType::get(2, b.getI64Type());
  return LLVM::InlineAsmOp::create(b, vecType, ValueRange{lhs, rhs},
                                   "pmull2 $0.1q, $1.2d, $2.2d", "=w,w,w",
                                   /*has_side_effects=*/false,
                                   /*is_align_stack=*/false,
                                   LLVM::TailCallKind::None,
                                   LLVM::AsmDialectAttr{},
                                   /*operand_attrs=*/ArrayAttr())
      .getResult(0);
}

// Emit pmull for 8-bit polynomial multiplication.
// pmull Vd.8h, Vn.8b, Vm.8b - multiply 8 pairs of 8-bit polynomials
// Produces 8 16-bit results
Value emitPMULL8(ImplicitLocOpBuilder &b, Value lhs, Value rhs) {
  auto vecType = VectorType::get(8, b.getI16Type());
  return LLVM::InlineAsmOp::create(b, vecType, ValueRange{lhs, rhs},
                                   "pmull $0.8h, $1.8b, $2.8b", "=w,w,w",
                                   /*has_side_effects=*/false,
                                   /*is_align_stack=*/false,
                                   LLVM::TailCallKind::None,
                                   LLVM::AsmDialectAttr{},
                                   /*operand_attrs=*/ArrayAttr())
      .getResult(0);
}

// Emit pmull2 for high 8-bit polynomial multiplication.
// pmull2 Vd.8h, Vn.16b, Vm.16b - multiply high 8 pairs of 8-bit polynomials
Value emitPMULL2_8(ImplicitLocOpBuilder &b, Value lhs, Value rhs) {
  auto vecType = VectorType::get(8, b.getI16Type());
  return LLVM::InlineAsmOp::create(b, vecType, ValueRange{lhs, rhs},
                                   "pmull2 $0.8h, $1.16b, $2.16b", "=w,w,w",
                                   /*has_side_effects=*/false,
                                   /*is_align_stack=*/false,
                                   LLVM::TailCallKind::None,
                                   LLVM::AsmDialectAttr{},
                                   /*operand_attrs=*/ArrayAttr())
      .getResult(0);
}

//===----------------------------------------------------------------------===//
// GHASH Polynomial Reduction Helper
//===----------------------------------------------------------------------===//

// Compute the GHASH reduction high * (x⁷ + x² + x + 1) for a 64-bit limb:
// returns the value to XOR into the low 64 bits, plus the overflow bits that
// spill past bit 63 (to be XORed into the next limb up).
std::pair<Value, Value> reduceGhash(ImplicitLocOpBuilder &b, Value high) {
  auto i64Type = b.getI64Type();

  Value h7 = arith::ShLIOp::create(
      b, high,
      arith::ConstantOp::create(b, i64Type,
                                b.getI64IntegerAttr(kGhashReductionShifts[0])));
  Value h2 = arith::ShLIOp::create(
      b, high,
      arith::ConstantOp::create(b, i64Type,
                                b.getI64IntegerAttr(kGhashReductionShifts[1])));
  Value h1 = arith::ShLIOp::create(
      b, high,
      arith::ConstantOp::create(b, i64Type,
                                b.getI64IntegerAttr(kGhashReductionShifts[2])));

  // XOR all together: h7 ^ h2 ^ h1 ^ high
  Value reduction = arith::XOrIOp::create(b, h7, h2);
  reduction = arith::XOrIOp::create(b, reduction, h1);
  reduction = arith::XOrIOp::create(b, reduction, high);

  // Overflow bits spilling past bit 63, using kGhashOverflowShifts = {57,62,63}
  Value h7_hi = arith::ShRUIOp::create(
      b, high,
      arith::ConstantOp::create(b, i64Type,
                                b.getI64IntegerAttr(kGhashOverflowShifts[0])));
  Value h2_hi = arith::ShRUIOp::create(
      b, high,
      arith::ConstantOp::create(b, i64Type,
                                b.getI64IntegerAttr(kGhashOverflowShifts[1])));
  Value h1_hi = arith::ShRUIOp::create(
      b, high,
      arith::ConstantOp::create(b, i64Type,
                                b.getI64IntegerAttr(kGhashOverflowShifts[2])));

  Value overflow = arith::XOrIOp::create(b, h7_hi, h2_hi);
  overflow = arith::XOrIOp::create(b, overflow, h1_hi);

  return {reduction, overflow};
}

//===----------------------------------------------------------------------===//
// PMULL-based GHASH Field Multiplication
//===----------------------------------------------------------------------===//

// Multiply two GHASH-basis i128 values (as vector<2xi64>) using PMULL.
// Karatsuba — 3 PMULL (the cross term a₀b₁ + a₁b₀ = (a₀+a₁)(b₀+b₁) + ll + hh),
// then reduce mod x¹²⁸ + x⁷ + x² + x + 1.
Value mulGhashPMULL(ImplicitLocOpBuilder &b, Value lhs, Value rhs) {
  auto i64Type = b.getI64Type();
  auto vec2i64Type = VectorType::get(2, i64Type);

  // Swap the two 64-bit lanes so each lane of *Xor holds (lo ^ hi); then
  // emitPMULL multiplies the low lanes, giving (a₀+a₁)·(b₀+b₁).
  Value lhsSwapped =
      vector::ShuffleOp::create(b, lhs, lhs, ArrayRef<int64_t>{1, 0});
  Value rhsSwapped =
      vector::ShuffleOp::create(b, rhs, rhs, ArrayRef<int64_t>{1, 0});
  Value lhsXor = arith::XOrIOp::create(b, lhs, lhsSwapped);
  Value rhsXor = arith::XOrIOp::create(b, rhs, rhsSwapped);

  Value ll = emitPMULL(b, lhs, rhs);      // low·low   = a₀·b₀
  Value hh = emitPMULL2(b, lhs, rhs);     // high·high = a₁·b₁
  Value m = emitPMULL(b, lhsXor, rhsXor); // (a₀+a₁)·(b₀+b₁)

  // mid = m ^ ll ^ hh = a₀b₁ + a₁b₀
  Value mid = arith::XOrIOp::create(b, arith::XOrIOp::create(b, m, ll), hh);

  // Extract parts
  Value llLow = vector::ExtractOp::create(b, ll, ArrayRef<int64_t>{0});
  Value llHigh = vector::ExtractOp::create(b, ll, ArrayRef<int64_t>{1});
  Value hhLow = vector::ExtractOp::create(b, hh, ArrayRef<int64_t>{0});
  Value hhHigh = vector::ExtractOp::create(b, hh, ArrayRef<int64_t>{1});
  Value midLow = vector::ExtractOp::create(b, mid, ArrayRef<int64_t>{0});
  Value midHigh = vector::ExtractOp::create(b, mid, ArrayRef<int64_t>{1});

  // 256-bit product as limbs r0..r3 (low to high).
  Value r0 = llLow;
  Value r1 = arith::XOrIOp::create(b, llHigh, midLow);
  Value r2 = arith::XOrIOp::create(b, hhLow, midHigh);
  Value r3 = hhHigh;

  // Fold the high half down via x¹²⁸ == x⁷ + x² + x + 1.
  auto [r3_red, r3_overflow] = reduceGhash(b, r3);
  r1 = arith::XOrIOp::create(b, r1, r3_red);
  r2 = arith::XOrIOp::create(b, r2, r3_overflow);

  auto [r2_red, r2_overflow] = reduceGhash(b, r2);
  r0 = arith::XOrIOp::create(b, r0, r2_red);
  r1 = arith::XOrIOp::create(b, r1, r2_overflow);

  // Pack result back into vector<2xi64>
  return vector::FromElementsOp::create(b, vec2i64Type, ValueRange{r0, r1});
}

//===----------------------------------------------------------------------===//
// PMULL-based Packed 8-bit Multiplication
//===----------------------------------------------------------------------===//

// Reduce 16-bit polynomial product to 8-bit using tower reduction for level 3.
// Tower level 3 reduction polynomial: x⁸ + x⁴ + x³ + x + 1
Value reduceTowerLevel3Packed(ImplicitLocOpBuilder &b, Value highBits) {
  auto vecType = mlir::cast<VectorType>(highBits.getType());

  // kTowerLevel3ReductionShifts = {4, 3, 1, 0}
  // But since we're working with the high byte shifted down, we apply
  // these shifts to create the reduction polynomial contribution

  // Create shift constants as vectors
  auto createShiftConst = [&](int64_t shift) {
    SmallVector<int16_t, 8> values(vecType.getNumElements(),
                                   static_cast<int16_t>(shift));
    return arith::ConstantOp::create(
        b, DenseElementsAttr::get(vecType, ArrayRef<int16_t>(values)));
  };

  Value h4 = arith::ShLIOp::create(b, highBits, createShiftConst(4));
  Value h3 = arith::ShLIOp::create(b, highBits, createShiftConst(3));
  Value h1 = arith::ShLIOp::create(b, highBits, createShiftConst(1));

  Value reduction = arith::XOrIOp::create(b, h4, h3);
  reduction = arith::XOrIOp::create(b, reduction, h1);
  reduction = arith::XOrIOp::create(b, reduction, highBits);

  return reduction;
}

// Multiply packed 8-bit binary field elements using PMULL.
// For 16 elements (128-bit vector), we split into two 8-element groups.
Value mulBF8PackedPMULL(ImplicitLocOpBuilder &b, Value lhs, Value rhs,
                        unsigned numElements) {
  assert(numElements == 16 && "Only 16 elements supported for now");

  auto i8Type = b.getI8Type();
  auto i16Type = b.getI16Type();
  auto vec8i16Type = VectorType::get(8, i16Type);

  // PMULL multiplies 8 pairs of 8-bit values, producing 8 16-bit results
  // pmull Vd.8h, Vn.8b, Vm.8b - low 8 bytes
  // pmull2 Vd.8h, Vn.16b, Vm.16b - high 8 bytes

  // Low 8 elements
  Value prodLow = emitPMULL8(b, lhs, rhs);
  // High 8 elements
  Value prodHigh = emitPMULL2_8(b, lhs, rhs);

  // Now we need to reduce each 16-bit result to 8-bit using tower reduction
  // Extract high byte (bits 8-15) from each 16-bit element
  auto createMask = [&](uint16_t mask) {
    SmallVector<int16_t, 8> values(8, static_cast<int16_t>(mask));
    return arith::ConstantOp::create(
        b, DenseElementsAttr::get(vec8i16Type, ArrayRef<int16_t>(values)));
  };

  Value shiftConst8 = createMask(8);
  Value lowMask = createMask(0xFF);

  // For low product
  Value lowHighBits = arith::ShRUIOp::create(b, prodLow, shiftConst8);
  Value lowLowBits = arith::AndIOp::create(b, prodLow, lowMask);
  Value lowReduction = reduceTowerLevel3Packed(b, lowHighBits);
  Value lowResult = arith::XOrIOp::create(b, lowLowBits, lowReduction);

  // For high product
  Value highHighBits = arith::ShRUIOp::create(b, prodHigh, shiftConst8);
  Value highLowBits = arith::AndIOp::create(b, prodHigh, lowMask);
  Value highReduction = reduceTowerLevel3Packed(b, highHighBits);
  Value highResult = arith::XOrIOp::create(b, highLowBits, highReduction);

  // Truncate i16 results to i8 and combine into 16-element vector
  auto vec8i8Type = VectorType::get(8, i8Type);
  Value lowTrunc = arith::TruncIOp::create(b, vec8i8Type, lowResult);
  Value highTrunc = arith::TruncIOp::create(b, vec8i8Type, highResult);

  // Concatenate low and high results using vector.shuffle
  SmallVector<int64_t, 16> indices;
  for (int64_t i = 0; i < 8; ++i)
    indices.push_back(i);
  for (int64_t i = 0; i < 8; ++i)
    indices.push_back(8 + i);

  return vector::ShuffleOp::create(b, lowTrunc, highTrunc, indices);
}

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

// Extract operands from MulOp (lhs, rhs) or SquareOp (input, input).
template <typename OpTy>
std::pair<Value, Value> getMulOperands(OpTy op) {
  if constexpr (std::is_same_v<OpTy, MulOp>)
    return {op.getLhs(), op.getRhs()};
  else
    return {op.getInput(), op.getInput()};
}

// Pattern for GHASH-field mul/square using PMULL. (The tower bf<6>/bf<7> no
// longer specialize to a carryless multiply -- that computes the flat GHASH
// product, not the tower -- so they lower via the portable recursive mulTower.)
template <typename OpTy>
struct ConvertGhashToPMULL : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    Type ghashType = op.getResult().getType();
    auto bfType = dyn_cast<BinaryFieldType>(ghashType);
    if (!bfType || !bfType.isGhash())
      return failure();

    addARMCryptoTargetFeatures(op);
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto i128Type = b.getIntegerType(128);
    auto vec2i64Type = VectorType::get(2, b.getI64Type());

    auto [lhs, rhs] = getMulOperands(op);

    // Cast ghash -> i128; BinaryFieldToArith later reconciles these casts.
    Value lhsI128 =
        UnrealizedConversionCastOp::create(b, i128Type, lhs).getResult(0);
    Value rhsI128 =
        UnrealizedConversionCastOp::create(b, i128Type, rhs).getResult(0);

    Value lhsVec = LLVM::BitcastOp::create(b, vec2i64Type, lhsI128);
    Value rhsVec = LLVM::BitcastOp::create(b, vec2i64Type, rhsI128);

    Value resultVec = mulGhashPMULL(b, lhsVec, rhsVec);

    Value resultI128 = LLVM::BitcastOp::create(b, i128Type, resultVec);
    Value resultGhash =
        UnrealizedConversionCastOp::create(b, ghashType, resultI128)
            .getResult(0);
    rewriter.replaceOp(op, resultGhash);
    return success();
  }
};

// Pattern for packed 8-bit binary field mul/square using PMULL.
template <typename OpTy>
struct ConvertPackedBF8ToPMULL : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    Type resultType = op.getResult().getType();

    // Only handle vector types
    auto vecType = dyn_cast<VectorType>(resultType);
    if (!vecType)
      return failure();

    auto bfType = dyn_cast<BinaryFieldType>(vecType.getElementType());
    if (!bfType || bfType.getTowerLevel() != 3)
      return failure();

    int64_t numElements = vecType.getNumElements();
    if (numElements != 16)
      return failure();

    // Add ARM crypto target features to parent function
    addARMCryptoTargetFeatures(op);

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto [lhs, rhs] = getMulOperands(op);
    auto vecI8Type = VectorType::get(numElements, b.getI8Type());
    Value lhsI8 =
        UnrealizedConversionCastOp::create(b, vecI8Type, lhs).getResult(0);
    Value rhsI8 =
        UnrealizedConversionCastOp::create(b, vecI8Type, rhs).getResult(0);

    Value result = mulBF8PackedPMULL(b, lhsI8, rhsI8, numElements);

    // Cast result back
    Value resultCast =
        UnrealizedConversionCastOp::create(b, vecType, result).getResult(0);
    rewriter.replaceOp(op, resultCast);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct SpecializeBinaryFieldToARM
    : impl::SpecializeBinaryFieldToARMBase<SpecializeBinaryFieldToARM> {
  using SpecializeBinaryFieldToARMBase::SpecializeBinaryFieldToARMBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();

    RewritePatternSet patterns(context);

    if (usePMULL) {
      // Packed bf<3> keeps its PMULL fast path. The tower bf<6>/bf<7> no longer
      // specialize -- a carryless multiply computes the flat GHASH product, not
      // the recursive x²+x+α tower -- so they fall through to the portable
      // mulTower in binary-field-to-arith. The GHASH field gets the carryless
      // multiply as its CPU fast path instead.
      patterns.add<ConvertPackedBF8ToPMULL<MulOp>,
                   ConvertPackedBF8ToPMULL<SquareOp>>(context);
      patterns.add<ConvertGhashToPMULL<MulOp>, ConvertGhashToPMULL<SquareOp>>(
          context);
    }

    // Use greedy pattern rewriting (not partial conversion)
    // This allows patterns to fail gracefully without marking ops illegal
    // Disable folding to avoid issues with tensor.from_elements folding
    // on binary field types (MLIR's folder doesn't understand custom types)
    GreedyRewriteConfig config;
    config.enableFolding(false);
    if (failed(applyPatternsGreedily(module, std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::prime_ir::field
