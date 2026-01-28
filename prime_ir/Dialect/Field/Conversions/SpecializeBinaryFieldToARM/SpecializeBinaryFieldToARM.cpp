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

  // Add target features for ARM crypto extensions (PMULL)
  // "+neon,+crypto" or "+neon,+aes" enables the PMULL instruction
  funcOp->setAttr("llvm.target_features",
                  StringAttr::get(funcOp.getContext(), "+neon,+crypto"));
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
  return b
      .create<LLVM::InlineAsmOp>(
          vecType, ValueRange{lhs, rhs}, "pmull $0.1q, $1.1d, $2.1d", "=w,w,w",
          /*has_side_effects=*/false,
          /*is_align_stack=*/false, LLVM::TailCallKind::None,
          LLVM::AsmDialectAttr{},
          /*operand_attrs=*/ArrayAttr())
      .getResult(0);
}

// Emit pmull2 instruction for high 64-bit halves.
// pmull2 Vd.1q, Vn.2d, Vm.2d - multiply high 64-bit halves, produce 128-bit
// result
Value emitPMULL2(ImplicitLocOpBuilder &b, Value lhs, Value rhs) {
  auto vecType = VectorType::get(2, b.getI64Type());
  return b
      .create<LLVM::InlineAsmOp>(
          vecType, ValueRange{lhs, rhs}, "pmull2 $0.1q, $1.2d, $2.2d", "=w,w,w",
          /*has_side_effects=*/false,
          /*is_align_stack=*/false, LLVM::TailCallKind::None,
          LLVM::AsmDialectAttr{},
          /*operand_attrs=*/ArrayAttr())
      .getResult(0);
}

// Emit pmull for 8-bit polynomial multiplication.
// pmull Vd.8h, Vn.8b, Vm.8b - multiply 8 pairs of 8-bit polynomials
// Produces 8 16-bit results
Value emitPMULL8(ImplicitLocOpBuilder &b, Value lhs, Value rhs) {
  auto vecType = VectorType::get(8, b.getI16Type());
  return b
      .create<LLVM::InlineAsmOp>(
          vecType, ValueRange{lhs, rhs}, "pmull $0.8h, $1.8b, $2.8b", "=w,w,w",
          /*has_side_effects=*/false,
          /*is_align_stack=*/false, LLVM::TailCallKind::None,
          LLVM::AsmDialectAttr{},
          /*operand_attrs=*/ArrayAttr())
      .getResult(0);
}

// Emit pmull2 for high 8-bit polynomial multiplication.
// pmull2 Vd.8h, Vn.16b, Vm.16b - multiply high 8 pairs of 8-bit polynomials
Value emitPMULL2_8(ImplicitLocOpBuilder &b, Value lhs, Value rhs) {
  auto vecType = VectorType::get(8, b.getI16Type());
  return b
      .create<LLVM::InlineAsmOp>(vecType, ValueRange{lhs, rhs},
                                 "pmull2 $0.8h, $1.16b, $2.16b", "=w,w,w",
                                 /*has_side_effects=*/false,
                                 /*is_align_stack=*/false,
                                 LLVM::TailCallKind::None,
                                 LLVM::AsmDialectAttr{},
                                 /*operand_attrs=*/ArrayAttr())
      .getResult(0);
}

//===----------------------------------------------------------------------===//
// Tower Polynomial Reduction Helpers
//===----------------------------------------------------------------------===//

// Compute tower level 7 reduction: high * (x^7 + x^2 + x + 1)
// Returns the value to XOR with the low 64 bits.
// Also computes overflow bits that need to be XORed with higher positions.
std::pair<Value, Value> reduceTowerLevel7(ImplicitLocOpBuilder &b, Value high) {
  auto i64Type = b.getI64Type();

  // Compute shifts using constants from BinaryFieldTables.h
  // kTowerLevel7ReductionShifts = {7, 2, 1, 0}
  Value h7 = b.create<arith::ShLIOp>(
      high, b.create<arith::ConstantOp>(
                i64Type, b.getI64IntegerAttr(kTowerLevel7ReductionShifts[0])));
  Value h2 = b.create<arith::ShLIOp>(
      high, b.create<arith::ConstantOp>(
                i64Type, b.getI64IntegerAttr(kTowerLevel7ReductionShifts[1])));
  Value h1 = b.create<arith::ShLIOp>(
      high, b.create<arith::ConstantOp>(
                i64Type, b.getI64IntegerAttr(kTowerLevel7ReductionShifts[2])));

  // XOR all together: h7 ^ h2 ^ h1 ^ high
  Value reduction = b.create<arith::XOrIOp>(h7, h2);
  reduction = b.create<arith::XOrIOp>(reduction, h1);
  reduction = b.create<arith::XOrIOp>(reduction, high);

  // Compute overflow bits using kTowerLevel7OverflowShifts = {57, 62, 63}
  Value h7_hi = b.create<arith::ShRUIOp>(
      high, b.create<arith::ConstantOp>(
                i64Type, b.getI64IntegerAttr(kTowerLevel7OverflowShifts[0])));
  Value h2_hi = b.create<arith::ShRUIOp>(
      high, b.create<arith::ConstantOp>(
                i64Type, b.getI64IntegerAttr(kTowerLevel7OverflowShifts[1])));
  Value h1_hi = b.create<arith::ShRUIOp>(
      high, b.create<arith::ConstantOp>(
                i64Type, b.getI64IntegerAttr(kTowerLevel7OverflowShifts[2])));

  Value overflow = b.create<arith::XOrIOp>(h7_hi, h2_hi);
  overflow = b.create<arith::XOrIOp>(overflow, h1_hi);

  return {reduction, overflow};
}

// Compute tower level 6 reduction: high * (x^4 + x^3 + x + 1)
Value reduceTowerLevel6(ImplicitLocOpBuilder &b, Value high) {
  auto i64Type = b.getI64Type();

  // kTowerLevel6ReductionShifts = {4, 3, 1, 0}
  Value h4 = b.create<arith::ShLIOp>(
      high, b.create<arith::ConstantOp>(
                i64Type, b.getI64IntegerAttr(kTowerLevel6ReductionShifts[0])));
  Value h3 = b.create<arith::ShLIOp>(
      high, b.create<arith::ConstantOp>(
                i64Type, b.getI64IntegerAttr(kTowerLevel6ReductionShifts[1])));
  Value h1 = b.create<arith::ShLIOp>(
      high, b.create<arith::ConstantOp>(
                i64Type, b.getI64IntegerAttr(kTowerLevel6ReductionShifts[2])));

  Value reduction = b.create<arith::XOrIOp>(h4, h3);
  reduction = b.create<arith::XOrIOp>(reduction, h1);
  reduction = b.create<arith::XOrIOp>(reduction, high);

  return reduction;
}

//===----------------------------------------------------------------------===//
// PMULL-based Binary Field Multiplication
//===----------------------------------------------------------------------===//

// Multiply two 64-bit tower field elements using PMULL.
// Returns 64-bit result after reduction.
Value mulBF64PMULL(ImplicitLocOpBuilder &b, Value lhs, Value rhs) {
  auto i64Type = b.getI64Type();
  auto vec2i64Type = VectorType::get(2, i64Type);

  // Pack inputs into 128-bit vectors (low 64 bits used)
  Value zero = b.create<arith::ConstantOp>(i64Type, b.getI64IntegerAttr(0));
  Value lhsVec =
      b.create<vector::FromElementsOp>(vec2i64Type, ValueRange{lhs, zero});
  Value rhsVec =
      b.create<vector::FromElementsOp>(vec2i64Type, ValueRange{rhs, zero});

  // Carryless multiply using PMULL: produces 128-bit result
  Value product = emitPMULL(b, lhsVec, rhsVec);

  // Extract low and high 64-bit parts
  Value prodLow = b.create<vector::ExtractOp>(product, ArrayRef<int64_t>{0});
  Value prodHigh = b.create<vector::ExtractOp>(product, ArrayRef<int64_t>{1});

  // Reduce using tower polynomial for level 6
  Value reduction = reduceTowerLevel6(b, prodHigh);
  return b.create<arith::XOrIOp>(prodLow, reduction);
}

// Multiply two 128-bit tower field elements using PMULL with Karatsuba.
Value mulBF128PMULL(ImplicitLocOpBuilder &b, Value lhs, Value rhs) {
  auto i64Type = b.getI64Type();
  auto vec2i64Type = VectorType::get(2, i64Type);

  // Karatsuba multiplication for 128-bit values
  // Let a = a_h * x^64 + a_l, b = b_h * x^64 + b_l
  // a * b = a_h*b_h * x^128 + (a_h*b_l + a_l*b_h) * x^64 + a_l*b_l

  // Low-low product using pmull (low 64-bit halves)
  Value ll = emitPMULL(b, lhs, rhs);
  // High-high product using pmull2 (high 64-bit halves)
  Value hh = emitPMULL2(b, lhs, rhs);

  // Cross products: need to extract halves and multiply
  // Extract individual 64-bit values
  Value lhsLow = b.create<vector::ExtractOp>(lhs, ArrayRef<int64_t>{0});
  Value lhsHigh = b.create<vector::ExtractOp>(lhs, ArrayRef<int64_t>{1});
  Value rhsLow = b.create<vector::ExtractOp>(rhs, ArrayRef<int64_t>{0});
  Value rhsHigh = b.create<vector::ExtractOp>(rhs, ArrayRef<int64_t>{1});

  // Pack for cross products
  Value zero = b.create<arith::ConstantOp>(i64Type, b.getI64IntegerAttr(0));
  Value lhsLowVec =
      b.create<vector::FromElementsOp>(vec2i64Type, ValueRange{lhsLow, zero});
  Value rhsHighVec =
      b.create<vector::FromElementsOp>(vec2i64Type, ValueRange{rhsHigh, zero});
  Value lhsHighVec =
      b.create<vector::FromElementsOp>(vec2i64Type, ValueRange{lhsHigh, zero});
  Value rhsLowVec =
      b.create<vector::FromElementsOp>(vec2i64Type, ValueRange{rhsLow, zero});

  // lhs_low * rhs_high
  Value lh = emitPMULL(b, lhsLowVec, rhsHighVec);
  // lhs_high * rhs_low
  Value hl = emitPMULL(b, lhsHighVec, rhsLowVec);

  // Middle term = lh XOR hl
  Value mid = b.create<arith::XOrIOp>(lh, hl);

  // Extract parts
  Value llLow = b.create<vector::ExtractOp>(ll, ArrayRef<int64_t>{0});
  Value llHigh = b.create<vector::ExtractOp>(ll, ArrayRef<int64_t>{1});
  Value hhLow = b.create<vector::ExtractOp>(hh, ArrayRef<int64_t>{0});
  Value hhHigh = b.create<vector::ExtractOp>(hh, ArrayRef<int64_t>{1});
  Value midLow = b.create<vector::ExtractOp>(mid, ArrayRef<int64_t>{0});
  Value midHigh = b.create<vector::ExtractOp>(mid, ArrayRef<int64_t>{1});

  // Combine: result is 256-bit, we need to reduce mod tower polynomial
  // Bits 0-63: llLow, Bits 64-127: llHigh XOR midLow
  // Bits 128-191: hhLow XOR midHigh, Bits 192-255: hhHigh
  Value r0 = llLow;
  Value r1 = b.create<arith::XOrIOp>(llHigh, midLow);
  Value r2 = b.create<arith::XOrIOp>(hhLow, midHigh);
  Value r3 = hhHigh;

  // Reduce 256-bit to 128-bit using tower polynomial for level 7
  // First reduce r3 (bits 192-255): r3 * x^64 * (x^7 + x^2 + x + 1)
  auto [r3_red, r3_overflow] = reduceTowerLevel7(b, r3);
  r1 = b.create<arith::XOrIOp>(r1, r3_red);
  r2 = b.create<arith::XOrIOp>(r2, r3_overflow);

  // Now reduce r2 (bits 128-191): r2 * (x^7 + x^2 + x + 1)
  auto [r2_red, r2_overflow] = reduceTowerLevel7(b, r2);
  r0 = b.create<arith::XOrIOp>(r0, r2_red);
  r1 = b.create<arith::XOrIOp>(r1, r2_overflow);

  // Pack result back into vector<2xi64>
  return b.create<vector::FromElementsOp>(vec2i64Type, ValueRange{r0, r1});
}

//===----------------------------------------------------------------------===//
// PMULL-based Packed 8-bit Multiplication
//===----------------------------------------------------------------------===//

// Reduce 16-bit polynomial product to 8-bit using tower reduction for level 3.
// Tower level 3 reduction polynomial: x^8 + x^4 + x^3 + x + 1
Value reduceTowerLevel3Packed(ImplicitLocOpBuilder &b, Value highBits) {
  auto vecType = mlir::cast<VectorType>(highBits.getType());

  // kTowerLevel3ReductionShifts = {4, 3, 1, 0}
  // But since we're working with the high byte shifted down, we apply
  // these shifts to create the reduction polynomial contribution

  // Create shift constants as vectors
  auto createShiftConst = [&](int64_t shift) {
    SmallVector<int16_t, 8> values(vecType.getNumElements(),
                                   static_cast<int16_t>(shift));
    return b.create<arith::ConstantOp>(
        DenseElementsAttr::get(vecType, ArrayRef<int16_t>(values)));
  };

  Value h4 = b.create<arith::ShLIOp>(highBits, createShiftConst(4));
  Value h3 = b.create<arith::ShLIOp>(highBits, createShiftConst(3));
  Value h1 = b.create<arith::ShLIOp>(highBits, createShiftConst(1));

  Value reduction = b.create<arith::XOrIOp>(h4, h3);
  reduction = b.create<arith::XOrIOp>(reduction, h1);
  reduction = b.create<arith::XOrIOp>(reduction, highBits);

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
    return b.create<arith::ConstantOp>(
        DenseElementsAttr::get(vec8i16Type, ArrayRef<int16_t>(values)));
  };

  Value shiftConst8 = createMask(8);
  Value lowMask = createMask(0xFF);

  // For low product
  Value lowHighBits = b.create<arith::ShRUIOp>(prodLow, shiftConst8);
  Value lowLowBits = b.create<arith::AndIOp>(prodLow, lowMask);
  Value lowReduction = reduceTowerLevel3Packed(b, lowHighBits);
  Value lowResult = b.create<arith::XOrIOp>(lowLowBits, lowReduction);

  // For high product
  Value highHighBits = b.create<arith::ShRUIOp>(prodHigh, shiftConst8);
  Value highLowBits = b.create<arith::AndIOp>(prodHigh, lowMask);
  Value highReduction = reduceTowerLevel3Packed(b, highHighBits);
  Value highResult = b.create<arith::XOrIOp>(highLowBits, highReduction);

  // Truncate i16 results to i8 and combine into 16-element vector
  auto vec8i8Type = VectorType::get(8, i8Type);
  Value lowTrunc = b.create<arith::TruncIOp>(vec8i8Type, lowResult);
  Value highTrunc = b.create<arith::TruncIOp>(vec8i8Type, highResult);

  // Concatenate low and high results using vector.shuffle
  SmallVector<int64_t, 16> indices;
  for (int64_t i = 0; i < 8; ++i)
    indices.push_back(i);
  for (int64_t i = 0; i < 8; ++i)
    indices.push_back(8 + i);

  return b.create<vector::ShuffleOp>(lowTrunc, highTrunc, indices);
}

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

// Pattern for 64-bit binary field multiplication using PMULL
struct ConvertBF64MulToPMULL : public OpRewritePattern<MulOp> {
  using OpRewritePattern<MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const override {
    Type resultType = op.getResult().getType();

    // Check if this is bf<6> (64-bit binary field)
    auto bfType = dyn_cast<BinaryFieldType>(resultType);
    if (!bfType || bfType.getTowerLevel() != 6)
      return failure();

    // Add ARM crypto target features to parent function
    addARMCryptoTargetFeatures(op);

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // Get original operands (still BinaryFieldType)
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();

    // Use unrealized_conversion_cast from bf<6> to i64
    auto i64Type = b.getI64Type();
    Value lhsI64 =
        b.create<UnrealizedConversionCastOp>(i64Type, lhs).getResult(0);
    Value rhsI64 =
        b.create<UnrealizedConversionCastOp>(i64Type, rhs).getResult(0);

    Value result = mulBF64PMULL(b, lhsI64, rhsI64);

    // Cast back to bf<6>
    Value resultBF =
        b.create<UnrealizedConversionCastOp>(bfType, result).getResult(0);
    rewriter.replaceOp(op, resultBF);
    return success();
  }
};

// Pattern for 128-bit binary field multiplication using PMULL
struct ConvertBF128MulToPMULL : public OpRewritePattern<MulOp> {
  using OpRewritePattern<MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const override {
    Type resultType = op.getResult().getType();

    // Check if this is bf<7> (128-bit binary field)
    auto bfType = dyn_cast<BinaryFieldType>(resultType);
    if (!bfType || bfType.getTowerLevel() != 7)
      return failure();

    // Add ARM crypto target features to parent function
    addARMCryptoTargetFeatures(op);

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // Get original operands (still BinaryFieldType)
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();

    auto i64Type = b.getI64Type();
    auto i128Type = b.getIntegerType(128);
    auto vec2i64Type = VectorType::get(2, i64Type);

    // Cast bf<7> -> i128 (matches BinaryFieldToArith type converter target)
    // After BinaryFieldToArith runs, these become i128 -> i128 no-ops
    Value lhsI128 =
        b.create<UnrealizedConversionCastOp>(i128Type, lhs).getResult(0);
    Value rhsI128 =
        b.create<UnrealizedConversionCastOp>(i128Type, rhs).getResult(0);

    // LLVM bitcast i128 -> vector<2xi64> for PMULL operations
    Value lhsVec = b.create<LLVM::BitcastOp>(vec2i64Type, lhsI128);
    Value rhsVec = b.create<LLVM::BitcastOp>(vec2i64Type, rhsI128);

    Value resultVec = mulBF128PMULL(b, lhsVec, rhsVec);

    // LLVM bitcast vector<2xi64> -> i128
    Value resultI128 = b.create<LLVM::BitcastOp>(i128Type, resultVec);

    // Cast i128 -> bf<7>
    Value resultBF =
        b.create<UnrealizedConversionCastOp>(bfType, resultI128).getResult(0);
    rewriter.replaceOp(op, resultBF);
    return success();
  }
};

// Pattern for packed 8-bit binary field multiplication using PMULL
struct ConvertPackedBF8MulToPMULL : public OpRewritePattern<MulOp> {
  using OpRewritePattern<MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const override {
    Type resultType = op.getResult().getType();

    // Only handle vector types (not tensors)
    auto vecType = dyn_cast<VectorType>(resultType);
    if (!vecType)
      return failure();

    auto bfType = dyn_cast<BinaryFieldType>(vecType.getElementType());
    if (!bfType || bfType.getTowerLevel() != 3)
      return failure();

    // Only support 16 elements (128-bit) for now
    int64_t numElements = vecType.getNumElements();
    if (numElements != 16)
      return failure();

    // Add ARM crypto target features to parent function
    addARMCryptoTargetFeatures(op);

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // Convert bf<3> vectors to i8 vectors for SIMD operation
    auto vecI8Type = VectorType::get(numElements, b.getI8Type());

    // Use unrealized_conversion_cast for type conversion
    Value lhs = b.create<UnrealizedConversionCastOp>(vecI8Type, op.getLhs())
                    .getResult(0);
    Value rhs = b.create<UnrealizedConversionCastOp>(vecI8Type, op.getRhs())
                    .getResult(0);

    // Perform PMULL multiplication
    Value result = mulBF8PackedPMULL(b, lhs, rhs, numElements);

    // Cast result back to bf<3> vector
    Value resultCast =
        b.create<UnrealizedConversionCastOp>(vecType, result).getResult(0);

    rewriter.replaceOp(op, resultCast);
    return success();
  }
};

// Pattern for packed 8-bit binary field squaring using PMULL
struct ConvertPackedBF8SquareToPMULL : public OpRewritePattern<SquareOp> {
  using OpRewritePattern<SquareOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SquareOp op,
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

    auto vecI8Type = VectorType::get(numElements, b.getI8Type());
    Value input = b.create<UnrealizedConversionCastOp>(vecI8Type, op.getInput())
                      .getResult(0);

    // Square = multiply by itself
    Value result = mulBF8PackedPMULL(b, input, input, numElements);

    // Cast result back
    Value resultCast =
        b.create<UnrealizedConversionCastOp>(vecType, result).getResult(0);

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
      patterns.add<ConvertBF64MulToPMULL, ConvertBF128MulToPMULL,
                   ConvertPackedBF8MulToPMULL, ConvertPackedBF8SquareToPMULL>(
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
