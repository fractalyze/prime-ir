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

#include "prime_ir/Dialect/Field/Conversions/SpecializeBinaryFieldToX86/SpecializeBinaryFieldToX86.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
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

#define GEN_PASS_DEF_SPECIALIZEBINARYFIELDTOX86
#include "prime_ir/Dialect/Field/Conversions/SpecializeBinaryFieldToX86/SpecializeBinaryFieldToX86.h.inc"

namespace {

// Tower <-> AES field transformation tables are defined in BinaryFieldTables.h

//===----------------------------------------------------------------------===//
// GFNI Intrinsic Helpers
//===----------------------------------------------------------------------===//

// Emit vgf2p8affineqb instruction for vectors of specified byte count.
// Performs affine transformation: result[i] = matrix * input[i] XOR imm
// NumBytes must be 16 (SSE), 32 (AVX2), or 64 (AVX-512).
template <unsigned NumBytes>
Value emitGF2P8AffineQB(ImplicitLocOpBuilder &b, Value input, uint64_t matrix,
                        uint8_t imm = 0) {
  static_assert(NumBytes == 16 || NumBytes == 32 || NumBytes == 64,
                "NumBytes must be 16, 32, or 64");
  constexpr unsigned NumI64 = NumBytes / 8;

  auto vecType = VectorType::get(NumBytes, b.getI8Type());
  auto i64Type = b.getI64Type();
  auto matrixVecType = VectorType::get(NumI64, i64Type);

  // Create replicated matrix constant
  SmallVector<int64_t, 8> matrixValues(NumI64, static_cast<int64_t>(matrix));
  Value matrixConst = b.create<arith::ConstantOp>(
      DenseElementsAttr::get(matrixVecType, ArrayRef<int64_t>(matrixValues)));

  std::string asmString = "vgf2p8affineqb $0, $1, $2, " + std::to_string(imm);
  return b
      .create<LLVM::InlineAsmOp>(
          vecType, ValueRange{input, matrixConst}, asmString, "=x,x,x",
          /*has_side_effects=*/false,
          /*is_align_stack=*/true, LLVM::TailCallKind::None,
          LLVM::AsmDialectAttr::get(b.getContext(), LLVM::AsmDialect::AD_Intel),
          /*operand_attrs=*/ArrayAttr())
      .getResult(0);
}

// Emit vgf2p8mulb instruction for vectors of specified byte count.
// Performs GF(2⁸) multiplication in AES field.
// NumBytes must be 16 (SSE), 32 (AVX2), or 64 (AVX-512).
template <unsigned NumBytes>
Value emitGF2P8MulB(ImplicitLocOpBuilder &b, Value lhs, Value rhs) {
  static_assert(NumBytes == 16 || NumBytes == 32 || NumBytes == 64,
                "NumBytes must be 16, 32, or 64");

  auto vecType = VectorType::get(NumBytes, b.getI8Type());
  return b
      .create<LLVM::InlineAsmOp>(
          vecType, ValueRange{lhs, rhs}, "vgf2p8mulb $0, $1, $2", "=x,x,x",
          /*has_side_effects=*/false,
          /*is_align_stack=*/true, LLVM::TailCallKind::None,
          LLVM::AsmDialectAttr::get(b.getContext(), LLVM::AsmDialect::AD_Intel),
          /*operand_attrs=*/ArrayAttr())
      .getResult(0);
}

//===----------------------------------------------------------------------===//
// PCLMULQDQ Intrinsic Helpers
//===----------------------------------------------------------------------===//

// Emit vpclmulqdq instruction for carryless multiplication
// imm8 selects which 64-bit halves to multiply:
//   0x00: low * low
//   0x01: low * high
//   0x10: high * low
//   0x11: high * high
Value emitPCLMULQDQ128(ImplicitLocOpBuilder &b, Value lhs, Value rhs,
                       uint8_t imm) {
  auto vecType = VectorType::get(2, b.getI64Type());
  std::string asmString = "vpclmulqdq $0, $1, $2, " + std::to_string(imm);
  return b
      .create<LLVM::InlineAsmOp>(
          vecType, ValueRange{lhs, rhs}, asmString, "=x,x,x",
          /*has_side_effects=*/false,
          /*is_align_stack=*/true, LLVM::TailCallKind::None,
          LLVM::AsmDialectAttr::get(b.getContext(), LLVM::AsmDialect::AD_Intel),
          /*operand_attrs=*/ArrayAttr())
      .getResult(0);
}

//===----------------------------------------------------------------------===//
// Tower Polynomial Reduction Helpers
//===----------------------------------------------------------------------===//

// Compute tower level 7 reduction: high * (x⁷ + x² + x + 1)
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

// Compute tower level 6 reduction: high * (x⁴ + x³ + x + 1)
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
// GFNI-based Binary Field Multiplication
//===----------------------------------------------------------------------===//

// Multiply packed 8-bit binary tower field elements using GFNI.
// Algorithm:
// 1. Transform inputs from tower field to AES field
// 2. Multiply in AES field using vgf2p8mulb
// 3. Transform result back to tower field
template <unsigned NumBytes>
Value mulBF8PackedGFNIImpl(ImplicitLocOpBuilder &b, Value lhs, Value rhs) {
  uint64_t towerToAesMatrix = getTowerToAesMatrixPacked();
  uint64_t aesToTowerMatrix = getAesToTowerMatrixPacked();

  Value lhsAes = emitGF2P8AffineQB<NumBytes>(b, lhs, towerToAesMatrix);
  Value rhsAes = emitGF2P8AffineQB<NumBytes>(b, rhs, towerToAesMatrix);
  Value productAes = emitGF2P8MulB<NumBytes>(b, lhsAes, rhsAes);
  return emitGF2P8AffineQB<NumBytes>(b, productAes, aesToTowerMatrix);
}

// Runtime dispatch wrapper for mulBF8PackedGFNI
Value mulBF8PackedGFNI(ImplicitLocOpBuilder &b, Value lhs, Value rhs,
                       unsigned numElements) {
  assert((numElements == 16 || numElements == 32 || numElements == 64) &&
         "numElements must be 16, 32, or 64");
  switch (numElements) {
  case 16:
    return mulBF8PackedGFNIImpl<16>(b, lhs, rhs);
  case 32:
    return mulBF8PackedGFNIImpl<32>(b, lhs, rhs);
  case 64:
    return mulBF8PackedGFNIImpl<64>(b, lhs, rhs);
  default:
    llvm_unreachable("Unsupported vector size for GFNI");
  }
}

//===----------------------------------------------------------------------===//
// PCLMULQDQ-based Binary Field Multiplication
//===----------------------------------------------------------------------===//

// Tower field reduction polynomial for level 6 (64-bit): x⁶⁴ + x⁴ + x³ + x +
// 1 This is different from POLYVAL - we need proper tower reduction

// Multiply two 64-bit tower field elements using PCLMULQDQ
// Returns 64-bit result after reduction
Value mulBF64PCLMULQDQ(ImplicitLocOpBuilder &b, Value lhs, Value rhs) {
  auto i64Type = b.getI64Type();
  auto vec2i64Type = VectorType::get(2, i64Type);

  // Pack inputs into 128-bit vectors (low 64 bits used)
  Value zero = b.create<arith::ConstantOp>(i64Type, b.getI64IntegerAttr(0));
  Value lhsVec =
      b.create<vector::FromElementsOp>(vec2i64Type, ValueRange{lhs, zero});
  Value rhsVec =
      b.create<vector::FromElementsOp>(vec2i64Type, ValueRange{rhs, zero});

  // Carryless multiply: produces 128-bit result
  Value product = emitPCLMULQDQ128(b, lhsVec, rhsVec, 0x00);

  // Extract low and high 64-bit parts
  Value prodLow = b.create<vector::ExtractOp>(product, ArrayRef<int64_t>{0});
  Value prodHigh = b.create<vector::ExtractOp>(product, ArrayRef<int64_t>{1});

  // Reduce using tower polynomial for level 6
  Value reduction = reduceTowerLevel6(b, prodHigh);
  return b.create<arith::XOrIOp>(prodLow, reduction);
}

// Multiply two 128-bit tower field elements using PCLMULQDQ with Karatsuba
Value mulBF128PCLMULQDQ(ImplicitLocOpBuilder &b, Value lhs, Value rhs) {
  auto i64Type = b.getI64Type();
  auto vec2i64Type = VectorType::get(2, i64Type);

  // Karatsuba multiplication for 128-bit values
  // Let a = a_h * x⁶⁴ + a_l, b = b_h * x⁶⁴ + b_l
  // a * b = a_h*b_h * x^128 + (a_h*b_l + a_l*b_h) * x⁶⁴ + a_l*b_l

  // Low-low product
  Value ll = emitPCLMULQDQ128(b, lhs, rhs, 0x00);
  // High-high product
  Value hh = emitPCLMULQDQ128(b, lhs, rhs, 0x11);
  // Cross products
  Value lh = emitPCLMULQDQ128(b, lhs, rhs, 0x01);
  Value hl = emitPCLMULQDQ128(b, lhs, rhs, 0x10);

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
  // First reduce r3 (bits 192-255): r3 * x⁶⁴ * (x⁷ + x² + x + 1)
  auto [r3_red, r3_overflow] = reduceTowerLevel7(b, r3);
  r1 = b.create<arith::XOrIOp>(r1, r3_red);
  r2 = b.create<arith::XOrIOp>(r2, r3_overflow);

  // Now reduce r2 (bits 128-191): r2 * (x⁷ + x² + x + 1)
  auto [r2_red, r2_overflow] = reduceTowerLevel7(b, r2);
  r0 = b.create<arith::XOrIOp>(r0, r2_red);
  r1 = b.create<arith::XOrIOp>(r1, r2_overflow);

  // Pack result back into vector<2xi64>
  return b.create<vector::FromElementsOp>(vec2i64Type, ValueRange{r0, r1});
}

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

// Pattern for packed 8-bit binary field multiplication using GFNI
// This pattern works on vector types (after tensor->vector conversion)
struct ConvertPackedBF8MulToGFNI : public OpRewritePattern<MulOp> {
  using OpRewritePattern<MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const override {
    Type resultType = op.getResult().getType();

    // Only handle vector types (not tensors)
    // Tensors should be converted to vectors first via bufferization
    auto vecType = dyn_cast<VectorType>(resultType);
    if (!vecType)
      return failure();

    auto bfType = dyn_cast<BinaryFieldType>(vecType.getElementType());
    if (!bfType || bfType.getTowerLevel() != 3)
      return failure();

    // Check for supported vector sizes (128, 256, 512 bits)
    int64_t numElements = vecType.getNumElements();
    if (numElements != 16 && numElements != 32 && numElements != 64)
      return failure();

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // Convert bf<3> vectors to i8 vectors for SIMD operation
    auto vecI8Type = VectorType::get(numElements, b.getI8Type());

    // Use unrealized_conversion_cast for type conversion
    // This allows reconcile-unrealized-casts to clean up after
    // BinaryFieldToArith
    Value lhs = b.create<UnrealizedConversionCastOp>(vecI8Type, op.getLhs())
                    .getResult(0);
    Value rhs = b.create<UnrealizedConversionCastOp>(vecI8Type, op.getRhs())
                    .getResult(0);

    // Perform GFNI multiplication
    Value result = mulBF8PackedGFNI(b, lhs, rhs, numElements);

    // Cast result back to bf<3> vector
    Value resultCast =
        b.create<UnrealizedConversionCastOp>(vecType, result).getResult(0);

    rewriter.replaceOp(op, resultCast);
    return success();
  }
};

// Pattern for 64-bit binary field multiplication using PCLMULQDQ
struct ConvertBF64MulToPCLMULQDQ : public OpRewritePattern<MulOp> {
  using OpRewritePattern<MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const override {
    Type resultType = op.getResult().getType();

    // Check if this is bf<6> (64-bit binary field)
    auto bfType = dyn_cast<BinaryFieldType>(resultType);
    if (!bfType || bfType.getTowerLevel() != 6)
      return failure();

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

    Value result = mulBF64PCLMULQDQ(b, lhsI64, rhsI64);

    // Cast back to bf<6>
    Value resultBF =
        b.create<UnrealizedConversionCastOp>(bfType, result).getResult(0);
    rewriter.replaceOp(op, resultBF);
    return success();
  }
};

// Pattern for 128-bit binary field multiplication using PCLMULQDQ
struct ConvertBF128MulToPCLMULQDQ : public OpRewritePattern<MulOp> {
  using OpRewritePattern<MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const override {
    Type resultType = op.getResult().getType();

    // Check if this is bf<7> (128-bit binary field)
    auto bfType = dyn_cast<BinaryFieldType>(resultType);
    if (!bfType || bfType.getTowerLevel() != 7)
      return failure();

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

    // LLVM bitcast i128 -> vector<2xi64> for PCLMULQDQ operations
    Value lhsVec = b.create<LLVM::BitcastOp>(vec2i64Type, lhsI128);
    Value rhsVec = b.create<LLVM::BitcastOp>(vec2i64Type, rhsI128);

    Value resultVec = mulBF128PCLMULQDQ(b, lhsVec, rhsVec);

    // LLVM bitcast vector<2xi64> -> i128
    Value resultI128 = b.create<LLVM::BitcastOp>(i128Type, resultVec);

    // Cast i128 -> bf<7>
    Value resultBF =
        b.create<UnrealizedConversionCastOp>(bfType, resultI128).getResult(0);
    rewriter.replaceOp(op, resultBF);
    return success();
  }
};

// Pattern for packed 8-bit binary field squaring using GFNI
struct ConvertPackedBF8SquareToGFNI : public OpRewritePattern<SquareOp> {
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
    if (numElements != 16 && numElements != 32 && numElements != 64)
      return failure();

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto vecI8Type = VectorType::get(numElements, b.getI8Type());
    Value input = b.create<UnrealizedConversionCastOp>(vecI8Type, op.getInput())
                      .getResult(0);

    // Square = multiply by itself
    Value result = mulBF8PackedGFNI(b, input, input, numElements);

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

struct SpecializeBinaryFieldToX86
    : impl::SpecializeBinaryFieldToX86Base<SpecializeBinaryFieldToX86> {
  using SpecializeBinaryFieldToX86Base::SpecializeBinaryFieldToX86Base;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();

    RewritePatternSet patterns(context);

    if (useGFNI) {
      patterns.add<ConvertPackedBF8MulToGFNI, ConvertPackedBF8SquareToGFNI>(
          context);
    }

    if (usePCLMULQDQ) {
      patterns.add<ConvertBF64MulToPCLMULQDQ, ConvertBF128MulToPCLMULQDQ>(
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
