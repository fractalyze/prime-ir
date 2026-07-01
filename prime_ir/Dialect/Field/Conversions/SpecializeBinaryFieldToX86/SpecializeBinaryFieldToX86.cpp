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
  Value matrixConst = arith::ConstantOp::create(
      b,
      DenseElementsAttr::get(matrixVecType, ArrayRef<int64_t>(matrixValues)));

  std::string asmString = "vgf2p8affineqb $0, $1, $2, " + std::to_string(imm);
  return LLVM::InlineAsmOp::create(
             b, vecType, ValueRange{input, matrixConst}, asmString, "=x,x,x",
             /*has_side_effects=*/false,
             /*is_align_stack=*/true, LLVM::TailCallKind::None,
             LLVM::AsmDialectAttr::get(b.getContext(),
                                       LLVM::AsmDialect::AD_Intel),
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
  return LLVM::InlineAsmOp::create(
             b, vecType, ValueRange{lhs, rhs}, "vgf2p8mulb $0, $1, $2",
             "=x,x,x",
             /*has_side_effects=*/false,
             /*is_align_stack=*/true, LLVM::TailCallKind::None,
             LLVM::AsmDialectAttr::get(b.getContext(),
                                       LLVM::AsmDialect::AD_Intel),
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
  return LLVM::InlineAsmOp::create(
             b, vecType, ValueRange{lhs, rhs}, asmString, "=x,x,x",
             /*has_side_effects=*/false,
             /*is_align_stack=*/true, LLVM::TailCallKind::None,
             LLVM::AsmDialectAttr::get(b.getContext(),
                                       LLVM::AsmDialect::AD_Intel),
             /*operand_attrs=*/ArrayAttr())
      .getResult(0);
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

// Multiply two GHASH-basis i128 values (as vector<2xi64>) using PCLMULQDQ.
// Karatsuba — 3 PCLMULQDQ (the cross term a₀b₁ + a₁b₀ = (a₀+a₁)(b₀+b₁) + ll +
// hh), then reduce mod x¹²⁸ + x⁷ + x² + x + 1.
Value mulGhashPCLMULQDQ(ImplicitLocOpBuilder &b, Value lhs, Value rhs) {
  auto i64Type = b.getI64Type();
  auto vec2i64Type = VectorType::get(2, i64Type);

  // Swap the two 64-bit lanes so each lane of *Xor holds (lo ^ hi).
  Value lhsSwapped =
      vector::ShuffleOp::create(b, lhs, lhs, ArrayRef<int64_t>{1, 0});
  Value rhsSwapped =
      vector::ShuffleOp::create(b, rhs, rhs, ArrayRef<int64_t>{1, 0});
  Value lhsXor = arith::XOrIOp::create(b, lhs, lhsSwapped);
  Value rhsXor = arith::XOrIOp::create(b, rhs, rhsSwapped);

  Value ll = emitPCLMULQDQ128(b, lhs, rhs, 0x00);
  Value hh = emitPCLMULQDQ128(b, lhs, rhs, 0x11);
  Value m = emitPCLMULQDQ128(b, lhsXor, rhsXor, 0x00);

  Value mid = arith::XOrIOp::create(b, arith::XOrIOp::create(b, m, ll), hh);

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

  return vector::FromElementsOp::create(b, vec2i64Type, ValueRange{r0, r1});
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
    Value lhs = UnrealizedConversionCastOp::create(b, vecI8Type, op.getLhs())
                    .getResult(0);
    Value rhs = UnrealizedConversionCastOp::create(b, vecI8Type, op.getRhs())
                    .getResult(0);

    // Perform GFNI multiplication
    Value result = mulBF8PackedGFNI(b, lhs, rhs, numElements);

    // Cast result back to bf<3> vector
    Value resultCast =
        UnrealizedConversionCastOp::create(b, vecType, result).getResult(0);

    rewriter.replaceOp(op, resultCast);
    return success();
  }
};

// Pattern for the GHASH-basis multiply (`bf<7, ghash>`) using PCLMULQDQ. (The
// tower bf<6>/bf<7> deliberately do NOT specialize to a carryless multiply --
// that computes the flat GHASH product, not the tower -- so they lower via the
// portable recursive mulTower.)
struct ConvertGhashMulToPCLMULQDQ : public OpRewritePattern<MulOp> {
  using OpRewritePattern<MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const override {
    Type ghashType = op.getResult().getType();
    auto bfType = dyn_cast<BinaryFieldType>(ghashType);
    if (!bfType || !bfType.isGhash())
      return failure();

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto i128Type = b.getIntegerType(128);
    auto vec2i64Type = VectorType::get(2, b.getI64Type());

    // Cast ghash -> i128; BinaryFieldToArith later reconciles these casts.
    Value lhsI128 = UnrealizedConversionCastOp::create(b, i128Type, op.getLhs())
                        .getResult(0);
    Value rhsI128 = UnrealizedConversionCastOp::create(b, i128Type, op.getRhs())
                        .getResult(0);

    Value lhsVec = LLVM::BitcastOp::create(b, vec2i64Type, lhsI128);
    Value rhsVec = LLVM::BitcastOp::create(b, vec2i64Type, rhsI128);

    Value resultVec = mulGhashPCLMULQDQ(b, lhsVec, rhsVec);

    Value resultI128 = LLVM::BitcastOp::create(b, i128Type, resultVec);
    Value resultGhash =
        UnrealizedConversionCastOp::create(b, ghashType, resultI128)
            .getResult(0);
    rewriter.replaceOp(op, resultGhash);
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
    Value input =
        UnrealizedConversionCastOp::create(b, vecI8Type, op.getInput())
            .getResult(0);

    // Square = multiply by itself
    Value result = mulBF8PackedGFNI(b, input, input, numElements);

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
      patterns.add<ConvertGhashMulToPCLMULQDQ>(context);
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
