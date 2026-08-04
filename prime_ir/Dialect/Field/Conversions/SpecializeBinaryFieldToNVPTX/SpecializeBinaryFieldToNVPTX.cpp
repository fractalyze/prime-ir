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

#include "prime_ir/Dialect/Field/Conversions/SpecializeBinaryFieldToNVPTX/SpecializeBinaryFieldToNVPTX.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"    // IWYU pragma: keep
#include "mlir/Dialect/Vector/IR/VectorOps.h" // IWYU pragma: keep
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "prime_ir/Dialect/Field/Conversions/BinaryFieldToArith/BinaryFieldTables.h"
#include "prime_ir/Dialect/Field/Conversions/SpecializeBinaryFieldToNVPTX/TowerFlatBasis.h"
#include "prime_ir/Dialect/Field/IR/FieldDialect.h"
#include "prime_ir/Dialect/Field/IR/FieldOps.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"
#include "prime_ir/Utils/ShapedFieldMul.h"

namespace mlir::prime_ir::field {

#define GEN_PASS_DEF_SPECIALIZEBINARYFIELDTONVPTX
#include "prime_ir/Dialect/Field/Conversions/SpecializeBinaryFieldToNVPTX/SpecializeBinaryFieldToNVPTX.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// clmad Intrinsic Helper
//===----------------------------------------------------------------------===//

// Emit a single PTX `clmad.{lo,hi}.u64` (carryless multiply-add, PTX ISA 9.3):
//   dst = carryless_product_{lo|hi}(a, b) XOR c
// `lo` selects bits 0..63 of the 64x64 carryless product, `hi` bits 64..127.
// The `c` operand chains the XOR-accumulation for free, which the schoolbook
// below uses to fold cross terms without extra XOR ops. Kept as opaque inline
// asm so `clmad` survives to PTX regardless of the LLVM NVPTX backend version;
// only ptxas (CUDA 13.3+) needs to know the instruction.
Value emitClmad(ImplicitLocOpBuilder &b, Value a, Value bv, Value c,
                bool isHi) {
  StringRef asmString =
      isHi ? "clmad.hi.u64 $0, $1, $2, $3;" : "clmad.lo.u64 $0, $1, $2, $3;";
  return LLVM::InlineAsmOp::create(
             b, b.getI64Type(), ValueRange{a, bv, c}, asmString, "=l,l,l,l",
             /*has_side_effects=*/false,
             /*is_align_stack=*/false, LLVM::TailCallKind::None,
             LLVM::AsmDialectAttr::get(b.getContext(),
                                       LLVM::AsmDialect::AD_ATT),
             /*operand_attrs=*/ArrayAttr())
      .getResult(0);
}

//===----------------------------------------------------------------------===//
// clmad-based GHASH Multiplication
//===----------------------------------------------------------------------===//

// Multiply two GHASH-basis i128 values using clmad. Eight clmad build the
// 128x128 -> 256-bit carryless product as limbs r0..r3 (low to high), folding
// the two cross terms via clmad's XOR-accumulate operand exactly as flock's
// clmad kernel (optim/clmad/ghash_mul.ptx):
//   r0 = ll_lo
//   r1 = ll_hi ^ (a1·b0)_lo ^ (a0·b1)_lo   (= ll_hi ^ mid_lo)
//   r2 = hh_lo ^ (a1·b0)_hi ^ (a0·b1)_hi   (= hh_lo ^ mid_hi)
//   r3 = hh_hi
// These are the same r0..r3 limbs the x86 PCLMULQDQ path produces, so the
// reduction reuses the shared `reduceGhash` — the carryless-multiply backends
// fold the high half identically and cannot drift.
Value mulGhashClmad(ImplicitLocOpBuilder &b, Value lhsI128, Value rhsI128) {
  auto i64Ty = b.getI64Type();
  auto i128Ty = b.getIntegerType(128);
  Value sh64 = arith::ConstantIntOp::create(b, 64, 128);

  Value a0 = arith::TruncIOp::create(b, i64Ty, lhsI128);
  Value a1 = arith::TruncIOp::create(b, i64Ty,
                                     arith::ShRUIOp::create(b, lhsI128, sh64));
  Value b0 = arith::TruncIOp::create(b, i64Ty, rhsI128);
  Value b1 = arith::TruncIOp::create(b, i64Ty,
                                     arith::ShRUIOp::create(b, rhsI128, sh64));

  Value z = arith::ConstantIntOp::create(b, 0, 64);

  Value r0 = emitClmad(b, a0, b0, z, /*isHi=*/false);
  Value t1 = emitClmad(b, a1, b0, z, /*isHi=*/false);
  t1 = emitClmad(b, a0, b1, t1, /*isHi=*/false);
  Value r1 = emitClmad(b, a0, b0, t1, /*isHi=*/true);
  Value t2 = emitClmad(b, a1, b0, z, /*isHi=*/true);
  t2 = emitClmad(b, a0, b1, t2, /*isHi=*/true);
  Value r2 = emitClmad(b, a1, b1, t2, /*isHi=*/false);
  Value r3 = emitClmad(b, a1, b1, z, /*isHi=*/true);

  // Fold the high half down via x¹²⁸ == x⁷ + x² + x + 1.
  auto [r3Red, r3Overflow] = reduceGhash(b, r3);
  r1 = arith::XOrIOp::create(b, r1, r3Red);
  r2 = arith::XOrIOp::create(b, r2, r3Overflow);

  auto [r2Red, r2Overflow] = reduceGhash(b, r2);
  r0 = arith::XOrIOp::create(b, r0, r2Red);
  r1 = arith::XOrIOp::create(b, r1, r2Overflow);

  // Reassemble the 128-bit result: r0 | (r1 << 64).
  Value r0Ext = arith::ExtUIOp::create(b, i128Ty, r0);
  Value r1Ext = arith::ExtUIOp::create(b, i128Ty, r1);
  Value r1Hi = arith::ShLIOp::create(b, r1Ext, sh64);
  return arith::OrIOp::create(b, r0Ext, r1Hi);
}

//===----------------------------------------------------------------------===//
// clmad-based Tower Multiplication (via flat-basis conversion)
//===----------------------------------------------------------------------===//

// Tower bf<4>/bf<5> multiply through the isomorphic flat polynomial basis
// (constants and rationale in TowerFlatBasis.h):
//
//   mul(a, b) = fromFlat(reduce(clmul(toFlat(a), toFlat(b))))

// Apply the GF(2)-linear map `cols` (column i = image of basis bit i) to an
// n-bit value: XOR together the columns selected by the set bits.
Value emitBitMatrix(ImplicitLocOpBuilder &b, Value x, ArrayRef<uint64_t> cols,
                    unsigned n) {
  Value zero = arith::ConstantIntOp::create(b, 0, n);
  Value acc = zero;
  for (unsigned i = 0; i < n; ++i) {
    Value shifted = x;
    if (i != 0) {
      Value sh = arith::ConstantIntOp::create(b, i, n);
      shifted = arith::ShRUIOp::create(b, x, sh);
    }
    Value bit = arith::TruncIOp::create(b, b.getI1Type(), shifted);
    Value col =
        arith::ConstantIntOp::create(b, static_cast<int64_t>(cols[i]), n);
    Value sel = arith::SelectOp::create(b, bit, col, zero);
    acc = arith::XOrIOp::create(b, acc, sel);
  }
  return acc;
}

// Multiply two flat-basis values of width n (16 or 32) held in i64, reducing
// mod y^n + fLow. The product fits 2n-1 <= 63 bits, so one clmad.lo covers
// it; two folds finish because fLow has degree < 8:
//   p  = clmul(a, b)                      (bits 0 .. 2n-2)
//   t1 = p ^ clmul(p >> n, fLow)          (bits 0..n-1 correct, junk above)
//   h2 = (t1 >> n) ^ (p >> n)             (= fold1 >> n, the second fold input)
//   t2 = t1 ^ clmul(h2, fLow)             (bits 0..n-1 are the result)
// The final truncation to n bits discards the junk the folds leave above.
Value mulFlatClmad(ImplicitLocOpBuilder &b, Value a64, Value b64, unsigned n,
                   uint64_t fLow) {
  Value z = arith::ConstantIntOp::create(b, 0, 64);
  Value shN = arith::ConstantIntOp::create(b, n, 64);
  Value fLowC = arith::ConstantIntOp::create(b, static_cast<int64_t>(fLow), 64);

  Value p = emitClmad(b, a64, b64, z, /*isHi=*/false);
  Value hi = arith::ShRUIOp::create(b, p, shN);
  Value t1 = emitClmad(b, hi, fLowC, p, /*isHi=*/false);
  Value h2 = arith::XOrIOp::create(b, arith::ShRUIOp::create(b, t1, shN), hi);
  return emitClmad(b, h2, fLowC, t1, /*isHi=*/false);
}

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

// Pattern for the GHASH-basis multiply (`bf<7, ghash>`) using clmad. As with
// the x86 PCLMULQDQ path, tower bf<6>/bf<7> deliberately do NOT specialize --
// the carryless product computes the flat GHASH product, not the tower -- so
// they lower via the portable recursive mulTower in binary-field-to-arith.
struct ConvertGhashMulToClmad : public OpRewritePattern<MulOp> {
  using OpRewritePattern<MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const override {
    // getElementTypeOrSelf so a shaped (tensor/vector) ghash mul matches too;
    // replaceFlatFieldMul unrolls it lane by lane.
    auto bfType = dyn_cast<BinaryFieldType>(
        getElementTypeOrSelf(op.getResult().getType()));
    if (!bfType || !bfType.isGhash())
      return failure();
    Type ghashType = bfType;

    auto mulScalar = [ghashType](ImplicitLocOpBuilder &b, Value lhs,
                                 Value rhs) -> Value {
      auto i128Type = b.getIntegerType(128);
      // Cast ghash -> i128; BinaryFieldToArith later reconciles these casts.
      Value lhsI128 =
          UnrealizedConversionCastOp::create(b, i128Type, lhs).getResult(0);
      Value rhsI128 =
          UnrealizedConversionCastOp::create(b, i128Type, rhs).getResult(0);
      Value resultI128 = mulGhashClmad(b, lhsI128, rhsI128);
      return UnrealizedConversionCastOp::create(b, ghashType, resultI128)
          .getResult(0);
    };
    return replaceFlatFieldMul(rewriter, op, op.getLhs(), op.getRhs(),
                               mulScalar);
  }
};

// Tower bf<4>/bf<5> multiply via flat-basis conversion + one clmad product.
// bf<6> would additionally need clmad.hi limbs and a degree-64 modulus.
struct ConvertTowerMulToClmad : public OpRewritePattern<MulOp> {
  using OpRewritePattern<MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const override {
    Type towerType = op.getResult().getType();
    auto bfType = dyn_cast<BinaryFieldType>(towerType);
    if (!bfType || !bfType.isTower())
      return failure();
    unsigned level = bfType.getTowerLevel();
    if (level != 4 && level != 5)
      return failure();
    const unsigned n = 1u << level;

    ArrayRef<uint64_t> toFlat =
        level == 4 ? ArrayRef(kTowerToFlat16) : ArrayRef(kTowerToFlat32);
    ArrayRef<uint64_t> fromFlat =
        level == 4 ? ArrayRef(kFlatToTower16) : ArrayRef(kFlatToTower32);
    uint64_t fLow = level == 4 ? kFlatModLow16 : kFlatModLow32;

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto intNType = b.getIntegerType(n);
    auto i64Type = b.getI64Type();

    // Cast tower -> iN; BinaryFieldToArith later reconciles these casts.
    Value lhsN = UnrealizedConversionCastOp::create(b, intNType, op.getLhs())
                     .getResult(0);
    Value rhsN = UnrealizedConversionCastOp::create(b, intNType, op.getRhs())
                     .getResult(0);

    Value lhsFlat =
        arith::ExtUIOp::create(b, i64Type, emitBitMatrix(b, lhsN, toFlat, n));
    Value rhsFlat =
        arith::ExtUIOp::create(b, i64Type, emitBitMatrix(b, rhsN, toFlat, n));

    Value prodFlat64 = mulFlatClmad(b, lhsFlat, rhsFlat, n, fLow);
    Value prodFlat = arith::TruncIOp::create(b, intNType, prodFlat64);

    Value resultN = emitBitMatrix(b, prodFlat, fromFlat, n);
    Value resultTower =
        UnrealizedConversionCastOp::create(b, towerType, resultN).getResult(0);
    rewriter.replaceOp(op, resultTower);
    return success();
  }
};

// AES-basis bf<3, aes> multiply via one clmad product. The AES basis is
// already the flat monomial basis GF(2)[x]/(x⁸ + x⁴ + x³ + x + 1) (0x11B), so
// -- unlike the tower path -- no toFlat/fromFlat bit-matrix is needed. The 8×8
// product is 15-bit, so one clmad.lo covers it and two folds finish. Result is
// byte-identical to BinaryFieldToArith's emitAesMul (the software fallback).
struct ConvertAesMulToClmad : public OpRewritePattern<MulOp> {
  using OpRewritePattern<MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const override {
    // getElementTypeOrSelf so a shaped (tensor/vector) aes mul matches too;
    // replaceFlatFieldMul unrolls it lane by lane.
    auto bfType = dyn_cast<BinaryFieldType>(
        getElementTypeOrSelf(op.getResult().getType()));
    if (!bfType || !bfType.isAes())
      return failure();
    Type aesType = bfType;

    auto mulScalar = [aesType](ImplicitLocOpBuilder &b, Value lhs,
                               Value rhs) -> Value {
      auto i8Type = b.getI8Type();
      auto i64Type = b.getI64Type();
      // Cast aes -> i8; BinaryFieldToArith later reconciles these casts.
      Value lhs8 =
          UnrealizedConversionCastOp::create(b, i8Type, lhs).getResult(0);
      Value rhs8 =
          UnrealizedConversionCastOp::create(b, i8Type, rhs).getResult(0);
      // 0x1b = x⁴ + x³ + x + 1, the low half of the AES modulus x⁸ + 0x1b.
      Value prodFlat64 =
          mulFlatClmad(b, arith::ExtUIOp::create(b, i64Type, lhs8),
                       arith::ExtUIOp::create(b, i64Type, rhs8),
                       /*n=*/8, /*fLow=*/0x1b);
      Value prod8 = arith::TruncIOp::create(b, i8Type, prodFlat64);
      return UnrealizedConversionCastOp::create(b, aesType, prod8).getResult(0);
    };
    return replaceFlatFieldMul(rewriter, op, op.getLhs(), op.getRhs(),
                               mulScalar);
  }
};

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct SpecializeBinaryFieldToNVPTX
    : impl::SpecializeBinaryFieldToNVPTXBase<SpecializeBinaryFieldToNVPTX> {
  using SpecializeBinaryFieldToNVPTXBase::SpecializeBinaryFieldToNVPTXBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();

    RewritePatternSet patterns(context);
    if (useClmad) {
      patterns.add<ConvertGhashMulToClmad, ConvertTowerMulToClmad,
                   ConvertAesMulToClmad>(context);
    }

    // Greedy rewriting (not partial conversion) so unmatched field.mul ops
    // fall through gracefully. Folding disabled: MLIR's folder doesn't
    // understand binary field types (matches the x86/ARM specializers).
    GreedyRewriteConfig config;
    config.enableFolding(false);
    if (failed(applyPatternsGreedily(module, std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::prime_ir::field
