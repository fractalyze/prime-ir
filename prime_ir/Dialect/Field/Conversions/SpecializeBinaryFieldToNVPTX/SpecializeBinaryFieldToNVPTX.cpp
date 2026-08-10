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
#include "prime_ir/Dialect/Field/Conversions/BinaryFieldToArith/TowerFlatBasis.h"
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
// The `c` operand chains one XOR into the multiply for free, which the
// schedules below use to absorb fold terms. Kept as opaque inline
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

// Fold the 256-bit carryless product limbs r0..r3 (low to high) down to a GHASH
// field element via x¹²⁸ == x⁷ + x² + x + 1, then repack as i128. Shared by the
// multiply and the square so the two cannot drift in how they reduce.
Value reduceGhashProduct(ImplicitLocOpBuilder &b, Value r0, Value r1, Value r2,
                         Value r3) {
  auto i128Ty = b.getIntegerType(128);
  Value sh64 = arith::ConstantIntOp::create(b, 64, 128);

  auto [r3Red, r3Overflow] = reduceGhash(b, r3);
  r1 = arith::XOrIOp::create(b, r1, r3Red);
  r2 = arith::XOrIOp::create(b, r2, r3Overflow);

  auto [r2Red, r2Overflow] = reduceGhash(b, r2);
  r0 = arith::XOrIOp::create(b, r0, r2Red);
  r1 = arith::XOrIOp::create(b, r1, r2Overflow);

  Value r0Ext = arith::ExtUIOp::create(b, i128Ty, r0);
  Value r1Ext = arith::ExtUIOp::create(b, i128Ty, r1);
  Value r1Hi = arith::ShLIOp::create(b, r1Ext, sh64);
  return arith::OrIOp::create(b, r0Ext, r1Hi);
}

// Multiply two GHASH-basis i128 values using clmad. Karatsuba — 3 sub-products
// instead of 4 (cross term a₀b₁ + a₁b₀ = (a₀+a₁)(b₀+b₁) + a₀b₀ + a₁b₁), which
// trades two clmad for six XOR. Worth it because this multiply is
// clmad-issue-bound on sm_120, so the XORs issue alongside for free; see the
// pass's Performance notes for the measurement and when to re-check it.
//
// These are the same r0..r3 limbs the x86 PCLMULQDQ, ARM PMULL, and portable
// `emitGhashMul` paths produce from the same identity, so the reduction reuses
// the shared `reduceGhash` — the carryless-multiply backends fold the high half
// identically and cannot drift.
Value mulGhashClmad(ImplicitLocOpBuilder &b, Value lhsI128, Value rhsI128) {
  auto i64Ty = b.getI64Type();
  Value sh64 = arith::ConstantIntOp::create(b, 64, 128);

  Value a0 = arith::TruncIOp::create(b, i64Ty, lhsI128);
  Value a1 = arith::TruncIOp::create(b, i64Ty,
                                     arith::ShRUIOp::create(b, lhsI128, sh64));
  Value b0 = arith::TruncIOp::create(b, i64Ty, rhsI128);
  Value b1 = arith::TruncIOp::create(b, i64Ty,
                                     arith::ShRUIOp::create(b, rhsI128, sh64));

  Value z = arith::ConstantIntOp::create(b, 0, 64);

  Value aXor = arith::XOrIOp::create(b, a0, a1);
  Value bXor = arith::XOrIOp::create(b, b0, b1);

  Value llLo = emitClmad(b, a0, b0, z, /*isHi=*/false);
  Value llHi = emitClmad(b, a0, b0, z, /*isHi=*/true);
  Value hhLo = emitClmad(b, a1, b1, z, /*isHi=*/false);
  Value hhHi = emitClmad(b, a1, b1, z, /*isHi=*/true);

  // The `^ ll ^ hh` half of the cross term rides clmad's accumulate operand;
  // the other half is these two XORs.
  Value foldLo = arith::XOrIOp::create(b, llLo, hhLo);
  Value foldHi = arith::XOrIOp::create(b, llHi, hhHi);
  Value midLo = emitClmad(b, aXor, bXor, foldLo, /*isHi=*/false);
  Value midHi = emitClmad(b, aXor, bXor, foldHi, /*isHi=*/true);

  return reduceGhashProduct(b, llLo, arith::XOrIOp::create(b, llHi, midLo),
                            arith::XOrIOp::create(b, hhLo, midHi), hhHi);
}

// Square a GHASH-basis i128 value. In characteristic 2 the cross term vanishes
// — a₀·a₁ + a₁·a₀ = 0 — so only the two diagonal sub-products survive: four
// clmad against the multiply's six.
//
// Routing a square through mulGhashClmad would still emit the cross-term clmads
// and then XOR their result to zero. `clmad` is opaque inline asm, so nothing
// downstream can prove that and delete them.
Value squareGhashClmad(ImplicitLocOpBuilder &b, Value valI128) {
  auto i64Ty = b.getI64Type();
  Value sh64 = arith::ConstantIntOp::create(b, 64, 128);

  Value a0 = arith::TruncIOp::create(b, i64Ty, valI128);
  Value a1 = arith::TruncIOp::create(b, i64Ty,
                                     arith::ShRUIOp::create(b, valI128, sh64));
  Value z = arith::ConstantIntOp::create(b, 0, 64);

  return reduceGhashProduct(b, emitClmad(b, a0, a0, z, /*isHi=*/false),
                            emitClmad(b, a0, a0, z, /*isHi=*/true),
                            emitClmad(b, a1, a1, z, /*isHi=*/false),
                            emitClmad(b, a1, a1, z, /*isHi=*/true));
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

// Multiply two flat-basis 64-bit values, reducing mod y^64 + fLow: the
// product is two limbs, so 5 clmads — two for the limbs, one folding the
// high limb via y^64 == fLow, and two for the spill of that fold (at most
// deg(fLow) bits), which one more fold absorbs. Schedule proven against
// long division by tools/derive_tower_flat_basis.py.
Value mulFlat64Clmad(ImplicitLocOpBuilder &b, Value a64, Value b64,
                     uint64_t fLow) {
  Value z = arith::ConstantIntOp::create(b, 0, 64);
  Value fLowC = arith::ConstantIntOp::create(b, static_cast<int64_t>(fLow), 64);
  Value pLo = emitClmad(b, a64, b64, z, /*isHi=*/false);
  Value pHi = emitClmad(b, a64, b64, z, /*isHi=*/true);
  Value r = emitClmad(b, pHi, fLowC, pLo, /*isHi=*/false);
  Value spill = emitClmad(b, pHi, fLowC, z, /*isHi=*/true);
  return emitClmad(b, spill, fLowC, r, /*isHi=*/false);
}

// Apply the GF(2)-linear map with 128-bit columns (given as lo/hi halves)
// to an i128 value: XOR together the columns selected by the set bits.
Value emitBitMatrix128(ImplicitLocOpBuilder &b, Value x,
                       ArrayRef<uint64_t> colsLo, ArrayRef<uint64_t> colsHi) {
  auto i128Ty = b.getIntegerType(128);
  auto i128Const = [&](const llvm::APInt &v) {
    return arith::ConstantOp::create(b, i128Ty, IntegerAttr::get(i128Ty, v));
  };
  Value zero = i128Const(llvm::APInt(128, 0));
  Value acc = zero;
  for (unsigned i = 0; i < 128; ++i) {
    Value shifted = x;
    if (i != 0) {
      shifted = arith::ShRUIOp::create(b, x, i128Const(llvm::APInt(128, i)));
    }
    Value bit = arith::TruncIOp::create(b, b.getI1Type(), shifted);
    llvm::APInt col(128, colsLo[i]);
    col |= llvm::APInt(128, colsHi[i]).shl(64);
    Value sel = arith::SelectOp::create(b, bit, i128Const(col), zero);
    acc = arith::XOrIOp::create(b, acc, sel);
  }
  return acc;
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

// `field.square` on a GHASH value gets its own pattern rather than being routed
// through the multiply with both operands equal, so it can drop the cross-term
// products (see squareGhashClmad). replaceFlatFieldMul is op-agnostic and takes
// the input twice; only the scalar callback differs.
struct ConvertGhashSquareToClmad : public OpRewritePattern<SquareOp> {
  using OpRewritePattern<SquareOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SquareOp op,
                                PatternRewriter &rewriter) const override {
    auto bfType = dyn_cast<BinaryFieldType>(
        getElementTypeOrSelf(op.getResult().getType()));
    if (!bfType || !bfType.isGhash())
      return failure();
    Type ghashType = bfType;

    auto squareScalar = [ghashType](ImplicitLocOpBuilder &b, Value val,
                                    Value) -> Value {
      auto i128Type = b.getIntegerType(128);
      Value valI128 =
          UnrealizedConversionCastOp::create(b, i128Type, val).getResult(0);
      Value resultI128 = squareGhashClmad(b, valI128);
      return UnrealizedConversionCastOp::create(b, ghashType, resultI128)
          .getResult(0);
    };
    return replaceFlatFieldMul(rewriter, op, op.getInput(), op.getInput(),
                               squareScalar);
  }
};

// Tower bf<4>/bf<5> multiply via flat-basis conversion + one clmad product.
// bf<6> would additionally need clmad.hi limbs and a degree-64 modulus.
struct ConvertTowerMulToClmad : public OpRewritePattern<MulOp> {
  using OpRewritePattern<MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const override {
    // getElementTypeOrSelf so a shaped (tensor/vector) tower mul matches too;
    // replaceFlatFieldMul unrolls it lane by lane, and the conversion ladders
    // then only ever see the extracted scalars.
    auto bfType = dyn_cast<BinaryFieldType>(
        getElementTypeOrSelf(op.getResult().getType()));
    if (!bfType || !bfType.isTower())
      return failure();
    unsigned level = bfType.getTowerLevel();
    if (level < 4 || level > 7)
      return failure();
    const unsigned n = 1u << level;
    Type towerType = bfType;

    auto mulScalar = [=](ImplicitLocOpBuilder &b, Value lhs,
                         Value rhs) -> Value {
      auto intNType = b.getIntegerType(n);

      // Cast tower -> iN; BinaryFieldToArith later reconciles these casts.
      Value lhsN =
          UnrealizedConversionCastOp::create(b, intNType, lhs).getResult(0);
      Value rhsN =
          UnrealizedConversionCastOp::create(b, intNType, rhs).getResult(0);

      Value resultN;
      if (level == 7) {
        // 128-bit: route through the GHASH basis so level 7 shares
        // mulGhashClmad instead of carrying a second product schedule.
        Value lhsFlat =
            emitBitMatrix128(b, lhsN, kTowerToFlat128Lo, kTowerToFlat128Hi);
        Value rhsFlat =
            emitBitMatrix128(b, rhsN, kTowerToFlat128Lo, kTowerToFlat128Hi);
        Value prod = mulGhashClmad(b, lhsFlat, rhsFlat);
        resultN =
            emitBitMatrix128(b, prod, kFlatToTower128Lo, kFlatToTower128Hi);
      } else {
        ArrayRef<uint64_t> toFlat = level == 4   ? ArrayRef(kTowerToFlat16)
                                    : level == 5 ? ArrayRef(kTowerToFlat32)
                                                 : ArrayRef(kTowerToFlat64);
        ArrayRef<uint64_t> fromFlat = level == 4   ? ArrayRef(kFlatToTower16)
                                      : level == 5 ? ArrayRef(kFlatToTower32)
                                                   : ArrayRef(kFlatToTower64);
        const uint64_t fLow = BinaryFieldType::kCanonicalFlatModLow[level];
        auto i64Type = b.getI64Type();
        Value lhsFlat = emitBitMatrix(b, lhsN, toFlat, n);
        Value rhsFlat = emitBitMatrix(b, rhsN, toFlat, n);
        Value prodFlat;
        if (n == 64) {
          prodFlat = mulFlat64Clmad(b, lhsFlat, rhsFlat, fLow);
        } else {
          Value prod64 = mulFlatClmad(
              b, arith::ExtUIOp::create(b, i64Type, lhsFlat),
              arith::ExtUIOp::create(b, i64Type, rhsFlat), n, fLow);
          prodFlat = arith::TruncIOp::create(b, intNType, prod64);
        }
        resultN = emitBitMatrix(b, prodFlat, fromFlat, n);
      }
      return UnrealizedConversionCastOp::create(b, towerType, resultN)
          .getResult(0);
    };
    return replaceFlatFieldMul(rewriter, op, op.getLhs(), op.getRhs(),
                               mulScalar);
  }
};

// Generic flat multiply (isGenericFlat, levels 1-6): clmad product plus
// constant-tap folds, modulus read off the type; no conversion ladders.
struct ConvertFlatGenericMulToClmad : public OpRewritePattern<MulOp> {
  using OpRewritePattern<MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const override {
    // getElementTypeOrSelf so a shaped (tensor/vector) flat mul matches too;
    // replaceFlatFieldMul unrolls it lane by lane.
    auto bfType = dyn_cast<BinaryFieldType>(
        getElementTypeOrSelf(op.getResult().getType()));
    if (!bfType || !bfType.isGenericFlat())
      return failure();
    // Verifier caps generic flat at level 6, so n <= 64.
    const unsigned n = bfType.getBitWidth();
    const uint64_t fLow = bfType.getFlatModLow();
    Type flatType = bfType;
    // The storage carrier, not the element width: sub-byte levels ride i8.
    auto storageType = bfType.getStorageType();

    auto mulScalar = [=](ImplicitLocOpBuilder &b, Value lhs,
                         Value rhs) -> Value {
      auto i64Type = b.getI64Type();
      // Cast flat -> storage int; BinaryFieldToArith later reconciles these
      // casts.
      Value lhsS =
          UnrealizedConversionCastOp::create(b, storageType, lhs).getResult(0);
      Value rhsS =
          UnrealizedConversionCastOp::create(b, storageType, rhs).getResult(0);
      // Sub-byte elements sit in a wider carrier that field.bitcast retags
      // without normalizing; clmad multiplies the whole carrier, so junk
      // above bit n-1 would fold into the result.
      if (storageType.getWidth() > n) {
        Value mask = arith::ConstantIntOp::create(
            b, static_cast<int64_t>((uint64_t{1} << n) - 1),
            storageType.getWidth());
        lhsS = arith::AndIOp::create(b, lhsS, mask);
        rhsS = arith::AndIOp::create(b, rhsS, mask);
      }

      Value prod;
      if (n == 64) {
        prod = mulFlat64Clmad(b, lhsS, rhsS, fLow);
      } else {
        Value prod64 =
            mulFlatClmad(b, arith::ExtUIOp::create(b, i64Type, lhsS),
                         arith::ExtUIOp::create(b, i64Type, rhsS), n, fLow);
        // Junk above bit n-1 must not leak into a wider storage carrier.
        if (storageType.getWidth() > n) {
          prod64 = arith::AndIOp::create(
              b, prod64,
              arith::ConstantIntOp::create(
                  b, static_cast<int64_t>((uint64_t{1} << n) - 1), 64));
        }
        prod = arith::TruncIOp::create(b, storageType, prod64);
      }
      return UnrealizedConversionCastOp::create(b, flatType, prod).getResult(0);
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
      patterns.add<ConvertGhashMulToClmad, ConvertGhashSquareToClmad,
                   ConvertTowerMulToClmad, ConvertFlatGenericMulToClmad>(
          context);
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
