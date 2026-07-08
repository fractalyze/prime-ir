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
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "prime_ir/Dialect/Field/Conversions/BinaryFieldToArith/BinaryFieldTables.h"
#include "prime_ir/Dialect/Field/IR/FieldDialect.h"
#include "prime_ir/Dialect/Field/IR/FieldOps.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"

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
// Conversion Pattern
//===----------------------------------------------------------------------===//

// Pattern for the GHASH-basis multiply (`bf<7, ghash>`) using clmad. As with
// the x86 PCLMULQDQ path, tower bf<6>/bf<7> deliberately do NOT specialize --
// the carryless product computes the flat GHASH product, not the tower -- so
// they lower via the portable recursive mulTower in binary-field-to-arith.
struct ConvertGhashMulToClmad : public OpRewritePattern<MulOp> {
  using OpRewritePattern<MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const override {
    Type ghashType = op.getResult().getType();
    auto bfType = dyn_cast<BinaryFieldType>(ghashType);
    if (!bfType || !bfType.isGhash())
      return failure();

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto i128Type = b.getIntegerType(128);

    // Cast ghash -> i128; BinaryFieldToArith later reconciles these casts.
    Value lhsI128 = UnrealizedConversionCastOp::create(b, i128Type, op.getLhs())
                        .getResult(0);
    Value rhsI128 = UnrealizedConversionCastOp::create(b, i128Type, op.getRhs())
                        .getResult(0);

    Value resultI128 = mulGhashClmad(b, lhsI128, rhsI128);

    Value resultGhash =
        UnrealizedConversionCastOp::create(b, ghashType, resultI128)
            .getResult(0);
    rewriter.replaceOp(op, resultGhash);
    return success();
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
      patterns.add<ConvertGhashMulToClmad>(context);
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
