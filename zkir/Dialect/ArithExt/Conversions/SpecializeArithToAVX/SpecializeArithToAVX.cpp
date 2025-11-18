/* Copyright 2025 The ZKIR Authors.

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

#include "zkir/Dialect/ArithExt/Conversions/SpecializeArithToAVX/SpecializeArithToAVX.h"

#include <optional>
#include <type_traits>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::zkir::arith_ext {

#define GEN_PASS_DEF_SPECIALIZEARITHTOAVX
#include "zkir/Dialect/ArithExt/Conversions/SpecializeArithToAVX/SpecializeArithToAVX.h.inc"

namespace {
inline bool isConstantSplat(Value value) {
  if (auto constantOp = value.getDefiningOp<arith::ConstantOp>()) {
    return isa<SplatElementsAttr>(constantOp.getValueAttr());
  }
  return false;
}

inline bool canSpecialize(Operation *op) {
  return isa<arith::AddIOp>(op) || isa<arith::SubIOp>(op) ||
         isa<arith::MulIOp>(op) || isa<arith::MulUIExtendedOp>(op) ||
         isa<arith::MulSIExtendedOp>(op);
}

// Multiplies two vector<16xi32> operands using the vpmuludq instruction.
//
// vpmuludq performs extended multiplication on only the even lanes, producing
// vector<8xi64> results. This function bitcasts the results and returns a pair
// of vector<16xi32> values representing the extended products:
// - First: even lane extended products
// - Second: odd lane extended products
std::pair<Value, Value> mulExtendedByOddEven(ImplicitLocOpBuilder &b,
                                             Value lhsEven, Value lhsOdd,
                                             Value rhsEven, Value rhsOdd,
                                             bool isSigned = false) {
  std::string asmMulString =
      isSigned ? "vpmuldq $0, $1, $2" : "vpmuludq $0, $1, $2";
  auto vecI32Type = VectorType::get(16, b.getI32Type());
  auto vecI64Type = VectorType::get(8, b.getI64Type());
  Value prodEven64 = b.create<LLVM::InlineAsmOp>(
                          vecI64Type, ValueRange{lhsEven, rhsEven},
                          asmMulString, "=x,x,x", /*has_side_effects=*/false,
                          /*is_align_stack=*/true, LLVM::TailCallKind::None,
                          /*asm_dialect=*/
                          LLVM::AsmDialectAttr::get(b.getContext(),
                                                    LLVM::AsmDialect::AD_Intel),
                          /*operand_attrs=*/ArrayAttr())
                         .getResult(0);
  Value prodOdd64 = b.create<LLVM::InlineAsmOp>(
                         vecI64Type, ValueRange{lhsOdd, rhsOdd}, asmMulString,
                         "=x,x,x", /*has_side_effects=*/false,
                         /*is_align_stack=*/true, LLVM::TailCallKind::None,
                         /*asm_dialect=*/
                         LLVM::AsmDialectAttr::get(b.getContext(),
                                                   LLVM::AsmDialect::AD_Intel),
                         /*operand_attrs=*/ArrayAttr())
                        .getResult(0);

  // cast them to vector<16xi32> so even lanes are the low parts and odd
  // lanes are the high parts
  auto prodEven32 = b.create<vector::BitCastOp>(vecI32Type, prodEven64);
  auto prodOdd32 = b.create<vector::BitCastOp>(vecI32Type, prodOdd64);
  return {prodEven32, prodOdd32};
}

// Helper for arith.addi and arith.subi
template <typename OpType>
std::pair<Value, Value> addSubByOddEven(ImplicitLocOpBuilder &b, Value lhsEven,
                                        Value lhsOdd, Value rhsEven,
                                        Value rhsOdd) {
  auto vecType = VectorType::get(16, b.getI32Type());

  const char *asmString;
  if constexpr (std::is_same_v<OpType, arith::AddIOp>) {
    asmString = "vpaddd $0, $1, $2";
  } else if constexpr (std::is_same_v<OpType, arith::SubIOp>) {
    asmString = "vpsubd $0, $1, $2";
  }

  Value resEven =
      b.create<LLVM::InlineAsmOp>(
           vecType, ValueRange{lhsEven, rhsEven}, asmString, "=x,x,x",
           /*has_side_effects=*/false,
           /*is_align_stack=*/true, LLVM::TailCallKind::None,
           /*asm_dialect=*/
           LLVM::AsmDialectAttr::get(b.getContext(),
                                     LLVM::AsmDialect::AD_Intel),
           /*operand_attrs=*/ArrayAttr())
          .getResult(0);

  Value resOdd = b.create<LLVM::InlineAsmOp>(
                      vecType, ValueRange{lhsOdd, rhsOdd}, asmString, "=x,x,x",
                      /*has_side_effects=*/false,
                      /*is_align_stack=*/true, LLVM::TailCallKind::None,
                      /*asm_dialect=*/
                      LLVM::AsmDialectAttr::get(b.getContext(),
                                                LLVM::AsmDialect::AD_Intel),
                      /*operand_attrs=*/ArrayAttr())
                     .getResult(0);

  return {resEven, resOdd};
}

// Gathers the low parts of two vectors of 16 32-bit integers.
// [a₀, a₁, a₂, a₃, a₄, a₅, a₆, a₇, a₈, a₉, a₁₀, a₁₁, a₁₂, a₁₃, a₁₄, a₁₅]
// [b₀, b₁, b₂, b₃, b₄, b₅, b₆, b₇, b₈, b₉, b₁₀, b₁₁, b₁₂, b₁₃, b₁₄, b₁₅]
// => [a₀, b₀, a₂, b₂, a₄, b₄, a₆, b₆, a₈, b₈, a₁₀, b₁₀, a₁₂, b₁₂, a₁₄, b₁₄]
Value gatherLowsInterleaved(ImplicitLocOpBuilder &b, Value even, Value odd) {
  // 0b1010101010101010 = 0xAAAA
  Value constOddMask = b.create<LLVM::ConstantOp>(b.getI16Type(), 0xAAAA);
  auto vecI32Type = VectorType::get(16, b.getI32Type());

  // Construct vector<16xi32> with the low parts
  return b
      .create<LLVM::InlineAsmOp>(
          vecI32Type, ValueRange{even, constOddMask, odd},
          "vmovsldup $0 {$2}, $3", "=x,0,^Yk,x",
          /*has_side_effects=*/false,
          /*is_align_stack=*/true, LLVM::TailCallKind::None,
          /*asm_dialect=*/
          LLVM::AsmDialectAttr::get(b.getContext(), LLVM::AsmDialect::AD_Intel),
          /*operand_attrs=*/ArrayAttr())
      .getResult(0);
}

inline bool isGatherLowsResult(Value value) {
  if (auto inlineAsmOp = value.getDefiningOp<LLVM::InlineAsmOp>()) {
    return inlineAsmOp.getAsmString() == "vmovsldup $0 {$2}, $3";
  }
  return false;
}

// Gather the high parts of two vectors of 16 32-bit integers.
// [a₀, a₁, a₂, a₃, a₄, a₅, a₆, a₇, a₈, a₉, a₁₀, a₁₁, a₁₂, a₁₃, a₁₄, a₁₅]
// [b₀, b₁, b₂, b₃, b₄, b₅, b₆, b₇, b₈, b₉, b₁₀, b₁₁, b₁₂, b₁₃, b₁₄, b₁₅]
// => [a₁, b₁, a₃, b₃, a₅, b₅, a₇, b₇, a₉, b₉, a₁₁, b₁₁, a₁₃, b₁₃, a₁₅, b₁₅]
Value gatherHighsInterleaved(ImplicitLocOpBuilder &b, Value even, Value odd) {
  // 0b0101010101010101 = 0x5555
  Value constEvenMask = b.create<LLVM::ConstantOp>(b.getI16Type(), 0x5555);
  auto vecI32Type = VectorType::get(16, b.getI32Type());

  // Construct vector<16xi32> with the low parts
  return b
      .create<LLVM::InlineAsmOp>(
          vecI32Type, ValueRange{odd, constEvenMask, even},
          "vmovshdup $0 {$2}, $3", "=x,0,^Yk,x",
          /*has_side_effects=*/false,
          /*is_align_stack=*/true, LLVM::TailCallKind::None,
          /*asm_dialect=*/
          LLVM::AsmDialectAttr::get(b.getContext(), LLVM::AsmDialect::AD_Intel),
          /*operand_attrs=*/ArrayAttr())
      .getResult(0);
}

inline bool isGatherHighsResult(Value value) {
  if (auto inlineAsmOp = value.getDefiningOp<LLVM::InlineAsmOp>()) {
    return inlineAsmOp.getAsmString() == "vmovshdup $0 {$2}, $3";
  }
  return false;
}

// Duplicates the odd lanes of a vector<16xi32> to the even lanes.
// [a₀, a₁, a₂, a₃, a₄, a₅, a₆, a₇, a₈, a₉, a₁₀, a₁₁, a₁₂, a₁₃, a₁₄, a₁₅]
// => [a₁, a₁, a₃, a₃, a₅, a₅, a₇, a₇, a₉, a₉, a₁₁, a₁₁, a₁₃, a₁₃, a₁₅, a₁₅]
Value duplicateOddLanesToEven(ImplicitLocOpBuilder &b, Value vec) {
  auto vecI32Type = VectorType::get(16, b.getI32Type());
  return b.create<vector::ShuffleOp>(
      vecI32Type, vec, vec,
      b.getDenseI64ArrayAttr(
          {1, 1, 3, 3, 5, 5, 7, 7, 9, 9, 11, 11, 13, 13, 15, 15}));
}

// Extracts even and odd lane values from a gatherLowsResult.
std::pair<Value, Value> extractEvenOddFromLows(Value value) {
  auto inlineAsm = value.getDefiningOp<LLVM::InlineAsmOp>();
  return {inlineAsm.getOperands()[0], inlineAsm.getOperands()[2]};
}

// Extracts even and odd lane values from a gatherHighsResult.
std::pair<Value, Value> extractEvenOddFromHighs(ImplicitLocOpBuilder &b,
                                                Value value, bool duplicate) {
  auto inlineAsm = value.getDefiningOp<LLVM::InlineAsmOp>();
  Value odd = inlineAsm.getOperands()[0];
  Value even = inlineAsm.getOperands()[2];
  if (duplicate) {
    odd = duplicateOddLanesToEven(b, odd);
    even = duplicateOddLanesToEven(b, even);
  }
  return {even, odd};
}

// Extracts even and odd lane values from a Value, handling gather operations,
// constants, and default cases.
// Returns std::nullopt if value is not on a path and duplicateForDefault is
// false.
std::optional<std::pair<Value, Value>>
extractEvenOdd(ImplicitLocOpBuilder &b, Value value,
               bool duplicateForHighs = false,
               bool duplicateForDefault = true) {
  if (isGatherLowsResult(value)) {
    return extractEvenOddFromLows(value);
  }
  if (isGatherHighsResult(value)) {
    return extractEvenOddFromHighs(b, value, duplicateForHighs);
  }
  if (isConstantSplat(value)) {
    return {{value, value}};
  }
  // Default case
  if (duplicateForDefault) {
    return {{value, duplicateOddLanesToEven(b, value)}};
  }
  return std::nullopt;
}

} // namespace

template <typename OpType>
struct SpecializeAddSubIOpToAVX512 : public OpConversionPattern<OpType> {
  explicit SpecializeAddSubIOpToAVX512(MLIRContext *context)
      : OpConversionPattern<OpType>(context) {}

  using OpConversionPattern<OpType>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpType op,
                  typename OpConversionPattern<OpType>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Rewrite gather -> op to opOdd, opEven -> gather if the next operation
    // is a dual-lane operation. This way, the next dual-lane operation can
    // avoid gathering/splitting the op result and directly use odd/even lanes.
    //
    // NOTE(batzor): This pattern only works if both operands are on the same
    // path.
    //
    // High path operands will look like this:
    // [a₁, b₁, a₃, b₃, a₅, b₅, a₇, b₇, a₉, b₉, a₁₁, b₁₁, a₁₃, b₁₃, a₁₅, b₁₅]
    // [c₁, d₁, c₃, d₃, c₅, d₅, c₇, d₇, c₉, d₉, c₁₁, d₁₁, c₁₃, d₁₃, c₁₅, d₁₅]
    //
    // So we can do op(a, c), op(b, d) and gather the odd lanes.
    //
    // Low path operands will look like this:
    // [a₀, b₀, a₂, b₂, a₄, b₄, a₆, b₆, a₈, b₈, a₁₀, b₁₀, a₁₂, b₁₂, a₁₄, b₁₄]
    // [c₀, d₀, c₂, d₂, c₄, d₄, c₆, d₆, c₈, d₈, c₁₀, d₁₀, c₁₂, d₁₂, c₁₄, d₁₄]
    //
    // So we can do op(a, c), op(b, d) and gather the even lanes.
    if (auto vecType = dyn_cast<VectorType>(op.getLhs().getType())) {
      if (vecType.getElementType().isInteger(32) &&
          vecType.getNumElements() == 16) {
        ImplicitLocOpBuilder b(op.getLoc(), rewriter);

        for (auto user : op->getUsers()) {
          // If the next operation is not a dual-lane operation, return failure.
          if (!canSpecialize(user)) {
            return failure();
          }
        }

        bool isLhsLow = isGatherLowsResult(adaptor.getLhs());
        bool isLhsHigh = isGatherHighsResult(adaptor.getLhs());
        // In the case of SubIOp, LHS can be a constant.
        bool isLhsConst = isConstantSplat(adaptor.getLhs());
        bool isRhsLow = isGatherLowsResult(adaptor.getRhs());
        bool isRhsHigh = isGatherHighsResult(adaptor.getRhs());
        bool isRhsConst = isConstantSplat(adaptor.getRhs());

        bool onLowPath = (isLhsLow || isLhsConst) && (isRhsLow || isRhsConst);
        bool onHighPath =
            (isLhsHigh || isLhsConst) && (isRhsHigh || isRhsConst);

        // LHS and RHS are not on the same path, return failure.
        if (!(onLowPath || onHighPath)) {
          return failure();
        }

        auto [lhsEven, lhsOdd] =
            extractEvenOdd(b, adaptor.getLhs(), false, false).value();
        auto [rhsEven, rhsOdd] =
            extractEvenOdd(b, adaptor.getRhs(), false, false).value();

        auto [resultEven32, resultOdd32] =
            addSubByOddEven<OpType>(b, lhsEven, lhsOdd, rhsEven, rhsOdd);

        if (onLowPath) {
          Value gatherLow = gatherLowsInterleaved(b, resultEven32, resultOdd32);
          rewriter.replaceOp(op, {gatherLow});
          return success();
        }

        if (onHighPath) {
          Value gatherHigh =
              gatherHighsInterleaved(b, resultEven32, resultOdd32);
          rewriter.replaceOp(op, {gatherHigh});
          return success();
        }

        return failure();
      }
    }
    return failure();
  }
};

using SpecializeAddIOpToAVX512 = SpecializeAddSubIOpToAVX512<arith::AddIOp>;
using SpecializeSubIOpToAVX512 = SpecializeAddSubIOpToAVX512<arith::SubIOp>;

template <typename OpType>
struct SpecializeMulIOpToAVX512Impl : public OpConversionPattern<OpType> {
  explicit SpecializeMulIOpToAVX512Impl(MLIRContext *context)
      : OpConversionPattern<OpType>(context) {}

  using OpConversionPattern<OpType>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpType op,
                  typename OpConversionPattern<OpType>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // if vector<16xi32> type, rewrite using vpmuludq, shuffle + vpmuludq
    if (auto vecType = dyn_cast<VectorType>(op.getLhs().getType())) {
      if (vecType.getElementType().isInteger(32) &&
          vecType.getNumElements() == 16) {
        ImplicitLocOpBuilder b(op.getLoc(), rewriter);

        auto [lhsEven, lhsOdd] =
            *extractEvenOdd(b, adaptor.getLhs(), true, true);
        auto [rhsEven, rhsOdd] =
            *extractEvenOdd(b, adaptor.getRhs(), true, true);

        Value prodEven32, prodOdd32;
        if constexpr (std::is_same_v<OpType, arith::MulSIExtendedOp>) {
          std::tie(prodEven32, prodOdd32) =
              mulExtendedByOddEven(b, lhsEven, lhsOdd, rhsEven, rhsOdd, true);
        } else {
          std::tie(prodEven32, prodOdd32) =
              mulExtendedByOddEven(b, lhsEven, lhsOdd, rhsEven, rhsOdd);
        }

        if constexpr (std::is_same_v<OpType, arith::MulIOp>) {
          Value prodLow = gatherLowsInterleaved(b, prodEven32, prodOdd32);
          rewriter.replaceOp(op, prodLow);
        } else {
          Value prodLow = gatherLowsInterleaved(b, prodEven32, prodOdd32);
          Value prodHi = gatherHighsInterleaved(b, prodEven32, prodOdd32);
          rewriter.replaceOp(op, {prodLow, prodHi});
        }
        return success();
      }
    }
    return failure();
  }
};

using SpecializeMulUIExtendedToAVX512 =
    SpecializeMulIOpToAVX512Impl<arith::MulUIExtendedOp>;
using SpecializeMulSIExtendedToAVX512 =
    SpecializeMulIOpToAVX512Impl<arith::MulSIExtendedOp>;
using SpecializeMulIOpToAVX512 = SpecializeMulIOpToAVX512Impl<arith::MulIOp>;

namespace {
#include "zkir/Dialect/ArithExt/Conversions/SpecializeArithToAVX/SpecializeArithToAVX.cpp.inc"
} // namespace

struct SpecializeArithToAVX
    : impl::SpecializeArithToAVXBase<SpecializeArithToAVX> {
  using SpecializeArithToAVXBase::SpecializeArithToAVXBase;

  void runOnOperation() override;
};

void SpecializeArithToAVX::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addLegalDialect<vector::VectorDialect>();

  RewritePatternSet patterns(context);
  populateWithGenerated(patterns);
  patterns.add<
      // clang-format off
      SpecializeAddIOpToAVX512,
      SpecializeMulIOpToAVX512,
      SpecializeMulSIExtendedToAVX512,
      SpecializeMulUIExtendedToAVX512,
      SpecializeSubIOpToAVX512
      // clang-format on
      >(context);
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

} // namespace mlir::zkir::arith_ext
