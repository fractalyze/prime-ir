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

#include "prime_ir/Dialect/ArithExt/Conversions/SpecializeArithToAVX/SpecializeArithToAVX.h"

#include <optional>
#include <type_traits>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::prime_ir::arith_ext {

#define GEN_PASS_DEF_SPECIALIZEARITHTOAVX
#include "prime_ir/Dialect/ArithExt/Conversions/SpecializeArithToAVX/SpecializeArithToAVX.h.inc"

namespace {
inline bool isConstantSplat(Value value) {
  if (auto constantOp = value.getDefiningOp<arith::ConstantOp>()) {
    return isa<SplatElementsAttr>(constantOp.getValueAttr());
  }
  return false;
}

inline bool canSpecialize(Operation *op) {
  return isa<arith::AddIOp, arith::SubIOp, arith::MulIOp,
             arith::MulUIExtendedOp, arith::MulSIExtendedOp>(op);
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
  Value prodEven64 =
      LLVM::InlineAsmOp::create(
          b, vecI64Type, ValueRange{lhsEven, rhsEven}, asmMulString, "=x,x,x",
          /*has_side_effects=*/false,
          /*is_align_stack=*/true, LLVM::TailCallKind::None,
          /*asm_dialect=*/
          LLVM::AsmDialectAttr::get(b.getContext(), LLVM::AsmDialect::AD_Intel),
          /*operand_attrs=*/ArrayAttr())
          .getResult(0);
  Value prodOdd64 =
      LLVM::InlineAsmOp::create(
          b, vecI64Type, ValueRange{lhsOdd, rhsOdd}, asmMulString, "=x,x,x",
          /*has_side_effects=*/false,
          /*is_align_stack=*/true, LLVM::TailCallKind::None,
          /*asm_dialect=*/
          LLVM::AsmDialectAttr::get(b.getContext(), LLVM::AsmDialect::AD_Intel),
          /*operand_attrs=*/ArrayAttr())
          .getResult(0);

  // cast them to vector<16xi32> so even lanes are the low parts and odd
  // lanes are the high parts
  auto prodEven32 = vector::BitCastOp::create(b, vecI32Type, prodEven64);
  auto prodOdd32 = vector::BitCastOp::create(b, vecI32Type, prodOdd64);
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
      LLVM::InlineAsmOp::create(
          b, vecType, ValueRange{lhsEven, rhsEven}, asmString, "=x,x,x",
          /*has_side_effects=*/false,
          /*is_align_stack=*/true, LLVM::TailCallKind::None,
          /*asm_dialect=*/
          LLVM::AsmDialectAttr::get(b.getContext(), LLVM::AsmDialect::AD_Intel),
          /*operand_attrs=*/ArrayAttr())
          .getResult(0);

  Value resOdd =
      LLVM::InlineAsmOp::create(
          b, vecType, ValueRange{lhsOdd, rhsOdd}, asmString, "=x,x,x",
          /*has_side_effects=*/false,
          /*is_align_stack=*/true, LLVM::TailCallKind::None,
          /*asm_dialect=*/
          LLVM::AsmDialectAttr::get(b.getContext(), LLVM::AsmDialect::AD_Intel),
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
  Value constOddMask = LLVM::ConstantOp::create(b, b.getI16Type(), 0xAAAA);
  auto vecI32Type = VectorType::get(16, b.getI32Type());

  // Construct vector<16xi32> with the low parts
  return LLVM::InlineAsmOp::create(
             b, vecI32Type, ValueRange{even, constOddMask, odd},
             "vmovsldup $0 {$2}, $3", "=x,0,^Yk,x",
             /*has_side_effects=*/false,
             /*is_align_stack=*/true, LLVM::TailCallKind::None,
             /*asm_dialect=*/
             LLVM::AsmDialectAttr::get(b.getContext(),
                                       LLVM::AsmDialect::AD_Intel),
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
  Value constEvenMask = LLVM::ConstantOp::create(b, b.getI16Type(), 0x5555);
  auto vecI32Type = VectorType::get(16, b.getI32Type());

  // Construct vector<16xi32> with the low parts
  return LLVM::InlineAsmOp::create(
             b, vecI32Type, ValueRange{odd, constEvenMask, even},
             "vmovshdup $0 {$2}, $3", "=x,0,^Yk,x",
             /*has_side_effects=*/false,
             /*is_align_stack=*/true, LLVM::TailCallKind::None,
             /*asm_dialect=*/
             LLVM::AsmDialectAttr::get(b.getContext(),
                                       LLVM::AsmDialect::AD_Intel),
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
  return vector::ShuffleOp::create(
      b, vecI32Type, vec, vec,
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

//===----------------------------------------------------------------------===//
// AVX2 (2x ymm) helpers
//===----------------------------------------------------------------------===//
//
// On AVX2, vector<16xi32> occupies two ymm registers and there are no opmask
// registers, so the zmm inline asm above is not encodable. The same dual-lane
// dataflow is expressed as target-independent IR instead:
// - Extended products use the masked 64-bit multiply idiom that the X86
//   backend selects to vpmuludq/vpmuldq (combineMulToPMULDQ; the
//   llvm.x86.*.pmul[u].dq intrinsics were removed in LLVM 7 in favor of this
//   form, which is also what clang emits for _mm256_mul_ep[iu]32).
// - Low/high interleaves are vector.shuffle ops; LLVM lowers each to a short
//   ymm sequence (e.g. vmovs[lh]dup + vpblendd) per half.
// - Adds/subs on dual-lane values stay as plain integer ops; they legalize
//   to vpaddd/vpsubd ymm pairs without help.
// Keeping the ops transparent (no asm) lets LLVM schedule them freely.
//
// The arithmetic is emitted in the LLVM dialect rather than arith: the
// conversion target only marks the LLVM and vector dialects legal, and ops
// created by a conversion pattern must be legal or the whole rewrite is
// rolled back. (arith.addi on vector<16xi32> also could not be marked legal
// without disabling the dual-lane chain pattern that matches that very form.)

// vector.shuffle masks implementing the interleaves documented on
// gatherLowsInterleaved/gatherHighsInterleaved above, as 2-input shuffles of
// (even, odd).
inline constexpr int64_t kGatherLowsMask[16] = {0, 16, 2,  18, 4,  20, 6,  22,
                                                8, 24, 10, 26, 12, 28, 14, 30};
inline constexpr int64_t kGatherHighsMask[16] = {1, 17, 3,  19, 5,  21, 7,  23,
                                                 9, 25, 11, 27, 13, 29, 15, 31};

// Multiplies the even 32-bit lanes of the given vector<16xi32> operands into
// 64-bit products, like mulExtendedByOddEven, but as the pmuludq/pmuldq IR
// idiom instead of zmm inline asm.
std::pair<Value, Value> mulExtendedByOddEvenAVX2(ImplicitLocOpBuilder &b,
                                                 Value lhsEven, Value lhsOdd,
                                                 Value rhsEven, Value rhsOdd,
                                                 bool isSigned = false) {
  auto vecI32Type = VectorType::get(16, b.getI32Type());
  auto vecI64Type = VectorType::get(8, b.getI64Type());

  // Reinterprets the vector as vector<8xi64> and extends each even i32 lane
  // (the low half of every i64 lane) to the full 64 bits.
  auto extendEvenLanes = [&](Value vec) -> Value {
    Value vec64 = vector::BitCastOp::create(b, vecI64Type, vec);
    if (isSigned) {
      // sext_inreg as shifts; matched to vpmuldq.
      Value c32 = LLVM::ConstantOp::create(
          b, vecI64Type,
          DenseElementsAttr::get(vecI64Type, b.getI64IntegerAttr(32)));
      Value shifted = LLVM::ShlOp::create(b, vec64, c32);
      return LLVM::AShrOp::create(b, shifted, c32);
    }
    // Zero-extend by masking; matched to vpmuludq.
    Value mask = LLVM::ConstantOp::create(
        b, vecI64Type,
        DenseElementsAttr::get(vecI64Type, b.getI64IntegerAttr(0xFFFFFFFF)));
    return LLVM::AndOp::create(b, vec64, mask);
  };

  Value prodEven64 = LLVM::MulOp::create(b, extendEvenLanes(lhsEven),
                                         extendEvenLanes(rhsEven));
  Value prodOdd64 =
      LLVM::MulOp::create(b, extendEvenLanes(lhsOdd), extendEvenLanes(rhsOdd));

  // cast them to vector<16xi32> so even lanes are the low parts and odd
  // lanes are the high parts
  auto prodEven32 = vector::BitCastOp::create(b, vecI32Type, prodEven64);
  auto prodOdd32 = vector::BitCastOp::create(b, vecI32Type, prodOdd64);
  return {prodEven32, prodOdd32};
}

// Shuffle-based equivalent of gatherLowsInterleaved.
Value gatherLowsInterleavedAVX2(ImplicitLocOpBuilder &b, Value even,
                                Value odd) {
  auto vecI32Type = VectorType::get(16, b.getI32Type());
  return vector::ShuffleOp::create(b, vecI32Type, even, odd,
                                   b.getDenseI64ArrayAttr(kGatherLowsMask));
}

inline bool isGatherLowsResultAVX2(Value value) {
  auto shuffleOp = value.getDefiningOp<vector::ShuffleOp>();
  return shuffleOp && shuffleOp.getMask() == ArrayRef<int64_t>(kGatherLowsMask);
}

// Shuffle-based equivalent of gatherHighsInterleaved.
Value gatherHighsInterleavedAVX2(ImplicitLocOpBuilder &b, Value even,
                                 Value odd) {
  auto vecI32Type = VectorType::get(16, b.getI32Type());
  return vector::ShuffleOp::create(b, vecI32Type, even, odd,
                                   b.getDenseI64ArrayAttr(kGatherHighsMask));
}

inline bool isGatherHighsResultAVX2(Value value) {
  auto shuffleOp = value.getDefiningOp<vector::ShuffleOp>();
  return shuffleOp &&
         shuffleOp.getMask() == ArrayRef<int64_t>(kGatherHighsMask);
}

// Shuffle-based equivalent of extractEvenOdd.
std::optional<std::pair<Value, Value>>
extractEvenOddAVX2(ImplicitLocOpBuilder &b, Value value,
                   bool duplicateForHighs = false,
                   bool duplicateForDefault = true) {
  if (isGatherLowsResultAVX2(value)) {
    auto shuffleOp = value.getDefiningOp<vector::ShuffleOp>();
    return {{shuffleOp.getV1(), shuffleOp.getV2()}};
  }
  if (isGatherHighsResultAVX2(value)) {
    auto shuffleOp = value.getDefiningOp<vector::ShuffleOp>();
    Value even = shuffleOp.getV1();
    Value odd = shuffleOp.getV2();
    if (duplicateForHighs) {
      even = duplicateOddLanesToEven(b, even);
      odd = duplicateOddLanesToEven(b, odd);
    }
    return {{even, odd}};
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

//===----------------------------------------------------------------------===//
// Flavor traits
//===----------------------------------------------------------------------===//

// Dispatch tables between the AVX-512 and AVX2 lowerings of the dual-lane
// building blocks. The conversion patterns below are templated over these.
struct Avx512Flavor {
  static bool matchGatherLows(Value value) { return isGatherLowsResult(value); }
  static bool matchGatherHighs(Value value) {
    return isGatherHighsResult(value);
  }
  static Value emitGatherLows(ImplicitLocOpBuilder &b, Value even, Value odd) {
    return gatherLowsInterleaved(b, even, odd);
  }
  static Value emitGatherHighs(ImplicitLocOpBuilder &b, Value even, Value odd) {
    return gatherHighsInterleaved(b, even, odd);
  }
  static std::optional<std::pair<Value, Value>>
  splitEvenOdd(ImplicitLocOpBuilder &b, Value value, bool duplicateForHighs,
               bool duplicateForDefault) {
    return extractEvenOdd(b, value, duplicateForHighs, duplicateForDefault);
  }
  static std::pair<Value, Value> emitMulExtended(ImplicitLocOpBuilder &b,
                                                 Value lhsEven, Value lhsOdd,
                                                 Value rhsEven, Value rhsOdd,
                                                 bool isSigned) {
    return mulExtendedByOddEven(b, lhsEven, lhsOdd, rhsEven, rhsOdd, isSigned);
  }
  template <typename OpType>
  static std::pair<Value, Value> emitAddSub(ImplicitLocOpBuilder &b,
                                            Value lhsEven, Value lhsOdd,
                                            Value rhsEven, Value rhsOdd) {
    return addSubByOddEven<OpType>(b, lhsEven, lhsOdd, rhsEven, rhsOdd);
  }
};

struct Avx2Flavor {
  static bool matchGatherLows(Value value) {
    return isGatherLowsResultAVX2(value);
  }
  static bool matchGatherHighs(Value value) {
    return isGatherHighsResultAVX2(value);
  }
  static Value emitGatherLows(ImplicitLocOpBuilder &b, Value even, Value odd) {
    return gatherLowsInterleavedAVX2(b, even, odd);
  }
  static Value emitGatherHighs(ImplicitLocOpBuilder &b, Value even, Value odd) {
    return gatherHighsInterleavedAVX2(b, even, odd);
  }
  static std::optional<std::pair<Value, Value>>
  splitEvenOdd(ImplicitLocOpBuilder &b, Value value, bool duplicateForHighs,
               bool duplicateForDefault) {
    return extractEvenOddAVX2(b, value, duplicateForHighs, duplicateForDefault);
  }
  static std::pair<Value, Value> emitMulExtended(ImplicitLocOpBuilder &b,
                                                 Value lhsEven, Value lhsOdd,
                                                 Value rhsEven, Value rhsOdd,
                                                 bool isSigned) {
    return mulExtendedByOddEvenAVX2(b, lhsEven, lhsOdd, rhsEven, rhsOdd,
                                    isSigned);
  }
  // Adds/subs on dual-lane values need no special instruction on AVX2;
  // vector<16xi32> integer add/sub legalizes to vpaddd/vpsubd ymm pairs.
  template <typename OpType>
  static std::pair<Value, Value> emitAddSub(ImplicitLocOpBuilder &b,
                                            Value lhsEven, Value lhsOdd,
                                            Value rhsEven, Value rhsOdd) {
    using LLVMOp = std::conditional_t<std::is_same_v<OpType, arith::AddIOp>,
                                      LLVM::AddOp, LLVM::SubOp>;
    Value resEven = LLVMOp::create(b, lhsEven, rhsEven);
    Value resOdd = LLVMOp::create(b, lhsOdd, rhsOdd);
    return {resEven, resOdd};
  }
};

} // namespace

template <typename OpType, typename Flavor>
struct SpecializeAddSubIOp : public OpConversionPattern<OpType> {
  explicit SpecializeAddSubIOp(MLIRContext *context)
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

        bool isLhsLow = Flavor::matchGatherLows(adaptor.getLhs());
        bool isLhsHigh = Flavor::matchGatherHighs(adaptor.getLhs());
        // In the case of SubIOp, LHS can be a constant.
        bool isLhsConst = isConstantSplat(adaptor.getLhs());
        bool isRhsLow = Flavor::matchGatherLows(adaptor.getRhs());
        bool isRhsHigh = Flavor::matchGatherHighs(adaptor.getRhs());
        bool isRhsConst = isConstantSplat(adaptor.getRhs());

        bool onLowPath = (isLhsLow || isLhsConst) && (isRhsLow || isRhsConst);
        bool onHighPath =
            (isLhsHigh || isLhsConst) && (isRhsHigh || isRhsConst);

        // LHS and RHS are not on the same path, return failure.
        if (!(onLowPath || onHighPath)) {
          return failure();
        }

        auto [lhsEven, lhsOdd] =
            Flavor::splitEvenOdd(b, adaptor.getLhs(), false, false).value();
        auto [rhsEven, rhsOdd] =
            Flavor::splitEvenOdd(b, adaptor.getRhs(), false, false).value();

        auto [resultEven32, resultOdd32] = Flavor::template emitAddSub<OpType>(
            b, lhsEven, lhsOdd, rhsEven, rhsOdd);

        if (onLowPath) {
          Value gatherLow =
              Flavor::emitGatherLows(b, resultEven32, resultOdd32);
          rewriter.replaceOp(op, {gatherLow});
          return success();
        }

        if (onHighPath) {
          Value gatherHigh =
              Flavor::emitGatherHighs(b, resultEven32, resultOdd32);
          rewriter.replaceOp(op, {gatherHigh});
          return success();
        }

        return failure();
      }
    }
    return failure();
  }
};

using SpecializeAddIOpToAVX512 =
    SpecializeAddSubIOp<arith::AddIOp, Avx512Flavor>;
using SpecializeSubIOpToAVX512 =
    SpecializeAddSubIOp<arith::SubIOp, Avx512Flavor>;
using SpecializeAddIOpToAVX2 = SpecializeAddSubIOp<arith::AddIOp, Avx2Flavor>;
using SpecializeSubIOpToAVX2 = SpecializeAddSubIOp<arith::SubIOp, Avx2Flavor>;

template <typename OpType, typename Flavor>
struct SpecializeMulIOpImpl : public OpConversionPattern<OpType> {
  explicit SpecializeMulIOpImpl(MLIRContext *context)
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
            *Flavor::splitEvenOdd(b, adaptor.getLhs(), true, true);
        auto [rhsEven, rhsOdd] =
            *Flavor::splitEvenOdd(b, adaptor.getRhs(), true, true);

        bool isSigned = std::is_same_v<OpType, arith::MulSIExtendedOp>;
        auto [prodEven32, prodOdd32] = Flavor::emitMulExtended(
            b, lhsEven, lhsOdd, rhsEven, rhsOdd, isSigned);

        if constexpr (std::is_same_v<OpType, arith::MulIOp>) {
          Value prodLow = Flavor::emitGatherLows(b, prodEven32, prodOdd32);
          rewriter.replaceOp(op, prodLow);
        } else {
          Value prodLow = Flavor::emitGatherLows(b, prodEven32, prodOdd32);
          Value prodHi = Flavor::emitGatherHighs(b, prodEven32, prodOdd32);
          rewriter.replaceOp(op, {prodLow, prodHi});
        }
        return success();
      }
    }
    return failure();
  }
};

using SpecializeMulUIExtendedToAVX512 =
    SpecializeMulIOpImpl<arith::MulUIExtendedOp, Avx512Flavor>;
using SpecializeMulSIExtendedToAVX512 =
    SpecializeMulIOpImpl<arith::MulSIExtendedOp, Avx512Flavor>;
using SpecializeMulIOpToAVX512 =
    SpecializeMulIOpImpl<arith::MulIOp, Avx512Flavor>;
using SpecializeMulUIExtendedToAVX2 =
    SpecializeMulIOpImpl<arith::MulUIExtendedOp, Avx2Flavor>;
using SpecializeMulSIExtendedToAVX2 =
    SpecializeMulIOpImpl<arith::MulSIExtendedOp, Avx2Flavor>;
using SpecializeMulIOpToAVX2 = SpecializeMulIOpImpl<arith::MulIOp, Avx2Flavor>;

namespace {
#include "prime_ir/Dialect/ArithExt/Conversions/SpecializeArithToAVX/SpecializeArithToAVX.cpp.inc"
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
  switch (flavor) {
  case AVXFlavor::kAVX512:
    patterns.add<
        // clang-format off
        SpecializeAddIOpToAVX512,
        SpecializeMulIOpToAVX512,
        SpecializeMulSIExtendedToAVX512,
        SpecializeMulUIExtendedToAVX512,
        SpecializeSubIOpToAVX512
        // clang-format on
        >(context);
    break;
  case AVXFlavor::kAVX2:
    patterns.add<
        // clang-format off
        SpecializeAddIOpToAVX2,
        SpecializeMulIOpToAVX2,
        SpecializeMulSIExtendedToAVX2,
        SpecializeMulUIExtendedToAVX2,
        SpecializeSubIOpToAVX2
        // clang-format on
        >(context);
    break;
  }
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

} // namespace mlir::prime_ir::arith_ext
