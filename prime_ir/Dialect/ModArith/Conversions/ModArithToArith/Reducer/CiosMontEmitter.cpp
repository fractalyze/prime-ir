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

#include "prime_ir/Dialect/ModArith/Conversions/ModArithToArith/Reducer/CiosMontEmitter.h"

#include <tuple>

#include "llvm/ADT/APInt.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "prime_ir/Dialect/ModArith/Conversions/ModArithToArith/Reducer/MontReducer.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithTypes.h" // IWYU pragma: keep

namespace mlir::prime_ir::mod_arith {

// Limb width for the CIOS schedule. Fixed at 32 bits so that each
// multiply-accumulate is a (i32 x i32 -> i64) widening — the IMAD.WIDE shape.
static constexpr unsigned kLimbBits = 32;

CiosMontEmitter::CiosMontEmitter(ImplicitLocOpBuilder &b,
                                 ModArithType modArithType)
    : b(b), modArithType(modArithType), modAttr(modArithType.getModulus()),
      montAttr(modArithType.getMontgomeryAttr()) {}

// static
bool CiosMontEmitter::isEligible(ModArithType modArithType,
                                 Type convertedType) {
  // Scalar integer storage only: the IMAD.WIDE rationale and the 32-bit limb
  // schedule are register-resident scalar codegen.
  auto intTy = dyn_cast<IntegerType>(convertedType);
  if (!intTy)
    return false;
  if (modArithType.getMontgomeryAttr().getNumLimbs() <= 1)
    return false;
  // Width must be a whole number of 64-bit words: the CIOS row count
  // 32*ceil(w/32) must equal the type's Montgomery radix exponent
  // 64*numLimbs (w=96 would divide by 2^96, not R=2^128), and the
  // 32-bit modulus-limb extraction must stay in bounds.
  if (intTy.getWidth() % 64 != 0)
    return false;
  // Spare-bit moduli only: with the sign bit set, 2p > 2ʷ and the cheap
  // skip-the-final-overflow-column invariant (t[N] == 0) no longer holds.
  return !modArithType.getModulus().getValue().isSignBitSet();
}

Value CiosMontEmitter::i32Const(uint64_t v) {
  return arith::ConstantOp::create(
      b, b.getIntegerAttr(b.getIntegerType(kLimbBits), v));
}

SmallVector<Value> CiosMontEmitter::decompose(Value wide) {
  auto wideTy = cast<IntegerType>(wide.getType());
  unsigned w = wideTy.getWidth();
  unsigned n = (w + kLimbBits - 1) / kLimbBits;
  Type i32 = b.getIntegerType(kLimbBits);
  SmallVector<Value> limbs;
  limbs.reserve(n);
  for (unsigned j = 0; j < n; ++j) {
    Value chunk = wide;
    if (j != 0) {
      auto shift =
          arith::ConstantOp::create(b, b.getIntegerAttr(wideTy, j * kLimbBits));
      chunk = arith::ShRUIOp::create(b, wide, shift);
    }
    limbs.push_back(arith::TruncIOp::create(b, i32, chunk));
  }
  return limbs;
}

Value CiosMontEmitter::recompose(ArrayRef<Value> limbs, Type wideTy) {
  auto wTy = cast<IntegerType>(wideTy);
  Value acc;
  for (unsigned j = 0; j < limbs.size(); ++j) {
    Value ext = arith::ExtUIOp::create(b, wTy, limbs[j]);
    if (j != 0) {
      auto shift =
          arith::ConstantOp::create(b, b.getIntegerAttr(wTy, j * kLimbBits));
      ext = arith::ShLIOp::create(b, ext, shift);
    }
    acc = acc ? arith::OrIOp::create(b, acc, ext).getResult() : ext;
  }
  return acc;
}

std::pair<Value, Value> CiosMontEmitter::mulAddStep(Value a32, Value b32,
                                                    Value t32, Value c32) {
  Type i64 = b.getIntegerType(2 * kLimbBits);
  Value aw = arith::ExtUIOp::create(b, i64, a32);
  Value bw = arith::ExtUIOp::create(b, i64, b32);
  Value p = arith::MulIOp::create(b, aw, bw);
  Value tw = arith::ExtUIOp::create(b, i64, t32);
  Value cw = arith::ExtUIOp::create(b, i64, c32);
  // s = a*b + t + c <= (2³²-1)² + 2(2³²-1) < 2⁶⁴, so no overflow.
  Value s = arith::AddIOp::create(b, arith::AddIOp::create(b, p, tw), cw);
  Value lo = arith::TruncIOp::create(b, b.getIntegerType(kLimbBits), s);
  auto shift = arith::ConstantOp::create(b, b.getIntegerAttr(i64, kLimbBits));
  Value hi = arith::TruncIOp::create(b, b.getIntegerType(kLimbBits),
                                     arith::ShRUIOp::create(b, s, shift));
  return {lo, hi};
}

std::pair<Value, Value> CiosMontEmitter::addCarryStep(Value t32, Value c32) {
  Type i64 = b.getIntegerType(2 * kLimbBits);
  Value tw = arith::ExtUIOp::create(b, i64, t32);
  Value cw = arith::ExtUIOp::create(b, i64, c32);
  Value s = arith::AddIOp::create(b, tw, cw);
  Value lo = arith::TruncIOp::create(b, b.getIntegerType(kLimbBits), s);
  auto shift = arith::ConstantOp::create(b, b.getIntegerAttr(i64, kLimbBits));
  Value hi = arith::TruncIOp::create(b, b.getIntegerType(kLimbBits),
                                     arith::ShRUIOp::create(b, s, shift));
  return {lo, hi};
}

Value CiosMontEmitter::reduceRow(MutableArrayRef<Value> t,
                                 ArrayRef<Value> nConst, Value np32,
                                 unsigned n) {
  // m = t[0] * n′₃₂ (low half); the resulting t[0] + m*n[0] has a zero low
  // limb, which is the limb dropped by the ÷2³² shift.
  Value m = arith::MulIOp::create(b, t[0], np32);
  Value c;
  // mulAddStep on (m, n[0], t[0], 0): lo is 0 by construction, discard it.
  std::tie(std::ignore, c) = mulAddStep(m, nConst[0], t[0], i32Const(0));
  for (unsigned j = 1; j < n; ++j)
    std::tie(t[j - 1], c) = mulAddStep(m, nConst[j], t[j], c);
  std::tie(t[n - 1], c) = addCarryStep(t[n], c);
  return c;
}

Value CiosMontEmitter::emitMontMul(Value lhs, Value rhs, bool lazy) {
  auto wideTy = cast<IntegerType>(lhs.getType());
  unsigned w = wideTy.getWidth();
  unsigned n = (w + kLimbBits - 1) / kLimbBits;

  SmallVector<Value> a = decompose(lhs);
  SmallVector<Value> bLimbs = decompose(rhs);

  // Compile-time constants: modulus limbs n[0..N) and n′₃₂ = low 32 bits of
  // n′ (x ≡ −n⁻¹ mod 2⁶⁴ ⇒ x ≡ −n⁻¹ mod 2³²).
  APInt mod = modAttr.getValue();
  SmallVector<Value> nConst;
  nConst.reserve(n);
  for (unsigned j = 0; j < n; ++j)
    nConst.push_back(
        i32Const(mod.extractBits(kLimbBits, j * kLimbBits).getZExtValue()));
  Value np32 = i32Const(
      montAttr.getNPrime().getValue().extractBits(kLimbBits, 0).getZExtValue());

  // t[0..N+2): accumulator with two overflow columns t[N], t[N+1].
  SmallVector<Value> t(n + 2, i32Const(0));

  for (unsigned i = 0; i < n; ++i) {
    // Multiply-accumulate row: t += a[i] * b.
    Value c = i32Const(0);
    for (unsigned j = 0; j < n; ++j)
      std::tie(t[j], c) = mulAddStep(a[i], bLimbs[j], t[j], c);
    auto [lo, hi] = addCarryStep(t[n], c);
    t[n] = lo;
    // hi add cannot overflow t[N+1]'s column, so a plain i32 add suffices.
    t[n + 1] = arith::AddIOp::create(b, t[n + 1], hi);

    // Reduction row: t += m * n, shift down one limb.
    c = reduceRow(t, nConst, np32, n);
    t[n] = arith::AddIOp::create(b, t[n + 1], c);
    t[n + 1] = i32Const(0);
  }

  // Spare-bit modulus: result is in t[0..N) with t[N) == 0 (range < 2p < 2ʷ).
  t.truncate(n);
  Value result = recompose(t, wideTy);

  if (lazy)
    return result;
  return MontReducer(b, modArithType)
      .getCanonicalFromExtended(result, /*bound=*/2);
}

Value CiosMontEmitter::emitRedc(Value tLow, Value tHigh, bool lazy) {
  auto wideTy = cast<IntegerType>(tLow.getType());
  unsigned w = wideTy.getWidth();
  unsigned n = (w + kLimbBits - 1) / kLimbBits;

  // T = tLow + tHigh·2ʷ as 2N i32 limbs: low N from tLow, high N from tHigh.
  SmallVector<Value> tl = decompose(tLow);
  SmallVector<Value> th = decompose(tHigh);
  tl.append(th.begin(), th.end());

  // Compile-time constants: modulus limbs n[0..N) and n′₃₂ = low 32 bits of n′.
  APInt mod = modAttr.getValue();
  SmallVector<Value> nConst;
  nConst.reserve(n);
  for (unsigned j = 0; j < n; ++j)
    nConst.push_back(
        i32Const(mod.extractBits(kLimbBits, j * kLimbBits).getZExtValue()));
  Value np32 = i32Const(
      montAttr.getNPrime().getValue().extractBits(kLimbBits, 0).getZExtValue());

  // Fixed window t[0..N+1] over the 2N-limb T, mirroring emitMontMul's
  // reduction accumulator: t[N] holds the limb consumed this row (a full T
  // limb plus a tiny carry), t[N+1] holds the residual carry (always ≤ 2).
  // The low N+1 limbs are pre-loaded; the higher limbs feed in one per row.
  SmallVector<Value> t(n + 2, i32Const(0));
  for (unsigned j = 0; j <= n; ++j)
    t[j] = tl[j];
  unsigned feedIdx = n + 1;

  for (unsigned i = 0; i < n; ++i) {
    // Reduction row consumes t[N] and returns the carry exiting column N.
    Value c = reduceRow(t, nConst, np32, n);
    // Slide the window up by one: t[N] ← next T limb + carry c + residual
    // t[N+1]; the carry-outs are ≤ 1 each, so t[N+1] stays ≤ 2.
    Value feed = feedIdx < 2 * n ? tl[feedIdx] : i32Const(0);
    ++feedIdx;
    auto [lo, hi] = addCarryStep(feed, c);
    auto [lo2, hi2] = addCarryStep(lo, t[n + 1]);
    t[n] = lo2;
    t[n + 1] = arith::AddIOp::create(b, hi, hi2);
  }

  // Spare-bit modulus: result is in t[0..N) with t[N) == 0 (range < 2p < 2ʷ).
  t.truncate(n);
  Value result = recompose(t, wideTy);

  if (lazy)
    return result;
  return MontReducer(b, modArithType)
      .getCanonicalFromExtended(result, /*bound=*/2);
}

} // namespace mlir::prime_ir::mod_arith
