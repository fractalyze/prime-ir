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

#include "prime_ir/Dialect/ModArith/Conversions/ModArithToArith/Reducer/SolinasReducer.h"

#include "llvm/ADT/APInt.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithTypes.h" // IWYU pragma: keep
#include "zk_dtypes/include/field/goldilocks/goldilocks.h"

namespace mlir::prime_ir::mod_arith {

SolinasReducer::SolinasReducer(ImplicitLocOpBuilder &b,
                               ModArithType modArithType)
    : b(b), modAttr(modArithType.getModulus()) {
  bitWidth = modAttr.getValue().getBitWidth();
  assert(bitWidth == 64 &&
         modAttr.getValue() ==
             APInt(64, zk_dtypes::Goldilocks::Config::kModulus) &&
         "SolinasReducer only supports the Goldilocks prime "
         "p = 2^64 - 2^32 + 1");
}

Value SolinasReducer::reduce(Value lo, Value hi) {
  Type t = lo.getType();
  auto konst = [&](uint64_t v) {
    return arith::ConstantOp::create(b, t, b.getIntegerAttr(t, v));
  };
  // ε = 2^32 - 1 = 2^64 mod p. With hi = hi_hi·2^32 + hi_lo and the
  // congruences 2^64 ≡ ε, 2^96 ≡ -1 (mod p):
  //   x = hi·2^64 + lo ≡ lo - hi_hi + hi_lo·ε   (mod p).
  Value half = konst(bitWidth / 2); // 32
  Value epsilon = konst((1ULL << 32) - 1);
  Value hiHi = arith::ShRUIOp::create(b, hi, half);   // hi >> 32   < 2^32
  Value hiLo = arith::AndIOp::create(b, hi, epsilon); // hi & ε     < 2^32

  // t0 = lo - hi_hi. The subtraction borrows iff lo < hi_hi; a borrow means
  // the true value wrapped by 2^64 ≡ ε, so correct by subtracting ε.
  Value borrow = arith::CmpIOp::create(b, arith::CmpIPredicate::ult, lo, hiHi);
  Value t0Raw = arith::SubIOp::create(b, lo, hiHi);
  Value t0Corr = arith::SubIOp::create(b, t0Raw, epsilon);
  Value t0 = arith::SelectOp::create(b, borrow, t0Corr, t0Raw);

  // t1 = hi_lo · ε. Since ε = 2^32 - 1 and hi_lo < 2^32, this is
  // (hi_lo << 32) - hi_lo, exact in 64 bits (hi_lo << 32 < 2^64, no borrow).
  // Strength-reduced to shift+sub: ptxas keeps the multiply-by-immediate as a
  // 64-bit IMAD otherwise, paid on every Goldilocks field multiply.
  Value t1 =
      arith::SubIOp::create(b, arith::ShLIOp::create(b, hiLo, half), hiLo);

  // t2 = t0 + t1; an add carry again means a 2^64 ≡ ε wrap, corrected by +ε.
  auto add = arith::AddUIExtendedOp::create(b, t0, t1);
  Value t2Corr = arith::AddIOp::create(b, add.getSum(), epsilon);
  Value t2 =
      arith::SelectOp::create(b, add.getOverflow(), t2Corr, add.getSum());

  // t2 < 2^64 < 2p, but may be ≥ p; one conditional subtract canonicalizes.
  Value p = konst(modAttr.getValue().getZExtValue());
  Value ge = arith::CmpIOp::create(b, arith::CmpIPredicate::uge, t2, p);
  Value sub = arith::SubIOp::create(b, t2, p);
  return arith::SelectOp::create(b, ge, sub, t2);
}

} // namespace mlir::prime_ir::mod_arith
