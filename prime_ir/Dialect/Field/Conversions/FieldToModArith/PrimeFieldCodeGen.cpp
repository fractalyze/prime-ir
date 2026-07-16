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

#include "prime_ir/Dialect/Field/Conversions/FieldToModArith/PrimeFieldCodeGen.h"

#include <algorithm>
#include <functional>
#include <map>
#include <optional>
#include <utility>

#include "llvm/ADT/APInt.h"
#include "mlir/IR/TypeUtilities.h"
#include "prime_ir/Dialect/ModArith/IR//ModArithOperation.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithOps.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithTypes.h"
#include "prime_ir/Utils/BuilderContext.h"

namespace mlir::prime_ir::field {

PrimeFieldCodeGen
PrimeFieldCodeGen::operator+(const PrimeFieldCodeGen &other) const {
  ImplicitLocOpBuilder *b = BuilderContext::GetInstance().Top();
  return PrimeFieldCodeGen(
      mod_arith::AddOp::create(*b, value, other.value).getOutput());
}

PrimeFieldCodeGen &
PrimeFieldCodeGen::operator+=(const PrimeFieldCodeGen &other) {
  ImplicitLocOpBuilder *b = BuilderContext::GetInstance().Top();
  value = mod_arith::AddOp::create(*b, value, other.value).getOutput();
  return *this;
}

PrimeFieldCodeGen
PrimeFieldCodeGen::operator-(const PrimeFieldCodeGen &other) const {
  ImplicitLocOpBuilder *b = BuilderContext::GetInstance().Top();
  return PrimeFieldCodeGen(
      mod_arith::SubOp::create(*b, value, other.value).getOutput());
}

PrimeFieldCodeGen &
PrimeFieldCodeGen::operator-=(const PrimeFieldCodeGen &other) {
  ImplicitLocOpBuilder *b = BuilderContext::GetInstance().Top();
  value = mod_arith::SubOp::create(*b, value, other.value).getOutput();
  return *this;
}

PrimeFieldCodeGen
PrimeFieldCodeGen::operator*(const PrimeFieldCodeGen &other) const {
  ImplicitLocOpBuilder *b = BuilderContext::GetInstance().Top();
  return PrimeFieldCodeGen(
      mod_arith::MulOp::create(*b, value, other.value).getOutput());
}

PrimeFieldCodeGen &
PrimeFieldCodeGen::operator*=(const PrimeFieldCodeGen &other) {
  ImplicitLocOpBuilder *b = BuilderContext::GetInstance().Top();
  value = mod_arith::MulOp::create(*b, value, other.value).getOutput();
  return *this;
}

PrimeFieldCodeGen PrimeFieldCodeGen::operator-() const {
  ImplicitLocOpBuilder *b = BuilderContext::GetInstance().Top();
  return PrimeFieldCodeGen(mod_arith::NegateOp::create(*b, value).getOutput());
}

PrimeFieldCodeGen PrimeFieldCodeGen::Double() const {
  ImplicitLocOpBuilder *b = BuilderContext::GetInstance().Top();
  return PrimeFieldCodeGen(mod_arith::DoubleOp::create(*b, value).getOutput());
}

PrimeFieldCodeGen PrimeFieldCodeGen::Square() const {
  ImplicitLocOpBuilder *b = BuilderContext::GetInstance().Top();
  return PrimeFieldCodeGen(mod_arith::SquareOp::create(*b, value).getOutput());
}

PrimeFieldCodeGen PrimeFieldCodeGen::Inverse() const {
  ImplicitLocOpBuilder *b = BuilderContext::GetInstance().Top();
  if (BuilderContext::GetInstance().PreferFermatChain()) {
    if (auto modType = dyn_cast<mod_arith::ModArithType>(
            getElementTypeOrSelf(value.getType())))
      if (auto ab = detectSolinasForm(modType.getModulus().getValue()))
        return solinasInverseChain(ab->first, ab->second);
  }
  return PrimeFieldCodeGen(mod_arith::InverseOp::create(*b, value).getOutput());
}

std::optional<std::pair<unsigned, unsigned>>
PrimeFieldCodeGen::detectSolinasForm(const APInt &modulus) {
  // p = 2^a - 2^b + 1  <=>  p - 2 = (2^a - 1) - 2^b, i.e. all bits in [0, a)
  // set except a single zero at bit b. Detect that shape and read off (a, b).
  if (modulus.ult(5))
    return std::nullopt;
  APInt e = modulus - 2;
  unsigned a = e.getActiveBits(); // bit a-1 is the top set bit
  APInt zeros = APInt::getLowBitsSet(e.getBitWidth(), a) ^ e; // zero bits < a
  if (!zeros.isPowerOf2()) // need exactly one zero bit in [0, a)
    return std::nullopt;
  unsigned b = zeros.logBase2();
  // Both runs must be non-empty: low = bits [0,b), high = bits (b,a).
  if (b == 0 || b + 1 >= a)
    return std::nullopt;
  return std::make_pair(a, b);
}

// x -> x^(p-2) for p = 2^a - 2^b + 1, branch-free. With R_k = 2^k - 1,
// p-2 = R_{a-1-b}·2^{b+1} + R_b, so x^(p-2) = (x^R_{a-1-b})^{2^{b+1}}·x^R_b.
PrimeFieldCodeGen PrimeFieldCodeGen::solinasInverseChain(unsigned a,
                                                         unsigned b) const {
  PrimeFieldCodeGen x = *this;
  auto sq = [](PrimeFieldCodeGen v, unsigned n) {
    for (unsigned i = 0; i < n; ++i)
      v = v.Square();
    return v;
  };
  std::map<unsigned, PrimeFieldCodeGen> memo;
  std::function<PrimeFieldCodeGen(unsigned)> rk =
      [&](unsigned k) -> PrimeFieldCodeGen {
    auto it = memo.find(k);
    if (it != memo.end())
      return it->second;
    PrimeFieldCodeGen v;
    if (k == 1)
      v = x; // R_1 = x
    else if (k % 2 == 0)
      v = sq(rk(k / 2), k / 2) * rk(k / 2); // R_{2m} = (R_m)^{2^m} · R_m
    else
      v = rk(k - 1).Square() * x; // R_{2m+1} = (R_{2m})^2 · x
    memo.emplace(k, v);
    return v;
  };

  unsigned hiRun = a - 1 - b; // high run of ones: bits (b, a)
  unsigned loRun = b;         // low run of ones:  bits [0, b)
  unsigned lo = std::min(hiRun, loRun), hi = std::max(hiRun, loRun);
  PrimeFieldCodeGen rLo = rk(lo);
  PrimeFieldCodeGen rHi = (hi == lo + 1) ? (rLo.Square() * x) : rk(hi);
  PrimeFieldCodeGen rHiRun = (hiRun == hi) ? rHi : rLo;
  PrimeFieldCodeGen rLoRun = (loRun == hi) ? rHi : rLo;
  return sq(rHiRun, b + 1) * rLoRun; // (x^R_{a-1-b})^{2^{b+1}} · x^R_b
}

bool PrimeFieldCodeGen::hasInverseChain(const APInt &modulus) {
  return detectSolinasForm(modulus).has_value();
}

PrimeFieldCodeGen PrimeFieldCodeGen::CreateConst(int64_t constant) const {
  ImplicitLocOpBuilder *b = BuilderContext::GetInstance().Top();
  mod_arith::ModArithOperation op(
      constant, cast<mod_arith::ModArithType>(value.getType()));
  return PrimeFieldCodeGen(
      mod_arith::ConstantOp::create(*b, value.getType(), op.getIntegerAttr())
          .getOutput());
}

PrimeFieldCodeGen PrimeFieldCodeGen::CreateRationalConst(int64_t num,
                                                         int64_t denom) const {
  ImplicitLocOpBuilder *b = BuilderContext::GetInstance().Top();
  auto modArithType = cast<mod_arith::ModArithType>(value.getType());
  mod_arith::ModArithOperation result =
      mod_arith::ModArithOperation(num, modArithType) /
      mod_arith::ModArithOperation(denom, modArithType);
  return PrimeFieldCodeGen(mod_arith::ConstantOp::create(
                               *b, value.getType(), result.getIntegerAttr())
                               .getOutput());
}

} // namespace mlir::prime_ir::field
