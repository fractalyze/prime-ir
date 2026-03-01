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

#ifndef PRIME_IR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_FIELDDIALECTARITHMETIC_H_
#define PRIME_IR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_FIELDDIALECTARITHMETIC_H_

#include "mlir/IR/Value.h"
#include "prime_ir/Dialect/Field/IR/FieldOps.h"
#include "prime_ir/Utils/BuilderContext.h"

namespace mlir::prime_ir::elliptic_curve {

// CRTP base providing field.* dialect arithmetic operators.
//
// Derived must provide:
//   - Value getValue() const;
//   - explicit Derived(Value);
template <typename Derived>
class FieldDialectArithmetic {
public:
  Derived operator+(const Derived &o) const {
    auto *b = BuilderContext::GetInstance().Top();
    return Derived(
        b->create<field::AddOp>(self().getValue(), o.getValue()).getOutput());
  }

  Derived operator-(const Derived &o) const {
    auto *b = BuilderContext::GetInstance().Top();
    return Derived(
        b->create<field::SubOp>(self().getValue(), o.getValue()).getOutput());
  }

  Derived operator*(const Derived &o) const {
    auto *b = BuilderContext::GetInstance().Top();
    return Derived(
        b->create<field::MulOp>(self().getValue(), o.getValue()).getOutput());
  }

  Derived operator-() const {
    auto *b = BuilderContext::GetInstance().Top();
    return Derived(b->create<field::NegateOp>(self().getValue()).getOutput());
  }

  Derived Double() const {
    auto *b = BuilderContext::GetInstance().Top();
    return Derived(b->create<field::DoubleOp>(self().getValue()).getOutput());
  }

  Derived Square() const {
    auto *b = BuilderContext::GetInstance().Top();
    return Derived(b->create<field::SquareOp>(self().getValue()).getOutput());
  }

  Derived Inverse() const {
    auto *b = BuilderContext::GetInstance().Top();
    return Derived(b->create<field::InverseOp>(self().getValue()).getOutput());
  }

  Derived &operator+=(const Derived &o) {
    auto &self = static_cast<Derived &>(*this);
    self = self + o;
    return self;
  }

  Derived &operator-=(const Derived &o) {
    auto &self = static_cast<Derived &>(*this);
    self = self - o;
    return self;
  }

  Derived &operator*=(const Derived &o) {
    auto &self = static_cast<Derived &>(*this);
    self = self * o;
    return self;
  }

private:
  const Derived &self() const { return static_cast<const Derived &>(*this); }
};

} // namespace mlir::prime_ir::elliptic_curve

// NOLINTNEXTLINE(whitespace/line_length)
#endif // PRIME_IR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_FIELDDIALECTARITHMETIC_H_
