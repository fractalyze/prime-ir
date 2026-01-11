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

#include "prime_ir/Dialect/ModArith/IR//ModArithOperation.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithOps.h"
#include "prime_ir/Utils/BuilderContext.h"

namespace mlir::prime_ir::field {

PrimeFieldCodeGen
PrimeFieldCodeGen::operator+(const PrimeFieldCodeGen &other) const {
  ImplicitLocOpBuilder *b = BuilderContext::GetInstance().Top();
  return PrimeFieldCodeGen(
      b->create<mod_arith::AddOp>(value, other.value).getOutput());
}

PrimeFieldCodeGen &
PrimeFieldCodeGen::operator+=(const PrimeFieldCodeGen &other) {
  ImplicitLocOpBuilder *b = BuilderContext::GetInstance().Top();
  value = b->create<mod_arith::AddOp>(value, other.value).getOutput();
  return *this;
}

PrimeFieldCodeGen
PrimeFieldCodeGen::operator-(const PrimeFieldCodeGen &other) const {
  ImplicitLocOpBuilder *b = BuilderContext::GetInstance().Top();
  return PrimeFieldCodeGen(
      b->create<mod_arith::SubOp>(value, other.value).getOutput());
}

PrimeFieldCodeGen &
PrimeFieldCodeGen::operator-=(const PrimeFieldCodeGen &other) {
  ImplicitLocOpBuilder *b = BuilderContext::GetInstance().Top();
  value = b->create<mod_arith::SubOp>(value, other.value).getOutput();
  return *this;
}

PrimeFieldCodeGen
PrimeFieldCodeGen::operator*(const PrimeFieldCodeGen &other) const {
  ImplicitLocOpBuilder *b = BuilderContext::GetInstance().Top();
  return PrimeFieldCodeGen(
      b->create<mod_arith::MulOp>(value, other.value).getOutput());
}

PrimeFieldCodeGen &
PrimeFieldCodeGen::operator*=(const PrimeFieldCodeGen &other) {
  ImplicitLocOpBuilder *b = BuilderContext::GetInstance().Top();
  value = b->create<mod_arith::MulOp>(value, other.value).getOutput();
  return *this;
}

PrimeFieldCodeGen PrimeFieldCodeGen::operator-() const {
  ImplicitLocOpBuilder *b = BuilderContext::GetInstance().Top();
  return PrimeFieldCodeGen(b->create<mod_arith::NegateOp>(value).getOutput());
}

PrimeFieldCodeGen PrimeFieldCodeGen::Double() const {
  ImplicitLocOpBuilder *b = BuilderContext::GetInstance().Top();
  return PrimeFieldCodeGen(b->create<mod_arith::DoubleOp>(value).getOutput());
}

PrimeFieldCodeGen PrimeFieldCodeGen::Square() const {
  ImplicitLocOpBuilder *b = BuilderContext::GetInstance().Top();
  return PrimeFieldCodeGen(b->create<mod_arith::SquareOp>(value).getOutput());
}

PrimeFieldCodeGen PrimeFieldCodeGen::Inverse() const {
  ImplicitLocOpBuilder *b = BuilderContext::GetInstance().Top();
  return PrimeFieldCodeGen(b->create<mod_arith::InverseOp>(value).getOutput());
}

PrimeFieldCodeGen PrimeFieldCodeGen::CreateConst(int64_t constant) const {
  ImplicitLocOpBuilder *b = BuilderContext::GetInstance().Top();
  mod_arith::ModArithOperation op(
      constant, cast<mod_arith::ModArithType>(value.getType()));
  return PrimeFieldCodeGen(
      b->create<mod_arith::ConstantOp>(value.getType(), op.getIntegerAttr())
          .getOutput());
}

} // namespace mlir::prime_ir::field
