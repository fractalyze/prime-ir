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

#include "zkir/Dialect/Field/Conversions/FieldToModArith/PrimeFieldCodeGen.h"

#include "zkir/Dialect/ModArith/IR/ModArithOps.h"

namespace mlir::zkir::field {

PrimeFieldCodeGen
PrimeFieldCodeGen::operator+(const PrimeFieldCodeGen &other) const {
  return PrimeFieldCodeGen(
      b, b->create<mod_arith::AddOp>(value, other.value).getOutput());
}

PrimeFieldCodeGen &
PrimeFieldCodeGen::operator+=(const PrimeFieldCodeGen &other) {
  value = b->create<mod_arith::AddOp>(value, other.value).getOutput();
  return *this;
}

PrimeFieldCodeGen
PrimeFieldCodeGen::operator-(const PrimeFieldCodeGen &other) const {
  return PrimeFieldCodeGen(
      b, b->create<mod_arith::SubOp>(value, other.value).getOutput());
}

PrimeFieldCodeGen &
PrimeFieldCodeGen::operator-=(const PrimeFieldCodeGen &other) {
  value = b->create<mod_arith::SubOp>(value, other.value).getOutput();
  return *this;
}

PrimeFieldCodeGen
PrimeFieldCodeGen::operator*(const PrimeFieldCodeGen &other) const {
  return PrimeFieldCodeGen(
      b, b->create<mod_arith::MulOp>(value, other.value).getOutput());
}

PrimeFieldCodeGen &
PrimeFieldCodeGen::operator*=(const PrimeFieldCodeGen &other) {
  value = b->create<mod_arith::MulOp>(value, other.value).getOutput();
  return *this;
}

PrimeFieldCodeGen PrimeFieldCodeGen::operator-() const {
  return PrimeFieldCodeGen(b,
                           b->create<mod_arith::NegateOp>(value).getOutput());
}

PrimeFieldCodeGen PrimeFieldCodeGen::Double() const {
  return PrimeFieldCodeGen(b,
                           b->create<mod_arith::DoubleOp>(value).getOutput());
}

PrimeFieldCodeGen PrimeFieldCodeGen::Square() const {
  return PrimeFieldCodeGen(b,
                           b->create<mod_arith::SquareOp>(value).getOutput());
}

PrimeFieldCodeGen PrimeFieldCodeGen::Inverse() const {
  return PrimeFieldCodeGen(b,
                           b->create<mod_arith::InverseOp>(value).getOutput());
}

} // namespace mlir::zkir::field
