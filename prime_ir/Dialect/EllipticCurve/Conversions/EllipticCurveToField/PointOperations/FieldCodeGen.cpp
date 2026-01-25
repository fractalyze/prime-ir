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

#include "prime_ir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/PointOperations/FieldCodeGen.h"

#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "prime_ir/Dialect/Field/IR/FieldOperation.h"
#include "prime_ir/Dialect/Field/IR/FieldOps.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"
#include "prime_ir/Utils/BuilderContext.h"

namespace mlir::prime_ir::elliptic_curve {

FieldCodeGen FieldCodeGen::operator+(const FieldCodeGen &other) const {
  ImplicitLocOpBuilder *b = BuilderContext::GetInstance().Top();
  return FieldCodeGen(b->create<field::AddOp>(value, other.value).getOutput());
}

FieldCodeGen &FieldCodeGen::operator+=(const FieldCodeGen &other) {
  ImplicitLocOpBuilder *b = BuilderContext::GetInstance().Top();
  value = b->create<field::AddOp>(value, other.value).getOutput();
  return *this;
}

FieldCodeGen FieldCodeGen::operator-(const FieldCodeGen &other) const {
  ImplicitLocOpBuilder *b = BuilderContext::GetInstance().Top();
  return FieldCodeGen(b->create<field::SubOp>(value, other.value).getOutput());
}

FieldCodeGen &FieldCodeGen::operator-=(const FieldCodeGen &other) {
  ImplicitLocOpBuilder *b = BuilderContext::GetInstance().Top();
  value = b->create<field::SubOp>(value, other.value).getOutput();
  return *this;
}

FieldCodeGen FieldCodeGen::operator*(const FieldCodeGen &other) const {
  ImplicitLocOpBuilder *b = BuilderContext::GetInstance().Top();
  return FieldCodeGen(b->create<field::MulOp>(value, other.value).getOutput());
}

FieldCodeGen &FieldCodeGen::operator*=(const FieldCodeGen &other) {
  ImplicitLocOpBuilder *b = BuilderContext::GetInstance().Top();
  value = b->create<field::MulOp>(value, other.value).getOutput();
  return *this;
}

FieldCodeGen FieldCodeGen::operator-() const {
  ImplicitLocOpBuilder *b = BuilderContext::GetInstance().Top();
  return FieldCodeGen(b->create<field::NegateOp>(value).getOutput());
}

FieldCodeGen FieldCodeGen::Double() const {
  ImplicitLocOpBuilder *b = BuilderContext::GetInstance().Top();
  return FieldCodeGen(b->create<field::DoubleOp>(value).getOutput());
}

FieldCodeGen FieldCodeGen::Square() const {
  ImplicitLocOpBuilder *b = BuilderContext::GetInstance().Top();
  return FieldCodeGen(b->create<field::SquareOp>(value).getOutput());
}

FieldCodeGen FieldCodeGen::Inverse() const {
  ImplicitLocOpBuilder *b = BuilderContext::GetInstance().Top();
  return FieldCodeGen(b->create<field::InverseOp>(value).getOutput());
}

Value FieldCodeGen::IsZero() const {
  ImplicitLocOpBuilder *b = BuilderContext::GetInstance().Top();
  Value zero = field::createFieldZero(value.getType(), *b);
  return b->create<field::CmpOp>(arith::CmpIPredicate::eq, value, zero);
}

namespace {

// Creates a constant in the given field type.
// For tower extensions, this recursively creates constants at each level.
Value createFieldConstant(ImplicitLocOpBuilder &b, Type fieldType,
                          int64_t constant) {
  if (auto pfType = dyn_cast<field::PrimeFieldType>(fieldType)) {
    return b.create<field::ConstantOp>(
        fieldType, field::PrimeFieldOperation(constant, pfType).getIntegerAttr());
  }

  auto efType = cast<field::ExtensionFieldType>(fieldType);
  Type baseFieldType = efType.getBaseField();
  unsigned degree = efType.getDegree();

  // Recursively create constant in base field (handles tower extensions)
  Value baseConstant = createFieldConstant(b, baseFieldType, constant);

  // Create [baseConstant, 0, 0, ...] with 'degree' coefficients
  SmallVector<Value> coeffs(degree);
  coeffs[0] = baseConstant;
  Value zero = createFieldConstant(b, baseFieldType, 0);
  for (unsigned i = 1; i < degree; ++i) {
    coeffs[i] = zero;
  }

  return b.create<field::ExtFromCoeffsOp>(efType, coeffs);
}

} // namespace

FieldCodeGen FieldCodeGen::CreateConst(int64_t constant) const {
  ImplicitLocOpBuilder *b = BuilderContext::GetInstance().Top();
  return FieldCodeGen(createFieldConstant(*b, value.getType(), constant));
}

} // namespace mlir::prime_ir::elliptic_curve
