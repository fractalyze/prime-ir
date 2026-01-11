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
  Value zero =
      cast<field::FieldTypeInterface>(value.getType()).createZeroConstant(*b);
  return b->create<field::CmpOp>(arith::CmpIPredicate::eq, value, zero);
}

namespace {

template <size_t N>
struct DegreeDispatcher {
  static std::optional<FieldCodeGen>
  dispatch(size_t degree, ImplicitLocOpBuilder *b, Type type, int64_t constant,
           field::ExtensionFieldTypeInterface efType) {
    if (degree == N) {
      return FieldCodeGen(b->create<field::ConstantOp>(
          type, field::ExtensionFieldOperation<N>(constant, efType)
                    .getDenseIntElementsAttr()));
    }
    if constexpr (N > field::kMinExtDegree) {
      return DegreeDispatcher<N - 1>::dispatch(degree, b, type, constant,
                                               efType);
    }
    return std::nullopt;
  }
};

} // namespace

FieldCodeGen FieldCodeGen::CreateConst(int64_t constant) const {
  ImplicitLocOpBuilder *b = BuilderContext::GetInstance().Top();
  if (auto pfType = dyn_cast<field::PrimeFieldType>(value.getType())) {
    return FieldCodeGen(b->create<field::ConstantOp>(
        value.getType(),
        field::PrimeFieldOperation(constant, pfType).getIntegerAttr()));
  } else if (auto efType = dyn_cast<field::ExtensionFieldTypeInterface>(
                 value.getType())) {
    size_t degree = efType.getDegreeOverPrime();
    auto result = DegreeDispatcher<field::kMaxExtDegree>::dispatch(
        degree, b, value.getType(), constant, efType);
    if (result)
      return *result;
    llvm_unreachable("Unsupported extension field degree");
  }
  llvm_unreachable("Unsupported field type");
}

} // namespace mlir::prime_ir::elliptic_curve
