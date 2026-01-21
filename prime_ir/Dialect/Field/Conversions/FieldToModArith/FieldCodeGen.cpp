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

#include "prime_ir/Dialect/Field/Conversions/FieldToModArith/FieldCodeGen.h"

#include "prime_ir/Dialect/Field/IR/FieldTypes.h"
#include "prime_ir/Utils/BuilderContext.h"

namespace mlir::prime_ir::field {

FieldCodeGen::FieldCodeGen(Type type, Value value,
                           const TypeConverter *converter) {
  if (isa<PrimeFieldType>(type)) {
    codeGen = PrimeFieldCodeGen(value);
    return;
  }

  ImplicitLocOpBuilder *b = BuilderContext::GetInstance().Top();
  auto efType = cast<ExtensionFieldType>(type);
  Value nonResidue = b->create<mod_arith::ConstantOp>(
      converter->convertType(efType.getBaseField()),
      cast<IntegerAttr>(efType.getNonResidue()));
  unsigned degree = efType.getDegree();
  switch (degree) {
  case 2:
    codeGen = ExtensionFieldCodeGen<2>(value, nonResidue);
    break;
  case 3:
    codeGen = ExtensionFieldCodeGen<3>(value, nonResidue);
    break;
  case 4:
    codeGen = ExtensionFieldCodeGen<4>(value, nonResidue);
    break;
  default:
    llvm_unreachable("Unsupported extension field degree");
  }
}

FieldCodeGen::operator Value() const {
  return std::visit(
      [](const auto &v) -> Value { return static_cast<Value>(v); }, codeGen);
}

namespace {

template <typename F>
FieldCodeGen applyUnaryOp(const FieldCodeGen::CodeGenType &codeGen, F op) {
  return std::visit([&](const auto &v) -> FieldCodeGen { return op(v); },
                    codeGen);
}

template <typename F>
FieldCodeGen applyBinaryOp(const FieldCodeGen::CodeGenType &a,
                           const FieldCodeGen::CodeGenType &b, F op) {
  return std::visit(
      [&](const auto &lhs, const auto &rhs) -> FieldCodeGen {
        if constexpr (std::is_same_v<std::decay_t<decltype(lhs)>,
                                     std::decay_t<decltype(rhs)>>) {
          return op(lhs, rhs);
        }
        llvm_unreachable("Unsupported field type in binary operator");
      },
      a, b);
}

} // namespace

FieldCodeGen FieldCodeGen::operator+(const FieldCodeGen &other) const {
  return applyBinaryOp(codeGen, other.codeGen,
                       [](const auto &a, const auto &b) { return a + b; });
}

FieldCodeGen FieldCodeGen::operator-(const FieldCodeGen &other) const {
  return applyBinaryOp(codeGen, other.codeGen,
                       [](const auto &a, const auto &b) { return a - b; });
}

FieldCodeGen FieldCodeGen::operator*(const FieldCodeGen &other) const {
  return applyBinaryOp(codeGen, other.codeGen,
                       [](const auto &a, const auto &b) { return a * b; });
}

FieldCodeGen FieldCodeGen::operator-() const {
  return applyUnaryOp(codeGen, [](const auto &v) { return -v; });
}

FieldCodeGen FieldCodeGen::dbl() const {
  return applyUnaryOp(codeGen, [](const auto &v) { return v.Double(); });
}

FieldCodeGen FieldCodeGen::square() const {
  return applyUnaryOp(codeGen, [](const auto &v) { return v.Square(); });
}

FieldCodeGen FieldCodeGen::inverse() const {
  return applyUnaryOp(codeGen, [](const auto &v) { return v.Inverse(); });
}

} // namespace mlir::prime_ir::field
