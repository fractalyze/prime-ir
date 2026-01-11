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
  auto efType = cast<ExtensionFieldTypeInterface>(type);
  Value nonResidue = b->create<mod_arith::ConstantOp>(
      converter->convertType(efType.getBaseFieldType()),
      cast<IntegerAttr>(efType.getNonResidue()));
  if (isa<QuadraticExtFieldType>(type)) {
    codeGen = QuadraticExtensionFieldCodeGen(value, nonResidue);
  } else if (isa<CubicExtFieldType>(type)) {
    codeGen = CubicExtensionFieldCodeGen(value, nonResidue);
  } else if (isa<QuarticExtFieldType>(type)) {
    codeGen = QuarticExtensionFieldCodeGen(value, nonResidue);
  } else {
    llvm_unreachable("Unsupported extension field type");
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

FieldCodeGen FieldCodeGen::dbl() const {
  return applyUnaryOp(codeGen, [](const auto &v) { return v.Double(); });
}

FieldCodeGen FieldCodeGen::square() const {
  return applyUnaryOp(codeGen, [](const auto &v) { return v.Square(); });
}

FieldCodeGen FieldCodeGen::negate() const {
  return applyUnaryOp(codeGen, [](const auto &v) { return -v; });
}

FieldCodeGen FieldCodeGen::inverse() const {
  return applyUnaryOp(codeGen, [](const auto &v) { return v.Inverse(); });
}

} // namespace mlir::prime_ir::field
