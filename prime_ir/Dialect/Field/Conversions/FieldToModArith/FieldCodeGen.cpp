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
  Value nonResidue = efType.createNonResidueValue(*b);

  auto sig = getTowerSignature(efType);
#define CREATE_CODEGEN(unused_sig, TypeName)                                   \
  codeGen = TypeName(value, nonResidue);
  DISPATCH_TOWER_BY_SIGNATURE(sig, CREATE_CODEGEN, ExtensionFieldCodeGen,
                              CodeGen)
#undef CREATE_CODEGEN
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

// Type trait to detect ExtensionFieldCodeGen and extract its BaseFieldT.
template <typename T>
struct IsExtensionFieldCodeGen : std::false_type {};

template <size_t N, typename B>
struct IsExtensionFieldCodeGen<ExtensionFieldCodeGen<N, B>> : std::true_type {
  using BaseFieldT = B;
};

// Mixed-type dispatch: one operand is an extension field, the other is its
// base field. ExtBaseOp handles (ext, base); BaseExtOp handles (base, ext)
// with arguments reordered as (ext, base) for convenience.
template <typename ExtBaseOp, typename BaseExtOp>
FieldCodeGen applyMixedBinaryOp(const FieldCodeGen::CodeGenType &a,
                                const FieldCodeGen::CodeGenType &b,
                                ExtBaseOp extBaseOp, BaseExtOp baseExtOp) {
  return std::visit(
      [&](const auto &lhs, const auto &rhs) -> FieldCodeGen {
        using LHS = std::decay_t<decltype(lhs)>;
        using RHS = std::decay_t<decltype(rhs)>;
        if constexpr (IsExtensionFieldCodeGen<LHS>::value) {
          if constexpr (std::is_same_v<
                            typename IsExtensionFieldCodeGen<LHS>::BaseFieldT,
                            RHS>) {
            return extBaseOp(lhs, rhs);
          }
        }
        if constexpr (IsExtensionFieldCodeGen<RHS>::value) {
          if constexpr (std::is_same_v<
                            typename IsExtensionFieldCodeGen<RHS>::BaseFieldT,
                            LHS>) {
            return baseExtOp(rhs, lhs);
          }
        }
        llvm_unreachable("Unsupported mixed-type field operation");
      },
      a, b);
}

} // namespace

FieldCodeGen FieldCodeGen::operator+(const FieldCodeGen &other) const {
  if (codeGen.index() == other.codeGen.index()) {
    return applyBinaryOp(codeGen, other.codeGen,
                         [](const auto &a, const auto &b) { return a + b; });
  }
  return applyMixedBinaryOp(
      codeGen, other.codeGen,
      [](const auto &ext, const auto &base) { return ext + base; },
      [](const auto &ext, const auto &base) { return ext + base; });
}

FieldCodeGen FieldCodeGen::operator-(const FieldCodeGen &other) const {
  if (codeGen.index() == other.codeGen.index()) {
    return applyBinaryOp(codeGen, other.codeGen,
                         [](const auto &a, const auto &b) { return a - b; });
  }
  return applyMixedBinaryOp(
      codeGen, other.codeGen,
      [](const auto &ext, const auto &base) { return ext - base; },
      [](const auto &ext, const auto &base) { return -(ext - base); });
}

FieldCodeGen FieldCodeGen::operator*(const FieldCodeGen &other) const {
  if (codeGen.index() == other.codeGen.index()) {
    return applyBinaryOp(codeGen, other.codeGen,
                         [](const auto &a, const auto &b) { return a * b; });
  }
  return applyMixedBinaryOp(
      codeGen, other.codeGen,
      [](const auto &ext, const auto &base) { return ext * base; },
      [](const auto &ext, const auto &base) { return ext * base; });
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
