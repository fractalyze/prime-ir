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

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "prime_ir/Dialect/Field/IR/FieldOps.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"
#include "prime_ir/Utils/BuilderContext.h"

namespace mlir::prime_ir::field {

namespace {

// Helper to create non-residue constant for extension fields
Value createNonResidueConstant(ImplicitLocOpBuilder &b,
                               ExtensionFieldType efType,
                               const TypeConverter *converter) {
  Type baseFieldType = efType.getBaseField();
  Attribute nonResidueAttr = efType.getNonResidue();

  if (isa<PrimeFieldType>(baseFieldType)) {
    // Non-tower: non-residue is IntegerAttr in the prime field
    return b.create<mod_arith::ConstantOp>(
        converter->convertType(baseFieldType),
        cast<IntegerAttr>(nonResidueAttr));
  }

  // Tower: non-residue is DenseIntElementsAttr in the base extension field
  auto baseEfType = cast<ExtensionFieldType>(baseFieldType);
  auto denseAttr = cast<DenseIntElementsAttr>(nonResidueAttr);
  return ConstantOp::materialize(b, denseAttr, baseEfType, b.getLoc());
}

// Returns a signature encoding the tower structure.
// e.g., Fp12 = ((Fp2)^3)^2 -> {2, 3, 2}  (top degree first)
// e.g., Fp6 = (Fp2)^3 -> {3, 2}
// e.g., Fp2 -> {2}
SmallVector<unsigned> getTowerSignature(ExtensionFieldType efType) {
  SmallVector<unsigned> signature;
  Type current = efType;
  while (auto ef = dyn_cast<ExtensionFieldType>(current)) {
    signature.push_back(ef.getDegree());
    current = ef.getBaseField();
  }
  return signature;
}

} // namespace

FieldCodeGen::FieldCodeGen(Type type, Value value,
                           const TypeConverter *converter) {
  if (isa<PrimeFieldType>(type)) {
    codeGen = PrimeFieldCodeGen(value);
    return;
  }

  ImplicitLocOpBuilder *b = BuilderContext::GetInstance().Top();
  auto efType = cast<ExtensionFieldType>(type);
  Value nonResidue = createNonResidueConstant(*b, efType, converter);

  // Get tower signature to dispatch to the correct type
  auto signature = getTowerSignature(efType);
  unsigned depth = signature.size();

  // Dispatch based on tower signature
  // signature[0] = top degree, signature[1] = next level, etc.
  if (depth == 1) {
    // Non-tower extension field
    switch (signature[0]) {
    case 2:
      codeGen = QuadraticExtensionFieldCodeGen(value, nonResidue);
      return;
    case 3:
      codeGen = CubicExtensionFieldCodeGen(value, nonResidue);
      return;
    case 4:
      codeGen = QuarticExtensionFieldCodeGen(value, nonResidue);
      return;
    }
  } else if (depth == 2) {
    // Depth-1 tower: signature = {topDegree, baseDegree}
    unsigned topDegree = signature[0];
    unsigned baseDegree = signature[1];
    if (baseDegree == 2) {
      switch (topDegree) {
      case 2:
        codeGen = TowerQuadraticOverQuadraticCodeGen(value, nonResidue);
        return;
      case 3:
        codeGen = TowerCubicOverQuadraticCodeGen(value, nonResidue);
        return;
      case 4:
        codeGen = TowerQuarticOverQuadraticCodeGen(value, nonResidue);
        return;
      }
    } else if (baseDegree == 3 && topDegree == 2) {
      codeGen = TowerQuadraticOverCubicCodeGen(value, nonResidue);
      return;
    }
  } else if (depth == 3) {
    // Depth-2 tower: signature = {d0, d1, d2}
    unsigned d0 = signature[0], d1 = signature[1], d2 = signature[2];
    // Fp12 = ((Fp2)^3)^2: {2, 3, 2}
    if (d0 == 2 && d1 == 3 && d2 == 2) {
      codeGen = TowerQuadraticOverCubicOverQuadraticCodeGen(value, nonResidue);
      return;
    }
    // Fp12 = ((Fp2)^2)^3: {3, 2, 2}
    if (d0 == 3 && d1 == 2 && d2 == 2) {
      codeGen = TowerCubicOverQuadraticOverQuadraticCodeGen(value, nonResidue);
      return;
    }
    // Fp8 = ((Fp2)^2)^2: {2, 2, 2}
    if (d0 == 2 && d1 == 2 && d2 == 2) {
      codeGen =
          TowerQuadraticOverQuadraticOverQuadraticCodeGen(value, nonResidue);
      return;
    }
    // Fp12 = ((Fp3)^2)^2: {2, 2, 3}
    if (d0 == 2 && d1 == 2 && d2 == 3) {
      codeGen = TowerQuadraticOverQuadraticOverCubicCodeGen(value, nonResidue);
      return;
    }
  } else if (depth == 4) {
    // Depth-3 tower: signature = {d0, d1, d2, d3}
    unsigned d0 = signature[0], d1 = signature[1], d2 = signature[2],
             d3 = signature[3];
    // Fp24 = (((Fp2)^3)^2)^2: {2, 2, 3, 2}
    if (d0 == 2 && d1 == 2 && d2 == 3 && d3 == 2) {
      codeGen = TowerQuadraticOverQuadraticOverCubicOverQuadraticCodeGen(
          value, nonResidue);
      return;
    }
  }

  llvm_unreachable("Unsupported tower extension field configuration");
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
