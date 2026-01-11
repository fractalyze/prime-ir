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

#include "prime_ir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/PointOperations/PointCodeGen.h"

#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveTypes.h"

namespace mlir::prime_ir::elliptic_curve {

template class PointCodeGenBase<PointKind::kAffine>;
template class PointCodeGenBase<PointKind::kJacobian>;
template class PointCodeGenBase<PointKind::kXYZZ>;

PointCodeGen::PointCodeGen(Type type, Value value) {
  if (isa<AffineType>(type)) {
    codeGen = AffinePointCodeGen(value);
  } else if (isa<JacobianType>(type)) {
    codeGen = JacobianPointCodeGen(value);
  } else if (isa<XYZZType>(type)) {
    codeGen = XYZZPointCodeGen(value);
  } else {
    llvm_unreachable("Unsupported extension field type");
  }
}

PointCodeGen::operator Value() const {
  return std::visit(
      [](const auto &v) -> Value { return static_cast<Value>(v); }, codeGen);
}

PointKind PointCodeGen::getKind() const {
  return std::visit([](const auto &v) -> PointKind { return v.getKind(); },
                    codeGen);
}

namespace {

template <typename F>
PointCodeGen applyUnaryOp(const PointCodeGen::CodeGenType &codeGen, F op) {
  return std::visit([&](const auto &v) -> PointCodeGen { return op(v); },
                    codeGen);
}

template <typename F>
PointCodeGen applyBinaryOp(const PointCodeGen::CodeGenType &a,
                           const PointCodeGen::CodeGenType &b, F op) {
  return std::visit(
      [&](const auto &lhs, const auto &rhs) -> PointCodeGen {
        if constexpr (std::is_same_v<std::decay_t<decltype(lhs)>,
                                     std::decay_t<decltype(rhs)>>) {
          return op(lhs, rhs);
          // NOLINTNEXTLINE(readability/braces)
        } else if constexpr (std::is_same_v<std::decay_t<decltype(lhs)>,
                                            AffinePointCodeGen> ||
                             std::is_same_v<std::decay_t<decltype(rhs)>,
                                            AffinePointCodeGen>) {
          return op(lhs, rhs);
        }
        llvm_unreachable("Unsupported field type in binary operator");
      },
      a, b);
}

} // namespace

PointCodeGen PointCodeGen::add(const PointCodeGen &other,
                               PointKind outputKind) const {
  PointKind lhsKind = getKind();
  PointKind rhsKind = other.getKind();
  if (lhsKind == PointKind::kAffine && lhsKind == rhsKind) {
    if (outputKind == PointKind::kXYZZ) {
      return std::get<AffinePointCodeGen>(codeGen).AddToXyzz(
          std::get<AffinePointCodeGen>(other.codeGen));
    }
  }

  return applyBinaryOp(codeGen, other.codeGen,
                       [](const auto &a, const auto &b) { return a + b; });
}

PointCodeGen PointCodeGen::dbl(PointKind outputKind) const {
  PointKind kind = getKind();
  if (kind == PointKind::kAffine) {
    if (outputKind == PointKind::kXYZZ) {
      return std::get<AffinePointCodeGen>(codeGen).DoubleToXyzz();
    }
  }
  return applyUnaryOp(codeGen, [](const auto &v) { return v.Double(); });
}

PointCodeGen PointCodeGen::convert(PointKind outputKind) const {
  PointKind kind = getKind();
  if (outputKind == PointKind::kAffine) {
    if (kind == PointKind::kAffine) {
      return *this;
    } else if (kind == PointKind::kJacobian) {
      return std::get<JacobianPointCodeGen>(codeGen).ToAffine();
    } else if (kind == PointKind::kXYZZ) {
      return std::get<XYZZPointCodeGen>(codeGen).ToAffine();
    }
  } else if (outputKind == PointKind::kJacobian) {
    if (kind == PointKind::kAffine) {
      return std::get<AffinePointCodeGen>(codeGen).ToJacobian();
    } else if (kind == PointKind::kJacobian) {
      return *this;
    } else if (kind == PointKind::kXYZZ) {
      return std::get<XYZZPointCodeGen>(codeGen).ToJacobian();
    }
  } else if (outputKind == PointKind::kXYZZ) {
    if (kind == PointKind::kAffine) {
      return std::get<AffinePointCodeGen>(codeGen).ToXyzz();
    } else if (kind == PointKind::kJacobian) {
      return std::get<JacobianPointCodeGen>(codeGen).ToXyzz();
    } else if (kind == PointKind::kXYZZ) {
      return *this;
    }
  }
  llvm_unreachable("Unsupported point kind");
}

} // namespace mlir::prime_ir::elliptic_curve
