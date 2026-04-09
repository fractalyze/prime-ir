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

#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveAttributes.h"

#include "mlir/IR/TypeUtilities.h"
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveTypes.h"
#include "prime_ir/Dialect/Field/IR/FieldOperation.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"

namespace mlir::prime_ir::elliptic_curve {

// static
Attribute ShortWeierstrassAttr::parse(AsmParser &parser, Type odsType) {
  Attribute a, b, gX, gY;
  Type baseField;
  if (failed(parser.parseLess()) || failed(parser.parseAttribute(a)) ||
      failed(parser.parseComma()) || failed(parser.parseAttribute(b)) ||
      failed(parser.parseComma()) || failed(parser.parseLParen()) ||
      failed(parser.parseAttribute(gX)) || failed(parser.parseComma()) ||
      failed(parser.parseAttribute(gY)) || failed(parser.parseRParen()) ||
      failed(parser.parseGreater()) ||
      failed(field::parseColonFieldType(parser, baseField)))
    return nullptr;

  if (failed(field::validateAttribute(parser, baseField, a, "a")) ||
      failed(field::validateAttribute(parser, baseField, b, "b")) ||
      failed(field::validateAttribute(parser, baseField, gX, "gX")) ||
      failed(field::validateAttribute(parser, baseField, gY, "gY")))
    return nullptr;

  a = field::maybeToMontgomery(baseField, a);
  b = field::maybeToMontgomery(baseField, b);
  gX = field::maybeToMontgomery(baseField, gX);
  gY = field::maybeToMontgomery(baseField, gY);

  return ShortWeierstrassAttr::get(a.getContext(), baseField,
                                   cast<TypedAttr>(a), cast<TypedAttr>(b),
                                   cast<TypedAttr>(gX), cast<TypedAttr>(gY));
}

void ShortWeierstrassAttr::print(AsmPrinter &printer) const {
  Attribute a = field::maybeToStandard(getBaseField(), getA());
  Attribute b = field::maybeToStandard(getBaseField(), getB());
  Attribute gX = field::maybeToStandard(getBaseField(), getGx());
  Attribute gY = field::maybeToStandard(getBaseField(), getGy());

  printer << '<' << a << ',' << b << ",(" << gX << ',' << gY
          << ")> : " << getBaseField();
}

// static
LogicalResult
ShortWeierstrassAttr::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                             Type baseField, TypedAttr a, TypedAttr b,
                             TypedAttr gX, TypedAttr gY) {
  auto aOp = field::FieldOperation::fromUnchecked(a, baseField);
  auto bOp = field::FieldOperation::fromUnchecked(b, baseField);
  auto gXOp = field::FieldOperation::fromUnchecked(gX, baseField);
  auto gYOp = field::FieldOperation::fromUnchecked(gY, baseField);
  if (gYOp.square() != gXOp.square() * gXOp + aOp * gXOp + bOp) {
    emitError()
        << "a, b, gX, and gY must satisfy the equation y² = x³ + ax + b";
    return failure();
  }
  return success();
}

// static
Attribute TwistedEdwardsAttr::parse(AsmParser &parser, Type odsType) {
  Attribute a, d, gX, gY;
  Type baseField;
  if (failed(parser.parseLess()) || failed(parser.parseAttribute(a)) ||
      failed(parser.parseComma()) || failed(parser.parseAttribute(d)) ||
      failed(parser.parseComma()) || failed(parser.parseLParen()) ||
      failed(parser.parseAttribute(gX)) || failed(parser.parseComma()) ||
      failed(parser.parseAttribute(gY)) || failed(parser.parseRParen()) ||
      failed(parser.parseGreater()) ||
      failed(field::parseColonFieldType(parser, baseField)))
    return nullptr;

  if (failed(field::validateAttribute(parser, baseField, a, "a")) ||
      failed(field::validateAttribute(parser, baseField, d, "d")) ||
      failed(field::validateAttribute(parser, baseField, gX, "gX")) ||
      failed(field::validateAttribute(parser, baseField, gY, "gY")))
    return nullptr;

  a = field::maybeToMontgomery(baseField, a);
  d = field::maybeToMontgomery(baseField, d);
  gX = field::maybeToMontgomery(baseField, gX);
  gY = field::maybeToMontgomery(baseField, gY);

  return TwistedEdwardsAttr::get(a.getContext(), baseField, cast<TypedAttr>(a),
                                 cast<TypedAttr>(d), cast<TypedAttr>(gX),
                                 cast<TypedAttr>(gY));
}

void TwistedEdwardsAttr::print(AsmPrinter &printer) const {
  Attribute a = field::maybeToStandard(getBaseField(), getA());
  Attribute d = field::maybeToStandard(getBaseField(), getD());
  Attribute gX = field::maybeToStandard(getBaseField(), getGx());
  Attribute gY = field::maybeToStandard(getBaseField(), getGy());

  printer << '<' << a << ',' << d << ",(" << gX << ',' << gY
          << ")> : " << getBaseField();
}

// static
LogicalResult
TwistedEdwardsAttr::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                           Type baseField, TypedAttr a, TypedAttr d,
                           TypedAttr gX, TypedAttr gY) {
  auto aOp = field::FieldOperation::fromUnchecked(a, baseField);
  auto dOp = field::FieldOperation::fromUnchecked(d, baseField);
  auto gXOp = field::FieldOperation::fromUnchecked(gX, baseField);
  auto gYOp = field::FieldOperation::fromUnchecked(gY, baseField);
  if (aOp.isZero()) {
    emitError() << "twisted Edwards parameter 'a' must be non-zero";
    return failure();
  }
  if (dOp.isZero()) {
    emitError() << "twisted Edwards parameter 'd' must be non-zero";
    return failure();
  }
  // Verify: a * Gx² + Gy² == 1 + d * Gx² * Gy²
  auto gx2 = gXOp.square();
  auto gy2 = gYOp.square();
  auto one = aOp.getOne();
  if (aOp * gx2 + gy2 != one + dOp * gx2 * gy2) {
    emitError() << "a, d, gX, and gY must satisfy the equation "
                   "a * x² + y² = 1 + d * x² * y²";
    return failure();
  }
  return success();
}

} // namespace mlir::prime_ir::elliptic_curve
