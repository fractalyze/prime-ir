/* Copyright 2026 The ZKIR Authors.

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

#ifndef ZKIR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_POINTOPERATIONS_FIELDCODEGEN_H_
#define ZKIR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_POINTOPERATIONS_FIELDCODEGEN_H_

#include "absl/status/statusor.h"
#include "mlir/IR/Value.h"
#include "zk_dtypes/include/geometry/point_declarations.h"
#include "zkir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/PointOperations/PointCodeGenBaseForward.h"

namespace mlir::zkir::elliptic_curve {

// TODO(chokobole): **Architectural Refactoring Required**
//
// Unlike `field::PrimeFieldCodeGen` which generates `mod_arith` dialect
// operations, this class directly generates operations within the `field`
// dialect.
//
// **Issue:**
// A naming conflict exists because `field::FieldCodeGen` is already used to
// wrap both prime and extension field operations. From an elliptic curve
// perspective, the higher-level logic only requires generic "field operations,"
// regardless of whether the underlying type is a prime field or an extension
// field.
//
// **Goal:**
// Once the naming collision and hierarchy are resolved, this class should be
// unified and moved to the `field` directory to serve as the standard
// code-generation interface for all field-based arithmetic.
class FieldCodeGen {
public:
  FieldCodeGen() = default;
  explicit FieldCodeGen(Value value) : value(value) {}
  ~FieldCodeGen() = default;

  operator Value() const { return value; }

  FieldCodeGen operator+(const FieldCodeGen &other) const;
  FieldCodeGen &operator+=(const FieldCodeGen &other);
  FieldCodeGen operator-(const FieldCodeGen &other) const;
  FieldCodeGen &operator-=(const FieldCodeGen &other);
  FieldCodeGen operator*(const FieldCodeGen &other) const;
  FieldCodeGen &operator*=(const FieldCodeGen &other);
  FieldCodeGen operator-() const;

private:
  template <PointKind Kind>
  friend class PointCodeGenBase;
  template <typename, typename>
  friend class zk_dtypes::AffinePointOperation;
  template <typename, typename>
  friend class zk_dtypes::JacobianPointOperation;
  template <typename, typename>
  friend class zk_dtypes::PointXyzzOperation;

  FieldCodeGen Double() const;
  FieldCodeGen Square() const;
  FieldCodeGen Inverse() const;
  Value IsZero() const;
  FieldCodeGen CreateConst(int64_t constant) const;

  Value value;
};

} // namespace mlir::zkir::elliptic_curve

// NOLINTNEXTLINE(whitespace/line_length)
#endif // ZKIR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_POINTOPERATIONS_FIELDCODEGEN_H_
