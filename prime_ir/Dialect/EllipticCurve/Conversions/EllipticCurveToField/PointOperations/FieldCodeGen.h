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

#ifndef PRIME_IR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_POINTOPERATIONS_FIELDCODEGEN_H_
#define PRIME_IR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_POINTOPERATIONS_FIELDCODEGEN_H_

#include "absl/status/statusor.h"
#include "mlir/IR/Value.h"
#include "prime_ir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/FieldDialectArithmetic.h"
#include "prime_ir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/PointOperations/PointCodeGenBaseForward.h"
#include "zk_dtypes/include/geometry/point_declarations.h"

namespace mlir::prime_ir::elliptic_curve {

// Generates field.* dialect operations for field arithmetic.
//
// Unlike `field::PrimeFieldCodeGen` which generates `mod_arith` dialect
// operations, this class directly generates operations within the `field`
// dialect. Arithmetic is provided by the FieldDialectArithmetic CRTP base.
class FieldCodeGen : public FieldDialectArithmetic<FieldCodeGen> {
public:
  FieldCodeGen() = default;
  explicit FieldCodeGen(Value value) : value(value) {}
  ~FieldCodeGen() = default;

  operator Value() const { return value; }
  Value getValue() const { return value; }

private:
  template <PointKind Kind>
  friend class PointCodeGenBase;
  template <typename, typename>
  friend class zk_dtypes::AffinePointOperation;
  template <typename, typename>
  friend class zk_dtypes::JacobianPointOperation;
  template <typename, typename>
  friend class zk_dtypes::PointXyzzOperation;

  Value IsZero() const;
  FieldCodeGen CreateConst(int64_t constant) const;

  Value value;
};

} // namespace mlir::prime_ir::elliptic_curve

// NOLINTNEXTLINE(whitespace/line_length)
#endif // PRIME_IR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_POINTOPERATIONS_FIELDCODEGEN_H_
