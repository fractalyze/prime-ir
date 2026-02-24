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

Value FieldCodeGen::IsZero() const {
  ImplicitLocOpBuilder *b = BuilderContext::GetInstance().Top();
  Value zero = field::createFieldZero(value.getType(), *b);
  return b->create<field::CmpOp>(arith::CmpIPredicate::eq, value, zero);
}

FieldCodeGen FieldCodeGen::CreateConst(int64_t constant) const {
  ImplicitLocOpBuilder *b = BuilderContext::GetInstance().Top();
  return FieldCodeGen(
      field::createFieldConstant(value.getType(), *b, constant));
}

} // namespace mlir::prime_ir::elliptic_curve
