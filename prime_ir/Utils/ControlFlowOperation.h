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

#ifndef PRIME_IR_UTILS_CONTROLFLOWOPERATION_H_
#define PRIME_IR_UTILS_CONTROLFLOWOPERATION_H_

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Value.h"
#include "prime_ir/Utils/BuilderContext.h"
#include "zk_dtypes/include/control_flow_operation_forward.h"

namespace zk_dtypes {

template <>
class ControlFlowOperation<mlir::Value> {
public:
  ControlFlowOperation() = default;

  template <typename F1, typename F2>
  auto If(mlir::Value condition, F1 &&then, F2 &&otherwise) {
    mlir::ImplicitLocOpBuilder *b =
        mlir::prime_ir::BuilderContext::GetInstance().Top();
    return b
        ->create<mlir::scf::IfOp>(
            condition,
            [&then](mlir::OpBuilder &builder, mlir::Location loc) {
              mlir::ImplicitLocOpBuilder b(loc, builder);
              mlir::prime_ir::ScopedBuilderContext scb(&b);

              b.create<mlir::scf::YieldOp>(mlir::ValueRange{then()});
            },
            [&otherwise](mlir::OpBuilder &builder, mlir::Location loc) {
              mlir::ImplicitLocOpBuilder b(loc, builder);
              mlir::prime_ir::ScopedBuilderContext scb(&b);

              b.create<mlir::scf::YieldOp>(mlir::ValueRange{otherwise()});
            })
        .getResults();
  }

  mlir::Value Equal(mlir::Value x, mlir::Value y);
  mlir::Value NotEqual(mlir::Value x, mlir::Value y);
  mlir::Value And(mlir::Value x, mlir::Value y);
  mlir::Value Or(mlir::Value x, mlir::Value y);
  mlir::Value Not(mlir::Value x);

  mlir::Value True();
  mlir::Value False();
};

} // namespace zk_dtypes

#endif // PRIME_IR_UTILS_CONTROLFLOWOPERATION_H_
