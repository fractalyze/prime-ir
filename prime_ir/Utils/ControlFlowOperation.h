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

#include <type_traits>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Value.h"
#include "prime_ir/Utils/BuilderContext.h"
#include "zk_dtypes/include/control_flow_operation_forward.h"

namespace zk_dtypes {

namespace detail {
template <typename T, typename = void>
struct HasToValues : std::false_type {};
template <typename T>
struct HasToValues<T, std::void_t<decltype(std::declval<T>().toValues())>>
    : std::true_type {};
} // namespace detail

template <>
class ControlFlowOperation<mlir::Value> {
public:
  ControlFlowOperation() = default;

  // If: dispatches to single-Value or State-based path based on return type.
  template <typename F1, typename F2>
  static auto If(mlir::Value condition, F1 &&then, F2 &&otherwise) {
    using R = std::invoke_result_t<F1>;
    if constexpr (detail::HasToValues<R>::value) {
      return IfState<R>(condition, std::forward<F1>(then),
                        std::forward<F2>(otherwise));
    } else {
      return IfBasic(condition, std::forward<F1>(then),
                     std::forward<F2>(otherwise));
    }
  }

  // Emits an scf.for loop: for iv = 0 to count step 1.
  // BodyFn: (mlir::Value iv, mlir::ValueRange args) -> SmallVector<mlir::Value>
  template <typename BodyFn>
  static llvm::SmallVector<mlir::Value>
  For(int64_t count, mlir::ValueRange initArgs, BodyFn &&body) {
    auto *b = mlir::prime_ir::BuilderContext::GetInstance().Top();
    auto c0 = b->create<mlir::arith::ConstantIndexOp>(0);
    auto cN = b->create<mlir::arith::ConstantIndexOp>(count);
    auto c1 = b->create<mlir::arith::ConstantIndexOp>(1);
    auto forOp = b->create<mlir::scf::ForOp>(
        c0, cN, c1, initArgs,
        [&body](mlir::OpBuilder &builder, mlir::Location loc, mlir::Value iv,
                mlir::ValueRange args) {
          mlir::ImplicitLocOpBuilder ib(loc, builder);
          mlir::prime_ir::ScopedBuilderContext scb(&ib);
          auto results = body(iv, args);
          ib.create<mlir::scf::YieldOp>(mlir::ValueRange(results));
        });
    return llvm::SmallVector<mlir::Value>(forOp.getResults());
  }

  // State-based For: when State has toValues()/fromValues().
  // BodyFn: (mlir::Value iv, State) -> State
  template <typename State, typename BodyFn,
            std::enable_if_t<detail::HasToValues<State>::value, int> = 0>
  static State For(int64_t count, State init, BodyFn &&body) {
    auto initVals = init.toValues();
    auto results =
        For(count, mlir::ValueRange(initVals),
            [&](mlir::Value iv,
                mlir::ValueRange args) -> llvm::SmallVector<mlir::Value> {
              State s = State::fromValues(args);
              State result = body(iv, std::move(s));
              auto vals = result.toValues();
              return {vals.begin(), vals.end()};
            });
    return State::fromValues(results);
  }

  // Select: component-wise arith.select.
  template <typename T>
  static T Select(mlir::Value condition, const T &a, const T &b) {
    auto *builder = mlir::prime_ir::BuilderContext::GetInstance().Top();
    return T(builder->create<mlir::arith::SelectOp>(
        condition, static_cast<mlir::Value>(a), static_cast<mlir::Value>(b)));
  }

  mlir::Value Equal(mlir::Value x, mlir::Value y);
  mlir::Value NotEqual(mlir::Value x, mlir::Value y);
  mlir::Value And(mlir::Value x, mlir::Value y);
  mlir::Value Or(mlir::Value x, mlir::Value y);
  mlir::Value Not(mlir::Value x);

  mlir::Value True();
  mlir::Value False();

private:
  // Basic If: lambdas return a single mlir::Value.
  template <typename F1, typename F2>
  static mlir::Value IfBasic(mlir::Value condition, F1 &&then, F2 &&otherwise) {
    auto *b = mlir::prime_ir::BuilderContext::GetInstance().Top();
    auto ifOp = b->create<mlir::scf::IfOp>(
        condition,
        [&then](mlir::OpBuilder &builder, mlir::Location loc) {
          mlir::ImplicitLocOpBuilder ib(loc, builder);
          mlir::prime_ir::ScopedBuilderContext scb(&ib);
          ib.create<mlir::scf::YieldOp>(mlir::ValueRange{then()});
        },
        [&otherwise](mlir::OpBuilder &builder, mlir::Location loc) {
          mlir::ImplicitLocOpBuilder ib(loc, builder);
          mlir::prime_ir::ScopedBuilderContext scb(&ib);
          ib.create<mlir::scf::YieldOp>(mlir::ValueRange{otherwise()});
        });
    return ifOp.getResult(0);
  }

  // State-based If: lambdas return a State with toValues()/fromValues().
  template <typename State, typename F1, typename F2>
  static State IfState(mlir::Value condition, F1 &&then, F2 &&otherwise) {
    auto *b = mlir::prime_ir::BuilderContext::GetInstance().Top();
    auto ifOp = b->create<mlir::scf::IfOp>(
        condition,
        [&then](mlir::OpBuilder &builder, mlir::Location loc) {
          mlir::ImplicitLocOpBuilder ib(loc, builder);
          mlir::prime_ir::ScopedBuilderContext scb(&ib);
          State result = then();
          auto vals = result.toValues();
          ib.create<mlir::scf::YieldOp>(mlir::ValueRange(vals));
        },
        [&otherwise](mlir::OpBuilder &builder, mlir::Location loc) {
          mlir::ImplicitLocOpBuilder ib(loc, builder);
          mlir::prime_ir::ScopedBuilderContext scb(&ib);
          State result = otherwise();
          auto vals = result.toValues();
          ib.create<mlir::scf::YieldOp>(mlir::ValueRange(vals));
        });
    return State::fromValues(ifOp.getResults());
  }
};

} // namespace zk_dtypes

#endif // PRIME_IR_UTILS_CONTROLFLOWOPERATION_H_
