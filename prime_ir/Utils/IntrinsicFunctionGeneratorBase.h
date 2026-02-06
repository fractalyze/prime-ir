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

#ifndef PRIME_IR_UTILS_INTRINSICFUNCTIONGENERATORBASE_H_
#define PRIME_IR_UTILS_INTRINSICFUNCTIONGENERATORBASE_H_

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir::prime_ir {

/// Check if an operation is inside an intrinsic function (has `__prime_ir_`
/// prefix). Used to prevent infinite recursion when lowering intrinsic function
/// bodies.
inline bool isInsideIntrinsicFunction(Operation *op) {
  if (auto parentFunc = op->getParentOfType<func::FuncOp>())
    return parentFunc.getName().starts_with("__prime_ir_");
  return false;
}

/// CRTP base class for intrinsic function generators.
///
/// Provides common infrastructure for creating and managing intrinsic functions
/// that wrap high-level operations. Derived classes customize behavior by
/// providing type-specific implementations.
///
/// Example usage:
/// @code
/// class MyIntrinsicGenerator
///     : public IntrinsicFunctionGeneratorBase<MyIntrinsicGenerator> {
///   using Base = IntrinsicFunctionGeneratorBase<MyIntrinsicGenerator>;
/// public:
///   explicit MyIntrinsicGenerator(ModuleOp module) : Base(module) {}
///
///   func::FuncOp getOrCreateMyFunction(Type type) {
///     return getOrCreateFunction("my_func", {type}, {type},
///         [&](func::FuncOp func) {
///           OpBuilder builder(func.getContext());
///           auto args = setupFunctionBody(func, builder);
///           Value result = builder.create<MyOp>(func.getLoc(), args[0]);
///           emitReturn(builder, func.getLoc(), result);
///         });
///   }
/// };
/// @endcode
template <typename Derived>
class IntrinsicFunctionGeneratorBase {
public:
  explicit IntrinsicFunctionGeneratorBase(ModuleOp module)
      : module_(module), symbolTable_(module) {}

  ModuleOp getModule() const { return module_; }
  SymbolTable &getSymbolTable() { return symbolTable_; }

protected:
  /// Get or create a private function with the given signature.
  /// Uses a callback for body generation to allow type-specific behavior.
  template <typename BodyGeneratorFn>
  func::FuncOp getOrCreateFunction(StringRef funcName, TypeRange inputs,
                                   TypeRange outputs,
                                   BodyGeneratorFn &&bodyGenerator) {
    if (auto existing = symbolTable_.lookup<func::FuncOp>(funcName))
      return existing;

    auto funcType = FunctionType::get(module_.getContext(), inputs, outputs);

    OpBuilder builder(module_.getContext());
    builder.setInsertionPointToEnd(module_.getBody());
    auto func =
        builder.create<func::FuncOp>(module_.getLoc(), funcName, funcType);
    func.setPrivate();

    bodyGenerator(func);
    symbolTable_.insert(func);
    return func;
  }

  /// Emit a function call and return the single result.
  static Value emitCall(OpBuilder &builder, Location loc, func::FuncOp func,
                        ValueRange operands) {
    auto callOp = builder.create<func::CallOp>(loc, func, operands);
    return callOp.getResult(0);
  }

  /// Set up function body and return entry block arguments.
  /// Sets builder insertion point to start of entry block.
  static SmallVector<Value> setupFunctionBody(func::FuncOp func,
                                              OpBuilder &builder) {
    Block *entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    SmallVector<Value> args;
    for (auto arg : entryBlock->getArguments())
      args.push_back(arg);
    return args;
  }

  /// Emit return op with the given result.
  static void emitReturn(OpBuilder &builder, Location loc, Value result) {
    builder.create<func::ReturnOp>(loc, result);
  }

  /// Access derived class (CRTP pattern).
  Derived &derived() { return static_cast<Derived &>(*this); }
  const Derived &derived() const { return static_cast<const Derived &>(*this); }

private:
  ModuleOp module_;
  SymbolTable symbolTable_;
};

} // namespace mlir::prime_ir

#endif // PRIME_IR_UTILS_INTRINSICFUNCTIONGENERATORBASE_H_
