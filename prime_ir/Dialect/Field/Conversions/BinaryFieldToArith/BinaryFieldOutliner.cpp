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

#include "prime_ir/Dialect/Field/Conversions/BinaryFieldToArith/BinaryFieldOutliner.h"

#include "llvm/ADT/Twine.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "prime_ir/Dialect/Field/Conversions/BinaryFieldToArith/BinaryFieldCodeGen.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"

namespace mlir::prime_ir::field {

namespace {

/// The storage type a level-k tower element is computed on: i(2ᵏ).
IntegerType elementTypeForLevel(MLIRContext *ctx, unsigned towerLevel) {
  return IntegerType::get(ctx, 1u << towerLevel);
}

/// Keep LLVM from re-inlining the helper. A tower helper body is small
/// enough that LLVM's inliner folds it back into every caller, which
/// reinstates the exponential IR this outliner exists to remove
/// (measured in fractalyze/prime-ir#367). `FuncToLLVM` strips the `llvm.`
/// prefix and sets the matching `noinline` on the lowered `llvm.func`.
void markNoInline(func::FuncOp func) {
  func->setAttr("llvm.no_inline", UnitAttr::get(func.getContext()));
}

} // namespace

Value BinaryFieldOutliner::emitMulCall(ImplicitLocOpBuilder &b, Value lhs,
                                       Value rhs, unsigned towerLevel) {
  MLIRContext *ctx = getModule().getContext();
  IntegerType elemType = elementTypeForLevel(ctx, towerLevel);
  std::string funcName = ("__prime_ir_bf_mul_l" + Twine(towerLevel)).str();

  auto func = getOrCreateFunction(
      funcName, {elemType, elemType}, {elemType}, [&](func::FuncOp func) {
        markNoInline(func);
        OpBuilder builder(ctx);
        auto args = setupFunctionBody(func, builder);

        ImplicitLocOpBuilder bodyBuilder(func.getLoc(), builder);
        // The body expands one Karatsuba level; its three sub-products
        // recurse through mulTower, which emits calls to the level-(k−1)
        // helper whenever that level is still at or above the threshold.
        BinaryFieldCodeGen cg(BinaryFieldType::get(ctx, towerLevel), args[0],
                              bodyBuilder, this);
        emitReturn(builder, func.getLoc(),
                   cg.expandMulTower(args[0], args[1], towerLevel));
      });

  return emitCall(b, b.getLoc(), func, {lhs, rhs});
}

Value BinaryFieldOutliner::emitSquareCall(ImplicitLocOpBuilder &b, Value input,
                                          unsigned towerLevel) {
  MLIRContext *ctx = getModule().getContext();
  IntegerType elemType = elementTypeForLevel(ctx, towerLevel);
  std::string funcName = ("__prime_ir_bf_square_l" + Twine(towerLevel)).str();

  auto func = getOrCreateFunction(
      funcName, {elemType}, {elemType}, [&](func::FuncOp func) {
        markNoInline(func);
        OpBuilder builder(ctx);
        auto args = setupFunctionBody(func, builder);

        ImplicitLocOpBuilder bodyBuilder(func.getLoc(), builder);
        BinaryFieldCodeGen cg(BinaryFieldType::get(ctx, towerLevel), args[0],
                              bodyBuilder, this);
        emitReturn(builder, func.getLoc(),
                   cg.expandSquareTower(args[0], towerLevel));
      });

  return emitCall(b, b.getLoc(), func, {input});
}

} // namespace mlir::prime_ir::field
