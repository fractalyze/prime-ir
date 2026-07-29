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

#include "prime_ir/Utils/FunctionOutlinerBase.h"

#include "gtest/gtest.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

namespace mlir::prime_ir {
namespace {

class TestOutliner : public FunctionOutlinerBase<TestOutliner> {
  using Base = FunctionOutlinerBase<TestOutliner>;

public:
  explicit TestOutliner(ModuleOp module) : Base(module) {}

  func::FuncOp getOrCreate(StringRef name, Type in, Type out) {
    return getOrCreateFunction(name, {in}, {out}, [](func::FuncOp func) {
      OpBuilder builder(func.getContext());
      auto args = Base::setupFunctionBody(func, builder);
      Base::emitReturn(builder, func.getLoc(), args[0]);
    });
  }
};

class FunctionOutlinerBaseTest : public testing::Test {
protected:
  void SetUp() override {
    context.loadDialect<func::FuncDialect, LLVM::LLVMDialect>();
    module = ModuleOp::create(UnknownLoc::get(&context));
  }

  MLIRContext context;
  OwningOpRef<ModuleOp> module;
};

TEST_F(FunctionOutlinerBaseTest, SameNameSameSignatureDedups) {
  TestOutliner outliner(*module);
  Type i64 = IntegerType::get(&context, 64);

  func::FuncOp first = outliner.getOrCreate("helper", i64, i64);
  func::FuncOp second = outliner.getOrCreate("helper", i64, i64);

  EXPECT_EQ(first, second);
}

#if !defined(NDEBUG) && defined(GTEST_HAS_DEATH_TEST)
TEST_F(FunctionOutlinerBaseTest, SameNameDifferentSignatureAsserts) {
  TestOutliner outliner(*module);
  Type i64 = IntegerType::get(&context, 64);
  Type i32 = IntegerType::get(&context, 32);

  outliner.getOrCreate("helper", i64, i64);
  EXPECT_DEATH(outliner.getOrCreate("helper", i32, i32),
               "reused with a different signature");
}
#endif

} // namespace
} // namespace mlir::prime_ir
