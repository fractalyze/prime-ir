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

// The invariant these cases hold: a `field.powui` builder call that names only
// the operands compiles, and means unrolled.
//
// It is easy to break. ODS gives a plain `OptionalAttr` a *required* trailing
// builder parameter, so declaring `unroll` that way silently turned every
// operand-only call into a compile error — for consumers only, since prime-ir
// has no C++ caller of its own outside this file, and no lit test can reach a
// builder signature. Declaring it `DefaultValuedOptionalAttr` is what puts the
// `= true` back on the parameter. The same trap waits for the next attribute
// added to any op in this pinned dialect.

#include "gtest/gtest.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "prime_ir/Dialect/Field/IR/FieldDialect.h"
#include "prime_ir/Dialect/Field/IR/FieldOps.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"

namespace mlir::prime_ir::field {
namespace {

class PowUIOpTest : public testing::Test {
protected:
  void SetUp() override {
    context.loadDialect<FieldDialect>();
    module = ModuleOp::create(UnknownLoc::get(&context));
  }

  Location loc() { return UnknownLoc::get(&context); }

  // A builder anchored in the module body, so created ops have somewhere to go.
  OpBuilder bodyBuilder() {
    OpBuilder b(&context);
    b.setInsertionPointToStart(module->getBody());
    return b;
  }

  // The op only needs operands of the right type, not defined values, so an
  // unrealized cast stands in for whatever would really produce them.
  Value makeValue(OpBuilder &b, Type type) {
    return UnrealizedConversionCastOp::create(b, loc(), TypeRange{type},
                                              ValueRange{})
        .getResult(0);
  }

  PrimeFieldType fieldType() {
    return PrimeFieldType::get(
        &context, IntegerAttr::get(IntegerType::get(&context, 32), 7),
        /*isMontgomery=*/false);
  }

  MLIRContext context;
  OwningOpRef<ModuleOp> module;
};

// The regression guard. This call is exactly the shape that broke in xla's
// Poseidon S-box emitter and `mhlo.pow` lowering: operands, no attribute.
TEST_F(PowUIOpTest, OperandOnlyBuilderDefaultsToUnrolled) {
  OpBuilder b = bodyBuilder();
  Value base = makeValue(b, fieldType());
  Value exp = makeValue(b, b.getI32Type());

  auto op = PowUIOp::create(b, loc(), base, exp);

  EXPECT_TRUE(op.getUnroll());
  EXPECT_EQ(op.getOutput().getType(), base.getType());
}

// The same call through the result-typed overload, which is the one a
// conversion pattern with an explicit target type reaches for.
TEST_F(PowUIOpTest, ResultTypedOperandOnlyBuilderDefaultsToUnrolled) {
  OpBuilder b = bodyBuilder();
  Value base = makeValue(b, fieldType());
  Value exp = makeValue(b, b.getI32Type());

  auto op = PowUIOp::create(b, loc(), fieldType(), base, exp);

  EXPECT_TRUE(op.getUnroll());
}

// ImplicitLocOpBuilder is what the in-tree lowerings use, so it has to drop
// the attribute too.
TEST_F(PowUIOpTest, ImplicitLocBuilderDefaultsToUnrolled) {
  OpBuilder outer = bodyBuilder();
  ImplicitLocOpBuilder b(loc(), outer);
  Value base = makeValue(outer, fieldType());
  Value exp = makeValue(outer, outer.getI32Type());

  auto op = PowUIOp::create(b, base, exp);

  EXPECT_TRUE(op.getUnroll());
}

// `false` is the only value a caller ever has to spell out, so it must survive
// the trip through the defaulted parameter.
TEST_F(PowUIOpTest, ExplicitFalseIsKept) {
  OpBuilder b = bodyBuilder();
  Value base = makeValue(b, fieldType());
  Value exp = makeValue(b, b.getI32Type());

  auto op = PowUIOp::create(b, loc(), base, exp, /*unroll=*/false);

  EXPECT_FALSE(op.getUnroll());
}

// An op whose attribute is genuinely absent — every `field.powui` in a .mlir
// file written before the attribute existed — still reads as unrolled. This is
// the accessor's default fallback, distinct from what the builder materializes.
TEST_F(PowUIOpTest, AbsentAttributeReadsAsUnrolled) {
  OpBuilder b = bodyBuilder();
  Value base = makeValue(b, fieldType());
  Value exp = makeValue(b, b.getI32Type());

  auto op = PowUIOp::create(b, loc(), base, exp, /*unroll=*/false);
  // The attribute is inherent, so it lives in the op's properties rather than
  // the discardable dictionary — `removeAttr` would not touch it.
  op.removeUnrollAttr();

  ASSERT_FALSE(op.getUnrollAttr());
  EXPECT_TRUE(op.getUnroll());
}

} // namespace
} // namespace mlir::prime_ir::field
