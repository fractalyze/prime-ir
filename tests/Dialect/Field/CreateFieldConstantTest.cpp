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

// `createFieldConstant(Type, ImplicitLocOpBuilder &, uint64_t)` takes a
// CANONICAL field element, so every bit of the argument is significant.
//
// It used to forward through `createConstantAttr(int64_t)`, whose
// magnitude-then-negate path reads any value with bit 63 set as negative and
// materializes `p - (2^64 - v)` — for Goldilocks, `v - (2^32 - 1)`. Nothing
// caught it: the internal `assert(value.ult(modulus))` is satisfied by the
// magnitude, and small-modulus fields (BabyBear, KoalaBear, Mersenne31) can
// never set bit 63, so no pre-existing test could observe it.
//
// These cases pin the top half of a 64-bit field's value range for each field
// flavor the helper dispatches over.

#include "gtest/gtest.h"
#include "llvm/ADT/APInt.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "prime_ir/Dialect/Field/IR/FieldDialect.h"
#include "prime_ir/Dialect/Field/IR/FieldOps.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"

namespace mlir::prime_ir::field {
namespace {

// Goldilocks: p = 2^64 - 2^32 + 1. Roughly half its canonical range has bit 63
// set, which is what makes it the first field to expose this.
constexpr uint64_t kGoldilocksModulus = UINT64_C(18446744069414584321);

// Canonical Goldilocks values: four with bit 63 set (including both boundary
// cases p-1 and exactly 2^63), and two small ones that must be unaffected.
constexpr uint64_t kValues[] = {
    kGoldilocksModulus - 1,       // 0xFFFFFFFF00000000
    UINT64_C(0x8000000000000000), // exactly 2^63, the first bad value
    UINT64_C(0x9E3779B97F4A7C15),
    UINT64_C(0xABCDEF0123456789),
    UINT64_C(7),
    UINT64_C(11),
};

class CreateFieldConstantTest : public testing::Test {
protected:
  void SetUp() override {
    context.loadDialect<FieldDialect>();
    module = ModuleOp::create(UnknownLoc::get(&context));
  }

  // A builder anchored in the module body, so created ops have somewhere to go.
  ImplicitLocOpBuilder createBodyBuilder() {
    ImplicitLocOpBuilder b(UnknownLoc::get(&context), &context);
    b.setInsertionPointToStart(module->getBody());
    return b;
  }

  IntegerAttr getGoldilocksModulusAttr() {
    return IntegerAttr::get(IntegerType::get(&context, 64),
                            APInt(64, kGoldilocksModulus));
  }

  MLIRContext context;
  OwningOpRef<ModuleOp> module;
};

// Standard (non-Montgomery) form: the stored attribute IS the canonical value,
// so this asserts the exact bit pattern survives.
TEST_F(CreateFieldConstantTest, PrimeFieldStandardFormKeepsWideValues) {
  auto pfType = PrimeFieldType::get(&context, getGoldilocksModulusAttr(),
                                    /*isMontgomery=*/false);
  ImplicitLocOpBuilder b = createBodyBuilder();

  for (uint64_t v : kValues) {
    Value c = createFieldConstant(pfType, b, v);
    auto constOp = c.getDefiningOp<ConstantOp>();
    ASSERT_TRUE(constOp) << "value " << v;
    auto attr = dyn_cast<IntegerAttr>(constOp.getValue());
    ASSERT_TRUE(attr) << "value " << v;
    EXPECT_EQ(attr.getValue().getZExtValue(), v)
        << "wrong constant for " << v << " (delta "
        << (v - attr.getValue().getZExtValue()) << ")";
  }
}

// Montgomery form: the stored attribute is v·R, so compare after converting
// back to standard form rather than against the raw value.
TEST_F(CreateFieldConstantTest, PrimeFieldMontgomeryFormKeepsWideValues) {
  auto pfType = PrimeFieldType::get(&context, getGoldilocksModulusAttr(),
                                    /*isMontgomery=*/true);
  ImplicitLocOpBuilder b = createBodyBuilder();

  for (uint64_t v : kValues) {
    Value c = createFieldConstant(pfType, b, v);
    auto constOp = c.getDefiningOp<ConstantOp>();
    ASSERT_TRUE(constOp) << "value " << v;
    auto standard =
        dyn_cast<IntegerAttr>(maybeToStandard(pfType, constOp.getValue()));
    ASSERT_TRUE(standard) << "value " << v;
    EXPECT_EQ(standard.getValue().getZExtValue(), v)
        << "wrong constant for " << v;
  }
}

// Extension field: the scalar must land in coefficient 0 with the rest zero,
// and must not be mangled on the way in.
TEST_F(CreateFieldConstantTest, ExtensionFieldEmbedsWideScalarInCoeffZero) {
  auto pfType = PrimeFieldType::get(&context, getGoldilocksModulusAttr(),
                                    /*isMontgomery=*/false);
  auto nonResidue = IntegerAttr::get(IntegerType::get(&context, 64), 7);
  auto efType =
      ExtensionFieldType::get(&context, /*degree=*/3, pfType, nonResidue);
  ImplicitLocOpBuilder b = createBodyBuilder();

  for (uint64_t v : kValues) {
    Value c = createFieldConstant(efType, b, v);
    auto constOp = c.getDefiningOp<ConstantOp>();
    ASSERT_TRUE(constOp) << "value " << v;
    auto dense = dyn_cast<DenseIntElementsAttr>(constOp.getValue());
    ASSERT_TRUE(dense) << "value " << v;

    SmallVector<APInt> coeffs(dense.value_begin<APInt>(),
                              dense.value_end<APInt>());
    ASSERT_EQ(coeffs.size(), efType.getDegreeOverPrime());
    EXPECT_EQ(coeffs[0].getZExtValue(), v) << "wrong coeff 0 for " << v;
    for (size_t i = 1; i < coeffs.size(); ++i) {
      EXPECT_TRUE(coeffs[i].isZero()) << "coeff " << i << " nonzero for " << v;
    }
  }
}

} // namespace
} // namespace mlir::prime_ir::field
