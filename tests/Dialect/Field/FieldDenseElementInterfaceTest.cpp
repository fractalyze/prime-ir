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

#include <cstring>
#include <vector>

#include "gtest/gtest.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "prime_ir/Dialect/Field/IR/FieldDialect.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"

namespace mlir::prime_ir::field {
namespace {

class FieldDenseElementInterfaceTest : public testing::Test {
protected:
  static void SetUpTestSuite() { context.loadDialect<FieldDialect>(); }

  // PF<31:i32>: a 4-byte storage int.
  PrimeFieldType pf() {
    auto modAttr = IntegerAttr::get(IntegerType::get(&context, 32), 31);
    return PrimeFieldType::get(&context, modAttr, /*isMontgomery=*/false);
  }
  // EF<2x!PF, 6>: degreeOverPrime=2, storage = 8 bytes.
  ExtensionFieldType ef() {
    auto nonResidue = IntegerAttr::get(IntegerType::get(&context, 32), 6);
    return ExtensionFieldType::get(&context, /*degree=*/2, pf(), nonResidue);
  }

  static MLIRContext context;
};

MLIRContext FieldDenseElementInterfaceTest::context;

//===----------------------------------------------------------------------===//
// getDenseElementBitSize
//===----------------------------------------------------------------------===//

TEST_F(FieldDenseElementInterfaceTest, BitSizeIsStorageWidth) {
  // EF total storage = degreeOverPrime * prime storage.
  EXPECT_EQ(cast<DenseElementType>(ef()).getDenseElementBitSize(),
            ef().getDegreeOverPrime() * 32u);
}

//===----------------------------------------------------------------------===//
// Extension field: per-coefficient storage round-trip.
//===----------------------------------------------------------------------===//

TEST_F(FieldDenseElementInterfaceTest, ExtensionFieldConvertToCoeffs) {
  auto ty = cast<DenseElementType>(ef());
  // Coeffs [3, 5] of a degree-2 EF, each in i32 little-endian storage.
  std::vector<char> bytes = {3, 0, 0, 0, 5, 0, 0, 0};
  Attribute attr = ty.convertToAttribute(bytes);
  auto dense = dyn_cast<DenseIntElementsAttr>(attr);
  ASSERT_TRUE(dense) << "expected DenseIntElementsAttr cover";
  ASSERT_EQ(dense.getNumElements(), 2);
  auto values = llvm::to_vector(dense.getValues<APInt>());
  EXPECT_EQ(values[0].getZExtValue(), 3u);
  EXPECT_EQ(values[1].getZExtValue(), 5u);
}

TEST_F(FieldDenseElementInterfaceTest, ExtensionFieldRoundTrip) {
  auto ty = cast<DenseElementType>(ef());
  std::vector<char> original = {7, 0, 0, 0, 11, 0, 0, 0};
  Attribute attr = ty.convertToAttribute(original);
  ASSERT_TRUE(attr);

  llvm::SmallVector<char> back;
  ASSERT_TRUE(succeeded(ty.convertFromAttribute(attr, back)));
  ASSERT_EQ(back.size(), original.size());
  EXPECT_EQ(std::memcmp(back.data(), original.data(), original.size()), 0);
}

TEST_F(FieldDenseElementInterfaceTest, ExtensionFieldRejectsWrongSize) {
  auto ty = cast<DenseElementType>(ef());
  std::vector<char> tooShort(7, 0); // 7 bytes for an 8-byte element
  EXPECT_FALSE(ty.convertToAttribute(tooShort));
}

//===----------------------------------------------------------------------===//
// DenseElementsAttr can carry field-typed tensors end-to-end.
//===----------------------------------------------------------------------===//

TEST_F(FieldDenseElementInterfaceTest, DenseElementsAttrOverEfTensor) {
  auto tensorTy = RankedTensorType::get({2}, ef());
  // Two EF elements: [3, 5] and [7, 11].
  std::vector<char> bytes = {3, 0, 0, 0, 5, 0, 0, 0, 7, 0, 0, 0, 11, 0, 0, 0};
  auto dense = DenseElementsAttr::getFromRawBuffer(tensorTy, bytes);
  ASSERT_TRUE(dense);
  EXPECT_EQ(dense.getNumElements(), 2);
  EXPECT_EQ(dense.getRawData().size(), bytes.size());
}

TEST_F(FieldDenseElementInterfaceTest, ExtensionFieldConvertFromSplat) {
  auto ty = cast<DenseElementType>(ef());
  // A splat coeff attr (tensor<2xi32> all = 9): convertFromAttribute must
  // expand the single stored coeff across the full degree.
  auto coeffTy = RankedTensorType::get({2}, pf().getStorageType());
  auto splat = DenseIntElementsAttr::get(coeffTy, APInt(32, 9));
  ASSERT_TRUE(splat.isSplat());

  llvm::SmallVector<char> back;
  ASSERT_TRUE(succeeded(ty.convertFromAttribute(splat, back)));
  ASSERT_EQ(back.size(), 8u); // degreeOverPrime * primeBytes
  EXPECT_EQ(static_cast<unsigned char>(back[0]), 9u);
  EXPECT_EQ(static_cast<unsigned char>(back[4]), 9u);
}

} // namespace
} // namespace mlir::prime_ir::field
