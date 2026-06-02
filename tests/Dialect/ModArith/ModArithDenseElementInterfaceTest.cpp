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
#include "prime_ir/Dialect/ModArith/IR/ModArithDialect.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithTypes.h"

namespace mlir::prime_ir::mod_arith {
namespace {

class ModArithDenseElementInterfaceTest : public testing::Test {
protected:
  static void SetUpTestSuite() { context.loadDialect<ModArithDialect>(); }

  // !mod_arith.int<65537 : i32>: a 4-byte storage int.
  ModArithType modArith() {
    auto modAttr = IntegerAttr::get(IntegerType::get(&context, 32), 65537);
    return ModArithType::get(&context, modAttr);
  }

  static MLIRContext context;
};

MLIRContext ModArithDenseElementInterfaceTest::context;

TEST_F(ModArithDenseElementInterfaceTest, BitSizeIsStorageWidth) {
  EXPECT_EQ(cast<DenseElementType>(modArith()).getDenseElementBitSize(), 32u);
}

TEST_F(ModArithDenseElementInterfaceTest, ScalarRoundTrip) {
  auto ty = cast<DenseElementType>(modArith());
  std::vector<char> bytes = {7, 0, 1, 0}; // 0x00010007 in i32 LE
  Attribute attr = ty.convertToAttribute(bytes);
  auto intAttr = dyn_cast<IntegerAttr>(attr);
  ASSERT_TRUE(intAttr);
  EXPECT_EQ(intAttr.getValue().getZExtValue(), 0x00010007u);

  llvm::SmallVector<char> back;
  ASSERT_TRUE(succeeded(ty.convertFromAttribute(attr, back)));
  ASSERT_EQ(back.size(), bytes.size());
  EXPECT_EQ(std::memcmp(back.data(), bytes.data(), bytes.size()), 0);
}

TEST_F(ModArithDenseElementInterfaceTest, DenseElementsAttrOverTensor) {
  auto tensorTy = RankedTensorType::get({2}, modArith());
  std::vector<char> bytes = {1, 0, 0, 0, 2, 0, 0, 0};
  auto dense = DenseElementsAttr::getFromRawBuffer(tensorTy, bytes);
  ASSERT_TRUE(dense);
  EXPECT_EQ(dense.getNumElements(), 2);
  EXPECT_EQ(dense.getRawData().size(), bytes.size());
}

TEST_F(ModArithDenseElementInterfaceTest, RejectsWrongSize) {
  auto ty = cast<DenseElementType>(modArith());
  std::vector<char> tooShort(3, 0);
  EXPECT_FALSE(ty.convertToAttribute(tooShort));
}

} // namespace
} // namespace mlir::prime_ir::mod_arith
