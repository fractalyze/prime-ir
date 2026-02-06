/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.
Copyright 2026 The ZKX Authors.

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

#include "zkx/service/gpu/model/affine_map_evaluator.h"

#include "gtest/gtest.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/MLIRContext.h"

#include "zkx/tests/hlo_test_base.h"

namespace zkx::gpu {
namespace {

using ::mlir::AffineExpr;
using ::mlir::AffineMap;
using ::mlir::bindDims;
using ::mlir::bindSymbols;

class AffineMapEvaluator : public HloTestBase {
 public:
  mlir::MLIRContext mlir_context_;
};

TEST_F(AffineMapEvaluator, EvaluateMap) {
  AffineExpr d0, d1, s0, s1;
  bindDims(&mlir_context_, d0, d1);
  bindSymbols(&mlir_context_, s0, s1);

  auto affine_map =
      AffineMap::get(2, 2, {d0 + d1.floorDiv(8), s0 + s1 % 16}, &mlir_context_);

  auto res = EvaluateAffineMap(affine_map, /*dim_values=*/{1, 2},
                               /*symbol_values=*/{3, 4});
  ASSERT_EQ(res.size(), 2);
  EXPECT_EQ(res[0], 1);
  EXPECT_EQ(res[1], 7);
}

}  // namespace
}  // namespace zkx::gpu
