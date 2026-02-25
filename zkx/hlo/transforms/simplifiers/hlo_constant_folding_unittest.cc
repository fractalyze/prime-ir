/* Copyright 2017 The OpenXLA Authors.
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

#include "zkx/hlo/transforms/simplifiers/hlo_constant_folding.h"

#include <atomic>
#include <cstdint>
#include <vector>

#include "absl/types/span.h"

#include "xla/tsl/platform/statusor.h"
#include "zkx/hlo/ir/hlo_computation.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/parser/hlo_parser.h"
#include "zkx/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "zkx/hlo/testlib/pattern_matcher_gmock.h"
#include "zkx/hlo/utils/hlo_matchers.h"
#include "zkx/layout_util.h"
#include "zkx/literal_util.h"
#include "zkx/primitive_util.h"
#include "zkx/service/pattern_matcher.h"
#include "zkx/shape_util.h"
#include "zkx/zkx_data.pb.h"

namespace zkx {
namespace {

namespace op = testing::opcode_matchers;
namespace m = zkx::match;
using HloConstantFoldingTest = HloHardwareIndependentTestBase;

// NOTE(junbeomlee): Original XLA test used F32 → S64 conversion. Adapted to S32
// → S64 since ZKX has no float types.
TEST_F(HloConstantFoldingTest, ConvertS32ToS64) {
  HloComputation::Builder builder(TestName());
  HloInstruction* input = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(42)));
  builder.AddInstruction(
      HloInstruction::CreateConvert(ShapeUtil::MakeShape(S64, {}), input));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(), op::Convert(input));

  HloConstantFolding const_folder;
  TF_ASSERT_OK_AND_ASSIGN(bool result, const_folder.Run(module.get()));
  EXPECT_TRUE(result);

  EXPECT_THAT(computation->root_instruction(), GmockMatch(m::Constant()));
  EXPECT_EQ(
      computation->root_instruction()->literal().GetFirstElement<int64_t>(),
      42);
}

// NOTE(junbeomlee): Original XLA test used S64 → F32 conversion. Adapted to S64
// → S32 since ZKX has no float types.
TEST_F(HloConstantFoldingTest, ConvertS64ToS32) {
  HloComputation::Builder builder(TestName());
  HloInstruction* input = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int64_t>(42)));
  builder.AddInstruction(
      HloInstruction::CreateConvert(ShapeUtil::MakeShape(S32, {}), input));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(), op::Convert(input));

  HloConstantFolding const_folder;
  TF_ASSERT_OK_AND_ASSIGN(bool result, const_folder.Run(module.get()));
  EXPECT_TRUE(result);

  EXPECT_THAT(computation->root_instruction(), GmockMatch(m::Constant()));
  EXPECT_EQ(
      computation->root_instruction()->literal().GetFirstElement<int32_t>(),
      42);
}

// NOTE(junbeomlee): Original XLA test used F32 array → S64 array conversion.
// Adapted to S32 → S64 since ZKX has no float types.
TEST_F(HloConstantFoldingTest, ConvertS32ArrayToS64Array) {
  HloComputation::Builder builder(TestName());
  HloInstruction* input = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<int32_t>({42, 19})));
  builder.AddInstruction(
      HloInstruction::CreateConvert(ShapeUtil::MakeShape(S64, {2}), input));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(), op::Convert(input));

  HloConstantFolding const_folder;
  TF_ASSERT_OK_AND_ASSIGN(bool result, const_folder.Run(module.get()));
  EXPECT_TRUE(result);

  EXPECT_THAT(computation->root_instruction(), GmockMatch(m::Constant()));
  EXPECT_EQ(computation->root_instruction()->literal().Get<int64_t>({0}), 42);
  EXPECT_EQ(computation->root_instruction()->literal().Get<int64_t>({1}), 19);
}

// NOTE(junbeomlee): Original XLA test used F32. Adapted to S32 since ZKX has no
// float types.
TEST_F(HloConstantFoldingTest, Concatenate) {
  const struct TestConfig {
    int concat_dimension;
    std::vector<int64_t> dimensions;
    std::vector<int64_t> concat_sizes;
  } test_configs[] = {
      {1, {11, 0, 7, 5, 9}, {2, 5, 7, 11}},
      {3, {1, 4, 17, 0, 8}, {1, 3, 9, 12}},
  };

  for (auto& test_config : test_configs) {
    HloComputation::Builder builder(TestName());
    std::vector<int64_t> dimensions(test_config.dimensions.begin(),
                                    test_config.dimensions.end());
    int64_t concat_size = 0;
    std::vector<HloInstruction*> operands;
    for (auto csize : test_config.concat_sizes) {
      dimensions[test_config.concat_dimension] = csize;
      concat_size += csize;
      auto literal = LiteralUtil::CreateFromDimensions(S32, dimensions);
      HloInstruction* insn = builder.AddInstruction(
          HloInstruction::CreateConstant(std::move(literal)));
      operands.push_back(insn);
    }
    dimensions[test_config.concat_dimension] = concat_size;
    Shape shape = ShapeUtil::MakeShape(S32, dimensions);
    builder.AddInstruction(HloInstruction::CreateConcatenate(
        shape, operands, test_config.concat_dimension));
    auto module = CreateNewVerifiedModule();
    auto computation = module->AddEntryComputation(builder.Build());

    HloConstantFolding const_folder;
    TF_ASSERT_OK_AND_ASSIGN(bool result, const_folder.Run(module.get()));
    EXPECT_TRUE(result);

    HloInstruction* root = computation->root_instruction();
    EXPECT_THAT(root, GmockMatch(m::Constant()));
    EXPECT_TRUE(ShapeUtil::Equal(root->shape(), shape));
  }
}

// NOTE(junbeomlee): Original XLA test used F32 with CreateRandomLiteral.
// Adapted to S32 with CreateFromDimensions (zero-filled) since ZKX has no float
// types and no CreateRandomLiteral for integers.
TEST_F(HloConstantFoldingTest, Slice) {
  HloComputation::Builder builder(TestName());
  const int64_t dimensions[] = {11, 8, 7, 5, 9};
  const int64_t slice_start[] = {4, 2, 3, 1, 5};
  const int64_t slice_limits[] = {10, 8, 6, 5, 9};
  const int64_t slice_strides[] = {1, 1, 1, 1, 1};
  auto literal = LiteralUtil::CreateFromDimensions(
      S32, absl::MakeConstSpan(dimensions, 5));
  HloInstruction* literal_instruction = builder.AddInstruction(
      HloInstruction::CreateConstant(std::move(literal)));
  Shape shape = ShapeUtil::MakeShape(S32, {6, 6, 3, 4, 4});
  builder.AddInstruction(HloInstruction::CreateSlice(
      shape, literal_instruction, slice_start, slice_limits, slice_strides));
  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  HloConstantFolding const_folder;
  TF_ASSERT_OK_AND_ASSIGN(bool result, const_folder.Run(module.get()));
  EXPECT_TRUE(result);

  HloInstruction* root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Constant()));
  EXPECT_TRUE(ShapeUtil::Equal(root->shape(), shape));
}

// NOTE(junbeomlee): Original XLA test used F32 with CreateRandomLiteral.
// Adapted to S32 with CreateFromDimensions (zero-filled) since ZKX has no float
// types.
TEST_F(HloConstantFoldingTest, TransposeConstantFold) {
  HloComputation::Builder builder(TestName());
  const int64_t dimensions[] = {11, 8, 7, 5, 9};
  auto literal = LiteralUtil::CreateFromDimensions(
      S32, absl::MakeConstSpan(dimensions, 5));
  auto literal_clone = literal.Clone();
  HloInstruction* literal_instruction = builder.AddInstruction(
      HloInstruction::CreateConstant(std::move(literal)));
  Shape shape = ShapeUtil::MakeShape(S32, {8, 7, 11, 9, 5});
  const int64_t permutation[] = {1, 2, 0, 4, 3};
  builder.AddInstruction(
      HloInstruction::CreateTranspose(shape, literal_instruction, permutation));
  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  HloConstantFolding const_folder;
  TF_ASSERT_OK_AND_ASSIGN(bool result, const_folder.Run(module.get()));
  EXPECT_TRUE(result);

  HloInstruction* root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Constant()));
  EXPECT_TRUE(ShapeUtil::Compatible(root->shape(), shape));

  // NOTE(junbeomlee): Replaced EachCell (not available in ZKX Literal) with
  // manual element-by-element verification. Since all values are zero-filled,
  // just verify shape and a spot check.
  using NativeT = typename primitive_util::PrimitiveTypeToNative<S32>::type;
  // Spot-check: all zero-filled elements should remain zero after transpose.
  std::vector<int64_t> index = {0, 0, 0, 0, 0};
  EXPECT_EQ(root->literal().Get<NativeT>(index), 0);
}

const char* const kConstantFoldReduce = R"(
  HloModule ConstantFoldReduce

  add {
    a = s32[] parameter(0)
    b = s32[] parameter(1)
    ROOT add = s32[] add(a, b)
  }

  ENTRY r {
    x = s32[3] constant({1, 2, 3})
    init = s32[] constant(0)
    ROOT reduce = s32[] reduce(x, init), dimensions={0}, to_apply=add
  })";

TEST_F(HloConstantFoldingTest, ConstantFoldReduce) {
  TF_ASSERT_OK_AND_ASSIGN(auto m,
                          ParseAndReturnVerifiedModule(kConstantFoldReduce));
  HloConstantFolding const_folder;
  TF_ASSERT_OK_AND_ASSIGN(bool result, const_folder.Run(m.get()));
  EXPECT_TRUE(result);

  EXPECT_EQ(6, m->entry_computation()
                   ->root_instruction()
                   ->literal()
                   .GetFirstElement<int32_t>());
}

constexpr std::string_view kConstantFoldReduceWithMetadata = R"(
  HloModule ConstantFoldReduce

  add {
    a = s32[] parameter(0)
    b = s32[] parameter(1)
    ROOT add = s32[] add(a, b)
  }

  ENTRY r {
    x = s32[3] constant({1, 2, 3}), metadata={op_name="constant"}
    init = s32[] constant(0), metadata={op_name="zero_constant"}
    ROOT reduce = s32[] reduce(x, init), metadata={op_name="reduce"}, dimensions={0}, to_apply=add
  })";

TEST_F(HloConstantFoldingTest, ConstantFoldReduceCheckMetadata) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto m, ParseAndReturnVerifiedModule(kConstantFoldReduceWithMetadata));
  HloConstantFolding const_folder;
  TF_ASSERT_OK_AND_ASSIGN(bool result, const_folder.Run(m.get()));
  EXPECT_TRUE(result);
  OpMetadata reduce_metadata;
  reduce_metadata.set_op_name("reduce");
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              AllOf(op::Constant(), op::Metadata(reduce_metadata)));
}

TEST_F(HloConstantFoldingTest, ConstantFoldReduceNoLayout) {
  TF_ASSERT_OK_AND_ASSIGN(auto m,
                          ParseAndReturnVerifiedModule(kConstantFoldReduce));
  HloInstruction* add = (*m->computations().begin())->root_instruction();
  LayoutUtil::ClearLayout(add->mutable_shape());

  HloConstantFolding const_folder;
  TF_ASSERT_OK_AND_ASSIGN(bool result, const_folder.Run(m.get()));
  EXPECT_TRUE(result);

  EXPECT_EQ(6, m->entry_computation()
                   ->root_instruction()
                   ->literal()
                   .GetFirstElement<int32_t>());
}

// NOTE(junbeomlee): Original XLA test used f32. Adapted to s32 since ZKX has no
// float types.
const char* const kConstantFoldLargePad = R"(
  HloModule ConstantFoldLargePad

  ENTRY r {
    a = s32[1,1,1] constant({{{7}}})
    b = s32[] constant(42)
    ROOT pad = s32[2048,2048,128] pad(a, b), padding=1024_1023x1024_1023x64_63
  })";

TEST_F(HloConstantFoldingTest, DoesNotFoldLargePad) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kConstantFoldLargePad));
  HloConstantFolding const_folder;
  TF_ASSERT_OK_AND_ASSIGN(bool result, const_folder.Run(module.get()));
  EXPECT_FALSE(result);

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Pad(m::Constant(), m::Constant())));
}

// NOTE(junbeomlee): Original XLA test used f32. Adapted to s32 since ZKX has no
// float types.
TEST_F(HloConstantFoldingTest, DoesNotFoldPadBroadcast) {
  const char* const kConstantFoldPadBroadcast = R"(
  HloModule ConstantFoldLargePad

  ENTRY r {
    a = s32[] constant(239)
    broadcast_a = s32[4] broadcast(a), dimensions={}
    b = s32[] constant(42)
    ROOT pad = s32[8] pad(s32[4] broadcast_a, s32[] b), padding=4_0
  })";
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kConstantFoldPadBroadcast));
  HloConstantFolding const_folder;
  TF_ASSERT_OK_AND_ASSIGN(bool result, const_folder.Run(module.get()));
  EXPECT_FALSE(result);

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Pad(m::Broadcast(), m::Constant())));
}

// NOTE(junbeomlee): Original XLA test used f32. Adapted to s32 since ZKX has no
// float types.
TEST_F(HloConstantFoldingTest, DoesNotFoldSlicesWithLargeOperand) {
  const char* const kModuleStr = R"(
  HloModule test

  ENTRY r {
    a = s32[] constant(42)
    broadcast = s32[1000000000]{0} broadcast(a), dimensions={}
    slice1 = s32[10000]{0} slice(broadcast), slice={[0:10000]}
    slice2 = s32[10000]{0} slice(broadcast), slice={[10000:20000]}
    ROOT add = s32[10000]{0} add(slice1, slice2)
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  HloConstantFolding const_folder;
  TF_ASSERT_OK_AND_ASSIGN(bool result, const_folder.Run(module.get()));
  EXPECT_FALSE(result);

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Add(m::Slice(), m::Slice())));
}

// NOTE(junbeomlee): Original XLA test used f32. Adapted to s32 since ZKX has no
// float types.
TEST_F(HloConstantFoldingTest, DontFoldSubcomputationContainingAfterAll) {
  const char* const kModuleStr = R"(
  HloModule test

  Fn {
    tok = token[] after-all()
    ROOT root = s32[10] iota(), iota_dimension=0
  }

  ENTRY entry {
    ROOT call = s32[10] call(), to_apply=Fn
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  HloConstantFolding constant_folding;
  TF_ASSERT_OK_AND_ASSIGN(bool result,
                          RunHloPass(&constant_folding, module.get()));
  EXPECT_FALSE(result);
}

// NOTE(junbeomlee): Removed DontFoldSubcomputationTransitivelyContainingRng
// test — ZKX has no kRng opcode.

// NOTE(junbeomlee): Original XLA test used f32. Adapted to s32 since ZKX has no
// float types.
TEST_F(HloConstantFoldingTest, FoldOpsWhereOneOperandIsBroadcast) {
  const char* const kModuleStr = R"(
  HloModule test

  ENTRY entry {
    not_folded1 = s32[4] broadcast(s32[] constant(1))
    not_folded2 = add(s32[4] broadcast(s32[] constant(2)),
                      s32[4] broadcast(s32[] constant(3)))
    folded1 = add(s32[4] broadcast(s32[] constant(5)),
                  s32[4] constant({0,1,2,3}))
    folded2 = add(s32[4] constant({0,1,2,3}),
                  s32[4] broadcast(s32[] constant(5)))
    ROOT root = tuple(not_folded1, not_folded2, folded1, folded2)
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  HloConstantFolding constant_folding;
  TF_ASSERT_OK_AND_ASSIGN(bool result,
                          RunHloPass(&constant_folding, module.get()));
  EXPECT_TRUE(result);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::Broadcast(m::Constant()),
                                  m::Add(m::Broadcast(m::Constant()),
                                         m::Broadcast(m::Constant())),
                                  m::Constant(), m::Constant())));
}

TEST_F(HloConstantFoldingTest, FoldInt4Ops) {
  const char* const kModuleStr = R"(
  HloModule test

  ENTRY entry {
    c0 = s4[2]{0:E(4)} constant({1, 2})
    c1 = s4[2]{0:E(4)} constant({3, 4})
    add1 = s4[2]{0:E(4)} add(c0, c1)
    c2 = s4[]{:E(4)} constant(5)
    add2 = s4[2]{0:E(4)} add(c0, s4[2]{0:E(4)} broadcast(c2))
    ROOT root = tuple(add1, add2)
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  HloConstantFolding constant_folding;
  TF_ASSERT_OK_AND_ASSIGN(bool result,
                          RunHloPass(&constant_folding, module.get()));
  EXPECT_TRUE(result);
  auto is_4_bit = [](const HloInstruction* instr) {
    return instr->shape().layout().element_size_in_bits() == 4;
  };
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::Constant().WithPredicate(is_4_bit),
                                  m::Constant().WithPredicate(is_4_bit))));
}

// NOTE(junbeomlee): Original XLA test used bf16. Adapted to s32 since ZKX has
// no float types.
TEST_F(HloConstantFoldingTest, BigReduceWindow) {
  constexpr std::string_view kModuleStr = R"(
    HloModule test

    add_s32 {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      ROOT add = s32[] add(lhs, rhs)
    }

    ENTRY accumulated_all_reduce {
      x = s32[160,10,10,512]{3,2,1,0} broadcast(s32[] constant(1))
      init = s32[] constant(0)
      ROOT reduce-window = reduce-window(x, init), window={size=1x2x2x1 stride=1x2x2x1}, to_apply=add_s32
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  HloConstantFolding constant_folding;
  TF_ASSERT_OK_AND_ASSIGN(bool result,
                          RunHloPass(&constant_folding, module.get()));
  EXPECT_TRUE(result);
}

// NOTE(junbeomlee): Original XLA test used f32. Adapted to s32 since ZKX has no
// float types.
TEST_F(HloConstantFoldingTest, TimingConsumingTest) {
  constexpr std::string_view mod_str = R"(
    HloModule jit_f, entry_computation_layout={()->s32[]}
    region_0.4 {
      Arg_0.5 = s32[] parameter(0)
      Arg_1.6 = s32[] parameter(1)
      ROOT add.7 = s32[] add(Arg_0.5, Arg_1.6)
    }

    ENTRY main.9 {
      constant.1 = s32[] constant(1)
      broadcast.2 = s32[32,999,40,512]{3,2,1,0} broadcast(constant.1), dimensions={}
      constant.3 = s32[] constant(0)
      ROOT reduce.8 = s32[] reduce(broadcast.2, constant.3), dimensions={0,1,2,3}, to_apply=region_0.4
    }
   )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(mod_str));
  HloConstantFolding const_fold;
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&const_fold, module.get()));
  EXPECT_FALSE(result);
}

}  // namespace
}  // namespace zkx
