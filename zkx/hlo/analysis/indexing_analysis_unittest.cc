/* Copyright 2023 The OpenXLA Authors.
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

#include "zkx/hlo/analysis/indexing_analysis.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "zkx/hlo/analysis/indexing_map_serialization.h"
#include "zkx/hlo/analysis/indexing_test_utils.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/utils/hlo_traversal.h"

namespace zkx::gpu {
namespace {

using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::ExplainMatchResult;
using ::testing::IsEmpty;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

MATCHER_P2(MatchInstrIndexing, operand_id, indexing_map_matchers, "") {
  return ExplainMatchResult(Eq(operand_id), arg.operand_id, result_listener) &&
         ExplainMatchResult(indexing_map_matchers, arg.indexing_maps,
                            result_listener);
}

using IndexingAnalysisTest = IndexingTestBase;

TEST_F(IndexingAnalysisTest, FuseProducerConsumerOutputToInputIndexing) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = s32[1000, 1000] parameter(0)
      transpose_p0 = s32[1000, 1000]{0, 1} transpose(p0), dimensions={1, 0}
      ROOT a0 = s32[1000, 1000] add(p0, transpose_p0)
    }
  )");
  const HloInstruction* parameter = root->operand(0);
  const HloInstruction* transpose = root->operand(1);

  auto root_indexing = GetOutputToInputIndexing(root);
  auto grouped_by_key = GroupIndexingMapsByProducers(root_indexing, root);

  EXPECT_THAT(
      grouped_by_key,
      UnorderedElementsAre(Pair(parameter, ElementsAre(MatchIndexingMap(R"(
                    (d0, d1) -> (d0, d1),
                    domain:
                    d0 in [0, 999],
                    d1 in [0, 999]
                  )"))),
                           Pair(transpose, ElementsAre(MatchIndexingMap(R"(
                    (d0, d1) -> (d0, d1),
                    domain:
                    d0 in [0, 999],
                    d1 in [0, 999]
                  )")))));
}

TEST_F(IndexingAnalysisTest, ComputeGroupedOutputToInputIndexing) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = s32[1000, 1000] parameter(0)
      transpose_p0 = s32[1000, 1000]{0, 1} transpose(p0), dimensions={1, 0}
      ROOT a0 = s32[1000, 1000] add(p0, transpose_p0)
    }
  )");
  const HloInstruction* parameter = root->operand(0);
  const HloInstruction* transpose = root->operand(1);

  auto fusion_adaptor = HloFusionAdaptor::ForProducerConsumer(transpose, root);

  auto grouped_indexing = ComputeGroupedOutputToInputIndexing(
      *fusion_adaptor, fusion_adaptor->GetRoots()[0], &mlir_context_);
  EXPECT_THAT(grouped_indexing,
              UnorderedElementsAre(
                  Pair(root, ElementsAre(MatchIndexingMap(R"(
                    (d0, d1) -> (d0, d1),
                    domain:
                    d0 in [0, 999],
                    d1 in [0, 999]
                  )"))),
                  Pair(transpose, ElementsAre(MatchIndexingMap(R"(
                    (d0, d1) -> (d0, d1),
                    domain:
                    d0 in [0, 999],
                    d1 in [0, 999]
                  )"))),
                  Pair(parameter, UnorderedElementsAre(MatchIndexingMap(R"(
                        (d0, d1) -> (d0, d1),
                        domain:
                        d0 in [0, 999],
                        d1 in [0, 999]
                      )"),
                                                       MatchIndexingMap(R"(
                        (d0, d1) -> (d1, d0),
                        domain:
                        d0 in [0, 999],
                        d1 in [0, 999]
                      )")))));
}

TEST_F(IndexingAnalysisTest,
       ComputeGroupedOutputToInputIndexing_VariadicReduce) {
  auto root = ParseAndGetRoot(R"(
    HloModule m

    add {
      param_0 = s32[] parameter(0)
      param_1 = s32[] parameter(1)
      param_2 = s32[] parameter(2)
      param_3 = s32[] parameter(3)
      add.0 = s32[] add(param_0, param_2)
      add.1 = s32[] add(param_1, param_3)
      ROOT t = (s32[], s32[]) tuple(add.0, add.1)
    }

    ENTRY entry_computation {
      param_0.3 = s32[32,40]{1,0} parameter(0)
      param_1.3 = s32[32,40]{1,0} parameter(1)
      param_2.2 = s32[] parameter(2)
      constant = s32[] constant(0)
      ROOT reduce = (s32[32]{0}, s32[32]{0})
        reduce(param_0.3, param_1.3, param_2.2, constant),
        dimensions={1}, to_apply=add
    }
  )");
  auto fusion_adaptor = HloFusionAdaptor::ForInstruction(root);

  auto grouped_indexing = ComputeGroupedOutputToInputIndexing(
      *fusion_adaptor, fusion_adaptor->GetRoots()[0], &mlir_context_);

  EXPECT_THAT(grouped_indexing,
              UnorderedElementsAre(
                  Pair(root, ElementsAre(MatchIndexingMap(R"(
                    (d0) -> (d0),
                    domain:
                    d0 in [0, 31]
                  )"))),
                  Pair(root->operand(0), ElementsAre(MatchIndexingMap(R"(
                    (d0)[s0] -> (d0, s0),
                    domain:
                    d0 in [0, 31],
                    s0 in [0, 39]
                  )"))),
                  Pair(root->operand(1), ElementsAre(MatchIndexingMap(R"(
                    (d0)[s0] -> (d0, s0),
                    domain:
                    d0 in [0, 31],
                    s0 in [0, 39]
                  )"))),
                  Pair(root->operand(2), ElementsAre(MatchIndexingMap(R"(
                    (d0) -> (),
                    domain:
                    d0 in [0, 31]
                  )"))),
                  Pair(root->operand(3), ElementsAre(MatchIndexingMap(R"(
                    (d0) -> (),
                    domain:
                    d0 in [0, 31]
                  )")))));
}

TEST_F(IndexingAnalysisTest, ComputeGroupedOutputToInputIndexing_SingleOp) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = s32[1000, 1000] parameter(0)
      p1 = s32[1000, 1000] parameter(1)
      neg0 = s32[1000, 1000] negate(p1)
      ROOT a0 = s32[1000, 1000] add(p0, neg0)
    }
  )");
  HloComputation* entry_computation = root->parent();
  const HloInstruction* negate =
      entry_computation->GetInstructionWithName("neg0");
  const HloInstruction* parameter =
      entry_computation->GetInstructionWithName("p1");

  auto fusion_adaptor = HloFusionAdaptor::ForInstruction(negate);
  HloInstructionAdaptor parameter_adaptor =
      fusion_adaptor->GetRoots()[0].GetOperand(0);
  auto grouped_indexing = ComputeGroupedOutputToInputIndexing(
      *fusion_adaptor, parameter_adaptor, &mlir_context_);
  EXPECT_THAT(grouped_indexing, UnorderedElementsAre(Pair(
                                    parameter, ElementsAre(MatchIndexingMap(R"(
                                                     (d0, d1) -> (d0, d1),
                                                     domain:
                                                     d0 in [0, 999],
                                                     d1 in [0, 999]
                                                   )")))));
}

TEST_F(IndexingAnalysisTest,
       ComputeGroupedOutputToInputIndexing_StartNotAtRoot) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    max {
      p0 = s32[] parameter(0)
      p1 = s32[] parameter(1)
      ROOT max = s32[] maximum(p0, p1)
    }
    f {
      p0 = s32[15, 20] parameter(0)
      p0_init = s32[] parameter(1)
      p0_bcast = s32[15, 32, 20, 64] broadcast(p0), dimensions={0, 2}

      ROOT reduce_2 = s32[15, 64] reduce(p0_bcast, p0_init),
        dimensions={1, 2}, to_apply=max
    }
    ENTRY e {
      p0 = s32[15, 20] parameter(0)
      p0_init = s32[] constant(100)
      ROOT fusion = s32[15, 64] fusion(p0, p0_init), kind=kLoop, calls=f
    }
  )");
  auto fusion_adaptor = HloFusionAdaptor::ForInstruction(root);
  auto root_adaptor = fusion_adaptor->GetRoots()[0];

  auto bcast = root_adaptor.GetOperand(0);
  auto parameter_0 = bcast.GetOperand(0);

  auto grouped_indexing = ComputeGroupedOutputToInputIndexing(
      *fusion_adaptor, bcast, &mlir_context_);
  EXPECT_THAT(
      grouped_indexing,
      UnorderedElementsAre(
          Pair(&bcast.instruction(), ElementsAre(MatchIndexingMap(R"(
            (d0, d1, d2, d3) -> (d0, d1, d2, d3),
            domain:
            d0 in [0, 14],
            d1 in [0, 31],
            d2 in [0, 19],
            d3 in [0, 63]
          )"))),
          Pair(&parameter_0.instruction(), ElementsAre(MatchIndexingMap(R"(
            (d0, d1, d2, d3) -> (d0, d2),
            domain:
            d0 in [0, 14],
            d1 in [0, 31],
            d2 in [0, 19],
            d3 in [0, 63]
          )")))));
}

TEST_F(IndexingAnalysisTest, PhysicalLayoutTestOutputPermutation) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = s32[10, 20, 30] parameter(0)
      ROOT neg0 = s32[10, 20, 30]{1, 0, 2} negate(p0)
    }
  )");
  auto input_indexing = GetOutputToInputIndexing(root, /*output_id=*/0,
                                                 /*use_physical_layout=*/true);
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                            operand id = 0
                              (d0, d1, d2) -> (d1, d2, d0),
                              domain:
                              d0 in [0, 29],
                              d1 in [0, 9],
                              d2 in [0, 19]
                          )"));

  auto output_indexing = GetInputToOutputIndexing(root, /*input_id=*/0,
                                                  /*use_physical_layout=*/true);
  EXPECT_THAT(output_indexing.ToString(), MatchIndexingString(R"(
                            operand id = 0
                              (d0, d1, d2) -> (d2, d0, d1),
                              domain:
                              d0 in [0, 9],
                              d1 in [0, 19],
                              d2 in [0, 29]
                          )"));
}

TEST_F(IndexingAnalysisTest, CopyNothing) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = s32[0, 0]{0,1} parameter(0)
      ROOT copy0 = s32[0, 0]{1,0} copy(p0)
    }
  )");
  auto input_indexing = GetOutputToInputIndexing(root, /*output_id=*/0);
  input_indexing.Simplify();
  EXPECT_THAT(input_indexing.ToString(),
              MatchIndexingString("operand id = 0 KNOWN EMPTY"));

  auto output_indexing = GetInputToOutputIndexing(root, /*input_id=*/0);
  output_indexing.Simplify();
  EXPECT_THAT(output_indexing.ToString(),
              MatchIndexingString("operand id = 0 KNOWN EMPTY"));
}

TEST_F(IndexingAnalysisTest, ReshapeNothing) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = s32[1,0,0] parameter(0)
      ROOT reshape = s32[0] reshape(p0)
    }
  )");
  auto input_indexing = GetOutputToInputIndexing(root, /*output_id=*/0);
  input_indexing.Simplify();
  EXPECT_THAT(input_indexing.ToString(),
              MatchIndexingString("operand id = 0 KNOWN EMPTY"));

  auto output_indexing = GetInputToOutputIndexing(root, /*input_id=*/0);
  output_indexing.Simplify();
  EXPECT_THAT(output_indexing.ToString(),
              MatchIndexingString("operand id = 0 KNOWN EMPTY"));
  // Even though the indexing is known empty, the rank of the map should still
  // be 1.
  EXPECT_EQ(
      output_indexing.indexing_maps[0].begin()->GetAffineMap().getNumResults(),
      1);
}

TEST_F(IndexingAnalysisTest, PhysicalLayoutTestInputPermutation) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = s32[10, 20, 30]{1, 0, 2} parameter(0)
      ROOT neg0 = s32[10, 20, 30] negate(p0)
    }
  )");
  auto input_indexing = GetOutputToInputIndexing(root, /*output_id=*/0,
                                                 /*use_physical_layout=*/true);
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                            operand id = 0
                              (d0, d1, d2) -> (d2, d0, d1),
                              domain:
                              d0 in [0, 9],
                              d1 in [0, 19],
                              d2 in [0, 29]
                          )"));

  auto output_indexing = GetInputToOutputIndexing(root, /*input_id=*/0,
                                                  /*use_physical_layout=*/true);
  EXPECT_THAT(output_indexing.ToString(), MatchIndexingString(R"(
                            operand id = 0
                              (d0, d1, d2) -> (d1, d2, d0),
                              domain:
                              d0 in [0, 29],
                              d1 in [0, 9],
                              d2 in [0, 19]
                          )"));
}

TEST_F(IndexingAnalysisTest, PhysicalLayoutTestInputAndOutputPermutation) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = s32[10, 20, 30]{1, 0, 2} parameter(0)
      ROOT neg0 = s32[10, 20, 30]{1, 0, 2} negate(p0)
    }
  )");
  auto input_indexing = GetOutputToInputIndexing(root, /*output_id=*/0,
                                                 /*use_physical_layout=*/true);
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                            operand id = 0
                              (d0, d1, d2) -> (d0, d1, d2),
                              domain:
                              d0 in [0, 29],
                              d1 in [0, 9],
                              d2 in [0, 19]
                          )"));

  auto output_indexing = GetInputToOutputIndexing(root, /*input_id=*/0,
                                                  /*use_physical_layout=*/true);
  EXPECT_THAT(output_indexing.ToString(), MatchIndexingString(R"(
                            operand id = 0
                              (d0, d1, d2) -> (d0, d1, d2),
                              domain:
                              d0 in [0, 29],
                              d1 in [0, 9],
                              d2 in [0, 19]
                          )"));
}

TEST_F(IndexingAnalysisTest, ElementwiseOp) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = s32[10, 20] parameter(0)
      p1 = s32[10, 20] parameter(1)
      ROOT add0 = s32[10, 20] add(p0, p1)
    }
  )");
  auto input_indexing = GetOutputToInputIndexing(root);
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                            operand id = 0
                              (d0, d1) -> (d0, d1),
                              domain:
                              d0 in [0, 9],
                              d1 in [0, 19]
                            operand id = 1
                              (d0, d1) -> (d0, d1),
                              domain:
                              d0 in [0, 9],
                              d1 in [0, 19]
                          )"));

  auto output_indexing_0 = GetInputToOutputIndexing(root, /*input_id=*/0);
  EXPECT_THAT(output_indexing_0.ToString(), MatchIndexingString(R"(
                            operand id = 0
                              (d0, d1) -> (d0, d1),
                              domain:
                              d0 in [0, 9],
                              d1 in [0, 19]
                          )"));

  auto output_indexing_1 = GetInputToOutputIndexing(root, /*input_id=*/1);
  EXPECT_THAT(output_indexing_1.ToString(), MatchIndexingString(R"(
                            operand id = 0
                              (d0, d1) -> (d0, d1),
                              domain:
                              d0 in [0, 9],
                              d1 in [0, 19]
                          )"));
}

TEST_F(IndexingAnalysisTest, Map) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    mapper {
      a = s32[] parameter(0)
      b = s32[] parameter(1)
      ROOT add = s32[] add(a, b)
    }
    ENTRY e {
      p0 = s32[10, 20] parameter(0)
      p1 = s32[10, 20] parameter(1)
      ROOT add0 = s32[10, 20] map(%p0, %p1), dimensions={}, to_apply=mapper
    }
  )");
  auto input_indexing = GetOutputToInputIndexing(root);
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                            operand id = 0
                              (d0, d1) -> (d0, d1),
                              domain:
                              d0 in [0, 9],
                              d1 in [0, 19]
                            operand id = 1
                              (d0, d1) -> (d0, d1),
                              domain:
                              d0 in [0, 9],
                              d1 in [0, 19]
                          )"));

  auto output_indexing_0 = GetInputToOutputIndexing(root, /*input_id=*/0);
  EXPECT_THAT(output_indexing_0.ToString(), MatchIndexingString(R"(
                            operand id = 0
                              (d0, d1) -> (d0, d1),
                              domain:
                              d0 in [0, 9],
                              d1 in [0, 19]
                          )"));

  auto output_indexing_1 = GetInputToOutputIndexing(root, /*input_id=*/1);
  EXPECT_THAT(output_indexing_1.ToString(), MatchIndexingString(R"(
                            operand id = 0
                              (d0, d1) -> (d0, d1),
                              domain:
                              d0 in [0, 9],
                              d1 in [0, 19]
                          )"));
}

TEST_F(IndexingAnalysisTest, BitcastIsReshape) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = s32[4, 32] parameter(0)
      ROOT bitcast = s32[4, 8, 4] bitcast(p0)
    }
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                            operand id = 0
                              (d0, d1, d2) -> (d0, d1 * 4 + d2),
                              domain:
                              d0 in [0, 3],
                              d1 in [0, 7],
                              d2 in [0, 3]
                          )"));
}

TEST_F(IndexingAnalysisTest, BitcastIsTranspose) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = s32[3, 12288, 6, 128] parameter(0)
      ROOT bitcast = s32[3, 6, 128, 12288] {2, 1, 3, 0} bitcast(p0)
    }
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                            operand id = 0
                              (d0, d1, d2, d3) -> (d0, d3, d1, d2),
                              domain:
                              d0 in [0, 2],
                              d1 in [0, 5],
                              d2 in [0, 127],
                              d3 in [0, 12287]
                          )"));
}

TEST_F(IndexingAnalysisTest, BitcastIsTransposeReshapeTranspose) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = s32[16, 17, 3] parameter(0)
      ROOT bitcast = s32[51, 16] {0, 1} bitcast(p0)
    }
  )");
  auto input_indexing = GetOutputToInputIndexing(root);
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                            operand id = 0
                              (d0, d1) -> (d1, d0 floordiv 3, d0 mod 3),
                              domain:
                              d0 in [0, 50],
                              d1 in [0, 15]
                          )"));
  auto output_indexing = GetInputToOutputIndexing(root);
  EXPECT_THAT(output_indexing.ToString(), MatchIndexingString(R"(
                            operand id = 0
                              (d0, d1, d2) -> (d1 * 3 + d2, d0),
                              domain:
                              d0 in [0, 15],
                              d1 in [0, 16],
                              d2 in [0, 2]
                          )"));
}

TEST_F(IndexingAnalysisTest, BroadcastOp) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = s32[20] parameter(0)
      ROOT bc0 = s32[10, 20, 30] broadcast(p0), dimensions={1}
    }
  )");
  auto input_indexing = GetOutputToInputIndexing(root);
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                            operand id = 0
                              (d0, d1, d2) -> (d1),
                              domain:
                              d0 in [0, 9],
                              d1 in [0, 19],
                              d2 in [0, 29]
                          )"));
  auto output_indexing = GetInputToOutputIndexing(root);
  EXPECT_THAT(output_indexing.ToString(), MatchIndexingString(R"(
                            operand id = 0
                              (d0)[s0, s1] -> (s0, d0, s1),
                              domain:
                              d0 in [0, 19],
                              s0 in [0, 9],
                              s1 in [0, 29]
                          )"));
}

TEST_F(IndexingAnalysisTest, ConstantOp) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      ROOT c1 = s16[17, 22] constant(1)
    }
  )");
  auto input_indexing = GetOutputToInputIndexing(root);
  EXPECT_THAT(input_indexing.ToString(), IsEmpty());
}

TEST_F(IndexingAnalysisTest, ConcatenateOp) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = s32[2, 5, 7] parameter(0)
      p1 = s32[2, 11, 7] parameter(1)
      p2 = s32[2, 17, 7] parameter(2)
      ROOT concat = s32[2, 33, 7] concatenate(
        s32[2, 5, 7] p0, s32[2, 11, 7] p1, s32[2, 17, 7] p2), dimensions={1}
    }
  )");
  auto input_indexing = GetOutputToInputIndexing(root);
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                            operand id = 0
                              (d0, d1, d2) -> (d0, d1, d2),
                              domain:
                              d0 in [0, 1],
                              d1 in [0, 4],
                              d2 in [0, 6]
                            operand id = 1
                              (d0, d1, d2) -> (d0, d1 - 5, d2),
                              domain:
                              d0 in [0, 1],
                              d1 in [5, 15],
                              d2 in [0, 6]
                            operand id = 2
                              (d0, d1, d2) -> (d0, d1 - 16, d2),
                              domain:
                              d0 in [0, 1],
                              d1 in [16, 32],
                              d2 in [0, 6]
                          )"));

  auto output_indexing_0 = GetInputToOutputIndexing(root, /*input_id=*/0);
  EXPECT_THAT(output_indexing_0.ToString(), MatchIndexingString(R"(
                            operand id = 0
                              (d0, d1, d2) -> (d0, d1, d2),
                              domain:
                              d0 in [0, 1],
                              d1 in [0, 4],
                              d2 in [0, 6]
                          )"));

  auto output_indexing_1 = GetInputToOutputIndexing(root, /*input_id=*/1);
  EXPECT_THAT(output_indexing_1.ToString(), MatchIndexingString(R"(
                            operand id = 0
                              (d0, d1, d2) -> (d0, d1 + 5, d2),
                              domain:
                              d0 in [0, 1],
                              d1 in [0, 10],
                              d2 in [0, 6]
                          )"));

  auto output_indexing_2 = GetInputToOutputIndexing(root, /*input_id=*/2);
  EXPECT_THAT(output_indexing_2.ToString(), MatchIndexingString(R"(
                            operand id = 0
                              (d0, d1, d2) -> (d0, d1 + 16, d2),
                              domain:
                              d0 in [0, 1],
                              d1 in [0, 16],
                              d2 in [0, 6]
                          )"));
}

TEST_F(IndexingAnalysisTest, DynamicSliceOp) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      %src = s32[2,2,258] parameter(0)
      %of1 = s32[] parameter(1)
      %of2 = s32[] parameter(2)
      %of3 = s32[] parameter(3)
      ROOT %ds = s32[1,2,32] dynamic-slice(s32[2,2,258] %src,
        s32[] %of1, s32[] %of2, s32[] %of3),
        dynamic_slice_sizes={1, 2, 32}
    }
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                operand id = 0
                  (d0, d1, d2){rt0, rt1, rt2} -> (d0 + rt0, d1 + rt1, d2 + rt2),
                  domain:
                  d0 in [0, 0],
                  d1 in [0, 1],
                  d2 in [0, 31],
                  rt0 in [0, 1],
                  rt1 in [0, 0],
                  rt2 in [0, 226]
                operand id = 1
                  (d0, d1, d2)  -> (),
                  domain:
                  d0 in [0, 0],
                  d1 in [0, 1],
                  d2 in [0, 31]
                operand id = 2
                  (d0, d1, d2)  -> (),
                  domain:
                  d0 in [0, 0],
                  d1 in [0, 1],
                  d2 in [0, 31]
                operand id = 3
                  (d0, d1, d2)  -> (),
                  domain:
                  d0 in [0, 0],
                  d1 in [0, 1],
                  d2 in [0, 31]
              )"));
}

TEST_F(IndexingAnalysisTest, DynamicUpdateSliceOp) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      %src = s32[20,30] parameter(0)
      %upd = s32[5,10] parameter(1)
      %of1 = s32[] parameter(2)
      %of2 = s32[] parameter(3)
      ROOT %dus = s32[20,30] dynamic-update-slice(
          s32[20,30] %src, s32[5,10] %upd, s32[] %of1, s32[] %of2)
    }
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                operand id = 0
                  (d0, d1) -> (d0, d1),
                  domain:
                  d0 in [0, 19],
                  d1 in [0, 29]
                operand id = 1
                  (d0, d1){rt0, rt1}  -> (d0 - rt0, d1 - rt1),
                  domain:
                  d0 in [0, 19],
                  d1 in [0, 29],
                  rt0 in [0, 15],
                  rt1 in [0, 20]
                operand id = 2
                  (d0, d1)  -> (),
                  domain:
                  d0 in [0, 19],
                  d1 in [0, 29]
                operand id = 3
                  (d0, d1)  -> (),
                  domain:
                  d0 in [0, 19],
                  d1 in [0, 29]
              )"));
}

TEST_F(IndexingAnalysisTest, FusionOpWithSingleBinaryOp) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    f {
      p0 = s32[100] parameter(0)
      p1 = s32[100] parameter(1)
      ROOT a0 = s32[100] add(p0, p1)
    }
    ENTRY e {
      p0 = s32[100] parameter(0)
      p1 = s32[100] parameter(1)
      ROOT fusion = s32[100] fusion(p0, p1), kind=kLoop, calls=f
    }
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                            operand id = 0
                              (d0) -> (d0),
                              domain:
                              d0 in [0, 99]
                            operand id = 1
                              (d0) -> (d0),
                              domain:
                              d0 in [0, 99]
                          )"));
}

TEST_F(IndexingAnalysisTest, FusionOpTensorPlusTransposedTensor) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    f {
      p0 = s32[1000, 1000] parameter(0)
      transpose_p0 = s32[1000, 1000]{0, 1} transpose(p0), dimensions={1, 0}
      ROOT a0 = s32[1000, 1000] add(p0, transpose_p0)
    }
    ENTRY e {
      p0 = s32[1000,1000] parameter(0)
      ROOT fusion = s32[1000,1000] fusion(p0), kind=kLoop, calls=f
    }
  )"));
  EXPECT_THAT(input_indexing.indexing_maps,
              ElementsAre(UnorderedElementsAre(MatchIndexingMap(R"(
                            (d0, d1) -> (d0, d1),
                            domain:
                            d0 in [0, 999],
                            d1 in [0, 999]
                          )"),
                                               MatchIndexingMap(R"(
                            (d0, d1) -> (d1, d0),
                            domain:
                            d0 in [0, 999],
                            d1 in [0, 999]
                          )"))));
}

TEST_F(IndexingAnalysisTest, FusionExponentialDuplication) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule test_module

    fused_computation {
      p0 = s32[4] parameter(0)
      p1 = s32[4] parameter(1)
      add0 = s32[4] add(p0, p1)
      slice1.0 = s32[3] slice(add0), slice={[0:3]}
      slice1.1 = s32[3] slice(add0), slice={[1:4]}
      add1 = s32[3]{0} add(slice1.0, slice1.1)
      slice2.0 = s32[2] slice(add1), slice={[0:2]}
      slice2.1 = s32[2] slice(add1), slice={[1:3]}
      ROOT add2 = s32[2] add(slice2.0, slice2.1)
    }

    ENTRY entry_computation {
      p0 = s32[4] parameter(0)
      p1 = s32[4] parameter(1)
      ROOT fusion = s32[2] fusion(p0, p1), kind=kLoop,
      calls=fused_computation
    })"));
  EXPECT_THAT(input_indexing.indexing_maps,
              ElementsAre(UnorderedElementsAre(MatchIndexingMap(R"(
                            (d0) -> (d0 + 1),
                            domain:
                            d0 in [0, 1]
                          )"),
                                               MatchIndexingMap(R"(
                            (d0) -> (d0),
                            domain:
                            d0 in [0, 1]
                          )"),
                                               MatchIndexingMap(R"(
                            (d0) -> (d0 + 2),
                            domain:
                            d0 in [0, 1]
                          )")),
                          UnorderedElementsAre(MatchIndexingMap(R"(
                            (d0) -> (d0 + 2),
                            domain:
                            d0 in [0, 1]
                          )"),
                                               MatchIndexingMap(R"(
                            (d0) -> (d0 + 1),
                            domain:
                            d0 in [0, 1]
                          )"),
                                               MatchIndexingMap(R"(
                            (d0) -> (d0),
                            domain:
                            d0 in [0, 1]
                          )"))));
}

// TODO(chokobole): Enable this test. Dependency: mhlo::GatherOp
TEST_F(IndexingAnalysisTest, DISABLED_GatherOp) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    ENTRY main {
      operand = s32[33,76,70] parameter(0)
      indices = s32[1806,2] parameter(1)
      ROOT r = s32[1806,7,8,4] gather(operand, indices), offset_dims={1,2,3},
                                 collapsed_slice_dims={}, start_index_map={0,1},
                                 index_vector_dim=1, slice_sizes={7,8,4}
    }
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
              operand id = 0
                (d0, d1, d2, d3){rt0, rt1} -> (d1 + rt0, d2 + rt1, d3),
                domain:
                d0 in [0, 1805],
                d1 in [0, 6],
                d2 in [0, 7],
                d3 in [0, 3],
                rt0 in [0, 26],
                rt1 in [0, 68]
              operand id = 1
                (d0, d1, d2, d3)[s0] -> (d0, s0),
                domain:
                d0 in [0, 1805],
                d1 in [0, 6],
                d2 in [0, 7],
                d3 in [0, 3],
                s0 in [0, 1]
              )"));
}

TEST_F(IndexingAnalysisTest, FusionOpWithReduceOfReduce) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    max {
      p0 = s32[] parameter(0)
      p1 = s32[] parameter(1)
      ROOT max = s32[] maximum(p0, p1)
    }
    f {
      p0 = s32[150, 20, 10, 50] parameter(0)
      p0_init = s32[] parameter(1)
      reduce_1 = s32[20, 10] reduce(p0, p0_init),
        dimensions={0, 3}, to_apply=max
      ROOT reduce_2 = s32[10] reduce(reduce_1, p0_init),
        dimensions={0}, to_apply=max
    }
    ENTRY e {
      p0 = s32[150, 20, 10, 50] parameter(0)
      p0_init = s32[] constant(-100)
      ROOT fusion = s32[10] fusion(p0, p0_init), kind=kLoop, calls=f
    }
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0)[s0, s1, s2] -> (s0, s2, d0, s1),
                            domain:
                            d0 in [0, 9],
                            s0 in [0, 149],
                            s1 in [0, 49],
                            s2 in [0, 19]
                          operand id = 1
                            (d0) -> (),
                            domain:
                            d0 in [0, 9]
                          )"));
}

TEST_F(IndexingAnalysisTest, FusionOpWithReduceOfBroadcast) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    max {
      p0 = s32[] parameter(0)
      p1 = s32[] parameter(1)
      ROOT max = s32[] maximum(p0, p1)
    }
    f {
      p0 = s32[15, 20] parameter(0)
      p0_init = s32[] parameter(1)
      p0_bcast = s32[15, 32, 20, 64] broadcast(p0), dimensions={0, 2}

      ROOT reduce_2 = s32[15, 64] reduce(p0_bcast, p0_init),
        dimensions={1, 2}, to_apply=max
    }
    ENTRY e {
      p0 = s32[15, 20] parameter(0)
      p0_init = s32[] constant(-100)
      ROOT fusion = s32[15, 64] fusion(p0, p0_init), kind=kLoop, calls=f
    }
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0, d1)[s0] -> (d0, s0),
                            domain:
                            d0 in [0, 14],
                            d1 in [0, 63],
                            s0 in [0, 19]
                          operand id = 1
                            (d0, d1) -> (),
                            domain:
                            d0 in [0, 14],
                            d1 in [0, 63]
                          )"));
}

TEST_F(IndexingAnalysisTest, FusionOpWithTransposeOfTranspose) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    f {
      p0 = s32[20, 10, 50] parameter(0)

      lhs_transpose_1 = s32[10, 20, 50]
             transpose(p0), dimensions={1, 0, 2}
      lhs_n = s32[10, 20, 50] negate(lhs_transpose_1)
      lhs_transpose_2 = s32[10, 50, 20]
             transpose(lhs_n), dimensions={0, 2, 1}

      rhs_transpose_1 = s32[50, 10, 20]
             transpose(p0), dimensions={2, 1, 0}
      rhs_n = s32[50, 10, 20] negate(rhs_transpose_1)
      rhs_transpose_2 = s32[10, 50, 20]
             transpose(rhs_n), dimensions={1, 0, 2}

      ROOT add = s32[10, 50, 20] add(lhs_transpose_2, rhs_transpose_2)
    }
    ENTRY e {
      p0 = s32[20, 10, 50] parameter(0)
      ROOT fusion = s32[10, 50, 20] fusion(p0), kind=kLoop, calls=f
    }
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0, d1, d2) -> (d2, d0, d1),
                            domain:
                            d0 in [0, 9],
                            d1 in [0, 49],
                            d2 in [0, 19]
                          )"));
}

TEST_F(IndexingAnalysisTest, FusionOpWithReducedSlice) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    max {
      p0 = s32[] parameter(0)
      p1 = s32[] parameter(1)
      ROOT max = s32[] maximum(p0, p1)
    }
    f {
      p0 = s32[150, 64, 1024] parameter(0)
      p0_init = s32[] parameter(1)
      p0_slice = s32[16, 32, 128] slice(s32[150, 64, 1024] p0),
                slice={[5:21:1], [0:64:2], [50:434:3]}
      ROOT reduce = s32[32] reduce(p0_slice, p0_init),
        dimensions={0, 2}, to_apply=max
    }
    ENTRY e {
      p0 = s32[150, 64, 1024] parameter(0)
      p0_init = s32[] constant(-100)
      ROOT fusion = s32[32] fusion(p0, p0_init), kind=kLoop, calls=f
    }
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0)[s0, s1] -> (s0 + 5, d0 * 2, s1 * 3 + 50),
                            domain:
                            d0 in [0, 31],
                            s0 in [0, 15],
                            s1 in [0, 127]
                          operand id = 1
                            (d0) -> (),
                            domain:
                            d0 in [0, 31]
                          )"));
}

TEST_F(IndexingAnalysisTest, FusionOpWithReshape_CollapseOfExpand) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    f {
      p0 = s32[128] parameter(0)
      expand = s32[8, 16] reshape(p0)
      ROOT collapse = s32[128] reshape(expand)
    }
    ENTRY e {
      p0 = s32[128] parameter(0)
      ROOT fusion = s32[128] fusion(p0), kind=kLoop, calls=f
    }
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0) -> (d0),
                            domain:
                            d0 in [0, 127]
                          )"));
}

TEST_F(IndexingAnalysisTest, FusionOpWithReshape_ExpandOfCollapse) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    f {
      p0 = s32[8, 16] parameter(0)
      collapse = s32[128] reshape(p0)
      ROOT expand = s32[8, 16] reshape(collapse)
    }
    ENTRY e {
      p0 = s32[8, 16] parameter(0)
      ROOT fusion = s32[8, 16] fusion(p0), kind=kLoop, calls=f
    }
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0, d1) -> (d0, d1),
                            domain:
                            d0 in [0, 7],
                            d1 in [0, 15]
                          )"));
}

TEST_F(IndexingAnalysisTest, FusionOpWithReshape_ChainedGenericReshapes) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    f {
      p0 = s32[10, 10, 10] parameter(0)
      reshape1 = s32[50, 20] reshape(p0)
      ROOT reshape2 = s32[10, 10, 10] reshape(reshape1)
    }
    ENTRY e {
      p0 = s32[10, 10, 10] parameter(0)
      ROOT fusion = s32[10, 10, 10] fusion(p0), kind=kLoop, calls=f
    }
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0, d1, d2) -> (d0, d1, d2),
                            domain:
                            d0 in [0, 9],
                            d1 in [0, 9],
                            d2 in [0, 9]
                          )"));
}

TEST_F(IndexingAnalysisTest, FusionOpWithSliceOfSlice) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    f {
      p0 = s32[150, 64, 1024] parameter(0)
      p0_slice_1 = s32[16, 32, 128] slice(s32[150, 64, 1024] p0),
                slice={[5:21:1], [0:64:2], [50:434:3]}
      ROOT p0_slice_2 = s32[7, 9, 24] slice(s32[16, 32, 128] p0_slice_1),
                slice={[3:16:2], [4:30:3], [5:100:4]}
    }
    ENTRY e {
      p0 = s32[150, 64, 1024] parameter(0)
      ROOT fusion = s32[7, 9, 24] fusion(p0), kind=kLoop, calls=f
    }
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
              operand id = 0
                (d0, d1, d2) -> (d0 * 2 + 8, d1 * 6 + 8, d2 * 12 + 65),
                domain:
                d0 in [0, 6],
                d1 in [0, 8],
                d2 in [0, 23]
              )"));
}

TEST_F(IndexingAnalysisTest, FusionOpWithDynSliceOfDynSlice) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    f {
      %src = s32[150, 64] parameter(0)
      %of11 = s32[] parameter(1)
      %of12 = s32[] parameter(2)
      %of21 = s32[] parameter(3)
      %of22 = s32[] parameter(4)

      %ds1 = s32[50, 32] dynamic-slice(s32[150, 64] %src,
        s32[] %of11, s32[] %of12), dynamic_slice_sizes={50, 32}

      ROOT %ds2 = s32[25, 16] dynamic-slice(s32[50, 32] %ds1,
        s32[] %of21, s32[] %of22), dynamic_slice_sizes={25, 16}
    }
    ENTRY e {
      %p0 = s32[150, 64] parameter(0)
      %p1 = s32[] parameter(1)
      %p2 = s32[] parameter(2)
      %p3 = s32[] parameter(3)
      %p4 = s32[] parameter(4)
      ROOT fusion = s32[25, 16] fusion(p0, p1, p2, p3, p4),
        kind=kLoop, calls=f
    }
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
              operand id = 0
                (d0, d1){rt0, rt1, rt2, rt3} -> (d0 + rt0 + rt2, d1 + rt1 + rt3),
                domain:
                d0 in [0, 24],
                d1 in [0, 15],
                rt0 in [0, 100],
                rt1 in [0, 32],
                rt2 in [0, 25],
                rt3 in [0, 16]
              operand id = 1
                  (d0, d1) -> (),
                  domain:
                  d0 in [0, 24],
                  d1 in [0, 15]
              operand id = 2
                  (d0, d1) -> (),
                  domain:
                  d0 in [0, 24],
                  d1 in [0, 15]
              operand id = 3
                  (d0, d1) -> (),
                  domain:
                  d0 in [0, 24],
                  d1 in [0, 15]
              operand id = 4
                  (d0, d1) -> (),
                  domain:
                  d0 in [0, 24],
                  d1 in [0, 15]
                )"));
}

TEST_F(IndexingAnalysisTest, FusionOpSliceOfAllConcatenateOpInputs) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    f {
      p0 = s32[2, 5, 7] parameter(0)
      p1 = s32[2, 11, 7] parameter(1)
      p2 = s32[2, 17, 7] parameter(2)
      concat = s32[2, 33, 7] concatenate(
        s32[2, 5, 7] p0, s32[2, 11, 7] p1, s32[2, 17, 7] p2), dimensions={1}
      ROOT slice = s32[2, 11, 7] slice(s32[2, 33, 7] concat),
        slice={[0:2:1], [0:33:3], [0:7:1]}
    }
    ENTRY e {
      p0 = s32[2, 5, 7] parameter(0)
      p1 = s32[2, 11, 7] parameter(1)
      p2 = s32[2, 17, 7] parameter(2)
      ROOT fusion = s32[2, 11, 7] fusion(p0, p1, p2), kind=kLoop, calls=f
    }
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0, d1, d2) -> (d0, d1 * 3, d2),
                            domain:
                            d0 in [0, 1],
                            d1 in [0, 1],
                            d2 in [0, 6]
                          operand id = 1
                            (d0, d1, d2) -> (d0, d1 * 3 - 5, d2),
                            domain:
                            d0 in [0, 1],
                            d1 in [2, 5],
                            d2 in [0, 6]
                          operand id = 2
                            (d0, d1, d2) -> (d0, d1 * 3 - 16, d2),
                            domain:
                            d0 in [0, 1],
                            d1 in [6, 10],
                            d2 in [0, 6]
                          )"));
}

TEST_F(IndexingAnalysisTest, FusionOpSliceOfOneOfConcatenateOpInputs) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    f {
      p0 = s32[2, 5, 7] parameter(0)
      p1 = s32[2, 11, 7] parameter(1)
      p2 = s32[2, 17, 7] parameter(2)
      concat = s32[2, 33, 7] concatenate(
        s32[2, 5, 7] p0, s32[2, 11, 7] p1, s32[2, 17, 7] p2), dimensions={1}
      ROOT slice = s32[2, 3, 7] slice(s32[2, 33, 7] concat),
        slice={[0:2:1], [0:5:2], [0:7:1]}
    }
    ENTRY e {
      p0 = s32[2, 5, 7] parameter(0)
      p1 = s32[2, 11, 7] parameter(1)
      p2 = s32[2, 17, 7] parameter(2)
      ROOT fusion = s32[2, 3, 7] fusion(p0, p1, p2), kind=kLoop, calls=f
    }
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0, d1, d2) -> (d0, d1 * 2, d2),
                            domain:
                            d0 in [0, 1],
                            d1 in [0, 2],
                            d2 in [0, 6]
                          operand id = 1
                            KNOWN EMPTY
                          operand id = 2
                            KNOWN EMPTY
                          )"));
}

TEST_F(IndexingAnalysisTest, FusionOpReshapeOfConcat) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    f {
      p0 = s32[2] parameter(0)
      p1 = s32[30] parameter(1)
      concat = s32[32] concatenate(s32[2] p0, s32[30] p1), dimensions={0}
      ROOT reshape = s32[4, 8] reshape(concat)
    }
    ENTRY e {
      p0 = s32[2] parameter(0)
      p1 = s32[30] parameter(1)
      ROOT fusion = s32[4, 8] fusion(p0, p1), kind=kLoop, calls=f
    }
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0, d1) -> (d0 * 8 + d1),
                            domain:
                            d0 in [0, 3],
                            d1 in [0, 7],
                            d0 * 8 + d1 in [0, 1]
                          operand id = 1
                            (d0, d1) -> (d0 * 8 + d1 - 2),
                            domain:
                            d0 in [0, 3],
                            d1 in [0, 7],
                            d0 * 8 + d1 in [2, 31]
                          )"));
}

TEST_F(IndexingAnalysisTest, IotaOp) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      ROOT iota = s32[5,5,111,42] iota(), iota_dimension=0
    }
  )");
  auto input_indexing = GetOutputToInputIndexing(root);
  EXPECT_THAT(input_indexing.indexing_maps, IsEmpty());
}

TEST_F(IndexingAnalysisTest, ReshapeOpCollapseShape) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = s32[4,8] parameter(0)
      ROOT reshape = s32[32] reshape(p0)
    }
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0) -> (d0 floordiv 8, d0 mod 8),
                            domain:
                            d0 in [0, 31]
                          )"));
}

TEST_F(IndexingAnalysisTest, ReshapeOpExpandShape) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = s32[32] parameter(0)
      ROOT reshape = s32[4, 8] reshape(p0)
    }
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0, d1) -> (d0 * 8 + d1),
                            domain:
                            d0 in [0, 3],
                            d1 in [0, 7]
                          )"));
}

TEST_F(IndexingAnalysisTest, ReshapeOpExpandAndCollapseShape) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = s32[4, 8, 12] parameter(0)
      ROOT reshape = s32[32, 3, 4] reshape(p0)
    }
  )");
  auto input_indexing = GetOutputToInputIndexing(root);
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
              operand id = 0
                (d0, d1, d2) -> (d0 floordiv 8, d0 mod 8, d1 * 4 + d2),
                domain:
                d0 in [0, 31],
                d1 in [0, 2],
                d2 in [0, 3]
              )"));

  auto output_indexing = GetInputToOutputIndexing(root);
  EXPECT_THAT(output_indexing.ToString(), MatchIndexingString(R"(
              operand id = 0
                (d0, d1, d2) -> (d0 * 8 + d1, d2 floordiv 4, d2 mod 4),
                domain:
                d0 in [0, 3],
                d1 in [0, 7],
                d2 in [0, 11]
              )"));
}

TEST_F(IndexingAnalysisTest, ReshapeOpExpandSubshapeOnly) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = s32[16, 8] parameter(0)
      ROOT reshape = s32[4, 4, 8] reshape(p0)
    }
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
              operand id = 0
                (d0, d1, d2) -> (d0 * 4 + d1, d2),
                domain:
                d0 in [0, 3],
                d1 in [0, 3],
                d2 in [0, 7]
              )"));
}

TEST_F(IndexingAnalysisTest, ReshapeOpGenericReshape2DTo3D) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = s32[4,8] parameter(0)
      ROOT reshape = s32[2, 4, 4] reshape(p0)
    }
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
              operand id = 0
                (d0, d1, d2) -> (d0 * 2 + d1 floordiv 2, (d1 mod 2) * 4 + d2),
                domain:
                d0 in [0, 1],
                d1 in [0, 3],
                d2 in [0, 3]
              )"));
}

TEST_F(IndexingAnalysisTest, ReshapeOpGenericReshape3DTo2D) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = s32[2, 4, 4] parameter(0)
      ROOT reshape = s32[4, 8] reshape(p0)
    }
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0, d1) -> (d0 floordiv 2,
                                        (d0 mod 2) * 2 + d1 floordiv 4,
                                        d1 mod 4),
                            domain:
                            d0 in [0, 3],
                            d1 in [0, 7]
                          )"));
}

TEST_F(IndexingAnalysisTest, PadOp) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = s32[4, 4] parameter(0)
      p1 = s32[] parameter(1)
      ROOT pad = s32[6, 8] pad(p0, p1), padding=1_1x4_0
    }
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                                  operand id = 0
                                    (d0, d1) -> (d0 - 1, d1 - 4),
                                    domain:
                                    d0 in [1, 4],
                                    d1 in [4, 7]
                                  operand id = 1
                                    (d0, d1) -> (),
                                    domain:
                                    d0 in [0, 5],
                                    d1 in [0, 7]
                                )"));
}

TEST_F(IndexingAnalysisTest, PadOpNoInterior) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = s32[2,8] parameter(0)
      p1 = s32[] parameter(1)
      ROOT pad = s32[10,8] pad(p0, p1), padding=1_7x0_0
    }
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                                  operand id = 0
                                    (d0, d1) -> (d0 - 1, d1),
                                    domain:
                                    d0 in [1, 2],
                                    d1 in [0, 7]
                                  operand id = 1
                                    (d0, d1) -> (),
                                    domain:
                                    d0 in [0, 9],
                                    d1 in [0, 7]
                                )"));
}

TEST_F(IndexingAnalysisTest, ReduceOp) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    max {
      p0 = s32[] parameter(0)
      p1 = s32[] parameter(1)
      ROOT max = s32[] maximum(p0, p1)
    }
    ENTRY e {
      p0 = s32[150, 20, 10, 50] parameter(0)
      p0_init = s32[] constant(-100)
      ROOT reduce = s32[150, 10] reduce(p0, p0_init),
        dimensions={3, 1}, to_apply=max
    }
  )");
  auto input_indexing = GetOutputToInputIndexing(root);
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0, d1)[s0, s1] -> (d0, s0, d1, s1),
                            domain:
                            d0 in [0, 149],
                            d1 in [0, 9],
                            s0 in [0, 19],
                            s1 in [0, 49]
                          operand id = 1
                            (d0, d1) -> (),
                            domain:
                            d0 in [0, 149],
                            d1 in [0, 9]
                          )"));

  auto output_indexing_0 = GetInputToOutputIndexing(root, 0);
  EXPECT_THAT(output_indexing_0.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0, d1, d2, d3) -> (d0, d2),
                            domain:
                            d0 in [0, 149],
                            d1 in [0, 19],
                            d2 in [0, 9],
                            d3 in [0, 49]
                          )"));
  auto output_indexing_1 = GetInputToOutputIndexing(root, 1);
  EXPECT_THAT(output_indexing_1.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            ()[s0, s1] -> (s0, s1),
                            domain:
                            s0 in [0, 149],
                            s1 in [0, 9]
                          )"));
}

TEST_F(IndexingAnalysisTest, VariadicReduceOp) {
  HloInstruction* root = ParseAndGetRoot(R"(
    HloModule m
    min {
      tmp_0 = s32[] parameter(0)
      tmp_1 = s32[] parameter(2)
      tmp_2 = s32[] parameter(1)
      tmp_3 = s32[] parameter(3)
      cmp = pred[] compare(tmp_0, tmp_1), direction=GE
      select1 = s32[] select(cmp, tmp_0, tmp_1)
      select2 = s32[] select(cmp, tmp_2, tmp_3)
      ROOT tmp_4 = (s32[], s32[]) tuple(select1, select2)
    }
    ENTRY e {
      p0 = s32[256,10] parameter(0)
      p0_init = s32[] constant(-100)
      p1 = s32[256,10] parameter(1)
      p1_init = s32[] constant(0)
      ROOT reduce = (s32[10], s32[10]) reduce(p0, p1, p0_init, p1_init),
        dimensions={0}, to_apply=min
    }
  )");

  auto output_indexing_0 = GetOutputToInputIndexing(root, /*output_id=*/0);
  EXPECT_THAT(output_indexing_0.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0)[s0] -> (s0, d0),
                            domain:
                            d0 in [0, 9],
                            s0 in [0, 255]
                          operand id = 1
                            (d0)[s0] -> (s0, d0),
                            domain:
                            d0 in [0, 9],
                            s0 in [0, 255]
                          operand id = 2
                            (d0) -> (),
                            domain:
                            d0 in [0, 9]
                          operand id = 3
                            (d0) -> (),
                            domain:
                            d0 in [0, 9]
                          )"));

  auto output_indexing_1 = GetOutputToInputIndexing(root, /*output_id=*/1);
  EXPECT_THAT(output_indexing_1.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0)[s0] -> (s0, d0),
                            domain:
                            d0 in [0, 9],
                            s0 in [0, 255]
                          operand id = 1
                            (d0)[s0] -> (s0, d0),
                            domain:
                            d0 in [0, 9],
                            s0 in [0, 255]
                          operand id = 2
                            (d0) -> (),
                            domain:
                            d0 in [0, 9]
                          operand id = 3
                            (d0) -> (),
                            domain:
                            d0 in [0, 9]
                          )"));

  constexpr std::string_view kInputToOutputIndexing = R"(
      (d0, d1) -> (d1),
      domain:
      d0 in [0, 255],
      d1 in [0, 9]
  )";
  auto input_indexing_0 = GetInputToOutputIndexing(root, /*input_id=*/0);
  EXPECT_THAT(
      input_indexing_0.indexing_maps,
      ElementsAre(ElementsAre(MatchIndexingMap(kInputToOutputIndexing)),
                  ElementsAre(MatchIndexingMap(kInputToOutputIndexing))));

  auto input_indexing_1 = GetInputToOutputIndexing(root, /*input_id=*/1);
  EXPECT_THAT(
      input_indexing_1.indexing_maps,
      ElementsAre(ElementsAre(MatchIndexingMap(kInputToOutputIndexing)),
                  ElementsAre(MatchIndexingMap(kInputToOutputIndexing))));

  constexpr std::string_view kInitToOutputIndexing = R"(
      ()[s0] -> (s0),
      domain:
      s0 in [0, 9]
  )";
  auto input_indexing_2 = GetInputToOutputIndexing(root, /*input_id=*/2);
  EXPECT_THAT(
      input_indexing_2.indexing_maps,
      ElementsAre(ElementsAre(MatchIndexingMap(kInitToOutputIndexing)),
                  ElementsAre(MatchIndexingMap(kInitToOutputIndexing))));
  auto input_indexing_3 = GetInputToOutputIndexing(root, /*input_id=*/2);
  EXPECT_THAT(
      input_indexing_3.indexing_maps,
      ElementsAre(ElementsAre(MatchIndexingMap(kInitToOutputIndexing)),
                  ElementsAre(MatchIndexingMap(kInitToOutputIndexing))));
}

TEST_F(IndexingAnalysisTest, ReduceWindowOp_NoPadding) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    max {
      p0 = s32[] parameter(0)
      p1 = s32[] parameter(1)
      ROOT max = s32[] maximum(p0, p1)
    }
    ENTRY e {
      c_inf = s32[] constant(-100)
      p0 = s32[1024, 514]parameter(0)
      ROOT reduce-window = s32[1024, 3] reduce-window(p0, c_inf),
        window={size=1x512 pad=0_0x0_0}, to_apply=max
    }
  )");
  auto input_indexing = GetOutputToInputIndexing(root);
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0, d1)[s0] -> (d0, d1 + s0),
                            domain:
                            d0 in [0, 1023],
                            d1 in [0, 2],
                            s0 in [0, 511]
                          operand id = 1
                            (d0, d1) -> (),
                            domain:
                            d0 in [0, 1023],
                            d1 in [0, 2]
                          )"));
}

TEST_F(IndexingAnalysisTest, ReduceWindowOp_PaddingAndWindowStride) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    max {
      p0 = s32[] parameter(0)
      p1 = s32[] parameter(1)
      ROOT max = s32[] maximum(p0, p1)
    }
    ENTRY e {
      c_inf = s32[] constant(-100)
      p0 = s32[13, 17] parameter(0)
      ROOT reduce-window = s32[7, 17] reduce-window(p0, c_inf),
       window={size=3x2 stride=2x1 pad=1_1x0_1}, to_apply=max
    }
  )");
  auto input_indexing = GetOutputToInputIndexing(root);
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0, d1)[s0, s1] -> (d0 * 2 + s0 - 1, d1 + s1),
                            domain:
                            d0 in [0, 6],
                            d1 in [0, 16],
                            s0 in [0, 2],
                            s1 in [0, 1],
                            d0 * 2 + s0 in [1, 13],
                            d1 + s1 in [0, 16]
                          operand id = 1
                            (d0, d1) -> (),
                            domain:
                            d0 in [0, 6],
                            d1 in [0, 16]
                          )"));
}

TEST_F(IndexingAnalysisTest, ReduceWindowOp_BaseDilation) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    max {
      p0 = s32[] parameter(0)
      p1 = s32[] parameter(1)
      ROOT max = s32[] maximum(p0, p1)
    }
    ENTRY e {
      c_inf = s32[] constant(-100)
      p0 = s32[2, 3] parameter(0)
      ROOT reduce-window = s32[3, 5] reduce-window(p0, c_inf),
       window={size=1x1 pad=0_0x0_0 lhs_dilate=2x2}, to_apply=max
    }
  )");
  auto input_indexing = GetOutputToInputIndexing(root);
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0, d1) -> (d0 floordiv 2, d1 floordiv 2),
                            domain:
                            d0 in [0, 2],
                            d1 in [0, 4],
                            d0 mod 2 in [0, 0],
                            d1 mod 2 in [0, 0]
                          operand id = 1
                            (d0, d1) -> (),
                            domain:
                            d0 in [0, 2],
                            d1 in [0, 4]
                          )"));
}

TEST_F(IndexingAnalysisTest, ReduceWindowOp_WindowDilation) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    max {
      p0 = s32[] parameter(0)
      p1 = s32[] parameter(1)
      ROOT max = s32[] maximum(p0, p1)
    }
    ENTRY e {
      c_inf = s32[] constant(-100)
      p0 = s32[7, 3] parameter(0)
      ROOT reduce-window = s32[4, 3] reduce-window(p0, c_inf),
       window={size=2x1 pad=0_0x0_0 rhs_dilate=3x1}, to_apply=max
    }
  )");
  auto input_indexing = GetOutputToInputIndexing(root);
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0, d1)[s0] -> (d0 + s0 * 3, d1),
                            domain:
                            d0 in [0, 3],
                            d1 in [0, 2],
                            s0 in [0, 1]
                          operand id = 1
                            (d0, d1) -> (),
                            domain:
                            d0 in [0, 3],
                            d1 in [0, 2]
                          )"));
}

// TODO(chokobole): Enable this test. Dependency: variadic ReduceWindow
TEST_F(IndexingAnalysisTest, DISABLED_ReduceWindowOp_Variadic) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    combiner {
      a0 = s32[] parameter(0)
      a1 = s32[] parameter(1)
      b0 = s32[] parameter(2)
      b1 = s32[] parameter(3)
      add0 = s32[] add(a0, b0)
      add1 = s32[] add(a1, b1)
      ROOT sum2 = (s32[], s32[]) tuple(add0, add1)
    }
    ENTRY e {
      c_s32 = s32[] constant(-100)
      c_s32 = s32[] constant(10)
      p0 = s32[2, 3] parameter(0)
      p1 = s32[2, 3] parameter(1)
      ROOT reduce-window = (s32[1, 2], s32[1, 2])
        reduce-window(p0, p1, c_s32, c_s32),
        window={size=2x2 pad=0_0x0_0}, to_apply=combiner
    }
  )");
  auto input_indexing_0 = GetOutputToInputIndexing(root, /*output_id=*/0);
  EXPECT_THAT(input_indexing_0.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0, d1)[s0, s1] -> (s0, d1 + s1),
                            domain:
                            d0 in [0, 0],
                            d1 in [0, 1],
                            s0 in [0, 1],
                            s1 in [0, 1]
                          operand id = 1
                            (d0, d1)[s0, s1] -> (s0, d1 + s1),
                            domain:
                            d0 in [0, 0],
                            d1 in [0, 1],
                            s0 in [0, 1],
                            s1 in [0, 1]
                          operand id = 2
                            (d0, d1) -> (),
                            domain:
                            d0 in [0, 0],
                            d1 in [0, 1]
                          operand id = 3
                           (d0, d1) -> (),
                            domain:
                            d0 in [0, 0],
                            d1 in [0, 1]
                          )"));
  auto input_indexing_1 = GetOutputToInputIndexing(root, /*output_id=*/1);
  EXPECT_THAT(input_indexing_1.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0, d1)[s0, s1] -> (s0, d1 + s1),
                            domain:
                            d0 in [0, 0],
                            d1 in [0, 1],
                            s0 in [0, 1],
                            s1 in [0, 1]
                          operand id = 1
                            (d0, d1)[s0, s1] -> (s0, d1 + s1),
                            domain:
                            d0 in [0, 0],
                            d1 in [0, 1],
                            s0 in [0, 1],
                            s1 in [0, 1]
                          operand id = 2
                            (d0, d1) -> (),
                            domain:
                            d0 in [0, 0],
                            d1 in [0, 1]
                          operand id = 3
                           (d0, d1) -> (),
                            domain:
                            d0 in [0, 0],
                            d1 in [0, 1]
                          )"));
}

TEST_F(IndexingAnalysisTest, ReverseOp) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = s32[1, 17, 9, 9] parameter(0)
      ROOT reverse = s32[1, 17, 9, 9] reverse(p0), dimensions={1, 2}
    }
  )");
  auto input_indexing = GetOutputToInputIndexing(root);
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0, d1, d2, d3) -> (d0, -d1 + 16, -d2 + 8, d3),
                            domain:
                            d0 in [0, 0],
                            d1 in [0, 16],
                            d2 in [0, 8],
                            d3 in [0, 8]
                          )"));

  auto output_indexing = GetInputToOutputIndexing(root);
  EXPECT_THAT(output_indexing.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0, d1, d2, d3) -> (d0, -d1 + 16, -d2 + 8, d3),
                            domain:
                            d0 in [0, 0],
                            d1 in [0, 16],
                            d2 in [0, 8],
                            d3 in [0, 8]
                          )"));
}

TEST_F(IndexingAnalysisTest, ReverseReshape) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    fused_computation {
      p0 = s32[10, 11] parameter(0)
      reverse.0 = s32[10, 11] reverse(p0), dimensions={0, 1}
      reshape.0 = s32[110] reshape(reverse.0)
      reverse.1 = s32[110] reverse(reshape.0), dimensions={0}
      ROOT reshape.1 = s32[10, 11] reshape(reverse.1)
    }
    ENTRY e {
      p0 = s32[10, 11] parameter(0)
      ROOT fusion = s32[10, 11] fusion(p0), kind=kLoop,
      calls=fused_computation
    }
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0, d1) -> (d0, d1),
                            domain:
                            d0 in [0, 9],
                            d1 in [0, 10]
                          )"));
}

TEST_F(IndexingAnalysisTest, SliceOp) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = s32[10, 20, 50] parameter(0)
      ROOT slice = s32[5, 3, 25] slice(s32[10, 20, 50] p0),
          slice={[5:10:1], [3:20:7], [0:50:2]}
    }
  )");
  auto input_indexing = GetOutputToInputIndexing(root);
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0, d1, d2) -> (d0 + 5, d1 * 7 + 3, d2 * 2),
                            domain:
                            d0 in [0, 4],
                            d1 in [0, 2],
                            d2 in [0, 24]
                          )"));
  auto output_indexing = GetInputToOutputIndexing(root);
  EXPECT_THAT(output_indexing.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0, d1, d2) -> (
                              d0 - 5,
                              (d1 - 3) floordiv 7,
                              d2 floordiv 2
                            ),
                            domain:
                            d0 in [5, 9],
                            d1 in [3, 17],
                            d2 in [0, 48],
                            (d1 - 3) mod 7 in [0, 0],
                            d2 mod 2 in [0, 0]
                          )"));
}

TEST_F(IndexingAnalysisTest, TransposeOp) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = s32[3, 12288, 6, 128] parameter(0)
      ROOT transpose = s32[3, 6, 128, 12288]
        transpose(p0), dimensions={0, 2, 3, 1}
    }
  )");
  EXPECT_THAT(GetOutputToInputIndexing(root).ToString(), MatchIndexingString(R"(
                            operand id = 0
                              (d0, d1, d2, d3) -> (d0, d3, d1, d2),
                              domain:
                              d0 in [0, 2],
                              d1 in [0, 5],
                              d2 in [0, 127],
                              d3 in [0, 12287]
                          )"));
  EXPECT_THAT(GetInputToOutputIndexing(root).ToString(), MatchIndexingString(R"(
                            operand id = 0
                              (d0, d1, d2, d3) -> (d0, d2, d3, d1),
                              domain:
                              d0 in [0, 2],
                              d1 in [0, 12287],
                              d2 in [0, 5],
                              d3 in [0, 127]
                          )"));
}

TEST_F(IndexingAnalysisTest, TransposeOp4D) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = s32[3, 12288, 6, 128] parameter(0)
      ROOT bitcast = s32[3, 6, 128, 12288] {2, 1, 3, 0} bitcast(p0)
    }
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                            operand id = 0
                              (d0, d1, d2, d3) -> (d0, d3, d1, d2),
                              domain:
                              d0 in [0, 2],
                              d1 in [0, 5],
                              d2 in [0, 127],
                              d3 in [0, 12287]
                          )"));
}

TEST_F(IndexingAnalysisTest, DotOp) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = s32[4, 38, 17, 11, 18, 10] parameter(0)
      p1 = s32[17, 10, 16, 18, 22, 38] parameter(1)
      ROOT dot = s32[10, 38, 4, 11, 16, 22] dot(p0, p1),
        lhs_batch_dims={5,1}, rhs_batch_dims={1,5},
        lhs_contracting_dims={4,2}, rhs_contracting_dims={3,0}
    }
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                operand id = 0
                  (d0, d1, d2, d3, d4, d5)[s0, s1] -> (d2, d1, s1, d3, s0, d0),
                  domain:
                  d0 in [0, 9],
                  d1 in [0, 37],
                  d2 in [0, 3],
                  d3 in [0, 10],
                  d4 in [0, 15],
                  d5 in [0, 21],
                  s0 in [0, 17],
                  s1 in [0, 16]
                operand id = 1
                  (d0, d1, d2, d3, d4, d5)[s0, s1] -> (s1, d0, d4, s0, d5, d1),
                  domain:
                  d0 in [0, 9],
                  d1 in [0, 37],
                  d2 in [0, 3],
                  d3 in [0, 10],
                  d4 in [0, 15],
                  d5 in [0, 21],
                  s0 in [0, 17],
                  s1 in [0, 16]
              )"));
}

TEST_F(IndexingAnalysisTest, EpilogueIndexing) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule m
    fused_computation {
      p0 = s32[1000, 1000] parameter(0)
      t = s32[1000, 1000]{0, 1} transpose(p0), dimensions={1, 0}
      a0 = s32[1000000] bitcast(t)
      ROOT negate = s32[1000000] negate(a0)
    }
    ENTRY e {
      p0 = s32[1000, 1000] parameter(0)
      ROOT fusion = s32[1000000] fusion(p0), kind=kLoop,
          calls=fused_computation
    }
  )");
  ASSERT_TRUE(module.ok());
  auto* computation = (*module)->GetComputationWithName("fused_computation");
  auto fusion = HloFusionAdaptor::ForComputation(computation);
  HloInstructionAdaptor transpose(*computation->GetInstructionWithName("t"),
                                  fusion.get());
  HloInstructionAdaptor negate(*computation->GetInstructionWithName("negate"),
                               fusion.get());

  EXPECT_THAT(ToString(ComputeEpilogueInputToOutputIndexing(transpose, negate,
                                                            &mlir_context_)),
              MatchIndexingString(R"(
                  (d0, d1) -> (d1 * 1000 + d0),
                  domain:
                  d0 in [0, 999],
                  d1 in [0, 999]
              )"));
}

TEST_F(IndexingAnalysisTest, EpilogueIndexing_NoEpilogue) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule m
    fused_computation {
      p0 = s32[1000, 1000] parameter(0)
      ROOT t = s32[1000, 1000]{0, 1} transpose(p0), dimensions={1, 0}
    }
    ENTRY e {
      p0 = s32[1000, 1000] parameter(0)
      ROOT fusion = s32[1000, 1000] fusion(p0), kind=kLoop,
          calls=fused_computation
    }
  )");
  ASSERT_TRUE(module.ok());
  auto* computation = (*module)->GetComputationWithName("fused_computation");
  auto fusion = HloFusionAdaptor::ForComputation(computation);
  HloInstructionAdaptor transpose(*computation->GetInstructionWithName("t"),
                                  fusion.get());

  EXPECT_THAT(ToString(ComputeEpilogueInputToOutputIndexing(
                  transpose, transpose, &mlir_context_)),
              MatchIndexingString(R"(
                  (d0, d1) -> (d0, d1),
                  domain:
                  d0 in [0, 999],
                  d1 in [0, 999]
              )"));
}

TEST_F(IndexingAnalysisTest, BroadcastingElementwise) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = pred[] parameter(0)
      p1 = s32[1000, 1000] parameter(1)
      p2 = s32[1000, 1000] parameter(2)
      ROOT select = s32[1000, 1000] select(p0, p1, p2)
    }
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                  operand id = 0 (d0, d1) -> (),
                    domain:
                    d0 in [0, 999],
                    d1 in [0, 999]
                  operand id = 1 (d0, d1) -> (d0, d1),
                    domain:
                    d0 in [0, 999],
                    d1 in [0, 999]
                  operand id = 2 (d0, d1) -> (d0, d1),
                    domain:
                    d0 in [0, 999],
                    d1 in [0, 999]
              )"));
}

TEST_F(IndexingAnalysisTest, FusionWithRTVarsSimplification_ScalarConstant) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"hlo(
      HloModule m
      fused_computation {
        p0 = s32[4096] parameter(0)
        offset = s64[] constant(42)
        ROOT dynamic-slice = s32[10]
          dynamic-slice(p0, offset), dynamic_slice_sizes={10}
      }
      ENTRY main {
        p0 = s32[4096] parameter(0)
        ROOT fusion = s32[10] fusion(p0), kind=kInput, calls=fused_computation
      }
    )hlo"));

  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
    operand id = 0
      (d0) -> (d0 + 42),
      domain:
      d0 in [0, 9]
  )"));
}

// TODO(chokobole): Enable this test. Dependency: HloGatherInstruction
TEST_F(IndexingAnalysisTest, DISABLED_FusionWithRTVarsSimplification_Iota) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"hlo(
      HloModule m
      fused_computation {
        p0 = s32[33,76] parameter(0)
        iota = s64[42,1] iota(), iota_dimension=0
        ROOT gather = s32[42,1,1] gather(p0, iota),
          offset_dims={1,2},
          collapsed_slice_dims={},
          start_index_map={0},
          index_vector_dim=1,
          slice_sizes={1,1}
      }
      ENTRY main {
        p0 = s32[33,76] parameter(0)
        ROOT fusion = s32[42,1,1] fusion(p0), kind=kInput, calls=fused_computation
      }
    )hlo"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
    operand id = 0
    (d0, d1, d2) -> (d0, 0),
     domain:
     d0 in [0, 41],
     d1 in [0, 0],
     d2 in [0, 0]
  )"));
}

// TODO(chokobole): Enable this test. Dependency: HloGatherInstruction
TEST_F(IndexingAnalysisTest,
       DISABLED_FusionWithRTVarsSimplification_IotaAsConstant) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"hlo(
      HloModule m
      fused_computation {
        p0 = s32[33,76] parameter(0)
        iota = s64[42,1] iota(), iota_dimension=1
        ROOT gather = s32[42,1,1] gather(p0, iota),
          offset_dims={1,2},
          collapsed_slice_dims={},
          start_index_map={0},
          index_vector_dim=1,
          slice_sizes={1,1}
      }
      ENTRY main {
        p0 = s32[33,76] parameter(0)
        ROOT fusion = s32[42,1,1] fusion(p0), kind=kInput, calls=fused_computation
      }
    )hlo"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
    operand id = 0
    (d0, d1, d2) -> (0, 0),
     domain:
     d0 in [0, 41],
     d1 in [0, 0],
     d2 in [0, 0]
  )"));
}

// TODO(chokobole): Enable this test. Dependency: HloGatherInstruction
TEST_F(IndexingAnalysisTest,
       DISABLED_FusionWithRTVarsSimplification_Broadcast) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"hlo(
      HloModule m
      fused_computation {
        p0 = s32[33,76] parameter(0)
        c42 = s64[] constant(42)
        bcast = s64[42, 1] broadcast(s64[] c42), dimensions={}
        ROOT gather = s32[42,1,1] gather(p0, bcast),
          offset_dims={1,2},
          collapsed_slice_dims={},
          start_index_map={0},
          index_vector_dim=1,
          slice_sizes={1,1}
      }
      ENTRY main {
        p0 = s32[33,76] parameter(0)
        ROOT fusion = s32[42,1,1] fusion(p0), kind=kInput, calls=fused_computation
      }
    )hlo"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
    operand id = 0
    (d0, d1, d2) -> (42, 0),
     domain:
     d0 in [0, 41],
     d1 in [0, 0],
     d2 in [0, 0]
  )"));
}

// TODO(chokobole): Enable this test. Dependency: HloGatherInstruction
TEST_F(IndexingAnalysisTest, DISABLED_FusionWithRTVarsSimplification_Reverse) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"hlo(
      HloModule m
      fused_computation {
        p0 = s32[33,76] parameter(0)
        iota = s64[42,1] iota(), iota_dimension=0
        reverse = s64[42,1] reverse(iota), dimensions={0}
        ROOT gather = s32[42,1,1] gather(p0, reverse),
          offset_dims={1,2},
          collapsed_slice_dims={},
          start_index_map={0},
          index_vector_dim=1,
          slice_sizes={1,1}
      }
      ENTRY main {
        p0 = s32[33,76] parameter(0)
        ROOT fusion = s32[42,1,1] fusion(p0), kind=kInput, calls=fused_computation
      }
    )hlo"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
    operand id = 0
    (d0, d1, d2) -> (-d0 + 41, 0),
     domain:
     d0 in [0, 41],
     d1 in [0, 0],
     d2 in [0, 0]
  )"));
}

TEST_F(IndexingAnalysisTest, FusionWithRTVarsSimplification_Add) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"hlo(
      HloModule m
      fused_computation {
        p0 = s32[4096] parameter(0)
        p1 = s64[] parameter(1)
        c42 = s64[] constant(42)
        add = s64[] add(c42, p1)
        ROOT dynamic-slice = s32[10]
          dynamic-slice(p0, add), dynamic_slice_sizes={10}
      }
      ENTRY main {
        p0 = s32[4096] parameter(0)
        p1 = s64[] parameter(1)
        ROOT fusion = s32[10] fusion(p0, p1), kind=kInput, calls=fused_computation
      }
    )hlo"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
    operand id = 0 (d0){rt0} -> (d0 + rt0 + 42),
      domain:
      d0 in [0, 9],
      rt0 in [0, 4086]
    operand id = 1
      (d0) -> (),
      domain:
      d0 in [0, 9]
  )"));
}

TEST_F(IndexingAnalysisTest, FusionWithRTVarsSimplification_Multiply) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"hlo(
      HloModule m
      fused_computation {
        p0 = s32[4096] parameter(0)
        p1 = s64[] parameter(1)
        c42 = s64[] constant(42)
        add = s64[] multiply(c42, p1)
        ROOT dynamic-slice = s32[10]
          dynamic-slice(p0, add), dynamic_slice_sizes={10}
      }
      ENTRY main {
        p0 = s32[4096] parameter(0)
        p1 = s64[] parameter(1)
        ROOT fusion = s32[10] fusion(p0, p1), kind=kInput, calls=fused_computation
      }
    )hlo"));
  // TODO: Figure out why the bounds are not updated.
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
    operand id = 0 (d0){rt0} -> (d0 + rt0 * 42),
      domain:
      d0 in [0, 9],
      rt0 in [0, 4086]
    operand id = 1
      (d0) -> (),
      domain:
      d0 in [0, 9]
  )"));
}

TEST_F(IndexingAnalysisTest, FusionWithRTVarsSimplification_ChainedOps) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"hlo(
      HloModule m
      fused_computation {
        p0 = s32[4096] parameter(0)
        p1 = s64[] parameter(1)
        c42 = s64[] constant(42)
        c2 = s64[] constant(2)
        add = s64[] add(c42, p1)
        multiply = s64[] multiply(c2, add)
        ROOT dynamic-slice = s32[10]
          dynamic-slice(p0, multiply), dynamic_slice_sizes={10}
      }
      ENTRY main {
        p0 = s32[4096] parameter(0)
        p1 = s64[] parameter(1)
        ROOT fusion = s32[10] fusion(p0, p1), kind=kInput, calls=fused_computation
      }
    )hlo"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
   operand id = 0
     (d0){rt0} -> (d0 + rt0 * 2 + 84),
     domain: d0 in [0, 9],
     rt0 in [0, 4086]
   operand id = 1
     (d0) -> (),
     domain:
     d0 in [0, 9]
  )"));
}

TEST_F(IndexingAnalysisTest, FusionOpWithDUS) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"hlo(
      HloModule m
      fused_computation {
        bitcast = s32[1,4096]{1,0} parameter(0)
        constant = s32[] constant(0)
        pad = s32[1,8192]{1,0} pad(bitcast, constant), padding=0_0x4096_0
        slice = s32[1]{0} parameter(1)
        bitcast.4 = s32[] bitcast(slice)
        ROOT dynamic-slice = s32[1,4096]{1,0}
          dynamic-slice(pad, constant, bitcast.4), dynamic_slice_sizes={1,4096}
      }
      ENTRY main {
        param_0 = s32[1,4096]{1,0} parameter(0)
        param_1 = s32[1]{0} parameter(1)
        ROOT fusion = s32[1,4096]{1,0} fusion(param_0, param_1), kind=kInput,
          calls=fused_computation
      }
    )hlo"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                            operand id = 0
                              (d0, d1){rt0} -> (0, d1 + rt0 - 4096),
                              domain:
                              d0 in [0, 0],
                              d1 in [0, 4095],
                              rt0 in [0, 4096],
                              d1 + rt0 in [4096, 8191]
                            operand id = 1
                              (d0, d1) -> (0),
                              domain:
                              d0 in [0, 0],
                              d1 in [0, 4095]
                          )"));
}

}  // namespace
}  // namespace zkx::gpu
