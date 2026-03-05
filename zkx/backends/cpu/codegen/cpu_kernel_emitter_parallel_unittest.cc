/* Copyright 2026 The ZKX Authors.

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

// Integration tests for multi-dimensional partitioned kernel parallelism.
// These tests run with HLO passes enabled so that ParallelTaskAssigner sets
// outer_dimension_partitions in BackendConfig. The shapes are chosen so that
// multi-dim partitioning is triggered on multi-core machines (small outermost
// dim + large total data).

#include <cstdint>

#include "absl/base/casts.h"
#include "gtest/gtest.h"

#include "zkx/backends/cpu/codegen/cpu_kernel_emitter_test.h"
#include "zkx/base/containers/container_util.h"
#include "zkx/base/random.h"
#include "zkx/literal_util.h"

namespace zkx::cpu {
namespace {

using ::testing::Test;

class ParallelKernelTest : public CpuKernelEmitterTest {};

// Elementwise add with shape (4, 64, 1024). The outermost dim (4) is too
// small to saturate all cores on a typical machine, so ParallelTaskAssigner
// should produce multi-dim partitions like [4, N] with total > 4.
TEST_F(ParallelKernelTest, MultiDimElementwiseAdd) {
  constexpr int64_t D0 = 4;
  constexpr int64_t D1 = 64;
  constexpr int64_t D2 = 1024;

  hlo_text_ = R"(
    ENTRY %main {
      %x = s32[4,64,1024] parameter(0)
      %y = s32[4,64,1024] parameter(1)
      ROOT %ret = s32[4,64,1024] add(%x, %y)
    }
  )";

  Array3D<int32_t> x_arr(D0, D1, D2);
  Array3D<int32_t> y_arr(D0, D1, D2);
  Array3D<int32_t> expected(D0, D1, D2);
  for (int64_t i = 0; i < D0; ++i) {
    for (int64_t j = 0; j < D1; ++j) {
      for (int64_t k = 0; k < D2; ++k) {
        x_arr({i, j, k}) = absl::bit_cast<int32_t>(base::Uniform<uint32_t>());
        y_arr({i, j, k}) = absl::bit_cast<int32_t>(base::Uniform<uint32_t>());
        expected({i, j, k}) = x_arr({i, j, k}) + y_arr({i, j, k});
      }
    }
  }
  literals_.push_back(LiteralUtil::CreateR3FromArray3D<int32_t>(x_arr));
  literals_.push_back(LiteralUtil::CreateR3FromArray3D<int32_t>(y_arr));
  expected_literal_ = LiteralUtil::CreateR3FromArray3D<int32_t>(expected);

  RunAndVerify(/*run_hlo_passes=*/true);
}

// Fusion (add + multiply) with shape (4, 64, 1024). The fusion pass merges
// the ops into a kLoop fusion, and ParallelTaskAssigner annotates it with
// multi-dim partitions. This tests GetScaledFusionShape inside the fused body.
TEST_F(ParallelKernelTest, MultiDimFusedAddMul) {
  constexpr int64_t D0 = 4;
  constexpr int64_t D1 = 64;
  constexpr int64_t D2 = 1024;

  hlo_text_ = R"(
    ENTRY %main {
      %x = s32[4,64,1024] parameter(0)
      %y = s32[4,64,1024] parameter(1)
      %c = s32[] constant(3)
      %bcast = s32[4,64,1024] broadcast(%c), dimensions={}
      %add = s32[4,64,1024] add(%x, %y)
      ROOT %ret = s32[4,64,1024] multiply(%add, %bcast)
    }
  )";

  Array3D<int32_t> x_arr(D0, D1, D2);
  Array3D<int32_t> y_arr(D0, D1, D2);
  Array3D<int32_t> expected(D0, D1, D2);
  for (int64_t i = 0; i < D0; ++i) {
    for (int64_t j = 0; j < D1; ++j) {
      for (int64_t k = 0; k < D2; ++k) {
        x_arr({i, j, k}) =
            absl::bit_cast<int32_t>(base::Uniform<uint32_t>()) % 1000;
        y_arr({i, j, k}) =
            absl::bit_cast<int32_t>(base::Uniform<uint32_t>()) % 1000;
        expected({i, j, k}) = (x_arr({i, j, k}) + y_arr({i, j, k})) * 3;
      }
    }
  }
  literals_.push_back(LiteralUtil::CreateR3FromArray3D<int32_t>(x_arr));
  literals_.push_back(LiteralUtil::CreateR3FromArray3D<int32_t>(y_arr));
  expected_literal_ = LiteralUtil::CreateR3FromArray3D<int32_t>(expected);

  RunAndVerify(/*run_hlo_passes=*/true);
}

// Broadcast source + elementwise add with multi-dim shape. The broadcast
// source (scalar → tensor) should not be partitioned; only the data-parallel
// operands get offset by chunk_byte_stride.
TEST_F(ParallelKernelTest, MultiDimBroadcastAdd) {
  constexpr int64_t D0 = 4;
  constexpr int64_t D1 = 64;
  constexpr int64_t D2 = 1024;

  hlo_text_ = R"(
    ENTRY %main {
      %x = s32[4,64,1024] parameter(0)
      %c = s32[] constant(42)
      %bcast = s32[4,64,1024] broadcast(%c), dimensions={}
      ROOT %ret = s32[4,64,1024] add(%x, %bcast)
    }
  )";

  Array3D<int32_t> x_arr(D0, D1, D2);
  Array3D<int32_t> expected(D0, D1, D2);
  for (int64_t i = 0; i < D0; ++i) {
    for (int64_t j = 0; j < D1; ++j) {
      for (int64_t k = 0; k < D2; ++k) {
        x_arr({i, j, k}) = absl::bit_cast<int32_t>(base::Uniform<uint32_t>());
        expected({i, j, k}) = x_arr({i, j, k}) + 42;
      }
    }
  }
  literals_.push_back(LiteralUtil::CreateR3FromArray3D<int32_t>(x_arr));
  expected_literal_ = LiteralUtil::CreateR3FromArray3D<int32_t>(expected);

  RunAndVerify(/*run_hlo_passes=*/true);
}

// R1 broadcast into R3 + elementwise multiply. The broadcast source [D2] is
// expanded to [D0, D1, D2] along dim 2, then multiplied with another [D0, D1,
// D2] input. Inside a fusion, the broadcast source operand has fewer total
// elements than the result and must NOT be offset.
TEST_F(ParallelKernelTest, MultiDimR1BroadcastMul) {
  constexpr int64_t D0 = 4;
  constexpr int64_t D1 = 64;
  constexpr int64_t D2 = 1024;

  hlo_text_ = R"(
    ENTRY %main {
      %x = s32[4,64,1024] parameter(0)
      %w = s32[1024] parameter(1)
      %bcast = s32[4,64,1024] broadcast(%w), dimensions={2}
      ROOT %ret = s32[4,64,1024] multiply(%x, %bcast)
    }
  )";

  Array3D<int32_t> x_arr(D0, D1, D2);
  auto w_vec = base::CreateVector(D2, []() {
    return absl::bit_cast<int32_t>(base::Uniform<uint32_t>()) % 100;
  });
  Array3D<int32_t> expected(D0, D1, D2);
  for (int64_t i = 0; i < D0; ++i) {
    for (int64_t j = 0; j < D1; ++j) {
      for (int64_t k = 0; k < D2; ++k) {
        x_arr({i, j, k}) =
            absl::bit_cast<int32_t>(base::Uniform<uint32_t>()) % 100;
        expected({i, j, k}) = x_arr({i, j, k}) * w_vec[k];
      }
    }
  }
  literals_.push_back(LiteralUtil::CreateR3FromArray3D<int32_t>(x_arr));
  literals_.push_back(LiteralUtil::CreateR1<int32_t>(w_vec));
  expected_literal_ = LiteralUtil::CreateR3FromArray3D<int32_t>(expected);

  RunAndVerify(/*run_hlo_passes=*/true);
}

// Slice + negate with multi-dim shape. The slice takes a full slice along all
// dimensions, so partitioning along any dimension is safe. This tests that the
// safety check in ParallelTaskAssigner correctly allows full slices.
TEST_F(ParallelKernelTest, MultiDimSliceNegate) {
  constexpr int64_t D0 = 4;
  constexpr int64_t D1 = 64;
  constexpr int64_t D2 = 1024;

  hlo_text_ = R"(
    ENTRY %main {
      %x = s32[4,64,1024] parameter(0)
      %slice = s32[4,64,512] slice(%x), slice={[0:4], [0:64], [0:512]}
      ROOT %ret = s32[4,64,512] negate(%slice)
    }
  )";

  Array3D<int32_t> x_arr(D0, D1, D2);
  Array3D<int32_t> expected(D0, D1, D2 / 2);
  for (int64_t i = 0; i < D0; ++i) {
    for (int64_t j = 0; j < D1; ++j) {
      for (int64_t k = 0; k < D2; ++k) {
        x_arr({i, j, k}) = absl::bit_cast<int32_t>(base::Uniform<uint32_t>());
        if (k < D2 / 2) {
          expected({i, j, k}) = -x_arr({i, j, k});
        }
      }
    }
  }
  literals_.push_back(LiteralUtil::CreateR3FromArray3D<int32_t>(x_arr));
  expected_literal_ = LiteralUtil::CreateR3FromArray3D<int32_t>(expected);

  RunAndVerify(/*run_hlo_passes=*/true);
}

// Negate with multi-dim shape. Simplest case: single unary op on a shape
// where multi-dim partitioning should activate.
TEST_F(ParallelKernelTest, MultiDimNegate) {
  constexpr int64_t D0 = 4;
  constexpr int64_t D1 = 64;
  constexpr int64_t D2 = 1024;

  hlo_text_ = R"(
    ENTRY %main {
      %x = s32[4,64,1024] parameter(0)
      ROOT %ret = s32[4,64,1024] negate(%x)
    }
  )";

  Array3D<int32_t> x_arr(D0, D1, D2);
  Array3D<int32_t> expected(D0, D1, D2);
  for (int64_t i = 0; i < D0; ++i) {
    for (int64_t j = 0; j < D1; ++j) {
      for (int64_t k = 0; k < D2; ++k) {
        x_arr({i, j, k}) = absl::bit_cast<int32_t>(base::Uniform<uint32_t>());
        expected({i, j, k}) = -x_arr({i, j, k});
      }
    }
  }
  literals_.push_back(LiteralUtil::CreateR3FromArray3D<int32_t>(x_arr));
  expected_literal_ = LiteralUtil::CreateR3FromArray3D<int32_t>(expected);

  RunAndVerify(/*run_hlo_passes=*/true);
}

// Large single-dim shape. With shape (1024, 1024), the outermost dim alone
// can saturate all cores, so only 1D partitioning should be used.
// Verifies correctness for the "normal" single-dim path.
TEST_F(ParallelKernelTest, SingleDimLargeAdd) {
  constexpr int64_t D0 = 1024;
  constexpr int64_t D1 = 1024;

  hlo_text_ = R"(
    ENTRY %main {
      %x = s32[1024,1024] parameter(0)
      %y = s32[1024,1024] parameter(1)
      ROOT %ret = s32[1024,1024] add(%x, %y)
    }
  )";

  Array2D<int32_t> x_arr(D0, D1);
  Array2D<int32_t> y_arr(D0, D1);
  Array2D<int32_t> expected(D0, D1);
  for (int64_t i = 0; i < D0; ++i) {
    for (int64_t j = 0; j < D1; ++j) {
      x_arr({i, j}) = absl::bit_cast<int32_t>(base::Uniform<uint32_t>());
      y_arr({i, j}) = absl::bit_cast<int32_t>(base::Uniform<uint32_t>());
      expected({i, j}) = x_arr({i, j}) + y_arr({i, j});
    }
  }
  literals_.push_back(LiteralUtil::CreateR2FromArray2D<int32_t>(x_arr));
  literals_.push_back(LiteralUtil::CreateR2FromArray2D<int32_t>(y_arr));
  expected_literal_ = LiteralUtil::CreateR2FromArray2D<int32_t>(expected);

  RunAndVerify(/*run_hlo_passes=*/true);
}

}  // namespace
}  // namespace zkx::cpu
