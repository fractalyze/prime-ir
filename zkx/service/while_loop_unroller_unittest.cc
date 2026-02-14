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

#include "zkx/service/while_loop_unroller.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "gtest/gtest.h"

#include "xla/tsl/platform/statusor.h"
#include "zkx/hlo/ir/hlo_computation.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/ir/hlo_opcode.h"
#include "zkx/hlo/testlib/verified_hlo_module.h"
#include "zkx/tests/hlo_test_base.h"
#include "zkx/tests/literal_test_util.h"

namespace zkx {
namespace {

class WhileLoopUnrollerTest : public HloTestBase {
 protected:
  [[nodiscard]] std::unique_ptr<VerifiedHloModule> MakeModuleWithSimpleLoop(
      int num_iters);
  [[nodiscard]] std::unique_ptr<VerifiedHloModule>
  MakeModuleWithLoopBodyIndirectInc(int num_iters);
  [[nodiscard]] std::unique_ptr<VerifiedHloModule>
  MakeModuleWithNestedLoopBodyIndirectInc(int num_iters);
  [[nodiscard]] std::unique_ptr<VerifiedHloModule>
  MakeModuleWithWhileFeedingAnotherWhile(int num_iters);
  [[nodiscard]] std::unique_ptr<VerifiedHloModule>
  MakeModuleWithSimpleLoopAllReduce(int num_iters);
  // These two methods make a module with a while loop over
  // (i = `start`; i < `stop`; i += `step`) whose iterations perform a
  // dynamic slice (or dynamic update slice) at position i with slice size
  // `slice_size` on a tensor whose dimension has size `dim_size`.
  [[nodiscard]] std::unique_ptr<VerifiedHloModule> MakeModuleWithDS(
      int start, int stop, int step, int slice_size, int dim_size);
  [[nodiscard]] std::unique_ptr<VerifiedHloModule> MakeModuleWithDUS(
      int start, int stop, int step, int slice_size, int dim_size);

 public:
  void UnrollAndCompare(std::unique_ptr<HloModule> module,
                        absl::Span<Literal* const> arguments,
                        int64_t unroll_factor = -1, bool wrap_in_loop = false) {
    Literal before_unroll = ExecuteAndTransfer(module->Clone(), arguments);
    VLOG(2) << "before unroll value: " << before_unroll.ToString();

    EXPECT_TRUE(WhileLoopUnroller(unroll_factor, wrap_in_loop)
                    .Run(module.get())
                    .value());

    Literal after_unroll = ExecuteAndTransfer(std::move(module), arguments);
    VLOG(2) << "after unroll value: " << after_unroll.ToString();

    ASSERT_TRUE(LiteralTestUtil::Equal(/*expected=*/before_unroll,
                                       /*actual=*/after_unroll));
  }
};

std::unique_ptr<VerifiedHloModule>
WhileLoopUnrollerTest::MakeModuleWithSimpleLoop(int num_iters) {
  std::string hlo_string_template = R"(
  HloModule SimpleLoop
  SimpleLoop.body {
    loop_var.1 = (s32[]{:T(128)}, s32[3]{0}) parameter(0)
    get-tuple-element.1 = s32[]{:T(128)} get-tuple-element(loop_var.1), index=0
    constant.1 = s32[]{:T(128)} constant(1)
    idx = s32[]{:T(128)} add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    output = s32[3]{0} add(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s32[]{:T(128)}, s32[3]{0}) tuple(idx, output)
  }
  SimpleLoop.condition {
    loop_var.2 = (s32[]{:T(128)}, s32[3]{0}) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[]{:T(128)} constant({{LOOP_BOUND}})
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  ENTRY SimpleLoop {
    constant.3 = s32[]{:T(128)} constant(0)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s32[]{:T(128)}, s32[3]{0}) tuple(constant.3, constant.4)
    ROOT while = (s32[]{:T(128)}, s32[3]{0}) while(tuple.1), condition=
      SimpleLoop.condition, body=SimpleLoop.body
  }
  )";
  std::string hlo_string = absl::StrReplaceAll(
      hlo_string_template, {{"{{LOOP_BOUND}}", absl::StrCat(num_iters)}});
  return ParseAndReturnVerifiedModule(hlo_string).value();
}

std::unique_ptr<VerifiedHloModule>
WhileLoopUnrollerTest::MakeModuleWithLoopBodyIndirectInc(int num_iters) {
  std::string hlo_string_template = R"(
  HloModule SimpleLoop
  SimpleLoop.body {
    loop_var.1 = (s32[], s32[], s32[3]{0}) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    get-tuple-element.2 = s32[] get-tuple-element(loop_var.1), index=1
    get-tuple-element.3 = s32[3]{0} get-tuple-element(loop_var.1), index=2
    output = s32[3]{0} add(get-tuple-element.3, get-tuple-element.3)
    inc = s32[] add(get-tuple-element.1, get-tuple-element.2)
    ROOT tuple = (s32[], s32[], s32[3]{0}) tuple(inc, get-tuple-element.2, output)
  }
  SimpleLoop.condition {
    loop_var.2 = (s32[], s32[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[] constant({{LOOP_BOUND}})
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  ENTRY SimpleLoop {
    constant.1 = s32[] constant(1)
    constant.3 = s32[] constant(0)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s32[], s32[], s32[3]{0}) tuple(constant.3, constant.1, constant.4)
    ROOT while = (s32[], s32[], s32[3]{0}) while(tuple.1), condition=
      SimpleLoop.condition, body=SimpleLoop.body
  }
  )";
  std::string hlo_string = absl::StrReplaceAll(
      hlo_string_template, {{"{{LOOP_BOUND}}", absl::StrCat(num_iters)}});
  return ParseAndReturnVerifiedModule(hlo_string).value();
}

std::unique_ptr<VerifiedHloModule>
WhileLoopUnrollerTest::MakeModuleWithNestedLoopBodyIndirectInc(int num_iters) {
  std::string hlo_string_template = R"(
  HloModule SimpleLoop
  SimpleLoop.body {
    loop_var.1 = (s32[], s32[], s32[3]{0}) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    get-tuple-element.2 = s32[] get-tuple-element(loop_var.1), index=1
    get-tuple-element.3 = s32[3]{0} get-tuple-element(loop_var.1), index=2
    output = s32[3]{0} add(get-tuple-element.3, get-tuple-element.3)
    inc = s32[] add(get-tuple-element.1, get-tuple-element.2)
    ROOT tuple = (s32[], s32[], s32[3]{0}) tuple(inc, get-tuple-element.2, output)
  }
  SimpleLoop.condition {
    loop_var.2 = (s32[], s32[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[] constant({{LOOP_BOUND}})
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  SimpleLoop {
    constant.1 = s32[] constant(1)
    constant.3 = s32[] constant(0)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s32[], s32[], s32[3]{0}) tuple(constant.3, constant.1, constant.4)
    ROOT while = (s32[], s32[], s32[3]{0}) while(tuple.1), condition=
      SimpleLoop.condition, body=SimpleLoop.body
  }
  OuterLoop.body {
    loop_var.1 = (s32[], s32[], s32[3]{0}, s32[10]{0}) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    get-tuple-element.2 = s32[] get-tuple-element(loop_var.1), index=1
    get-tuple-element.22 = s32[3]{0} get-tuple-element(loop_var.1), index=2
    get-tuple-element.3 = s32[10]{0} get-tuple-element(loop_var.1), index=3
    output = s32[10]{0} add(get-tuple-element.3, get-tuple-element.3)
    /* inner loop call*/
    constant.1 = s32[] constant(1)
    constant.3 = s32[] constant(0)
    tuple.1 = (s32[], s32[], s32[3]{0}) tuple(constant.3, constant.1, get-tuple-element.22)
    inner-while = (s32[], s32[], s32[3]{0}) while(tuple.1), condition=
        SimpleLoop.condition, body=SimpleLoop.body
    get-tuple-element.6 = s32[3]{0} get-tuple-element(inner-while), index=2
    inc = s32[] add(get-tuple-element.1, get-tuple-element.2)
    ROOT tuple = (s32[], s32[], s32[3]{0}, s32[10]{0}) tuple(inc, get-tuple-element.2, get-tuple-element.6, output)
  }
  OuterLoop.condition {
    loop_var.2 = (s32[], s32[], s32[3]{0}, s32[10]{0}) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[] constant({{LOOP_BOUND}})
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  ENTRY OuterLoop {
    constant.1 = s32[] constant(1)
    constant.3 = s32[] constant(0)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    constant.5 = s32[10]{0} constant({0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
    tuple.1 = (s32[], s32[], s32[3]{0}, s32[10]{0}) tuple(constant.3, constant.1, constant.4, constant.5)
    ROOT while = (s32[], s32[], s32[3]{0}, s32[10]{0}) while(tuple.1), condition=
        OuterLoop.condition, body=OuterLoop.body
  }
  )";
  std::string hlo_string = absl::StrReplaceAll(
      hlo_string_template, {{"{{LOOP_BOUND}}", absl::StrCat(num_iters)}});
  return ParseAndReturnVerifiedModule(hlo_string).value();
}

std::unique_ptr<VerifiedHloModule>
WhileLoopUnrollerTest::MakeModuleWithWhileFeedingAnotherWhile(int num_iters) {
  std::string hlo_string_template = R"(
  HloModule SimpleLoop
  SimpleLoop.body {
    loop_var.1 = (s32[], s32[3]{0}) parameter(0)
    const1 = s32[] constant(1)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    get-tuple-element.3 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    output = s32[3]{0} add(get-tuple-element.3, get-tuple-element.3)
    inc = s32[] add(get-tuple-element.1, const1)
    ROOT tuple = (s32[], s32[3]{0}) tuple(inc, output)
  }
  SimpleLoop.condition {
    loop_var.2 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[] constant({{LOOP_BOUND}})
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  SimpleLoop {
    constant.3 = s32[] constant(0)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s32[], s32[3]{0}) tuple(constant.3, constant.4)
    ROOT while = (s32[], s32[3]{0}) while(tuple.1), condition=
      SimpleLoop.condition, body=SimpleLoop.body
  }
  OuterLoop.body {
    loop_var.1 = (s32[], s32[3]{0}, s32[10]{0}) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    get-tuple-element.22 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    get-tuple-element.3 = s32[10]{0} get-tuple-element(loop_var.1), index=2
    output1 = s32[3]{0} add(get-tuple-element.22, get-tuple-element.22)
    output2 = s32[10]{0} add(get-tuple-element.3, get-tuple-element.3)
    one = s32[] constant(1)
    inc = s32[] add(get-tuple-element.1, one)
    ROOT tuple = (s32[], s32[3]{0}, s32[10]{0}) tuple(inc, output1, output2)
  }
  OuterLoop.condition {
    loop_var.2 = (s32[], s32[3]{0}, s32[10]{0}) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[] constant({{LOOP_BOUND}})
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  ENTRY entry.comp {
    constant.3 = s32[] constant(0)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    constant.5 = s32[10]{0} constant({0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
    /* inner loop call*/
    tuple.1 = (s32[], s32[3]{0}) tuple(constant.3, constant.4)
    inner-while = (s32[], s32[3]{0}) while(tuple.1), condition=
        SimpleLoop.condition, body=SimpleLoop.body
    get-tuple-element.6 = s32[3]{0} get-tuple-element(inner-while), index=1
    tuple.2 = (s32[], s32[3]{0}, s32[10]{0}) tuple(constant.3, get-tuple-element.6, constant.5)
    ROOT while = (s32[], s32[3]{0}, s32[10]{0}) while(tuple.2), condition=
        OuterLoop.condition, body=OuterLoop.body
  }
  )";
  std::string hlo_string = absl::StrReplaceAll(
      hlo_string_template, {{"{{LOOP_BOUND}}", absl::StrCat(num_iters)}});
  return ParseAndReturnVerifiedModule(hlo_string).value();
}

std::unique_ptr<VerifiedHloModule>
WhileLoopUnrollerTest::MakeModuleWithSimpleLoopAllReduce(int num_iters) {
  // Changed from f32 to s32 for ZKX compatibility.
  std::string hlo_string_template = R"(
  HloModule SimpleLoop

  %reduction {
    %x = s32[] parameter(0)
    %y = s32[] parameter(1)
    ROOT %add = s32[] add(s32[] %x, s32[] %y)
  }

  SimpleLoop.body {
    loop_var.1 = (s32[], s32[1024, 1024], s32[1024, 1024]) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    get-tuple-element.2 = s32[1024, 1024] get-tuple-element(loop_var.1), index=1
    get-tuple-element.3 = s32[1024, 1024] get-tuple-element(loop_var.1), index=2

    %all-reduce = s32[1024, 1024] all-reduce(s32[1024, 1024] get-tuple-element.2), channel_id=1, replica_groups={{0}}, to_apply=%reduction
    %accumulation = s32[1024, 1024] add(s32[1024, 1024] %all-reduce, s32[1024, 1024] get-tuple-element.3)

    constant.1 = s32[] constant(1)
    add = s32[] add(get-tuple-element.1, constant.1)
    ROOT tuple = (s32[], s32[1024, 1024], s32[1024, 1024]) tuple(add, get-tuple-element.2, %accumulation)
  }
  SimpleLoop.condition {
    loop_var.2 = (s32[], s32[1024, 1024], s32[1024, 1024]) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[] constant({{LOOP_BOUND}})
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  ENTRY SimpleLoop {
    %param.1 = s32[1024, 1024] parameter(0)
    constant.3 = s32[] constant(0)

    %accumulation_buffer_init = s32[] constant(0)
    %accumulation_buffer = s32[1024, 1024] broadcast(s32[] %accumulation_buffer_init), dimensions={}

    tuple.1 = (s32[], s32[1024, 1024], s32[1024, 1024]) tuple(constant.3, %param.1, %accumulation_buffer)
    ROOT while = (s32[], s32[1024, 1024], s32[1024, 1024]) while(tuple.1), condition=SimpleLoop.condition, body=SimpleLoop.body
  }
  )";
  std::string hlo_string = absl::StrReplaceAll(
      hlo_string_template, {{"{{LOOP_BOUND}}", absl::StrCat(num_iters)}});
  return ParseAndReturnVerifiedModule(hlo_string).value();
}

std::unique_ptr<VerifiedHloModule> WhileLoopUnrollerTest::MakeModuleWithDS(
    int start, int stop, int step, int slice_size, int dim_size) {
  std::string hlo_string_template = R"(
  HloModule SimpleLoop
  SimpleLoop.body {
    loop_var.1 = (s32[]{:T(128)}, s32[{{DIM_SIZE}},10]{1,0}) parameter(0)
    get-tuple-element.1 = s32[]{:T(128)} get-tuple-element(loop_var.1), index=0
    constant.1 = s32[]{:T(128)} constant({{STEP}})
    idx = s32[]{:T(128)} add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[{{DIM_SIZE}},10]{1,0} get-tuple-element(loop_var.1), index=1
    zero = s32[] constant(0)
    slice = s32[{{SLICE_SIZE}},10] dynamic-slice(get-tuple-element.2, get-tuple-element.1, zero), dynamic_slice_sizes={{{SLICE_SIZE}},10}
    output = s32[{{DIM_SIZE}},10]{1,0} add(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s32[]{:T(128)}, s32[{{DIM_SIZE}},10]{1,0}) tuple(idx, output)
  }
  SimpleLoop.condition {
    loop_var.2 = (s32[]{:T(128)}, s32[{{DIM_SIZE}},10]{1,0}) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[]{:T(128)} constant({{STOP}})
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  ENTRY SimpleLoop {
    constant.3 = s32[]{:T(128)} constant({{START}})
    constant.4 = s32[{{DIM_SIZE}},10]{1,0} constant({...})
    tuple.1 = (s32[]{:T(128)}, s32[{{DIM_SIZE}},10]{1,0}) tuple(constant.3, constant.4)
    ROOT while = (s32[]{:T(128)}, s32[{{DIM_SIZE}},10]{1,0}) while(tuple.1), condition= SimpleLoop.condition, body=SimpleLoop.body
  }
  )";
  std::string hlo_string = absl::StrReplaceAll(
      hlo_string_template, {{"{{START}}", absl::StrCat(start)},
                            {"{{STOP}}", absl::StrCat(stop)},
                            {"{{STEP}}", absl::StrCat(step)},
                            {"{{SLICE_SIZE}}", absl::StrCat(slice_size)},
                            {"{{DIM_SIZE}}", absl::StrCat(dim_size)}});
  return ParseAndReturnVerifiedModule(hlo_string).value();
}

std::unique_ptr<VerifiedHloModule> WhileLoopUnrollerTest::MakeModuleWithDUS(
    int start, int stop, int step, int slice_size, int dim_size) {
  std::string hlo_string_template = R"(
  HloModule SimpleLoop
  SimpleLoop.body {
    loop_var.1 = (s32[]{:T(128)}, s32[{{DIM_SIZE}},10]{1,0}) parameter(0)
    get-tuple-element.1 = s32[]{:T(128)} get-tuple-element(loop_var.1), index=0
    constant.1 = s32[]{:T(128)} constant({{STEP}})
    idx = s32[]{:T(128)} add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[{{DIM_SIZE}},10]{1,0} get-tuple-element(loop_var.1), index=1
    zero = s32[] constant(0)
    broadcast = s32[{{SLICE_SIZE}},10] broadcast(zero)
    slice = s32[{{DIM_SIZE}},10] dynamic-update-slice(get-tuple-element.2, broadcast, get-tuple-element.1, zero)
    output = s32[{{DIM_SIZE}},10]{1,0} add(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s32[]{:T(128)}, s32[{{DIM_SIZE}},10]{1,0}) tuple(idx, output)
  }
  SimpleLoop.condition {
    loop_var.2 = (s32[]{:T(128)}, s32[{{DIM_SIZE}},10]{1,0}) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[]{:T(128)} constant({{STOP}})
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  ENTRY SimpleLoop {
    constant.3 = s32[]{:T(128)} constant({{START}})
    constant.4 = s32[{{DIM_SIZE}},10]{1,0} constant({...})
    tuple.1 = (s32[]{:T(128)}, s32[{{DIM_SIZE}},10]{1,0}) tuple(constant.3, constant.4)
    ROOT while = (s32[]{:T(128)}, s32[{{DIM_SIZE}},10]{1,0}) while(tuple.1), condition= SimpleLoop.condition, body=SimpleLoop.body
  }
  )";
  std::string hlo_string = absl::StrReplaceAll(
      hlo_string_template, {{"{{START}}", absl::StrCat(start)},
                            {"{{STOP}}", absl::StrCat(stop)},
                            {"{{STEP}}", absl::StrCat(step)},
                            {"{{SLICE_SIZE}}", absl::StrCat(slice_size)},
                            {"{{DIM_SIZE}}", absl::StrCat(dim_size)}});
  return ParseAndReturnVerifiedModule(hlo_string).value();
}

TEST_F(WhileLoopUnrollerTest, SimpleLoopUnroll) {
  UnrollAndCompare(MakeModuleWithSimpleLoop(/*num_iters=*/5), {}, -1, false);
  UnrollAndCompare(MakeModuleWithSimpleLoop(/*num_iters=*/5), {}, -1, true);
}

// This test passes because we run WhileLoopConstantSinking before unrolling.
TEST_F(WhileLoopUnrollerTest, SimpleLoopUnrollNeedPrepare) {
  std::string hlo_string = R"(
  HloModule SimpleLoop
  SimpleLoop.body {
    loop_var.1 = (s64[], s32[3]{0}, s64[]) parameter(0)
    get-tuple-element.1 = s64[] get-tuple-element(loop_var.1), index=0
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    get-tuple-element.3 = s64[] get-tuple-element(loop_var.1), index=2
    add = s64[] add(get-tuple-element.1, get-tuple-element.3)
    multiply = s32[3]{0} add(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s64[], s32[3]{0}, s64[]) tuple(add, multiply, get-tuple-element.3)
  }
  SimpleLoop.condition {
    loop_var.2 = (s64[], s32[3]{0}, s64[]) parameter(0)
    get-tuple-element.3 = s64[] get-tuple-element(loop_var.2), index=0
    /* number of iterations is 10 */
    constant.2 = s64[] constant(10)
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  ENTRY SimpleLoop {
    constant.3 = s64[] constant(0)
    one = s64[] constant(1)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s64[], s32[3]{0}, s64[]) tuple(constant.3, constant.4, one)
    while = (s64[], s32[3]{0}, s64[]) while(tuple.1), condition=
      SimpleLoop.condition, body=SimpleLoop.body
    ROOT result = s32[3]{0} get-tuple-element(while), index=1
  }
  )";
  UnrollAndCompare(ParseAndReturnVerifiedModule(hlo_string).value(), {}, -1,
                   false);
  UnrollAndCompare(ParseAndReturnVerifiedModule(hlo_string).value(), {}, -1,
                   true);
}

// This test passes because we run TupleSimplifier before unrolling.
TEST_F(WhileLoopUnrollerTest, SimpleLoopUnrollNeedPrepare2) {
  std::string hlo_string = R"(
  HloModule SimpleLoop
  SimpleLoop.body {
    loop_var.1 = (s64[], s32[3]{0}, s64[]) parameter(0)
    get-tuple-element.1 = s64[] get-tuple-element(loop_var.1), index=0
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    get-tuple-element.3 = s64[] get-tuple-element(loop_var.1), index=2
    add = s64[] add(get-tuple-element.1, get-tuple-element.3)
    multiply = s32[3]{0} add(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s64[], s32[3]{0}, s64[]) tuple(add, multiply, get-tuple-element.3)
  }
  SimpleLoop.condition {
    loop_var.2 = (s64[], s32[3]{0}, s64[]) parameter(0)
    get-tuple-element.3 = s64[] get-tuple-element(loop_var.2), index=0
    /* number of iterations is 10 */
    constant.2 = s64[] constant(10)
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  ENTRY SimpleLoop {
    constant.3 = s64[] constant(0)
    one = s64[] constant(1)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s64[], s32[3]{0}, s64[]) tuple(constant.3, constant.4, one)
    gte1 = s64[] get-tuple-element(tuple.1), index=0
    gte2 = s32[3]{0} get-tuple-element(tuple.1), index=1
    gte3 = s64[] get-tuple-element(tuple.1), index=2
    tuple = (s64[], s32[3]{0}, s64[]) tuple(gte1, gte2, gte3)
    while = (s64[], s32[3]{0}, s64[]) while(tuple), condition=
      SimpleLoop.condition, body=SimpleLoop.body
    ROOT result = s32[3]{0} get-tuple-element(while), index=1
  }
  )";
  UnrollAndCompare(ParseAndReturnVerifiedModule(hlo_string).value(), {}, -1,
                   false);
  UnrollAndCompare(ParseAndReturnVerifiedModule(hlo_string).value(), {}, -1,
                   true);
}

TEST_F(WhileLoopUnrollerTest, SimpleLoopNotRoot) {
  std::string hlo_string = R"(
  HloModule SimpleLoop
  SimpleLoop.body {
    loop_var.1 = (s64[], s32[3]{0}) parameter(0)
    get-tuple-element.1 = s64[] get-tuple-element(loop_var.1), index=0
    constant.1 = s64[] constant(1)
    add = s64[] add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    multiply = s32[3]{0} add(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s64[], s32[3]{0}) tuple(add, multiply)
  }
  SimpleLoop.condition {
    loop_var.2 = (s64[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s64[] get-tuple-element(loop_var.2), index=0
    /* number of iterations is 10 */
    constant.2 = s64[] constant(10)
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  ENTRY SimpleLoop {
    constant.3 = s64[] constant(0)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s64[], s32[3]{0}) tuple(constant.3, constant.4)
    while = (s64[], s32[3]{0}) while(tuple.1), condition=
      SimpleLoop.condition, body=SimpleLoop.body
    ROOT result = s32[3]{0} get-tuple-element(while), index=1
  }
  )";
  UnrollAndCompare(ParseAndReturnVerifiedModule(hlo_string).value(), {}, -1,
                   false);
  UnrollAndCompare(ParseAndReturnVerifiedModule(hlo_string).value(), {}, -1,
                   true);
}

TEST_F(WhileLoopUnrollerTest, GetUnrollableLoops) {
  std::string hlo_string = R"(
  HloModule SimpleLoop
  SimpleLoop.body {
    loop_var.1 = (s64[], s32[3]{0}) parameter(0)
    get-tuple-element.1 = s64[] get-tuple-element(loop_var.1), index=0
    constant.1 = s64[] constant(1)
    add = s64[] add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    multiply = s32[3]{0} add(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s64[], s32[3]{0}) tuple(add, multiply)
  }
  SimpleLoop.condition {
    loop_var.2 = (s64[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s64[] get-tuple-element(loop_var.2), index=0
    /* number of iterations is 10 */
    constant.2 = s64[] constant(10)
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  SimpleLoop.body.2 {
    loop_var.1 = (s64[], s32[3]{0}) parameter(0)
    get-tuple-element.1 = s64[] get-tuple-element(loop_var.1), index=0
    constant.1 = s64[] constant(1)
    add = s64[] add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    multiply = s32[3]{0} add(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s64[], s32[3]{0}) tuple(add, multiply)
  }
  SimpleLoop.condition.2 {
    loop_var.2 = (s64[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s64[] get-tuple-element(loop_var.2), index=0
    /* number of iterations is 10 */
    constant.2 = s64[] constant(10)
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  SimpleLoop.body.3 {
    loop_var.1 = (s64[], s32[3]{0}) parameter(0)
    get-tuple-element.1 = s64[] get-tuple-element(loop_var.1), index=0
    constant.1 = s64[] constant(1)
    add = s64[] multiply(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    multiply = s32[3]{0} add(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s64[], s32[3]{0}) tuple(add, multiply)
  }
  SimpleLoop.condition.3 {
    loop_var.2 = (s64[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s64[] get-tuple-element(loop_var.2), index=0
    /* number of iterations is 10 */
    constant.2 = s64[] constant(10)
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  ENTRY SimpleLoop {
    constant.3 = s64[] constant(0)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s64[], s32[3]{0}) tuple(constant.3, constant.4)
    while1 = (s64[], s32[3]{0}) while(tuple.1), condition=
      SimpleLoop.condition, body=SimpleLoop.body
    while3 = (s64[], s32[3]{0}) while(tuple.1), condition=
      SimpleLoop.condition.3, body=SimpleLoop.body.3
    while2 = (s64[], s32[3]{0}) while(tuple.1), condition=
      SimpleLoop.condition.2, body=SimpleLoop.body.2
    o1 = s32[3]{0} get-tuple-element(while1), index=1
    o2 = s32[3]{0} get-tuple-element(while2), index=1
    ROOT result = (s32[3]{0}, s32[3]{0}) tuple(o1,o2)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto unrollable_loops = WhileLoopUnroller::GetUnrollableLoops(
      module.get(), {}, /*unroll_config=*/std::nullopt);
  // Only while1 and while2 are unrollable
  EXPECT_EQ(unrollable_loops.size(), 2);
}

TEST_F(WhileLoopUnrollerTest, UnrollMutipleLoops) {
  std::string hlo_string = R"(
  HloModule SimpleLoop
  SimpleLoop.body {
    loop_var.1 = (s64[], s32[3]{0}) parameter(0)
    get-tuple-element.1 = s64[] get-tuple-element(loop_var.1), index=0
    constant.1 = s64[] constant(1)
    add = s64[] add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    multiply = s32[3]{0} add(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s64[], s32[3]{0}) tuple(add, multiply)
  }
  SimpleLoop.condition {
    loop_var.2 = (s64[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s64[] get-tuple-element(loop_var.2), index=0
    /* number of iterations is 10 */
    constant.2 = s64[] constant(10)
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  SimpleLoop.body.2 {
    loop_var.1 = (s64[], s32[3]{0}) parameter(0)
    get-tuple-element.1 = s64[] get-tuple-element(loop_var.1), index=0
    constant.1 = s64[] constant(1)
    add = s64[] add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    multiply = s32[3]{0} add(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s64[], s32[3]{0}) tuple(add, multiply)
  }
  SimpleLoop.condition.2 {
    loop_var.2 = (s64[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s64[] get-tuple-element(loop_var.2), index=0
    /* number of iterations is 10 */
    constant.2 = s64[] constant(10)
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  ENTRY SimpleLoop {
    constant.3 = s64[] constant(0)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s64[], s32[3]{0}) tuple(constant.3, constant.4)
    while1 = (s64[], s32[3]{0}) while(tuple.1), condition=
      SimpleLoop.condition, body=SimpleLoop.body
    input = s32[3]{0} get-tuple-element(while1), index=1
    tuple.2 = (s64[], s32[3]{0}) tuple(constant.3, input)
    while2 = (s64[], s32[3]{0}) while(tuple.2), condition=
      SimpleLoop.condition.2, body=SimpleLoop.body.2
    o1 = s32[3]{0} get-tuple-element(while1), index=1
    o2 = s32[3]{0} get-tuple-element(while2), index=1
    ROOT result = (s32[3]{0}, s32[3]{0}) tuple(o1,o2)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  // Unroll the first loop
  TF_ASSERT_OK_AND_ASSIGN(
      UnrollResult unrolled_result,
      WhileLoopUnroller::UnrollAndReturnReplacement(
          module->entry_computation()->GetInstructionWithName("while1")));
  bool unrolled1 = unrolled_result.unrolled;
  EXPECT_TRUE(unrolled1);

  // There should be no call instructions after unrolling either loops since we
  // inline all the calls after unrolling.
  std::vector<HloInstruction*> call_instrs_1;
  for (auto* comp : module->MakeComputationPostOrder()) {
    absl::c_copy_if(comp->instructions(), std::back_inserter(call_instrs_1),
                    HloPredicateIsOp<HloOpcode::kCall>);
  }
  EXPECT_EQ(call_instrs_1.size(), 0);

  // Unroll the second loop
  TF_ASSERT_OK_AND_ASSIGN(
      UnrollResult unrolled_result2,
      WhileLoopUnroller::UnrollAndReturnReplacement(
          module->entry_computation()->GetInstructionWithName("while2")));
  bool unrolled2 = unrolled_result2.unrolled;
  EXPECT_TRUE(unrolled2);
  std::vector<HloInstruction*> call_instrs_2;
  for (auto* comp : module->MakeComputationPostOrder()) {
    absl::c_copy_if(comp->instructions(), std::back_inserter(call_instrs_2),
                    HloPredicateIsOp<HloOpcode::kCall>);
  }
  EXPECT_EQ(call_instrs_2.size(), 0);
}

TEST_F(WhileLoopUnrollerTest, SimpleLoopNonZeroInit) {
  std::string hlo_string = R"(
  HloModule SimpleLoop
  SimpleLoop.body {
    loop_var.1 = (s64[], s32[3]{0}) parameter(0)
    get-tuple-element.1 = s64[] get-tuple-element(loop_var.1), index=0
    constant.1 = s64[] constant(1)
    add = s64[] add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    multiply = s32[3]{0} add(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s64[], s32[3]{0}) tuple(add, multiply)
  }
  SimpleLoop.condition {
    loop_var.2 = (s64[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s64[] get-tuple-element(loop_var.2), index=0
    constant.2 = s64[] constant(10)
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  ENTRY SimpleLoop {
    constant.3 = s64[] constant(4)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s64[], s32[3]{0}) tuple(constant.3, constant.4)
    while = (s64[], s32[3]{0}) while(tuple.1), condition=
      SimpleLoop.condition, body=SimpleLoop.body
    ROOT result = s32[3]{0} get-tuple-element(while), index=1
  }
  )";
  UnrollAndCompare(ParseAndReturnVerifiedModule(hlo_string).value(), {}, -1,
                   false);
  UnrollAndCompare(ParseAndReturnVerifiedModule(hlo_string).value(), {}, -1,
                   true);
}

TEST_F(WhileLoopUnrollerTest, SimpleLoopS16IndVar) {
  std::string hlo_string = R"(
  HloModule SimpleLoop
  SimpleLoop.body {
    loop_var.1 = (s16[], s32[3]{0}) parameter(0)
    get-tuple-element.1 = s16[] get-tuple-element(loop_var.1), index=0
    constant.1 = s16[] constant(1)
    add = s16[] add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    multiply = s32[3]{0} add(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s16[], s32[3]{0}) tuple(add, multiply)
  }
  SimpleLoop.condition {
    loop_var.2 = (s16[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s16[] get-tuple-element(loop_var.2), index=0
    /* number of iterations is 10 */
    constant.2 = s16[] constant(10)
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  ENTRY SimpleLoop {
    constant.3 = s16[] constant(0)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s16[], s32[3]{0}) tuple(constant.3, constant.4)
    ROOT while = (s16[], s32[3]{0}) while(tuple.1), condition=
      SimpleLoop.condition, body=SimpleLoop.body
  }
  )";
  UnrollAndCompare(ParseAndReturnVerifiedModule(hlo_string).value(), {}, -1,
                   false);
  UnrollAndCompare(ParseAndReturnVerifiedModule(hlo_string).value(), {}, -1,
                   true);
}

TEST_F(WhileLoopUnrollerTest, LoopWithControlDep) {
  std::string hlo_string = R"(
  HloModule SimpleLoop
  SimpleLoop.body {
    loop_var.1 = (s16[], s32[3]{0}) parameter(0)
    get-tuple-element.1 = s16[] get-tuple-element(loop_var.1), index=0
    constant.1 = s16[] constant(1)
    add = s16[] add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    multiply = s32[3]{0} add(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s16[], s32[3]{0}) tuple(add, multiply)
  }
  SimpleLoop.condition {
    loop_var.2 = (s16[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s16[] get-tuple-element(loop_var.2), index=0
    /* number of iterations is 10 */
    constant.2 = s16[] constant(10)
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  ENTRY SimpleLoop {
    constant.3 = s16[] constant(0)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s16[], s32[3]{0}) tuple(constant.3, constant.4)
    while1 = (s16[], s32[3]{0}) while(tuple.1), condition=
      SimpleLoop.condition, body=SimpleLoop.body
    copy1 = copy(constant.3), control-predecessors={while1}
    ROOT add = add(copy1, constant.3)
  }
  )";
  EXPECT_FALSE(WhileLoopUnroller()
                   .Run(ParseAndReturnVerifiedModule(hlo_string).value().get())
                   .value());
}

TEST_F(WhileLoopUnrollerTest, SimpleLoopPartialUnroll) {
  auto m = MakeModuleWithSimpleLoop(/*num_iters=*/5);
  EXPECT_FALSE(WhileLoopUnroller(/*unroll_factor=*/3).Run(m.get()).value());
}

TEST_F(WhileLoopUnrollerTest, SimpleLoopNoUnrollDueToTripCountThreshold) {
  auto m = MakeModuleWithSimpleLoop(/*num_iters=*/5);
  UnrollConfig config;
  config.trip_count_threshold = 0;  // Set the trip count threshold to 0.
  EXPECT_FALSE(WhileLoopUnroller(/*unroll_factor=*/-1,
                                 /*wrap_in_trivial_loop=*/false, config)
                   .Run(m.get())
                   .value());
}

TEST_F(WhileLoopUnrollerTest, IndirectBodyInc) {
  std::unique_ptr<HloModule> module =
      MakeModuleWithLoopBodyIndirectInc(/*num_iters=*/5);
  UnrollAndCompare(MakeModuleWithLoopBodyIndirectInc(/*num_iters=*/5), {}, -1,
                   false);
  UnrollAndCompare(MakeModuleWithLoopBodyIndirectInc(/*num_iters=*/5), {}, -1,
                   true);
}

TEST_F(WhileLoopUnrollerTest, NestedIndirectBodyInc) {
  std::unique_ptr<HloModule> module =
      MakeModuleWithNestedLoopBodyIndirectInc(/*num_iters=*/5);
  UnrollAndCompare(MakeModuleWithNestedLoopBodyIndirectInc(/*num_iters=*/5), {},
                   -1, false);
  UnrollAndCompare(MakeModuleWithNestedLoopBodyIndirectInc(/*num_iters=*/5), {},
                   -1, true);
}

TEST_F(WhileLoopUnrollerTest, WhileFeedingWhile) {
  UnrollAndCompare(MakeModuleWithWhileFeedingAnotherWhile(/*num_iters=*/5), {},
                   -1, false);
  UnrollAndCompare(MakeModuleWithWhileFeedingAnotherWhile(/*num_iters=*/5), {},
                   -1, true);
}

// Changed from f32 to s32 types for ZKX compatibility.
TEST_F(WhileLoopUnrollerTest, LoopWithCollective) {
  int64_t num_iters = 5;
  auto module = MakeModuleWithSimpleLoopAllReduce(num_iters);

  EXPECT_TRUE(
      WhileLoopUnroller(/*unroll_factor=*/-1).Run(module.get()).value());

  EXPECT_EQ(absl::c_count_if(module->entry_computation()->instructions(),
                             [](const HloInstruction* instruction) {
                               return instruction->opcode() ==
                                      HloOpcode::kAllReduce;
                             }),
            num_iters);
}

// LoopWithCollective2 is removed: it uses convolution which is not available
// in ZKX.

TEST_F(WhileLoopUnrollerTest, MatchShapeCoveringDS) {
  auto module = MakeModuleWithDS(/*start=*/0, /*stop=*/3, /*step=*/1,
                                 /*slice_size=*/1, /*dim_size=*/3);
  HloInstruction* loop = module->entry_computation()->root_instruction();
  auto config = WhileLoopUnroller::IsLoopUnrollable(loop);
  EXPECT_TRUE(config.has_value());
  HloComputation* body = module->GetComputationWithName("SimpleLoop.body");
  HloInstruction* input = body->GetInstructionWithName("get-tuple-element.2");
  HloInstruction* instr = body->GetInstructionWithName("slice");
  EXPECT_TRUE(MatchShapeCoveringDynamicIndexInstruction(
                  instr, input, HloOpcode::kDynamicSlice, config.value())
                  .has_value());
}

TEST_F(WhileLoopUnrollerTest, MatchShapeCoveringDSShapeMismatch) {
  const std::string hlo_string = R"(
  HloModule SimpleLoop
  body {
    param = (s32[]{:T(128)}, s32[3,10]{1,0}, s32[3,11]{1,0}) parameter(0)
    idx = s32[]{:T(128)} get-tuple-element(param), index=0
    constant1 = s32[]{:T(128)} constant(1)
    new-idx = s32[]{:T(128)} add(idx, constant1)
    update = s32[3,10]{1,0} get-tuple-element(param), index=1
    input = s32[3,11]{1,0} get-tuple-element(param), index=2
    zero = s32[] constant(0)
    slice = s32[1,10] dynamic-slice(input, idx, zero), dynamic_slice_sizes={1,10}
    new-update = s32[3,10]{1,0} dynamic-update-slice(update, slice, idx, zero)
    ROOT tuple = (s32[]{:T(128)}, s32[3,10]{1,0}, s32[3,11]{1,0}) tuple(new-idx, new-update, input)
  }
  condition {
    param = (s32[]{:T(128)}, s32[3,10]{1,0}, s32[3,11]{1,0}) parameter(0)
    idx = s32[] get-tuple-element(param), index=0
    constant3 = s32[]{:T(128)} constant(3)
    ROOT less-than = pred[] compare(idx, constant3), direction=LT
  }
  ENTRY main {
    constant0 = s32[]{:T(128)} constant(0)
    init-update = s32[3,10]{1,0} constant({...})
    init-input = s32[3,11]{1,0} constant({...})
    init-while = (s32[]{:T(128)}, s32[3,10]{1,0}, s32[3,11]{1,0}) tuple(constant0, init-update, init-input)
    ROOT while = (s32[]{:T(128)}, s32[3,10]{1,0}, s32[3,11]{1,0}) while(init-while), condition=
      condition, body=body
  }
  )";

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  HloInstruction* loop = module->entry_computation()->root_instruction();
  auto config = WhileLoopUnroller::IsLoopUnrollable(loop);
  EXPECT_TRUE(config.has_value());
  HloComputation* body = module->GetComputationWithName("body");
  HloInstruction* input = body->GetInstructionWithName("input");
  HloInstruction* instr = body->GetInstructionWithName("slice");
  EXPECT_FALSE(MatchShapeCoveringDynamicIndexInstruction(
                   instr, input, HloOpcode::kDynamicSlice, config.value())
                   .has_value());
}

TEST_F(WhileLoopUnrollerTest, MatchShapeCoveringDSNested) {
  std::string hlo_string_template = R"(
  HloModule SimpleLoop
  %fused_computation.slice (param_0.51117: s32[3,10], p1: s32[]) -> s32[10] {
    %param_0.51117 = s32[3,10] parameter(0)
    p1 = s32[] parameter(1)
    %constant.85694 = s32[] constant(0)
    slice = s32[1,10] dynamic-slice(s32[3,10] %param_0.51117, p1, s32[] %constant.85694), dynamic_slice_sizes={1,10}
    ROOT %bitcast.31250 = s32[10] bitcast(s32[1,10] slice)
  }

  %fused_computation.outer (param_1.30691: s32[3,10], p2: s32[]) -> s32[10] {
    %param_1.30691 = s32[3,10] parameter(0)
    p2 = s32[] parameter(1)
    inner.fusion = s32[10] fusion(s32[3,10] %param_1.30691, p2), kind=kLoop, calls=%fused_computation.slice
    ROOT out = s32[10] add(inner.fusion, inner.fusion)
  }
  SimpleLoop.body {
    loop_var.1 = (s32[]{:T(128)}, s32[3,10]{1,0}) parameter(0)
    get-tuple-element.1 = s32[]{:T(128)} get-tuple-element(loop_var.1), index=0
    constant.1 = s32[]{:T(128)} constant(1)
    idx = s32[]{:T(128)} add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3,10]{1,0} get-tuple-element(loop_var.1), index=1
    zero = s32[] constant(0)
    outer.fusion = s32[10] fusion(get-tuple-element.2, get-tuple-element.1), kind=kOutput, calls=%fused_computation.outer
    output = s32[3,10]{1,0} add(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s32[]{:T(128)}, s32[3,10]{1,0}) tuple(idx, output)
  }
  SimpleLoop.condition {
    loop_var.2 = (s32[]{:T(128)}, s32[3,10]{1,0}) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[]{:T(128)} constant({{LOOP_BOUND}})
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  ENTRY SimpleLoop {
    constant.3 = s32[]{:T(128)} constant(0)
    constant.4 = s32[3,10]{1,0} constant({...})
    tuple.1 = (s32[]{:T(128)}, s32[3,10]{1,0}) tuple(constant.3, constant.4)
    ROOT while = (s32[]{:T(128)}, s32[3,10]{1,0}) while(tuple.1), condition=
      SimpleLoop.condition, body=SimpleLoop.body
  }
  )";

  std::string hlo_string = absl::StrReplaceAll(
      hlo_string_template, {{"{{LOOP_BOUND}}", absl::StrCat(3)}});
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  HloInstruction* loop = module->entry_computation()->root_instruction();
  auto config = WhileLoopUnroller::IsLoopUnrollable(loop);
  EXPECT_TRUE(config.has_value());
  HloComputation* inner_fusion_comp =
      module->GetComputationWithName("fused_computation.slice");
  HloInstruction* instr = inner_fusion_comp->GetInstructionWithName("slice");
  EXPECT_TRUE(MatchShapeCoveringDynamicIndexInstruction(
                  instr, inner_fusion_comp->parameter_instruction(0),
                  HloOpcode::kDynamicSlice, config.value())
                  .has_value());
}

TEST_F(WhileLoopUnrollerTest, AdvancedMatchShapeCoveringDSIncrementByTwo) {
  // In this version of the test, our dimension of interest gets incremented by
  // two at a time so that it takes on values {0, 2, 4}. The DS has slice size
  // two, so indeed all index values {0, 1, 2, 3, 4, 5} are retrieved by the
  // DS.
  auto module = MakeModuleWithDS(/*start=*/0, /*stop=*/6, /*step=*/2,
                                 /*slice_size=*/2, /*dim_size=*/6);
  HloInstruction* loop = module->entry_computation()->root_instruction();
  auto config = WhileLoopUnroller::IsLoopUnrollable(loop);
  EXPECT_TRUE(config.has_value());
  HloComputation* body = module->GetComputationWithName("SimpleLoop.body");
  HloInstruction* input = body->GetInstructionWithName("get-tuple-element.2");
  HloInstruction* instr = body->GetInstructionWithName("slice");
  EXPECT_TRUE(AdvancedMatchShapeCoveringDynamicIndexInstruction(
                  instr, input, HloOpcode::kDynamicSlice, config.value())
                  .has_value());
}

TEST_F(WhileLoopUnrollerTest,
       AdvancedMatchShapeCoveringDSIncrementByTwoMismatch) {
  // In this version of the test, our dimension of interest gets incremented by
  // two at a time so that it takes on values {0, 2, 4}. The DS has slice size
  // two, so only index values {0, 1, 2, 3, 4, 5} are retrieved by the DS and
  // index value 6 is not.
  auto module = MakeModuleWithDS(/*start=*/0, /*stop=*/6, /*step=*/2,
                                 /*slice_size=*/2, /*dim_size=*/7);
  HloInstruction* loop = module->entry_computation()->root_instruction();
  auto config = WhileLoopUnroller::IsLoopUnrollable(loop);
  EXPECT_TRUE(config.has_value());
  HloComputation* body = module->GetComputationWithName("SimpleLoop.body");
  HloInstruction* input = body->GetInstructionWithName("get-tuple-element.2");
  HloInstruction* instr = body->GetInstructionWithName("slice");
  EXPECT_FALSE(AdvancedMatchShapeCoveringDynamicIndexInstruction(
                   instr, input, HloOpcode::kDynamicSlice, config.value())
                   .has_value());
}

TEST_F(WhileLoopUnrollerTest, AdvancedMatchShapeCoveringDUS) {
  auto module = MakeModuleWithDUS(/*start=*/0, /*stop=*/3, /*step=*/1,
                                  /*slice_size=*/1, /*dim_size=*/3);
  HloInstruction* loop = module->entry_computation()->root_instruction();
  auto config = WhileLoopUnroller::IsLoopUnrollable(loop);
  EXPECT_TRUE(config.has_value());
  HloComputation* body = module->GetComputationWithName("SimpleLoop.body");
  HloInstruction* input = body->GetInstructionWithName("get-tuple-element.2");
  HloInstruction* instr = body->GetInstructionWithName("slice");
  EXPECT_TRUE(AdvancedMatchShapeCoveringDynamicIndexInstruction(
                  instr, input, HloOpcode::kDynamicUpdateSlice, config.value())
                  .has_value());
}

TEST_F(WhileLoopUnrollerTest, AdvancedMatchShapeCoveringDUSIncrementByTwo) {
  auto module = MakeModuleWithDUS(/*start=*/0, /*stop=*/6, /*step=*/2,
                                  /*slice_size=*/2, /*dim_size=*/6);
  HloInstruction* loop = module->entry_computation()->root_instruction();
  auto config = WhileLoopUnroller::IsLoopUnrollable(loop);
  EXPECT_TRUE(config.has_value());
  HloComputation* body = module->GetComputationWithName("SimpleLoop.body");
  HloInstruction* input = body->GetInstructionWithName("get-tuple-element.2");
  HloInstruction* instr = body->GetInstructionWithName("slice");
  EXPECT_TRUE(AdvancedMatchShapeCoveringDynamicIndexInstruction(
                  instr, input, HloOpcode::kDynamicUpdateSlice, config.value())
                  .has_value());
}

TEST_F(WhileLoopUnrollerTest,
       AdvancedMatchShapeCoveringDUSIncrementByTwoMismatch) {
  auto module = MakeModuleWithDUS(/*start=*/0, /*stop=*/6, /*step=*/2,
                                  /*slice_size=*/2, /*dim_size=*/7);
  HloInstruction* loop = module->entry_computation()->root_instruction();
  auto config = WhileLoopUnroller::IsLoopUnrollable(loop);
  EXPECT_TRUE(config.has_value());
  HloComputation* body = module->GetComputationWithName("SimpleLoop.body");
  HloInstruction* input = body->GetInstructionWithName("get-tuple-element.2");
  HloInstruction* instr = body->GetInstructionWithName("slice");
  EXPECT_FALSE(AdvancedMatchShapeCoveringDynamicIndexInstruction(
                   instr, input, HloOpcode::kDynamicUpdateSlice, config.value())
                   .has_value());
}

// UnrollLoopWithDynamicGte is removed: it uses bf16 and convolution which are
// not available in ZKX.

// IsEffectivelyStaticDynamicSlice is removed: it uses bf16 and convolution
// which are not available in ZKX.

// We do not support case where there is no tuple for input.
TEST_F(WhileLoopUnrollerTest, SimpleLoopWithCustomCallNoTuple) {
  std::string hlo_string = R"(
  HloModule SimpleLoop
  SimpleLoop.body {
    loop_var.1 = (s32[]{:T(128)}, s32[3]{0}) parameter(0)
    get-tuple-element.1 = s32[]{:T(128)} get-tuple-element(loop_var.1), index=0
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    custom-call.1 = (s32[]{:T(128)}, s32[3]{0}) custom-call(get-tuple-element.1, get-tuple-element.2), custom_call_target="CustomCallStart"
    get-tuple-element.3 = s32[]{:T(128)} get-tuple-element(custom-call.1), index=0
    constant.1 = s32[]{:T(128)} constant(1)
    idx = s32[]{:T(128)} add(get-tuple-element.3, constant.1)
    get-tuple-element.4 = s32[3]{0} get-tuple-element(custom-call.1), index=1
    output = s32[3]{0} add(get-tuple-element.4, get-tuple-element.4)
    tuple = (s32[]{:T(128)}, s32[3]{0}) tuple(idx, output)
    ROOT custom-call.2 = (s32[]{:T(128)}, s32[3]{0}) custom-call(idx, output), custom_call_target="CustomCallEnd"
  }
  SimpleLoop.condition {
    loop_var.2 = (s32[]{:T(128)}, s32[3]{0}) parameter(0)
    get-tuple-element.5 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[]{:T(128)} constant(5)
    ROOT less-than = pred[] compare(get-tuple-element.5, constant.2), direction=LT
  }
  ENTRY SimpleLoop {
    constant.3 = s32[]{:T(128)} constant(0)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s32[]{:T(128)}, s32[3]{0}) tuple(constant.3, constant.4)
    ROOT while = (s32[]{:T(128)}, s32[3]{0}) while(tuple.1), condition=
      SimpleLoop.condition, body=SimpleLoop.body
  }
  )";
  auto m = ParseAndReturnVerifiedModule(hlo_string).value();
  UnrollConfig config;
  EXPECT_FALSE(WhileLoopUnroller(/*unroll_factor=*/-1,
                                 /*wrap_in_trivial_loop=*/false, config)
                   .Run(m.get())
                   .value());
}

TEST_F(WhileLoopUnrollerTest, SimpleLoopWithCustomCallNonTupleForRoot) {
  std::string hlo_string = R"(
  HloModule SimpleLoop
  SimpleLoop.body {
    loop_var.1 = (s32[]{:T(128)}, s32[3]{0}) parameter(0)
    custom-call.1 = (s32[]{:T(128)}, s32[3]{0}) custom-call(loop_var.1), custom_call_target="CustomCallStart"
    get-tuple-element.1 = s32[]{:T(128)} get-tuple-element(custom-call.1), index=0
    constant.1 = s32[]{:T(128)} constant(1)
    idx = s32[]{:T(128)} add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(custom-call.1), index=1
    output = s32[3]{0} add(get-tuple-element.2, get-tuple-element.2)
    ROOT custom-call.2 = (s32[]{:T(128)}, s32[3]{0}) custom-call(idx, output), custom_call_target="CustomCallEnd"
  }
  SimpleLoop.condition {
    loop_var.2 = (s32[]{:T(128)}, s32[3]{0}) parameter(0)
    get-tuple-element.5 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[]{:T(128)} constant(5)
    ROOT less-than = pred[] compare(get-tuple-element.5, constant.2), direction=LT
  }
  ENTRY SimpleLoop {
    constant.3 = s32[]{:T(128)} constant(0)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s32[]{:T(128)}, s32[3]{0}) tuple(constant.3, constant.4)
    ROOT while = (s32[]{:T(128)}, s32[3]{0}) while(tuple.1), condition=
      SimpleLoop.condition, body=SimpleLoop.body
  }
  )";
  auto m = ParseAndReturnVerifiedModule(hlo_string).value();
  UnrollConfig config;
  EXPECT_FALSE(WhileLoopUnroller(/*unroll_factor=*/-1,
                                 /*wrap_in_trivial_loop=*/false, config)
                   .Run(m.get())
                   .value());
}

TEST_F(WhileLoopUnrollerTest, SimpleLoopWithCustomCall) {
  std::string hlo_string = R"(
  HloModule SimpleLoop
  SimpleLoop.body {
    loop_var.1 = (s32[]{:T(128)}, s32[3]{0}) parameter(0)
    custom-call.1 = (s32[]{:T(128)}, s32[3]{0}) custom-call(loop_var.1), custom_call_target="CustomCallStart"
    get-tuple-element.1 = s32[]{:T(128)} get-tuple-element(custom-call.1), index=0
    constant.1 = s32[]{:T(128)} constant(1)
    idx = s32[]{:T(128)} add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(custom-call.1), index=1
    output = s32[3]{0} add(get-tuple-element.2, get-tuple-element.2)
    tuple = (s32[]{:T(128)}, s32[3]{0}) tuple(idx, output)
    ROOT custom-call.2 = (s32[]{:T(128)}, s32[3]{0}) custom-call(tuple), custom_call_target="CustomCallEnd"
  }
  SimpleLoop.condition {
    loop_var.2 = (s32[]{:T(128)}, s32[3]{0}) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[]{:T(128)} constant(5)
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  ENTRY SimpleLoop {
    constant.3 = s32[]{:T(128)} constant(0)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s32[]{:T(128)}, s32[3]{0}) tuple(constant.3, constant.4)
    ROOT while = (s32[]{:T(128)}, s32[3]{0}) while(tuple.1), condition=
      SimpleLoop.condition, body=SimpleLoop.body
  }
  )";
  auto m = ParseAndReturnVerifiedModule(hlo_string).value();
  UnrollConfig config;
  EXPECT_FALSE(WhileLoopUnroller(/*unroll_factor=*/-1,
                                 /*wrap_in_trivial_loop=*/false, config)
                   .Run(m.get())
                   .value());
}

}  // namespace
}  // namespace zkx
