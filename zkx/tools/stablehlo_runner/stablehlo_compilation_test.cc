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

#include <memory>
#include <string>
#include <string_view>

#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"

#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test_util.h"
#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/tests/hlo_test_base.h"
#include "zkx/tools/stablehlo_runner/stablehlo_utils.h"

namespace zkx {
namespace {

using ::absl_testing::IsOk;

class StablehloCompilationTest : public HloTestBase {
 protected:
  absl::StatusOr<std::unique_ptr<HloModule>> LoadStablehloModule(
      std::string_view filename) {
    std::string path = tsl::testing::GetDataDependencyFilepath(
        absl::StrCat("zkx/tools/stablehlo_runner/", filename));
    std::string module_text;
    auto status =
        tsl::ReadFileToString(tsl::Env::Default(), path, &module_text);
    if (!status.ok()) return status;

    mlir::MLIRContext context;
    auto stablehlo_module = ParseStablehloModule(module_text, &context);
    if (!stablehlo_module.ok()) return stablehlo_module.status();

    return ConvertStablehloToHloModule(**stablehlo_module);
  }
};

TEST_F(StablehloCompilationTest, Poseidon2Compiles) {
  auto hlo_module = LoadStablehloModule("poseidon2_permutation.stablehlo.mlir");
  ASSERT_THAT(hlo_module, IsOk());
  EXPECT_THAT(GetOptimizedModule(std::move(*hlo_module)), IsOk());
}

TEST_F(StablehloCompilationTest, NttCompiles) {
  auto hlo_module = LoadStablehloModule("ntt.stablehlo.mlir");
  ASSERT_THAT(hlo_module, IsOk());
  EXPECT_THAT(GetOptimizedModule(std::move(*hlo_module)), IsOk());
}

}  // namespace
}  // namespace zkx
