/* Copyright 2025 The ZKX Authors.

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

#ifndef ZKX_BACKENDS_KERNEL_EMITTER_TEST_H_
#define ZKX_BACKENDS_KERNEL_EMITTER_TEST_H_

#include <string>
#include <string_view>
#include <vector>

#include "gtest/gtest.h"

#include "zkx/literal.h"
#include "zkx/service/hlo_runner.h"

namespace zkx {

class KernelEmitterTest : public testing::Test {
 public:
  explicit KernelEmitterTest(std::string_view platform_name);

  void RunAndVerify(bool run_hlo_passes = false);

 protected:
  virtual void Verify(const Literal& ret_literal) const;

  HloRunner runner_;
  std::string_view x_typename_;
  std::string_view x_std_typename_;
  std::vector<Literal> literals_;
  std::string hlo_text_;
  Literal expected_literal_;
  absl::StatusCode expected_status_code_ = absl::StatusCode::kOk;
};

}  // namespace zkx

#endif  // ZKX_BACKENDS_KERNEL_EMITTER_TEST_H_
