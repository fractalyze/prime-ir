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

#include <iostream>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/strings/substitute.h"

#include "zk_dtypes/include/elliptic_curve/bn/bn254/fr.h"

// clang-format off
#include "benchmark/field_flag.h"
#include "benchmark/runner.h"
// clang-format on

#include "zkx/literal_util.h"
#include "zkx/math/field/prime_field_serde.h"

namespace zkx::benchmark {

std::string GenerateHlo(Field field, uint32_t degree) {
  size_t size = size_t{1} << degree;
  return absl::Substitute(
      R"(
       ENTRY %main {
         %x = $1[$0] parameter(0)

         %mul = $1[$0] multiply(%x, %x)
         ROOT %ret = $1[$0] subtract(%mul, %x)
       }
       )",
      size, FieldToHloString(field));
}

template <typename F>
std::vector<Literal> GenerateLiterals(absl::Span<const F> scalars) {
  std::vector<Literal> literals;
  literals.push_back(LiteralUtil::CreateR1<F>(scalars));
  return literals;
}

template <typename F>
absl::Status RunBenchmark(Runner& runner, Field field) {
  TF_RETURN_IF_ERROR(runner.Run<F>(
      [field](uint32_t degree) { return GenerateHlo(field, degree); },
      [](absl::Span<const F> scalars) { return GenerateLiterals(scalars); }));
  runner.PrintResults();
  return runner.MaybeWriteCsv();
}

absl::Status RealMain(int argc, char** argv) {
  Runner runner;
  Field field;

  AddFieldFlag(runner.parser(), &field);
  runner.AddPositionalFlags();
  runner.AddOptionalFlags(Runner::Options{
      .multi_degrees = true,
  });

  TF_RETURN_IF_ERROR(runner.Parse(argc, argv));

  switch (field) {
    case Field::kBn254Fr:
      return RunBenchmark<zk_dtypes::bn254::Fr>(runner, field);
  }
  ABSL_UNREACHABLE();
  return absl::InternalError("Invalid field");
}

}  // namespace zkx::benchmark

int main(int argc, char* argv[]) {
  absl::Status status = zkx::benchmark::RealMain(argc, argv);
  if (!status.ok()) {
    std::cerr << status.message() << std::endl;
    return 1;
  }
  return 0;
}
