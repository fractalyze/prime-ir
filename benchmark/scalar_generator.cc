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

#include <stdint.h>

#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"

// clang-format off
#include "benchmark/field_flag.h"
// clang-format on
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "zk_dtypes/include/elliptic_curve/bn/bn254/fr.h"
#include "zkx/base/buffer/vector_buffer.h"
#include "zkx/base/flag/flag_parser.h"
#include "zkx/base/flag/numeric_flags.h"
#include "zkx/base/openmp_util.h"
#include "zkx/base/random.h"
#include "zkx/math/field/prime_field_serde.h"

namespace zkx::benchmark {

template <typename F>
absl::StatusOr<base::Uint8VectorBuffer> GenerateScalars(int32_t degree,
                                                        bool condensed) {
  size_t num_scalars = size_t{1} << degree;

  std::vector<F> scalars;
  scalars.resize(num_scalars);
  if (condensed) {
    int64_t a = num_scalars / 3;
    int64_t b = 2 * a;
    OMP_PARALLEL_FOR(int64_t i = 0; i < num_scalars; ++i) {
      if (i < a) {
        scalars[i] = F::Zero();
      } else if (i < b) {
        scalars[i] = F::One();
      } else {
        scalars[i] = F::Random();
      }
    }
    base::Shuffle(scalars);
  } else {
    OMP_PARALLEL_FOR(int64_t i = 0; i < num_scalars; ++i) {
      scalars[i] = F::Random();
    }
  }

  base::Uint8VectorBuffer write_buf;
  TF_RETURN_IF_ERROR(write_buf.Grow(base::EstimateSize(scalars)));
  TF_RETURN_IF_ERROR(write_buf.Write(scalars));
  CHECK(write_buf.Done());

  return write_buf;
}

absl::Status RealMain(int argc, char** argv) {
  base::FlagParser parser;
  Field field;
  uint32_t degree;
  std::string output_path;
  bool condensed;

  AddFieldFlag(parser, &field);
  parser.AddFlag<base::Uint32Flag>(&degree, &base::ParsePositiveValue<uint32_t>)
      .set_name("degree")
      .set_help("Degree of the inputs");
  parser.AddFlag<base::StringFlag>(&output_path)
      .set_name("output")
      .set_help("Output file");
  parser.AddFlag<base::BoolFlag>(&condensed)
      .set_long_name("--condensed")
      .set_default_value(false)
      .set_help(
          "Generate condensed scalars: 1/3 zeros, 1/3 ones, 1/3 small random "
          "values.");

  TF_RETURN_IF_ERROR(parser.Parse(argc, argv));

  base::Uint8VectorBuffer write_buf;

  switch (field) {
    case Field::kBn254Fr:
      TF_ASSIGN_OR_RETURN(
          write_buf, GenerateScalars<zk_dtypes::bn254::Fr>(degree, condensed));
      break;
  }

  std::string_view output_string(
      reinterpret_cast<const char*>(write_buf.buffer()),
      write_buf.buffer_len());

  TF_RETURN_IF_ERROR(
      tsl::WriteStringToFile(tsl::Env::Default(), output_path, output_string));

  return absl::OkStatus();
}

}  // namespace zkx::benchmark

int main(int argc, char** argv) {
  absl::Status status = zkx::benchmark::RealMain(argc, argv);
  if (!status.ok()) {
    std::cerr << status.message() << std::endl;
    return 1;
  }
  return 0;
}
