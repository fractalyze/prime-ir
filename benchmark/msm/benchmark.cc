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

// clang-format off
#include "benchmark/curve_flag.h"
#include "benchmark/runner.h"
// clang-format on
#include "zk_dtypes/include/elliptic_curve/bn/bn254/g1.h"
#include "zk_dtypes/include/elliptic_curve/bn/bn254/g2.h"
#include "zkx/literal_util.h"
#include "zkx/math/field/prime_field_serde.h"

namespace zkx::benchmark {

std::string GenerateHlo(Curve curve, uint32_t degree) {
  size_t size = size_t{1} << degree;
  return absl::Substitute(
      R"(
     ENTRY %main {
       %x = $1[$0] parameter(0)
       %y = $2_affine[$0] parameter(1)

       %ret.xyzz = $2_xyzz[] msm(%x, %y)
       ROOT %ret = $2_affine[] convert(%ret.xyzz)
     }
     )",
      size, CurveToScalarFieldHloString(curve), CurveToHloString(curve));
}

template <typename ScalarField, typename AffinePoint>
std::vector<Literal> GenerateLiterals(absl::Span<const ScalarField> scalars,
                                      absl::Span<const AffinePoint> bases) {
  std::vector<Literal> literals;
  literals.push_back(LiteralUtil::CreateR1<ScalarField>(scalars));
  literals.push_back(LiteralUtil::CreateR1<AffinePoint>(bases));
  return literals;
}

template <typename AffinePoint>
absl::Status RunBenchmark(Runner& runner, Curve curve) {
  using ScalarField = typename AffinePoint::ScalarField;

  std::vector<AffinePoint> generators(runner.GetMaxSize(),
                                      AffinePoint::Generator());

  TF_RETURN_IF_ERROR(runner.Run<ScalarField>(
      [curve](uint32_t degree) { return GenerateHlo(curve, degree); },
      [&generators](absl::Span<const ScalarField> scalars) {
        return GenerateLiterals(
            scalars,
            absl::MakeConstSpan(generators).subspan(0, scalars.size()));
      }));
  runner.PrintResults();
  return runner.MaybeWriteCsv();
}

absl::Status RealMain(int argc, char** argv) {
  Runner runner;
  Curve curve;
  AddCurveFlag(runner.parser(), &curve);
  runner.AddPositionalFlags();
  runner.AddOptionalFlags(Runner::Options{
      .multi_degrees = true,
  });

  TF_RETURN_IF_ERROR(runner.Parse(argc, argv));

  switch (curve) {
    case Curve::kBn254G1:
      return RunBenchmark<zk_dtypes::bn254::G1AffinePoint>(runner, curve);
    case Curve::kBn254G2:
      return RunBenchmark<zk_dtypes::bn254::G2AffinePoint>(runner, curve);
  }
  ABSL_UNREACHABLE();
  return absl::InternalError("Invalid curve");
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
