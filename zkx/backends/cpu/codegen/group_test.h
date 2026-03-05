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

#ifndef ZKX_BACKENDS_CPU_CODEGEN_GROUP_TEST_H_
#define ZKX_BACKENDS_CPU_CODEGEN_GROUP_TEST_H_

#include <string_view>

#include "absl/strings/substitute.h"

#include "zkx/array2d.h"
#include "zkx/backends/cpu/codegen/cpu_kernel_emitter_test.h"
#include "zkx/base/containers/container_util.h"
#include "zkx/base/random.h"
#include "zkx/comparison_util.h"
#include "zkx/literal_util.h"
#include "zkx/primitive_util.h"

namespace zkx::cpu {

template <typename AffinePoint>
class GroupScalarUnaryTest : public CpuKernelEmitterTest {
 public:
  using JacobianPoint = typename AffinePoint::JacobianPoint;

  void SetUp() override {
    CpuKernelEmitterTest::SetUp();
    x_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<AffinePoint>());
    ret_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<JacobianPoint>());
    x_ = AffinePoint::Random();
    literals_.push_back(LiteralUtil::CreateR0<AffinePoint>(x_));
  }

 protected:
  void SetUpConvert() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)

        ROOT %ret = $1[] convert(%x)
      }
    )",
                                 x_typename_, ret_typename_);
    expected_literal_ = LiteralUtil::CreateR0<JacobianPoint>(x_.ToJacobian());
  }

  void SetUpNegate() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)

        ROOT %ret = $0[] negate(%x)
      }
    )",
                                 x_typename_);
    expected_literal_ = LiteralUtil::CreateR0<AffinePoint>(-x_);
  }

 private:
  std::string_view ret_typename_;
  AffinePoint x_;
};

template <typename AffinePoint>
class GroupScalarBinaryTest : public CpuKernelEmitterTest {
 public:
  using JacobianPoint = typename AffinePoint::JacobianPoint;
  using ScalarField = typename AffinePoint::ScalarField;

  void SetUp() override {
    CpuKernelEmitterTest::SetUp();
    x_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<AffinePoint>());
    ret_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<JacobianPoint>());
    x_ = AffinePoint::Random();
    y_ = AffinePoint::Random();
    literals_.push_back(LiteralUtil::CreateR0<AffinePoint>(x_));
    literals_.push_back(LiteralUtil::CreateR0<AffinePoint>(y_));
  }

 protected:
  void SetUpAdd() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = $1[] add(%x, %y)
      }
    )",
                                 x_typename_, ret_typename_);
    expected_literal_ = LiteralUtil::CreateR0<JacobianPoint>(x_ + y_);
  }

  void SetUpCompare() {
    ComparisonDirection direction = RandomEqualityComparisonDirection();
    std::string direction_str = ComparisonDirectionToString(direction);

    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = pred[] compare(%x, %y), direction=$1
      }
    )",
                                 x_typename_, direction_str);
    switch (direction) {
      case ComparisonDirection::kEq:
        expected_literal_ = LiteralUtil::CreateR0<bool>(x_ == y_);
        break;
      case ComparisonDirection::kNe:
        expected_literal_ = LiteralUtil::CreateR0<bool>(x_ != y_);
        break;
      default:
        LOG(FATAL) << "Unexpected comparison direction: " << direction_str;
        break;
    }
  }

  void SetUpDouble() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)

        ROOT %ret = $1[] add(%x, %x)
      }
    )",
                                 x_typename_, ret_typename_);
    literals_.pop_back();
    expected_literal_ = LiteralUtil::CreateR0<JacobianPoint>(x_ + x_);
  }

  void SetUpScalarMul() {
    hlo_text_ = absl::Substitute(
        R"(
      ENTRY %main {
        %x = $0[] parameter(0)
        %y = $1[] parameter(1)

        ROOT %ret = $2[] multiply(%x, %y)
      }
    )",
        primitive_util::LowercasePrimitiveTypeName(
            primitive_util::NativeToPrimitiveType<ScalarField>()),
        primitive_util::LowercasePrimitiveTypeName(
            primitive_util::NativeToPrimitiveType<AffinePoint>()),
        ret_typename_);
    auto x = ScalarField::Random();
    literals_[0] = LiteralUtil::CreateR0<ScalarField>(x);
    expected_literal_ = LiteralUtil::CreateR0<JacobianPoint>(x * y_);
  }

  void SetUpSub() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = $1[] subtract(%x, %y)
      }
    )",
                                 x_typename_, ret_typename_);
    expected_literal_ = LiteralUtil::CreateR0<JacobianPoint>(x_ - y_);
  }

 private:
  std::string_view ret_typename_;
  AffinePoint x_;
  AffinePoint y_;
};

template <typename AffinePoint>
class GroupScalarTernaryTest : public CpuKernelEmitterTest {
 public:
  void SetUp() override {
    CpuKernelEmitterTest::SetUp();
    x_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<AffinePoint>());
  }

 protected:
  void SetUpSelect() {
    AffinePoint x = AffinePoint::Random();
    AffinePoint y = AffinePoint::Random();
    bool cond = base::Uniform<uint32_t>() % 2 == 0;
    literals_.push_back(LiteralUtil::CreateR0<bool>(cond));
    literals_.push_back(LiteralUtil::CreateR0<AffinePoint>(x));
    literals_.push_back(LiteralUtil::CreateR0<AffinePoint>(y));

    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %cond = pred[] parameter(0)
        %x = $0[] parameter(1)
        %y = $0[] parameter(2)

        ROOT %ret = $0[] select(%cond, %x, %y)
      }
    )",
                                 x_typename_);
    expected_literal_ = LiteralUtil::CreateR0<AffinePoint>(cond ? x : y);
  }
};

template <typename AffinePoint>
class GroupR2TensorUnaryTest : public CpuKernelEmitterTest {
 public:
  constexpr static int64_t M = 2;
  constexpr static int64_t N = 3;

  void SetUp() override {
    CpuKernelEmitterTest::SetUp();
    x_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<AffinePoint>());
    x_ = base::CreateVector(M, []() {
      return base::CreateVector(N, []() { return AffinePoint::Random(); });
    });
    Array2D<AffinePoint> x_array(M, N);
    for (int64_t i = 0; i < M; ++i) {
      for (int64_t j = 0; j < N; ++j) {
        x_array({i, j}) = x_[i][j];
      }
    }
    literals_.push_back(LiteralUtil::CreateR2FromArray2D(x_array));
  }

 protected:
  void Verify(const Literal& ret_literal) const override {
    for (int64_t i = 0; i < M; ++i) {
      for (int64_t j = 0; j < N; ++j) {
        EXPECT_EQ(ret_literal.Get<AffinePoint>({i, j}), expected_[i][j]);
      }
    }
  }

  void SetUpNegate() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[$1, $2] parameter(0)

        ROOT %ret = $0[$1, $2] negate(%x)
      }
    )",
                                 x_typename_, M, N);
    expected_ = base::CreateVector(M, [this](size_t i) {
      return base::CreateVector(N, [this, i](size_t j) { return -x_[i][j]; });
    });
  }

 private:
  std::vector<std::vector<AffinePoint>> x_;
  std::vector<std::vector<AffinePoint>> expected_;
};

template <typename AffinePoint>
class GroupR2TensorBinaryTest : public CpuKernelEmitterTest {
 public:
  constexpr static int64_t M = 2;
  constexpr static int64_t N = 3;

  using JacobianPoint = typename AffinePoint::JacobianPoint;

  void SetUp() override {
    CpuKernelEmitterTest::SetUp();
    x_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<AffinePoint>());
    ret_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<JacobianPoint>());
    x_ = base::CreateVector(M, []() {
      return base::CreateVector(N, []() { return AffinePoint::Random(); });
    });
    y_ = base::CreateVector(M, []() {
      return base::CreateVector(N, []() { return AffinePoint::Random(); });
    });
    Array2D<AffinePoint> x_array(M, N);
    Array2D<AffinePoint> y_array(M, N);
    for (int64_t i = 0; i < M; ++i) {
      for (int64_t j = 0; j < N; ++j) {
        x_array({i, j}) = x_[i][j];
        y_array({i, j}) = y_[i][j];
      }
    }
    literals_.push_back(LiteralUtil::CreateR2FromArray2D(x_array));
    literals_.push_back(LiteralUtil::CreateR2FromArray2D(y_array));
  }

 protected:
  void Verify(const Literal& ret_literal) const override {
    for (int64_t i = 0; i < M; ++i) {
      for (int64_t j = 0; j < N; ++j) {
        EXPECT_EQ(ret_literal.Get<JacobianPoint>({i, j}), expected_[i][j]);
      }
    }
  }

  void SetUpAdd() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[$2, $3] parameter(0)
        %y = $0[$2, $3] parameter(1)

        ROOT %ret = $1[$2, $3] add(%x, %y)
      }
    )",
                                 x_typename_, ret_typename_, M, N);
    expected_ = base::CreateVector(M, [this](size_t i) {
      return base::CreateVector(
          N, [this, i](size_t j) { return x_[i][j] + y_[i][j]; });
    });
  }

 private:
  std::string_view ret_typename_;
  std::vector<std::vector<AffinePoint>> x_;
  std::vector<std::vector<AffinePoint>> y_;
  std::vector<std::vector<JacobianPoint>> expected_;
};

// ---------------------------------------------------------------------------
// Bitcast-convert tests: EC ↔ integer round-trip.
// Uses unsigned integer type (not base field) as intermediate since
// BaseField (e.g., bn254::FqMont) may not be in the public type list.
// ---------------------------------------------------------------------------

template <typename AffinePoint>
class GroupBitcastConvertTest : public CpuKernelEmitterTest {
 public:
  using BaseField = typename AffinePoint::BaseField;
  constexpr static int64_t N = 4;
  constexpr static int64_t D = sizeof(AffinePoint) / sizeof(BaseField);

  void SetUp() override {
    CpuKernelEmitterTest::SetUp();
    point_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<AffinePoint>());
    auto int_type =
        primitive_util::UnsignedIntegralTypeForBitWidth(sizeof(BaseField) * 8);
    int_typename_ = primitive_util::LowercasePrimitiveTypeName(int_type);
    x_ = base::CreateVector(N, []() { return AffinePoint::Random(); });
    literals_.push_back(LiteralUtil::CreateR1<AffinePoint>(x_));
  }

 protected:
  // AffinePoint[N] → uXXX[N,D] bitcast-convert → AffinePoint[N]
  // bitcast-convert: round-trip.
  void SetUpEcToIntRoundTrip() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %p = $0[$2] parameter(0)
        %bc1 = $1[$2,$3] bitcast-convert(%p)
        ROOT %ret = $0[$2] bitcast-convert(%bc1)
      }
    )",
                                 point_typename_, int_typename_, N, D);
    expected_literal_ = LiteralUtil::CreateR1<AffinePoint>(x_);
  }

 private:
  std::string_view point_typename_;
  std::string_view int_typename_;
  std::vector<AffinePoint> x_;
};

}  // namespace zkx::cpu

#endif  // ZKX_BACKENDS_CPU_CODEGEN_GROUP_TEST_H_
