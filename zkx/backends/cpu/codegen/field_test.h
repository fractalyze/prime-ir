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

#ifndef ZKX_BACKENDS_CPU_CODEGEN_FIELD_TEST_H_
#define ZKX_BACKENDS_CPU_CODEGEN_FIELD_TEST_H_

#include <algorithm>
#include <optional>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/substitute.h"

#include "xla/tsl/platform/status.h"
#include "zk_dtypes/include/batch_inverse.h"
#include "zk_dtypes/include/field/root_of_unity.h"
#include "zkx/array2d.h"
#include "zkx/backends/cpu/codegen/cpu_kernel_emitter_test.h"
#include "zkx/base/containers/container_util.h"
#include "zkx/comparison_util.h"
#include "zkx/literal_util.h"
#include "zkx/math/base/sparse_matrix.h"
#include "zkx/math/field/prime_field_serde.h"
#include "zkx/primitive_util.h"

namespace zkx::cpu {

template <typename F>
class FieldScalarUnaryTest : public CpuKernelEmitterTest {
 public:
  void SetUp() override {
    CpuKernelEmitterTest::SetUp();
    x_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<F>());
    x_std_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<typename F::StdType>());
    x_ = F::Random();
    literals_.push_back(LiteralUtil::CreateR0<F>(x_));
  }

 protected:
  void SetUpConvertFromInt() {
    hlo_text_ = absl::Substitute(R"(
    ENTRY %main {
      %x = u32[] parameter(0)

      ROOT %ret = $0[] convert(%x)
    }
  )",
                                 x_typename_);
    uint32_t x;
    if (F::Config::kModulus < std::numeric_limits<uint32_t>::max()) {
      x = base::Uniform<uint32_t>(
          0, static_cast<uint32_t>(static_cast<uint64_t>(F::Config::kModulus)));
    } else {
      x = base::Uniform<uint32_t>();
    }
    literals_[0] = LiteralUtil::CreateR0<uint32_t>(x);
    expected_literal_ = LiteralUtil::CreateR0<F>(F(x));
  }

  void SetUpConvertFromIntToStd() {
    using FStd = typename F::StdType;

    hlo_text_ = absl::Substitute(R"(
    ENTRY %main {
      %x = u32[] parameter(0)

      ROOT %ret = $0[] convert(%x)
    }
  )",
                                 x_std_typename_);
    uint32_t x;
    if (F::Config::kModulus < std::numeric_limits<uint32_t>::max()) {
      x = base::Uniform<uint32_t>(
          0, static_cast<uint32_t>(static_cast<uint64_t>(F::Config::kModulus)));
    } else {
      x = base::Uniform<uint32_t>();
    }
    literals_[0] = LiteralUtil::CreateR0<uint32_t>(x);
    expected_literal_ = LiteralUtil::CreateR0<FStd>(FStd(x));
  }

  void SetUpConvertToStd() {
    using FStd = typename F::StdType;

    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)

        ROOT %ret = $1[] convert(%x)
      }
    )",
                                 x_typename_, x_std_typename_);
    expected_literal_ = LiteralUtil::CreateR0<FStd>(x_.MontReduce());
  }

  void SetUpNegate() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)

        ROOT %ret = $0[] negate(%x)
      }
    )",
                                 x_typename_);
    expected_literal_ = LiteralUtil::CreateR0<F>(-x_);
  }

  void SetUpInverse() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)

        ROOT %ret = $0[] inverse(%x)
      }
    )",
                                 x_typename_);
    if (x_.IsZero()) {
      expected_literal_ = LiteralUtil::CreateR0<F>(0);
    } else {
      expected_literal_ = LiteralUtil::CreateR0<F>(x_.Inverse());
    }
  }

 private:
  F x_;
};

template <typename F>
class FieldScalarBinaryTest : public CpuKernelEmitterTest {
 public:
  void SetUp() override {
    CpuKernelEmitterTest::SetUp();
    x_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<F>());
    x_ = F::Random();
    y_ = F::Random();
    literals_.push_back(LiteralUtil::CreateR0<F>(x_));
    literals_.push_back(LiteralUtil::CreateR0<F>(y_));
  }

 protected:
  void SetUpAdd() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = $0[] add(%x, %y)
      }
    )",
                                 x_typename_);
    expected_literal_ = LiteralUtil::CreateR0<F>(x_ + y_);
  }

  void SetUpCompare() {
    ComparisonDirection direction = RandomComparisonDirection();
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
      case ComparisonDirection::kGe:
        expected_literal_ = LiteralUtil::CreateR0<bool>(x_ >= y_);
        break;
      case ComparisonDirection::kGt:
        expected_literal_ = LiteralUtil::CreateR0<bool>(x_ > y_);
        break;
      case ComparisonDirection::kLe:
        expected_literal_ = LiteralUtil::CreateR0<bool>(x_ <= y_);
        break;
      case ComparisonDirection::kLt:
        expected_literal_ = LiteralUtil::CreateR0<bool>(x_ < y_);
        break;
    }
  }

  void SetUpDiv() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = $0[] divide(%x, %y)
      }
    )",
                                 x_typename_);
    if (y_.IsZero()) {
      expected_literal_ = LiteralUtil::CreateR0<F>(0);
    } else {
      expected_literal_ = LiteralUtil::CreateR0<F>(x_ / y_);
    }
  }

  void SetUpDouble() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)

        ROOT %ret = $0[] add(%x, %x)
      }
    )",
                                 x_typename_);
    literals_.pop_back();
    expected_literal_ = LiteralUtil::CreateR0<F>(x_.Double());
  }

  void SetUpFusion() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)
        %c = $0[] constant(2)

        %add = $0[] add(%x, %y)
        %mul = $0[] multiply(%add, %x)
        ROOT %ret = $0[] multiply(%mul, %c)
      }
    )",
                                 x_typename_);
    expected_literal_ = LiteralUtil::CreateR0<F>((x_ + y_) * x_ * 2);
  }

  void SetUpMaximum() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = $0[] maximum(%x, %y)
      }
    )",
                                 x_typename_);
    expected_literal_ = LiteralUtil::CreateR0<F>(std::max(x_, y_));
  }

  void SetUpMinimum() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = $0[] minimum(%x, %y)
      }
    )",
                                 x_typename_);
    expected_literal_ = LiteralUtil::CreateR0<F>(std::min(x_, y_));
  }

  void SetUpMul() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = $0[] multiply(%x, %y)
      }
    )",
                                 x_typename_);
    expected_literal_ = LiteralUtil::CreateR0<F>(x_ * y_);
  }

  void SetUpPow() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)
        %y = u32[] parameter(1)

        ROOT %ret = $0[] power(%x, %y)
      }
    )",
                                 x_typename_);

    auto y = base::Uniform<uint32_t>();
    literals_[1] = LiteralUtil::CreateR0<uint32_t>(y);
    expected_literal_ = LiteralUtil::CreateR0<F>(x_.Pow(y));
  }

  void SetUpPowWithSignedExponentShouldFail() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)
        %y = s32[] parameter(1)

        ROOT %ret = $0[] power(%x, %y)
      }
    )",
                                 x_typename_);
    expected_status_code_ = absl::StatusCode::kInvalidArgument;
  }

  void SetUpSquare() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)

        ROOT %ret = $0[] multiply(%x, %x)
      }
    )",
                                 x_typename_);
    literals_.pop_back();
    expected_literal_ = LiteralUtil::CreateR0<F>(x_.Square());
  }

  void SetUpSub() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = $0[] subtract(%x, %y)
      }
    )",
                                 x_typename_);
    expected_literal_ = LiteralUtil::CreateR0<F>(x_ - y_);
  }

 private:
  F x_;
  F y_;
};

template <typename F>
class FieldScalarTernaryTest : public CpuKernelEmitterTest {
 public:
  void SetUp() override {
    CpuKernelEmitterTest::SetUp();
    x_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<F>());
    x_ = F::Random();
    y_ = F::Random();
    z_ = F::Random();
    literals_.push_back(LiteralUtil::CreateR0<F>(x_));
    literals_.push_back(LiteralUtil::CreateR0<F>(y_));
    literals_.push_back(LiteralUtil::CreateR0<F>(z_));
  }

 protected:
  void SetUpClamp() {
    if (x_ > z_) {
      std::swap(x_, z_);
      std::swap(literals_[0], literals_[2]);
    }
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %min = $0[] parameter(0)
        %operand = $0[] parameter(1)
        %max = $0[] parameter(2)

        ROOT %ret = $0[] clamp(%min, %operand, %max)
      }
    )",
                                 x_typename_);
    expected_literal_ = LiteralUtil::CreateR0<F>(std::clamp(y_, x_, z_));
  }

  void SetUpSelect() {
    bool cond = static_cast<uint64_t>(x_.value()) % 2 == 0;
    literals_[0] = LiteralUtil::CreateR0<bool>(cond);
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %cond = pred[] parameter(0)
        %x = $0[] parameter(1)
        %y = $0[] parameter(2)

        ROOT %ret = $0[] select(%cond, %x, %y)
      }
    )",
                                 x_typename_);
    expected_literal_ = LiteralUtil::CreateR0<F>(cond ? y_ : z_);
  }

 private:
  F x_;
  F y_;
  F z_;
};

template <typename F>
class FieldR1TensorUnaryTest : public CpuKernelEmitterTest {
 public:
  constexpr static int64_t N = 4;

  void SetUp() override {
    CpuKernelEmitterTest::SetUp();
    x_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<F>());
    x_ = base::CreateVector(N, []() { return F::Random(); });
    literals_.push_back(LiteralUtil::CreateR1<F>(x_));
  }

 protected:
  void Verify(const Literal& ret_literal) const override {
    for (int64_t i = 0; i < N; ++i) {
      if (expected_[i].has_value()) {
        EXPECT_EQ(ret_literal.Get<F>({i}), expected_[i].value());
      }
    }
  }

  void SetUpBatchInverse() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[$1] parameter(0)

        ROOT %ret = $0[$1] inverse(%x)
      }
    )",
                                 x_typename_, N);

    std::vector<F> expected;
    TF_ASSERT_OK(zk_dtypes::BatchInverse(x_, &expected));

    expected_ = base::Map(expected, [](const absl::StatusOr<F>& x) {
      return std::optional<F>(x.value());
    });
  }

  void SetUpFFT() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[$1] parameter(0)

        ROOT %ret = $0[$1] fft(%x), fft_type=FFT, fft_length=$1
      }
    )",
                                 x_typename_, N);
    expected_ = base::Map(ComputeFFT(x_),
                          [](const F& x) { return std::optional<F>(x); });
  }

  void SetUpFFTWithTwiddles() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[$1] parameter(0)
        %y = $0[$1] parameter(1)

        ROOT %ret = $0[$1] fft(%x, %y), fft_type=FFT, fft_length=$1
      }
    )",
                                 x_typename_, N);
    std::vector<F> twiddles = ComputeTwiddles(x_);
    literals_.push_back(LiteralUtil::CreateR1<F>(twiddles));
    expected_ = base::Map(ComputeFFT(x_, twiddles),
                          [](const F& x) { return std::optional<F>(x); });
  }

  void SetUpIFFT() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[$1] parameter(0)

        ROOT %ret = $0[$1] fft(%x), fft_type=IFFT, fft_length=$1
      }
    )",
                                 x_typename_, N);
    expected_ = base::Map(ComputeInverseFFT(x_),
                          [](const F& x) { return std::optional<F>(x); });
  }

  void SetUpIFFTWithTwiddles() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[$1] parameter(0)
        %y = $0[$1] parameter(1)

        ROOT %ret = $0[$1] fft(%x, %y), fft_type=IFFT, fft_length=$1
      }
    )",
                                 x_typename_, N);
    std::vector<F> twiddles = ComputeInverseTwiddles(x_);
    literals_.push_back(LiteralUtil::CreateR1<F>(twiddles));
    expected_ = base::Map(ComputeInverseFFT(x_, twiddles),
                          [](const F& x) { return std::optional<F>(x); });
  }

 private:
  std::vector<F> ComputeTwiddles(const std::vector<F>& x) {
    absl::StatusOr<F> w_or = zk_dtypes::GetRootOfUnity<F>(x.size());
    CHECK_OK(w_or);
    F w = w_or.value();
    std::vector<F> twiddles(x.size());
    F twiddle = 1;
    for (int64_t i = 0; i < x.size(); ++i) {
      twiddles[i] = twiddle;
      twiddle = twiddle * w;
    }
    return twiddles;
  }

  std::vector<F> ComputeFFT(const std::vector<F>& x,
                            const std::vector<F>& twiddles) {
    std::vector<F> ret(x.size());
    for (int64_t i = 0; i < x.size(); ++i) {
      F v = 0;
      for (int64_t j = 0; j < x.size(); ++j) {
        v += x[j] * twiddles[(i * j) % x.size()];
      }
      ret[i] = v;
    }
    return ret;
  }

  std::vector<F> ComputeFFT(const std::vector<F>& x) {
    return ComputeFFT(x, ComputeTwiddles(x));
  }

  std::vector<F> ComputeInverseTwiddles(const std::vector<F>& x) {
    std::vector<F> twiddles = ComputeTwiddles(x);
    CHECK_OK(zk_dtypes::BatchInverse(twiddles, &twiddles));
    return twiddles;
  }

  std::vector<F> ComputeInverseFFT(const std::vector<F>& x,
                                   const std::vector<F>& twiddles) {
    F n_inv = F(x.size()).Inverse();
    std::vector<F> ret(x.size());
    for (int64_t i = 0; i < x.size(); ++i) {
      F v = 0;
      for (int64_t j = 0; j < x.size(); ++j) {
        v += x[j] * twiddles[(i * j) % x.size()];
      }
      ret[i] = v * n_inv;
    }
    return ret;
  }

  std::vector<F> ComputeInverseFFT(const std::vector<F>& x) {
    return ComputeInverseFFT(x, ComputeInverseTwiddles(x));
  }

  std::vector<F> x_;
  // TODO(chokobole): PrimeIR BatchInverse returns garbage value when the input
  // value is zero, which behaves differently from single inverse. If the
  // batch inverse also returns zero output, we should compare the result with
  // Literal.
  std::vector<std::optional<F>> expected_;
};

template <typename F>
class FieldR2TensorBinaryTest : public CpuKernelEmitterTest {
 public:
  constexpr static int64_t M = 2;
  constexpr static int64_t N = 3;

  void SetUp() override {
    CpuKernelEmitterTest::SetUp();
    x_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<F>());
    x_ = base::CreateVector(M, []() {
      return base::CreateVector(N, []() { return F::Random(); });
    });
    y_ = base::CreateVector(M, []() {
      return base::CreateVector(N, []() { return F::Random(); });
    });
    Array2D<F> x_array(M, N);
    Array2D<F> y_array(M, N);
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
        EXPECT_EQ(ret_literal.Get<F>({i, j}), expected_[i][j]);
      }
    }
  }

  void SetUpAdd() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[$1, $2] parameter(0)
        %y = $0[$1, $2] parameter(1)

        ROOT %ret = $0[$1, $2] add(%x, %y)
      }
    )",
                                 x_typename_, M, N);

    expected_ = base::CreateVector(M, [this](size_t i) {
      return base::CreateVector(
          N, [this, i](size_t j) { return x_[i][j] + y_[i][j]; });
    });
  }

 private:
  std::vector<std::vector<F>> x_;
  std::vector<std::vector<F>> y_;
  std::vector<std::vector<F>> expected_;
};

template <typename F>
class FieldTest : public CpuKernelEmitterTest {
 public:
  void SetUp() override {
    CpuKernelEmitterTest::SetUp();
    x_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<F>());
  }

 protected:
  void SetUpCSRMatrixVectorMultiplication() {
    constexpr static int64_t M = 4;
    constexpr static int64_t N = 3;
    constexpr static int64_t NNZ = 8;

    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[$1, $2]{1,0:D(D, C) NNZ($3)} parameter(0)
        %y = $0[$2] parameter(1)

        ROOT %ret = $0[$1] dot(%x, %y)
      }
    )",
                                 x_typename_, M, N, NNZ);

    math::SparseMatrix<F> x = math::SparseMatrix<F>::Random(M, N, NNZ);

    std::vector<uint32_t> x_row_ptrs, x_col_indices;
    std::vector<F> x_values;
    x.ToCSR(x_row_ptrs, x_col_indices, x_values);

    TF_ASSERT_OK_AND_ASSIGN(std::vector<uint8_t> x_buffer, x.ToCSRBuffer());
    std::vector<F> y = base::CreateVector(N, []() { return F::Random(); });

    literals_.push_back(LiteralUtil::CreateR1<uint8_t>(x_buffer));
    literals_.push_back(LiteralUtil::CreateR1<F>(y));

    std::vector<F> expected(x_row_ptrs.size() - 1);
    for (int64_t i = 0; i < expected.size(); ++i) {
      for (int64_t j = x_row_ptrs[i]; j < x_row_ptrs[i + 1]; ++j) {
        expected[i] += x_values[j] * y[x_col_indices[j]];
      }
    }
    expected_literal_ = LiteralUtil::CreateR1<F>(expected);
  }

  void SetUpDynamicUpdateSliceBug() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[16] parameter(0)
        %update = $0[1] parameter(1)
        %offset = s32[] parameter(2)

        ROOT %ret = $0[16] dynamic-update-slice(%x, %update, %offset)
      }
    )",
                                 x_typename_);
    size_t size = 16;
    int32_t offset = 3;
    std::vector<F> x = base::CreateVector(size, []() { return F::Random(); });
    std::vector<F> update = base::CreateVector(1, []() { return F::Random(); });
    literals_.push_back(LiteralUtil::CreateR1<F>(x));
    literals_.push_back(LiteralUtil::CreateR1<F>(update));
    literals_.push_back(LiteralUtil::CreateR0<int32_t>(offset));
    std::vector<F> expected = base::CreateVector(size, [&](size_t i) {
      if (i == offset) {
        return update[0];
      } else {
        return x[i];
      }
    });
    expected_literal_ = LiteralUtil::CreateR1<F>(expected);
  }
};

}  // namespace zkx::cpu

#endif  // ZKX_BACKENDS_CPU_CODEGEN_FIELD_TEST_H_
