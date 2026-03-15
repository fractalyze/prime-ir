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

#ifndef ZKX_BACKENDS_GPU_CODEGEN_FIELD_TEST_H_
#define ZKX_BACKENDS_GPU_CODEGEN_FIELD_TEST_H_

#include "absl/strings/substitute.h"

#include "zkx/array2d.h"
#include "zkx/backends/gpu/codegen/cuda_kernel_emitter_test.h"
#include "zkx/base/containers/container_util.h"
#include "zkx/base/random.h"
#include "zkx/literal_util.h"
#include "zkx/primitive_util.h"

namespace zkx::gpu {

template <typename F>
class FieldScalarUnaryTest : public CudaKernelEmitterTest {
 public:
  void SetUp() override {
    CudaKernelEmitterTest::SetUp();
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
      %f {
        %x = u32[] parameter(0)

        ROOT %ret = $0[] convert(%x)
      }

      ENTRY %main {
        %x = u32[] parameter(0)

        ROOT %ret = $0[] fusion(%x), kind=kLoop, calls=%f
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
      %f {
        %x = u32[] parameter(0)

        ROOT %ret = $0[] convert(%x)
      }

      ENTRY %main {
        %x = u32[] parameter(0)

        ROOT %ret = $0[] fusion(%x), kind=kLoop, calls=%f
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
      %f {
        %x = $0[] parameter(0)

        ROOT %ret = $1[] convert(%x)
      }

      ENTRY %main {
        %x = $0[] parameter(0)

        ROOT %ret = $1[] fusion(%x), kind=kLoop, calls=%f
      }
    )",
                                 x_typename_, x_std_typename_);
    expected_literal_ = LiteralUtil::CreateR0<FStd>(x_.MontReduce());
  }

  void SetUpNegate() {
    hlo_text_ = absl::Substitute(R"(
      %f {
        %x = $0[] parameter(0)

        ROOT %ret = $0[] negate(%x)
      }

      ENTRY %main {
        %x = $0[] parameter(0)

        ROOT %ret = $0[] fusion(%x), kind=kLoop, calls=%f
      }
    )",
                                 x_typename_);
    expected_literal_ = LiteralUtil::CreateR0<F>(-x_);
  }

 private:
  F x_;
};

template <typename F>
class FieldScalarBinaryTest : public CudaKernelEmitterTest {
 public:
  void SetUp() override {
    CudaKernelEmitterTest::SetUp();
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
      %f {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = $0[] add(%x, %y)
      }

      ENTRY %main {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = $0[] fusion(%x, %y), kind=kLoop, calls=%f
      }
    )",
                                 x_typename_);
    expected_literal_ = LiteralUtil::CreateR0<F>(x_ + y_);
  }

  void SetUpCompareEq() {
    hlo_text_ = absl::Substitute(R"(
      %f {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = pred[] compare(%x, %y), direction=EQ
      }

      ENTRY %main {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = pred[] fusion(%x, %y), kind=kLoop, calls=%f
      }
    )",
                                 x_typename_);
    expected_literal_ = LiteralUtil::CreateR0<bool>(x_ == y_);
  }

  void SetUpCompareNe() {
    hlo_text_ = absl::Substitute(R"(
      %f {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = pred[] compare(%x, %y), direction=NE
      }

      ENTRY %main {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = pred[] fusion(%x, %y), kind=kLoop, calls=%f
      }
    )",
                                 x_typename_);
    expected_literal_ = LiteralUtil::CreateR0<bool>(x_ != y_);
  }

  void SetUpDiv() {
    hlo_text_ = absl::Substitute(R"(
      %f {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = $0[] divide(%x, %y)
      }

      ENTRY %main {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = $0[] fusion(%x, %y), kind=kLoop, calls=%f
      }
    )",
                                 x_typename_);
    if (y_.IsZero()) {
      expected_literal_ = LiteralUtil::CreateR0<F>(F::Zero());
    } else {
      expected_literal_ = LiteralUtil::CreateR0<F>(x_ / y_);
    }
  }

  void SetUpDouble() {
    hlo_text_ = absl::Substitute(R"(
      %f {
        %x = $0[] parameter(0)

        ROOT %ret = $0[] add(%x, %x)
      }

      ENTRY %main {
        %x = $0[] parameter(0)

        ROOT %ret = $0[] fusion(%x), kind=kLoop, calls=%f
      }
    )",
                                 x_typename_);
    literals_.pop_back();
    expected_literal_ = LiteralUtil::CreateR0<F>(x_.Double());
  }

  void SetUpSub() {
    hlo_text_ = absl::Substitute(R"(
      %f {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = $0[] subtract(%x, %y)
      }

      ENTRY %main {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = $0[] fusion(%x, %y), kind=kLoop, calls=%f
      }
    )",
                                 x_typename_);
    expected_literal_ = LiteralUtil::CreateR0<F>(x_ - y_);
  }

  void SetUpPow() {
    hlo_text_ = absl::Substitute(R"(
      %f {
        %x = $0[] parameter(0)
        %y = u32[] parameter(1)

        ROOT %ret = $0[] power(%x, %y)
      }

      ENTRY %main {
        %x = $0[] parameter(0)
        %y = u32[] parameter(1)

        ROOT %ret = $0[] fusion(%x, %y), kind=kLoop, calls=%f
      }
    )",
                                 x_typename_);

    auto y = base::Uniform<uint32_t>();
    literals_[1] = LiteralUtil::CreateR0<uint32_t>(y);
    expected_literal_ = LiteralUtil::CreateR0<F>(x_.Pow(y));
  }

  void SetUpMul() {
    hlo_text_ = absl::Substitute(R"(
      %f {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = $0[] multiply(%x, %y)
      }

      ENTRY %main {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = $0[] fusion(%x, %y), kind=kLoop, calls=%f
      }
    )",
                                 x_typename_);
    expected_literal_ = LiteralUtil::CreateR0<F>(x_ * y_);
  }

  void SetUpSquare() {
    hlo_text_ = absl::Substitute(R"(
      %f {
        %x = $0[] parameter(0)

        ROOT %ret = $0[] multiply(%x, %x)
      }

      ENTRY %main {
        %x = $0[] parameter(0)

        ROOT %ret = $0[] fusion(%x), kind=kLoop, calls=%f
      }
    )",
                                 x_typename_);
    literals_.pop_back();
    expected_literal_ = LiteralUtil::CreateR0<F>(x_.Square());
  }

 private:
  F x_;
  F y_;
};

template <typename F>
class FieldScalarTernaryTest : public CudaKernelEmitterTest {
 public:
  void SetUp() override {
    CudaKernelEmitterTest::SetUp();
    x_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<F>());
    x_ = F::Random();
    y_ = F::Random();
    literals_.emplace_back();  // Placeholder for the predicate.
    literals_.push_back(LiteralUtil::CreateR0<F>(x_));
    literals_.push_back(LiteralUtil::CreateR0<F>(y_));
  }

 protected:
  void SetUpSelectTrue() {
    literals_[0] = LiteralUtil::CreateR0<bool>(true);
    hlo_text_ = absl::Substitute(R"(
      %f {
        %cond = pred[] parameter(0)
        %x = $0[] parameter(1)
        %y = $0[] parameter(2)

        ROOT %ret = $0[] select(%cond, %x, %y)
      }

      ENTRY %main {
        %cond = pred[] parameter(0)
        %x = $0[] parameter(1)
        %y = $0[] parameter(2)

        ROOT %ret = $0[] fusion(%cond, %x, %y), kind=kLoop, calls=%f
      }
    )",
                                 x_typename_);
    expected_literal_ = LiteralUtil::CreateR0<F>(x_);
  }

  void SetUpSelectFalse() {
    literals_[0] = LiteralUtil::CreateR0<bool>(false);
    hlo_text_ = absl::Substitute(R"(
      %f {
        %cond = pred[] parameter(0)
        %x = $0[] parameter(1)
        %y = $0[] parameter(2)

        ROOT %ret = $0[] select(%cond, %x, %y)
      }

      ENTRY %main {
        %cond = pred[] parameter(0)
        %x = $0[] parameter(1)
        %y = $0[] parameter(2)

        ROOT %ret = $0[] fusion(%cond, %x, %y), kind=kLoop, calls=%f
      }
    )",
                                 x_typename_);
    expected_literal_ = LiteralUtil::CreateR0<F>(y_);
  }

 private:
  F x_;
  F y_;
};

template <typename F>
class FieldR2TensorBinaryTest : public CudaKernelEmitterTest {
 public:
  constexpr static int64_t M = 2;
  constexpr static int64_t N = 3;

  void SetUp() override {
    CudaKernelEmitterTest::SetUp();
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
      %f {
        %x = $0[$1, $2] parameter(0)
        %y = $0[$1, $2] parameter(1)

        ROOT %ret = $0[$1, $2] add(%x, %y)
      }

      ENTRY %main {
        %x = $0[$1, $2] parameter(0)
        %y = $0[$1, $2] parameter(1)

        ROOT %ret = $0[$1, $2] fusion(%x, %y), kind=kLoop, calls=%f
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
class FieldR2TransposeTest : public CudaKernelEmitterTest {
 public:
  constexpr static int64_t M = 32;
  constexpr static int64_t N = 64;

  void SetUp() override {
    CudaKernelEmitterTest::SetUp();
    x_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<F>());
    x_ = base::CreateVector(M, []() {
      return base::CreateVector(N, []() { return F::Random(); });
    });
    Array2D<F> x_array(M, N);
    for (int64_t i = 0; i < M; ++i) {
      for (int64_t j = 0; j < N; ++j) {
        x_array({i, j}) = x_[i][j];
      }
    }
    literals_.push_back(LiteralUtil::CreateR2FromArray2D(x_array));
  }

 protected:
  void Verify(const Literal& ret_literal) const override {
    for (int64_t i = 0; i < N; ++i) {
      for (int64_t j = 0; j < M; ++j) {
        EXPECT_EQ(ret_literal.Get<F>({i, j}), expected_[i][j]);
      }
    }
  }

  void SetUpTranspose() {
    hlo_text_ = absl::Substitute(R"(
      %f {
        %x = $0[$1, $2]{1, 0} parameter(0)

        ROOT %transpose = $0[$2, $1]{1, 0} transpose(%x), dimensions={1, 0}
      }

      ENTRY %main {
        %x = $0[$1, $2]{1, 0} parameter(0)

        ROOT %ret = $0[$2, $1]{1, 0} fusion(%x), kind=kInput, calls=%f
      }
    )",
                                 x_typename_, M, N);

    expected_ = base::CreateVector(N, [this](size_t i) {
      return base::CreateVector(M, [this, i](size_t j) { return x_[j][i]; });
    });
  }

 private:
  std::vector<std::vector<F>> x_;
  std::vector<std::vector<F>> expected_;
};

template <typename EF>
class ExtFieldBitcastConvertTest : public CudaKernelEmitterTest {
 public:
  using F = typename EF::Config::BaseField;
  constexpr static int64_t N = 2;
  constexpr static int64_t D = sizeof(EF) / sizeof(F);

  void SetUp() override {
    CudaKernelEmitterTest::SetUp();
    ef_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<EF>());
    f_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<F>());
    x_ = base::CreateVector(N, []() { return EF::Random(); });
    literals_.push_back(LiteralUtil::CreateR1<EF>(x_));
  }

 protected:
  void SetUpExtFieldToField() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %p = $0[$2] parameter(0)
        ROOT %ret = $1[$2,$3] bitcast-convert(%p)
      }
    )",
                                 ef_typename_, f_typename_, N, D);

    Array2D<F> expected(N, D);
    for (int64_t i = 0; i < N; ++i) {
      F coeffs[D];
      std::memcpy(coeffs, &x_[i], sizeof(EF));
      for (int64_t j = 0; j < D; ++j) {
        expected({i, j}) = coeffs[j];
      }
    }
    expected_literal_ = LiteralUtil::CreateR2FromArray2D<F>(expected);
  }

  void SetUpFieldToExtField() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %p = $0[$2,$3] parameter(0)
        ROOT %ret = $1[$2] bitcast-convert(%p)
      }
    )",
                                 f_typename_, ef_typename_, N, D);

    Array2D<F> input(N, D);
    std::vector<EF> expected(N);
    for (int64_t i = 0; i < N; ++i) {
      F coeffs[D];
      for (int64_t j = 0; j < D; ++j) {
        coeffs[j] = F::Random();
        input({i, j}) = coeffs[j];
      }
      std::memcpy(&expected[i], coeffs, sizeof(EF));
    }
    literals_[0] = LiteralUtil::CreateR2FromArray2D<F>(input);
    expected_literal_ = LiteralUtil::CreateR1<EF>(expected);
  }

 private:
  std::string_view ef_typename_;
  std::string_view f_typename_;
  std::vector<EF> x_;
};

}  // namespace zkx::gpu

#endif  // ZKX_BACKENDS_GPU_CODEGEN_FIELD_TEST_H_
