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

#ifndef ZKX_BACKENDS_GPU_CODEGEN_INT_TEST_H_
#define ZKX_BACKENDS_GPU_CODEGEN_INT_TEST_H_

#include <type_traits>
#include <vector>

#include "absl/base/casts.h"
#include "absl/strings/substitute.h"

#include "zkx/array2d.h"
#include "zkx/backends/gpu/codegen/cuda_kernel_emitter_test.h"
#include "zkx/base/containers/container_util.h"
#include "zkx/base/random.h"
#include "zkx/literal_util.h"
#include "zkx/primitive_util.h"

namespace zkx::gpu {

template <typename T>
class BaseIntTest {
 protected:
  using UnsignedT =
      std::conditional_t<std::is_signed_v<T>, std::make_unsigned_t<T>, T>;

  static T GetRandomValue() {
    return absl::bit_cast<T>(base::Uniform<UnsignedT>());
  }
};

template <typename T>
class IntScalarUnaryTest : public BaseIntTest<T>, public CudaKernelEmitterTest {
 public:
  void SetUp() override {
    CudaKernelEmitterTest::SetUp();
    x_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<T>());
    x_ = BaseIntTest<T>::GetRandomValue();
    literals_.push_back(LiteralUtil::CreateR0<T>(x_));
  }

 protected:
  void SetUpConvertUp() {
    using DstType =
        typename std::conditional_t<std::is_signed_v<T>, int64_t, uint64_t>;
    static_assert(sizeof(T) < sizeof(DstType),
                  "T must be smaller than DstType");
    std::string_view dst_typename = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<DstType>());

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
                                 x_typename_, dst_typename);

    expected_literal_ =
        LiteralUtil::CreateR0<DstType>(static_cast<DstType>(x_));
  }

  void SetUpConvertDown() {
    using DstType =
        typename std::conditional_t<std::is_signed_v<T>, int16_t, uint16_t>;
    static_assert(sizeof(T) > sizeof(DstType), "T must be larger than DstType");
    std::string_view dst_typename = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<DstType>());

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
                                 x_typename_, dst_typename);
    expected_literal_ =
        LiteralUtil::CreateR0<DstType>(static_cast<DstType>(x_));
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
    expected_literal_ = LiteralUtil::CreateR0<T>(-x_);
  }

 private:
  T x_;
};

template <typename T>
class IntScalarBinaryTest : public BaseIntTest<T>,
                            public CudaKernelEmitterTest {
 public:
  void SetUp() override {
    CudaKernelEmitterTest::SetUp();
    x_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<T>());
    x_ = BaseIntTest<T>::GetRandomValue();
    y_ = BaseIntTest<T>::GetRandomValue();
    literals_.push_back(LiteralUtil::CreateR0<T>(x_));
    literals_.push_back(LiteralUtil::CreateR0<T>(y_));
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
    expected_literal_ = LiteralUtil::CreateR0<T>(x_ + y_);
  }

  void SetUpDiv() {
    uint32_t case_num = base::Uniform<uint32_t>() % 3;
    if (case_num == 0) {
      y_ = 0;
      literals_[1] = LiteralUtil::CreateR0<T>(y_);
    } else if (case_num == 1) {
      if (std::is_signed_v<T>) {
        x_ = std::numeric_limits<T>::min();
        literals_[0] = LiteralUtil::CreateR0<T>(x_);
        y_ = -1;
        literals_[1] = LiteralUtil::CreateR0<T>(y_);
      }
    }

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

    T expected;
    if (y_ == 0) {
      expected = -1;
    } else if (std::is_signed_v<T> && x_ == std::numeric_limits<T>::min() &&
               y_ == -1) {
      expected = std::numeric_limits<T>::min();
    } else {
      expected = x_ / y_;
    }
    expected_literal_ = LiteralUtil::CreateR0<T>(expected);
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
    expected_literal_ = LiteralUtil::CreateR0<T>(x_ * y_);
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
    expected_literal_ = LiteralUtil::CreateR0<T>(x_ - y_);
  }

 private:
  T x_;
  T y_;
};

template <typename T>
class IntR2TensorBinaryTest : public BaseIntTest<T>,
                              public CudaKernelEmitterTest {
 public:
  constexpr static int64_t M = 2;
  constexpr static int64_t N = 3;

  void SetUp() override {
    CudaKernelEmitterTest::SetUp();
    x_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<T>());
    x_ = base::CreateVector(M, []() {
      return base::CreateVector(
          N, []() { return BaseIntTest<T>::GetRandomValue(); });
    });
    y_ = base::CreateVector(M, []() {
      return base::CreateVector(
          N, []() { return BaseIntTest<T>::GetRandomValue(); });
    });
    Array2D<T> x_array(M, N);
    Array2D<T> y_array(M, N);
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
        EXPECT_EQ(ret_literal.Get<T>({i, j}), expected_[i][j]);
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
  std::vector<std::vector<T>> x_;
  std::vector<std::vector<T>> y_;
  std::vector<std::vector<T>> expected_;
};

template <typename T>
class IntTest : public BaseIntTest<T>, public CudaKernelEmitterTest {
 public:
  void SetUp() override {
    CudaKernelEmitterTest::SetUp();
    x_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<T>());
  }

 protected:
  void SetUpSlice() {
    constexpr static int64_t N = 6;
    constexpr static int64_t S = 2;
    constexpr static int64_t E = 5;

    hlo_text_ = absl::Substitute(R"(
      %f {
        %x = $0[$1] parameter(0)

        ROOT %ret = $0[$2] slice(%x), slice={[$3:$4]}
      }

      ENTRY %main {
        %x = $0[$1] parameter(0)

        ROOT %ret = $0[$2] fusion(%x), kind=kLoop, calls=%f
      }
    )",
                                 x_typename_, N, E - S, S, E);

    auto x = base::CreateVector(
        N, []() { return BaseIntTest<T>::GetRandomValue(); });
    literals_.push_back(LiteralUtil::CreateR1<T>(x));
    expected_literal_ = LiteralUtil::CreateR1<T>(
        base::CreateVector(E - S, [&x](size_t i) { return x[i + S]; }));
  }

  // Cross-dtype bitcast: T[4] → int8_t[4, sizeof(T)].
  void SetUpBitcastConvertDtypesToSmaller() {
    constexpr static int64_t N = 4;
    constexpr static int64_t ratio = sizeof(T) / sizeof(int8_t);

    std::string_view dst_typename = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<int8_t>());

    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[$1] parameter(0)
        ROOT %ret = $2[$1, $3] bitcast-convert(%x)
      }
    )",
                                 x_typename_, N, dst_typename, ratio);

    std::vector<T> input(N);
    Array2D<int8_t> expected(N, ratio);
    for (int64_t i = 0; i < N; ++i) {
      input[i] = BaseIntTest<T>::GetRandomValue();
      int8_t bytes[sizeof(T)];
      std::memcpy(bytes, &input[i], sizeof(T));
      for (int64_t j = 0; j < ratio; ++j) {
        expected({i, j}) = bytes[j];
      }
    }
    literals_.push_back(LiteralUtil::CreateR1<T>(input));
    expected_literal_ = LiteralUtil::CreateR2FromArray2D<int8_t>(expected);
  }

  // Cross-dtype bitcast: int8_t[4, sizeof(T)] → T[4].
  void SetUpBitcastConvertDtypesToLarger() {
    constexpr static int64_t N = 4;
    constexpr static int64_t ratio = sizeof(T) / sizeof(int8_t);

    std::string_view src_typename = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<int8_t>());

    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[$1, $2] parameter(0)
        ROOT %ret = $3[$1] bitcast-convert(%x)
      }
    )",
                                 src_typename, N, ratio, x_typename_);

    Array2D<int8_t> input(N, ratio);
    std::vector<T> expected(N);
    for (int64_t i = 0; i < N; ++i) {
      int8_t bytes[sizeof(T)];
      for (int64_t j = 0; j < ratio; ++j) {
        bytes[j] = static_cast<int8_t>(base::Uniform<uint8_t>());
        input({i, j}) = bytes[j];
      }
      std::memcpy(&expected[i], bytes, sizeof(T));
    }
    literals_.push_back(LiteralUtil::CreateR2FromArray2D<int8_t>(input));
    expected_literal_ = LiteralUtil::CreateR1<T>(expected);
  }
};

}  // namespace zkx::gpu

#endif  // ZKX_BACKENDS_GPU_CODEGEN_INT_TEST_H_
