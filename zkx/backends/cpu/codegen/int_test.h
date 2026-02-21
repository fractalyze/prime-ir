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

#ifndef ZKX_BACKENDS_CPU_CODEGEN_INT_TEST_H_
#define ZKX_BACKENDS_CPU_CODEGEN_INT_TEST_H_

#include <algorithm>
#include <cmath>
#include <numeric>
#include <type_traits>
#include <vector>

#include "absl/base/casts.h"
#include "absl/strings/substitute.h"

#include "zk_dtypes/include/big_int.h"
#include "zk_dtypes/include/pow.h"
#include "zkx/array2d.h"
#include "zkx/backends/cpu/codegen/cpu_kernel_emitter_test.h"
#include "zkx/base/containers/container_util.h"
#include "zkx/base/random.h"
#include "zkx/comparison_util.h"
#include "zkx/literal_util.h"
#include "zkx/primitive_util.h"
#include "zkx/types.h"

namespace zkx::cpu {

template <typename T>
class BaseIntTest {
 protected:
  using UnsignedT =
      std::conditional_t<std::is_signed_v<T>, std::make_unsigned_t<T>, T>;

  static T GetRandomValue() {
    return absl::bit_cast<T>(base::Uniform<UnsignedT>());
  }
};

// Specialization for u128 (BigInt<2>).
template <>
class BaseIntTest<u128> {
 protected:
  using UnsignedT = u128;

  static u128 GetRandomValue() { return u128::Random(); }
};

template <typename T>
class IntScalarUnaryTest : public BaseIntTest<T>, public CpuKernelEmitterTest {
 public:
  void SetUp() override {
    CpuKernelEmitterTest::SetUp();
    x_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<T>());
    x_ = BaseIntTest<T>::GetRandomValue();
    literals_.push_back(LiteralUtil::CreateR0<T>(x_));
  }

 protected:
  void SetUpAbs() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)

        ROOT %ret = $0[] abs(%x)
      }
    )",
                                 x_typename_);
    if constexpr (std::is_signed_v<T>) {
      expected_literal_ = LiteralUtil::CreateR0<T>(std::abs<T>(x_));
    } else {
      LOG(FATAL) << "abs is not supported for unsigned type";
    }
  }

  void SetUpBitcastConvert() {
    using DstType = typename std::conditional_t<
        std::is_signed_v<T>, std::make_unsigned_t<T>, std::make_signed_t<T>>;
    std::string_view dst_typename = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<DstType>());

    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)

        ROOT %ret = $1[] bitcast-convert(%x)
      }
    )",
                                 x_typename_, dst_typename);
    expected_literal_ =
        LiteralUtil::CreateR0<DstType>(absl::bit_cast<DstType>(x_));
  }

  void SetUpCountLeadingZeros() {
    uint32_t case_num = base::Uniform<uint32_t>() % 2;
    if (case_num == 0) {
      x_ = 0;
      literals_[0] = LiteralUtil::CreateR0<T>(x_);
    }
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)

        ROOT %ret = $0[] count-leading-zeros(%x)
      }
    )",
                                 x_typename_);
    if (x_ == 0) {
      // NOTE(chokobole): __builtin_clz is undefined behavior for 0.
      // See https://gcc.gnu.org/onlinedocs/gcc/Bit-Operation-Builtins.html
      expected_literal_ = LiteralUtil::CreateR0<T>(sizeof(T) * 8);
    } else {
      expected_literal_ = LiteralUtil::CreateR0<T>(__builtin_clz(x_));
    }
  }

  void SetUpConvertUp() {
    using DstType =
        typename std::conditional_t<std::is_signed_v<T>, int64_t, uint64_t>;
    static_assert(sizeof(T) < sizeof(DstType),
                  "T must be smaller than DstType");
    std::string_view dst_typename = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<DstType>());

    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)

        ROOT %ret = $1[] convert(%x)
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
      ENTRY %main {
        %x = $0[] parameter(0)

        ROOT %ret = $1[] convert(%x)
      }
    )",
                                 x_typename_, dst_typename);
    expected_literal_ =
        LiteralUtil::CreateR0<DstType>(static_cast<DstType>(x_));
  }

  void SetUpNegate() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)

        ROOT %ret = $0[] negate(%x)
      }
    )",
                                 x_typename_);
    expected_literal_ = LiteralUtil::CreateR0<T>(-x_);
  }

  void SetUpNot() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)

        ROOT %ret = $0[] not(%x)
      }
    )",
                                 x_typename_);
    expected_literal_ = LiteralUtil::CreateR0<T>(~x_);
  }

  void SetUpPopulationCount() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)

        ROOT %ret = $0[] popcnt(%x)
      }
    )",
                                 x_typename_);
    expected_literal_ = LiteralUtil::CreateR0<T>(__builtin_popcount(x_));
  }

  void SetUpSign() {
    uint32_t case_num = base::Uniform<uint32_t>() % 2;
    if (case_num == 0) {
      x_ = 0;
      literals_[0] = LiteralUtil::CreateR0<T>(x_);
    }
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)

        ROOT %ret = $0[] sign(%x)
      }
    )",
                                 x_typename_);
    T sign;
    if (x_ == 0) {
      sign = 0;
    } else if (x_ > 0) {
      sign = 1;
    } else {
      sign = -1;
    }
    expected_literal_ = LiteralUtil::CreateR0<T>(sign);
  }

 private:
  T x_;
};

template <typename T>
class IntScalarBinaryTest : public BaseIntTest<T>, public CpuKernelEmitterTest {
 public:
  void SetUp() override {
    CpuKernelEmitterTest::SetUp();
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
      ENTRY %main {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = $0[] add(%x, %y)
      }
    )",
                                 x_typename_);
    expected_literal_ = LiteralUtil::CreateR0<T>(x_ + y_);
  }

  void SetUpAnd() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = $0[] and(%x, %y)
      }
    )",
                                 x_typename_);
    expected_literal_ = LiteralUtil::CreateR0<T>(x_ & y_);
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
      ENTRY %main {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = $0[] divide(%x, %y)
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
    expected_literal_ = LiteralUtil::CreateR0<T>((x_ + y_) * x_ * 2);
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
    expected_literal_ = LiteralUtil::CreateR0<T>(std::max(x_, y_));
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
    expected_literal_ = LiteralUtil::CreateR0<T>(std::min(x_, y_));
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
    expected_literal_ = LiteralUtil::CreateR0<T>(x_ * y_);
  }

  void SetUpOr() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = $0[] or(%x, %y)
      }
    )",
                                 x_typename_);
    expected_literal_ = LiteralUtil::CreateR0<T>(x_ | y_);
  }

  void SetUpPower() {
    uint32_t case_num = base::Uniform<uint32_t>() % 3;
    if (case_num == 0) {
      x_ = -1;
      literals_[0] = LiteralUtil::CreateR0<T>(x_);
    } else if (case_num == 1) {
      x_ = 1;
      literals_[0] = LiteralUtil::CreateR0<T>(x_);
    }
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = $0[] power(%x, %y)
      }
    )",
                                 x_typename_);
    T expected;
    if (x_ == -1) {
      expected = y_ % 2 == 0 ? 1 : -1;
    } else if (x_ == 1) {
      expected = 1;
    } else if (y_ >= 0) {
      expected = zk_dtypes::Pow(x_, y_);
    } else {
      expected = 0;
    }
    expected_literal_ = LiteralUtil::CreateR0<T>(expected);
  }

  void SetUpRemainder() {
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
      ENTRY %main {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = $0[] remainder(%x, %y)
      }
    )",
                                 x_typename_);

    T expected;
    if (y_ == 0) {
      expected = x_;
    } else if (std::is_signed_v<T> && x_ == std::numeric_limits<T>::min() &&
               y_ == -1) {
      expected = 0;
    } else {
      expected = x_ % y_;
    }
    expected_literal_ = LiteralUtil::CreateR0<T>(expected);
  }

  void SetUpShiftLeft() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = $0[] shift-left(%x, %y)
      }
    )",
                                 x_typename_);

    T expected;
    if (IsShiftOutOfBounds(y_)) {
      expected = 0;
    } else {
      expected = x_ << y_;
    }
    expected_literal_ = LiteralUtil::CreateR0<T>(expected);
  }

  void SetUpShiftRightArithmetic() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = $0[] shift-right-arithmetic(%x, %y)
      }
    )",
                                 x_typename_);

    T expected;
    if (IsShiftOutOfBounds(y_)) {
      using SignedT = std::make_signed_t<T>;
      auto x_signed = static_cast<SignedT>(x_);
      if (x_signed < 0) {
        expected = -1;
      } else {
        expected = 0;
      }
    } else {
      expected = x_ >> y_;
    }
    expected_literal_ = LiteralUtil::CreateR0<T>(expected);
  }

  void SetUpShiftRightLogical() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = $0[] shift-right-logical(%x, %y)
      }
    )",
                                 x_typename_);

    T expected;
    if (IsShiftOutOfBounds(y_)) {
      expected = 0;
    } else {
      expected = x_ >> y_;
    }
    expected_literal_ = LiteralUtil::CreateR0<T>(expected);
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
    expected_literal_ = LiteralUtil::CreateR0<T>(x_ - y_);
  }

  void SetUpXor() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = $0[] xor(%x, %y)
      }
    )",
                                 x_typename_);
    expected_literal_ = LiteralUtil::CreateR0<T>(x_ ^ y_);
  }

 private:
  static bool IsShiftOutOfBounds(T rhs) {
    using UnsignedT = std::make_unsigned_t<T>;
    UnsignedT lhs_bits_unsigned =
        static_cast<UnsignedT>(std::numeric_limits<UnsignedT>::digits);
    UnsignedT rhs_unsigned = static_cast<UnsignedT>(rhs);
    return rhs_unsigned >= lhs_bits_unsigned;
  }

  T x_;
  T y_;
};

template <typename T>
class IntScalarTernaryTest : public BaseIntTest<T>,
                             public CpuKernelEmitterTest {
 public:
  void SetUp() override {
    CpuKernelEmitterTest::SetUp();
    x_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<T>());
    x_ = BaseIntTest<T>::GetRandomValue();
    y_ = BaseIntTest<T>::GetRandomValue();
    z_ = BaseIntTest<T>::GetRandomValue();
    literals_.push_back(LiteralUtil::CreateR0<T>(x_));
    literals_.push_back(LiteralUtil::CreateR0<T>(y_));
    literals_.push_back(LiteralUtil::CreateR0<T>(z_));
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
    expected_literal_ = LiteralUtil::CreateR0<T>(std::clamp(y_, x_, z_));
  }

  void SetUpSelect() {
    bool cond = x_ % 2 == 0;
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
    expected_literal_ = LiteralUtil::CreateR0<T>(cond ? y_ : z_);
  }

 private:
  T x_;
  T y_;
  T z_;
};

template <typename T>
class IntR2TensorBinaryTest : public BaseIntTest<T>,
                              public CpuKernelEmitterTest {
 public:
  constexpr static int64_t M = 2;
  constexpr static int64_t N = 3;

  void SetUp() override {
    CpuKernelEmitterTest::SetUp();
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

  void SetUpAddWithLayout() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[$1, $2]{0, 1} parameter(0)
        %y = $0[$1, $2]{0, 1} parameter(1)

        ROOT %ret = $0[$1, $2]{0, 1} add(%x, %y)
      }
    )",
                                 x_typename_, M, N);
    literals_[0] = literals_[0].Relayout(LayoutUtil::MakeLayout({0, 1}));
    literals_[1] = literals_[1].Relayout(LayoutUtil::MakeLayout({0, 1}));
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
class IntTest : public BaseIntTest<T>, public CpuKernelEmitterTest {
 public:
  void SetUp() override {
    CpuKernelEmitterTest::SetUp();
    x_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<T>());
  }

  void SetUpBitReverse() {
    // Use power-of-2 for the dimension to bit-reverse
    constexpr static int64_t D0 = 2;
    constexpr static int64_t D1 = 8;  // power of 2
    constexpr static int64_t D2 = 3;

    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[$1, $2, $3] parameter(0)

        ROOT %ret = $0[$1, $2, $3] bit-reverse(%x), dimensions={1}
      }
    )",
                                 x_typename_, D0, D1, D2);

    Array3D<T> x_array(D0, D1, D2);
    for (int64_t i = 0; i < D0; ++i) {
      for (int64_t j = 0; j < D1; ++j) {
        for (int64_t k = 0; k < D2; ++k) {
          x_array({i, j, k}) = BaseIntTest<T>::GetRandomValue();
        }
      }
    }
    literals_.push_back(LiteralUtil::CreateR3FromArray3D<T>(x_array));
    expected_literal_ = literals_[0].BitReverse({1});
  }

  void SetUpBroadcastScalar() {
    constexpr static int64_t N = 4;

    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)

        ROOT %ret = $0[$1] broadcast(%x)
      }
    )",
                                 x_typename_, N);

    auto x = BaseIntTest<T>::GetRandomValue();
    literals_.push_back(LiteralUtil::CreateR0<T>(x));
    expected_literal_ =
        LiteralUtil::CreateR1<T>(base::CreateVector(N, [x]() { return x; }));
  }

  void SetUpBroadcastTensorR1ToR3WithD0() {
    SetUpBroadcastTensorR1ToR3Helper(2, 0, [](const std::vector<T>& x) {
      return LiteralUtil::CreateR3<T>(
          {{{x[0], x[0]}, {x[0], x[0]}}, {{x[1], x[1]}, {x[1], x[1]}}});
    });
  }

  void SetUpBroadcastTensorR1ToR3WithD1() {
    SetUpBroadcastTensorR1ToR3Helper(2, 1, [](const std::vector<T>& x) {
      return LiteralUtil::CreateR3<T>(
          {{{x[0], x[0]}, {x[1], x[1]}}, {{x[0], x[0]}, {x[1], x[1]}}});
    });
  }

  void SetUpBroadcastTensorR1ToR3WithD2() {
    SetUpBroadcastTensorR1ToR3Helper(2, 2, [](const std::vector<T>& x) {
      return LiteralUtil::CreateR3<T>(
          {{{x[0], x[1]}, {x[0], x[1]}}, {{x[0], x[1]}, {x[0], x[1]}}});
    });
  }

  void SetUpCall() {
    hlo_text_ = absl::Substitute(R"(
      %func {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = $0[] add(%x, %y)
      }

      ENTRY %main {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = $0[] call(%x, %y), to_apply=%func
      }
    )",
                                 x_typename_);

    auto x = BaseIntTest<T>::GetRandomValue();
    auto y = BaseIntTest<T>::GetRandomValue();
    literals_.push_back(LiteralUtil::CreateR0<T>(x));
    literals_.push_back(LiteralUtil::CreateR0<T>(y));
    expected_literal_ = LiteralUtil::CreateR0<T>(x + y);
  }

  void SetUpConcatenate() {
    constexpr static int64_t D0 = 2;
    constexpr static int64_t D1 = 3;
    constexpr static int64_t D2 = 4;
    constexpr static int64_t D = D0 + D1 + D2;

    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[2, $1, 2] parameter(0)
        %y = $0[2, $2, 2] parameter(1)
        %z = $0[2, $3, 2] parameter(2)

        ROOT %ret = $0[2, $4, 2] concatenate(%x, %y, %z), dimensions={1}
      }
    )",
                                 x_typename_, D0, D1, D2, D);
    Array3D<T> x_array(2, D0, 2);
    Array3D<T> y_array(2, D1, 2);
    Array3D<T> z_array(2, D2, 2);
    for (int64_t i = 0; i < 2; ++i) {
      for (int64_t j = 0; j < D0; ++j) {
        for (int64_t k = 0; k < 2; ++k) {
          x_array({i, j, k}) = BaseIntTest<T>::GetRandomValue();
        }
      }
    }
    for (int64_t i = 0; i < 2; ++i) {
      for (int64_t j = 0; j < D1; ++j) {
        for (int64_t k = 0; k < 2; ++k) {
          y_array({i, j, k}) = BaseIntTest<T>::GetRandomValue();
        }
      }
    }
    for (int64_t i = 0; i < 2; ++i) {
      for (int64_t j = 0; j < D2; ++j) {
        for (int64_t k = 0; k < 2; ++k) {
          z_array({i, j, k}) = BaseIntTest<T>::GetRandomValue();
        }
      }
    }
    literals_.push_back(LiteralUtil::CreateR3FromArray3D<T>(x_array));
    literals_.push_back(LiteralUtil::CreateR3FromArray3D<T>(y_array));
    literals_.push_back(LiteralUtil::CreateR3FromArray3D<T>(z_array));
    Array3D<T> expected_array(2, D, 2);
    for (int64_t i = 0; i < 2; ++i) {
      for (int64_t j = 0; j < D; ++j) {
        for (int64_t k = 0; k < 2; ++k) {
          if (j < D0) {
            expected_array({i, j, k}) = x_array({i, j, k});
          } else if (j < D0 + D1) {
            expected_array({i, j, k}) = y_array({i, j - D0, k});
          } else {
            expected_array({i, j, k}) = z_array({i, j - D0 - D1, k});
          }
        }
      }
    }
    expected_literal_ = LiteralUtil::CreateR3FromArray3D<T>(expected_array);
  }

  void SetUpConditional() {
    hlo_text_ = absl::Substitute(R"(
      %identity {
        ROOT %ret = $0[] parameter(0)
      }

      %negate {
        %x = $0[] parameter(0)

        ROOT %ret = $0[] negate(%x)
      }

      ENTRY %main {
        %cond = pred[] parameter(0)
        %x = $0[] parameter(1)

        ROOT %ret = $0[] conditional(%cond, %x, %x), true_computation=%identity, false_computation=%negate
      }
    )",
                                 x_typename_);

    auto x = BaseIntTest<T>::GetRandomValue();
    bool cond = base::Uniform<uint32_t>() % 2 == 0;
    literals_.push_back(LiteralUtil::CreateR0<bool>(cond));
    literals_.push_back(LiteralUtil::CreateR0<T>(x));
    expected_literal_ = LiteralUtil::CreateR0<T>(cond ? x : -x);
  }

  void SetUpDynamicSlice() {
    constexpr static int64_t D0 = 5;
    constexpr static int64_t D1 = 6;
    constexpr static int64_t D2 = 7;

    constexpr static int64_t S0 = 1;
    constexpr static int64_t S1 = 2;
    constexpr static int64_t S2 = 3;

    constexpr static int64_t L0 = 2;
    constexpr static int64_t L1 = 3;
    constexpr static int64_t L2 = 4;

    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[$1, $2, $3] parameter(0)
        %offset0 = $0[] parameter(1)
        %offset1 = $0[] parameter(2)
        %offset2 = $0[] parameter(3)

        ROOT %ret = $0[$4, $5, $6] dynamic-slice(%x, %offset0, %offset1, %offset2), dynamic_slice_sizes={$4, $5, $6}
      }
    )",
                                 x_typename_, D0, D1, D2, L0, L1, L2);

    Array3D<T> x_array(D0, D1, D2);
    for (int64_t i = 0; i < D0; ++i) {
      for (int64_t j = 0; j < D1; ++j) {
        for (int64_t k = 0; k < D2; ++k) {
          x_array({i, j, k}) = BaseIntTest<T>::GetRandomValue();
        }
      }
    }
    literals_.push_back(LiteralUtil::CreateR3FromArray3D<T>(x_array));
    literals_.push_back(LiteralUtil::CreateR0<T>(S0));
    literals_.push_back(LiteralUtil::CreateR0<T>(S1));
    literals_.push_back(LiteralUtil::CreateR0<T>(S2));
    Array3D<T> expected_array(L0, L1, L2);
    for (int64_t i = 0; i < L0; ++i) {
      for (int64_t j = 0; j < L1; ++j) {
        for (int64_t k = 0; k < L2; ++k) {
          expected_array({i, j, k}) = x_array({S0 + i, S1 + j, S2 + k});
        }
      }
    }
    expected_literal_ = LiteralUtil::CreateR3FromArray3D<T>(expected_array);
  }

  void SetUpDynamicUpdateSlice() {
    constexpr static int64_t D0 = 5;
    constexpr static int64_t D1 = 6;
    constexpr static int64_t D2 = 7;

    constexpr static int64_t S0 = 1;
    constexpr static int64_t S1 = 2;
    constexpr static int64_t S2 = 3;

    constexpr static int64_t L0 = 2;
    constexpr static int64_t L1 = 3;
    constexpr static int64_t L2 = 4;

    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[$1, $2, $3] parameter(0)
        %update = $0[$4, $5, $6] parameter(1)
        %offset0 = $0[] parameter(2)
        %offset1 = $0[] parameter(3)
        %offset2 = $0[] parameter(4)

        ROOT %ret = $0[$1, $2, $3] dynamic-update-slice(%x, %update, %offset0, %offset1, %offset2)
      }
    )",
                                 x_typename_, D0, D1, D2, L0, L1, L2);

    Array3D<T> x_array(D0, D1, D2);
    for (int64_t i = 0; i < D0; ++i) {
      for (int64_t j = 0; j < D1; ++j) {
        for (int64_t k = 0; k < D2; ++k) {
          x_array({i, j, k}) = BaseIntTest<T>::GetRandomValue();
        }
      }
    }
    literals_.push_back(LiteralUtil::CreateR3FromArray3D<T>(x_array));
    Array3D<T> update_array(L0, L1, L2);
    for (int64_t i = 0; i < L0; ++i) {
      for (int64_t j = 0; j < L1; ++j) {
        for (int64_t k = 0; k < L2; ++k) {
          update_array({i, j, k}) = BaseIntTest<T>::GetRandomValue();
        }
      }
    }
    literals_.push_back(LiteralUtil::CreateR3FromArray3D<T>(update_array));
    literals_.push_back(LiteralUtil::CreateR0<T>(S0));
    literals_.push_back(LiteralUtil::CreateR0<T>(S1));
    literals_.push_back(LiteralUtil::CreateR0<T>(S2));
    Array3D<T> expected_array(D0, D1, D2);
    for (int64_t i = 0; i < D0; ++i) {
      for (int64_t j = 0; j < D1; ++j) {
        for (int64_t k = 0; k < D2; ++k) {
          if (i < S0 || i >= S0 + L0 || j < S1 || j >= S1 + L1 || k < S2 ||
              k >= S2 + L2) {
            expected_array({i, j, k}) = x_array({i, j, k});
          } else {
            expected_array({i, j, k}) = update_array({i - S0, j - S1, k - S2});
          }
        }
      }
    }
    expected_literal_ = LiteralUtil::CreateR3FromArray3D<T>(expected_array);
  }

  // Gather with unbatched scalar index into a 2D tensor.
  // Reproduces the segfault fixed in LookUpGatherStartValues where
  // batch_idx is empty and the old rank-1 special case accessed
  // batch_idx[0] unconditionally.
  // Equivalent to: jnp.take(operand[4,8], 0, axis=0) → result[8]
  void SetUpGather() {
    constexpr static int64_t D0 = 4;
    constexpr static int64_t D1 = 8;

    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %operand = $0[$1, $2] parameter(0)
        %indices = s32[] parameter(1)

        ROOT %ret = $0[$2] gather(%operand, %indices),
          offset_dims={0},
          collapsed_slice_dims={0},
          start_index_map={0},
          index_vector_dim=0,
          slice_sizes={1, $2}
      }
    )",
                                 x_typename_, D0, D1);

    Array2D<T> operand_array(D0, D1);
    for (int64_t i = 0; i < D0; ++i) {
      for (int64_t j = 0; j < D1; ++j) {
        operand_array({i, j}) = BaseIntTest<T>::GetRandomValue();
      }
    }
    literals_.push_back(LiteralUtil::CreateR2FromArray2D<T>(operand_array));

    int32_t index = base::Uniform<uint32_t>() % D0;
    literals_.push_back(LiteralUtil::CreateR0<int32_t>(index));

    std::vector<T> expected_vec(D1);
    for (int64_t j = 0; j < D1; ++j) {
      expected_vec[j] = operand_array({index, j});
    }
    expected_literal_ = LiteralUtil::CreateR1<T>(expected_vec);
  }

  void SetUpIotaWithD0() { SetUpIotaHelper(0); }

  void SetUpIotaWithD1() { SetUpIotaHelper(1); }

  void SetUpMap() {
    constexpr static int64_t D0 = 4;

    hlo_text_ = absl::Substitute(R"(
      %func {
        %a = $0[] parameter(0)
        %b = $0[] parameter(1)

        %mul = $0[] multiply(%a, %b)
        ROOT %ret = $0[] add(%mul, %a)
      }

      ENTRY %main {
        %x = $0[$1] parameter(0)
        %y = $0[$1] parameter(1)

        ROOT %ret = $0[$1] map(%x, %y), dimensions={0}, to_apply=%func
      }
    )",
                                 x_typename_, D0);
    std::vector<T> x = base::CreateVector(
        D0, []() { return BaseIntTest<T>::GetRandomValue(); });
    std::vector<T> y = base::CreateVector(
        D0, []() { return BaseIntTest<T>::GetRandomValue(); });
    literals_.push_back(LiteralUtil::CreateR1<T>(x));
    literals_.push_back(LiteralUtil::CreateR1<T>(y));
    std::vector<T> expected =
        base::CreateVector(D0, [&](size_t i) { return x[i] * y[i] + x[i]; });
    expected_literal_ = LiteralUtil::CreateR1<T>(expected);
  }

  void SetUpPad() {
    constexpr static int64_t M = 4;
    constexpr static int64_t LO = 2;
    constexpr static int64_t HI = 3;
    constexpr static int64_t N = M + LO + HI;

    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[$1] parameter(0)
        %padding_value = $0[] parameter(1)

        ROOT %ret = $0[$4] pad(%x, %padding_value), padding=$2_$3
      }
    )",
                                 x_typename_, M, LO, HI, N);

    std::vector<T> x = base::CreateVector(
        M, []() { return BaseIntTest<T>::GetRandomValue(); });
    T padding_value = BaseIntTest<T>::GetRandomValue();
    literals_.push_back(LiteralUtil::CreateR1<T>(x));
    literals_.push_back(LiteralUtil::CreateR0<T>(padding_value));
    std::vector<T> expected =
        base::CreateVector(N, [padding_value, &x](size_t i) {
          if (i < LO) {
            return padding_value;
          } else if (i < M + LO) {
            return x[i - LO];
          } else {
            return padding_value;
          }
        });
    expected_literal_ = LiteralUtil::CreateR1<T>(expected);
  }

  void SetUpReduce() {
    constexpr static int64_t D0 = 4;

    hlo_text_ = absl::Substitute(R"(
      %func {
        %a = $0[] parameter(0)
        %b = $0[] parameter(1)

        %mul = $0[] multiply(%a, %b)
        ROOT %ret = $0[] add(%mul, %a)
      }

      ENTRY %main {
        %x = $0[$1] parameter(0)
        %init = $0[] parameter(1)

        ROOT %ret = $0[] reduce(%x, %init), dimensions={0}, to_apply=%func
      }
    )",
                                 x_typename_, D0);
    std::vector<T> x = base::CreateVector(
        D0, []() { return BaseIntTest<T>::GetRandomValue(); });
    T init = BaseIntTest<T>::GetRandomValue();
    literals_.push_back(LiteralUtil::CreateR1<T>(x));
    literals_.push_back(LiteralUtil::CreateR0<T>(init));
    T result = std::accumulate(x.begin(), x.end(), init, [](T acc, T value) {
      return acc * value + acc;
    });
    expected_literal_ = LiteralUtil::CreateR0<T>(result);
  }

  // Tests partial reduction (2D→1D) to cover scalar init broadcast fix.
  void SetUpReducePartial() {
    hlo_text_ = absl::Substitute(R"(
      %add {
        %a = $0[] parameter(0)
        %b = $0[] parameter(1)
        ROOT %ret = $0[] add(%a, %b)
      }

      ENTRY %main {
        %x = $0[2, 3] parameter(0)
        %init = $0[] parameter(1)
        ROOT %ret = $0[2] reduce(%x, %init), dimensions={1}, to_apply=%add
      }
    )",
                                 x_typename_);

    // x = [[1, 2, 3], [4, 5, 6]], init = 10
    // expected = [10+1+2+3, 10+4+5+6] = [16, 25]
    literals_.push_back(LiteralUtil::CreateR2<T>({{1, 2, 3}, {4, 5, 6}}));
    literals_.push_back(LiteralUtil::CreateR0<T>(10));
    expected_literal_ = LiteralUtil::CreateR1<T>({16, 25});
  }

  void SetUpReduceWindow() {
    constexpr static int64_t D0 = 8;
    constexpr static int64_t W = 3;
    constexpr static int64_t STRIDE = 2;
    constexpr static int64_t PAD_LOW = 2;
    constexpr static int64_t PAD_HIGH = 2;
    constexpr static int64_t BASE_DIL = 2;
    constexpr static int64_t WIN_DIL = 2;
    // dilated_base = 15
    // padded = 19
    // dilated_window = 5
    // output = 8
    constexpr static int64_t DILATED_BASE = (D0 - 1) * BASE_DIL + 1;
    constexpr static int64_t PADDED = PAD_LOW + DILATED_BASE + PAD_HIGH;
    constexpr static int64_t DILATED_WIN = (W - 1) * WIN_DIL + 1;
    constexpr static int64_t OUT = (PADDED - DILATED_WIN) / STRIDE + 1;

    // f(a, b) = 2*a + b is non-commutative in fold order, so window reversal
    // produces a different result: e.g. for [x0, x1, x2]:
    //   forward: 4*x0 + 2*x1 + x2
    //   reversed: 4*x2 + 2*x1 + x0
    hlo_text_ =
        absl::Substitute(R"(
      %reduce_func {
        %a = $0[] parameter(0)
        %b = $0[] parameter(1)

        %doubled = $0[] add(%a, %a)
        ROOT %ret = $0[] add(%doubled, %b)
      }

      ENTRY %main {
        %x = $0[$1] parameter(0)
        %init = $0[] parameter(1)

        ROOT %ret = $0[$2] reduce-window(%x, %init), window={size=$3 stride=$4 pad=2_2 lhs_dilate=$5 rhs_dilate=$6 rhs_reversal=1}, to_apply=%reduce_func
      }
    )",
                         x_typename_, D0, OUT, W, STRIDE, BASE_DIL, WIN_DIL);

    std::vector<T> x = base::CreateVector(
        D0, []() { return BaseIntTest<T>::GetRandomValue(); });
    T init = 0;
    literals_.push_back(LiteralUtil::CreateR1<T>(x));
    literals_.push_back(LiteralUtil::CreateR0<T>(init));

    // Build dilated and padded input
    std::vector<T> padded(PADDED, init);
    for (int64_t i = 0; i < D0; ++i) {
      padded[PAD_LOW + i * BASE_DIL] = x[i];
    }

    // Compute expected with window reversal and dilation
    // reduction function: f(a, b) = 2*a + b
    std::vector<T> expected(OUT);
    for (int64_t i = 0; i < OUT; ++i) {
      int64_t window_start = i * STRIDE;
      T acc = init;
      // With rhs_reversal=1, iterate window in reverse order
      for (int64_t j = W - 1; j >= 0; --j) {
        int64_t pos = window_start + j * WIN_DIL;
        T val = (pos < PADDED) ? padded[pos] : init;
        acc = acc + acc + val;  // f(a, b) = 2*a + b
      }
      expected[i] = acc;
    }
    expected_literal_ = LiteralUtil::CreateR1<T>(expected);
  }

  void SetUpReshapeScalar() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[1, 1, 1] parameter(0)

        ROOT %ret = $0[] reshape(%x)
    }
    )",
                                 x_typename_);

    Array3D<T> x_array(1, 1, 1);
    x_array({0, 0, 0}) = BaseIntTest<T>::GetRandomValue();
    literals_.push_back(LiteralUtil::CreateR3FromArray3D<T>(x_array));
    expected_literal_ = LiteralUtil::CreateR0<T>(x_array({0, 0, 0}));
  }

  void SetUpReshape() {
    constexpr static int64_t D0 = 2;
    constexpr static int64_t D1 = 3;
    constexpr static int64_t D2 = 4;
    constexpr static int64_t D0Prime = 8;
    constexpr static int64_t D1Prime = 3;

    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[$1, $2, $3] parameter(0)

        ROOT %ret = $0[$4, $5] reshape(%x)
      }
    )",
                                 x_typename_, D0, D1, D2, D0Prime, D1Prime);

    Array3D<T> x_array(D0, D1, D2);
    for (int64_t i = 0; i < D0; ++i) {
      for (int64_t j = 0; j < D1; ++j) {
        for (int64_t k = 0; k < D2; ++k) {
          x_array({i, j, k}) = BaseIntTest<T>::GetRandomValue();
        }
      }
    }
    literals_.push_back(LiteralUtil::CreateR3FromArray3D<T>(x_array));
    TF_ASSERT_OK_AND_ASSIGN(expected_literal_,
                            literals_[0].Reshape({D0Prime, D1Prime}));
  }

  void SetUpReverse() {
    constexpr static int64_t D0 = 2;
    constexpr static int64_t D1 = 3;
    constexpr static int64_t D2 = 4;

    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[$1, $2, $3] parameter(0)

        ROOT %ret = $0[$1, $2, $3] reverse(%x), dimensions={0, 2}
      }
    )",
                                 x_typename_, D0, D1, D2);
    SetUpReverseHelper(D0, D1, D2, true, false, true);
  }

  void SetUpReverseWithEmptyDimensions() {
    constexpr static int64_t D0 = 2;
    constexpr static int64_t D1 = 3;
    constexpr static int64_t D2 = 4;

    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[$1, $2, $3] parameter(0)

        ROOT %ret = $0[$1, $2, $3] reverse(%x), dimensions={}
      }
    )",
                                 x_typename_, D0, D1, D2);

    SetUpReverseHelper(D0, D1, D2, false, false, false);
  }

  void SetUpScatter() {
    constexpr static int64_t M = 10;  // Operand size
    constexpr static int64_t N = 3;   // Number of updates

    hlo_text_ = absl::Substitute(R"(
      %update_func {
        %p0 = $0[] parameter(0)
        %p1 = $0[] parameter(1)
        ROOT %ret = $0[] add(%p0, %p1)
      }

      ENTRY %main {
        %operand = $0[$1] parameter(0)
        %indices = s32[$2, 1] parameter(1)
        %updates = $0[$2] parameter(2)

        ROOT %ret = $0[$1] scatter(%operand, %indices, %updates),
          update_window_dims={},
          inserted_window_dims={0},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1,
          to_apply=%update_func
      }
    )",
                                 x_typename_, M, N);

    std::vector<T> operand_vec = base::CreateVector(
        M, []() { return BaseIntTest<T>::GetRandomValue(); });

    std::vector<int32_t> indices_vec = {1, 4, 8};
    std::vector<T> updates_vec = base::CreateVector(
        N, []() { return BaseIntTest<T>::GetRandomValue(); });

    literals_.push_back(LiteralUtil::CreateR1<T>(operand_vec));

    Array2D<int32_t> indices_array(N, 1);
    for (int64_t i = 0; i < N; ++i) {
      indices_array({i, 0}) = indices_vec[i];
    }
    literals_.push_back(
        LiteralUtil::CreateR2FromArray2D<int32_t>(indices_array));
    literals_.push_back(LiteralUtil::CreateR1<T>(updates_vec));

    std::vector<T> expected_vec = operand_vec;
    for (int64_t i = 0; i < N; ++i) {
      expected_vec[indices_vec[i]] += updates_vec[i];
    }
    expected_literal_ = LiteralUtil::CreateR1<T>(expected_vec);
  }

  void SetUpSlice() {
    constexpr static int64_t N = 6;
    constexpr static int64_t S = 2;
    constexpr static int64_t E = 5;

    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[$1] parameter(0)

        ROOT %ret = $0[$2] slice(%x), slice={[$3:$4]}
      }
    )",
                                 x_typename_, N, E - S, S, E);

    auto x = base::CreateVector(
        N, []() { return BaseIntTest<T>::GetRandomValue(); });
    literals_.push_back(LiteralUtil::CreateR1<T>(x));
    expected_literal_ = LiteralUtil::CreateR1<T>(
        base::CreateVector(E - S, [&x](size_t i) { return x[i + S]; }));
  }

  void SetUpSort() {
    constexpr static int64_t D0 = 8;

    hlo_text_ = absl::Substitute(R"(
      %compare {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT ret = pred[] compare(%x, %y), direction=LT
      }

      ENTRY %main {
        %x = $0[$1] parameter(0)

        ROOT %ret = $0[$1] sort(%x), dimensions={0}, to_apply=%compare
      }
    )",
                                 x_typename_, D0);

    auto x = base::CreateVector(
        D0, []() { return BaseIntTest<T>::GetRandomValue(); });
    literals_.push_back(LiteralUtil::CreateR1<T>(x));
    auto expected = x;
    std::sort(expected.begin(), expected.end(), std::less<T>());
    expected_literal_ = LiteralUtil::CreateR1<T>(expected);
  }

  void SetUpTranspose() {
    constexpr static int64_t D0 = 2;
    constexpr static int64_t D1 = 3;
    constexpr static int64_t D2 = 4;

    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[$1, $2, $3] parameter(0)

        ROOT %ret = $0[$3, $1, $2] transpose(%x), dimensions={2, 0, 1}
      }
    )",
                                 x_typename_, D0, D1, D2);

    SetUpTransposeHelper(D0, D1, D2);
  }

  void SetUpTransposeWithLayout() {
    constexpr static int64_t D0 = 2;
    constexpr static int64_t D1 = 3;
    constexpr static int64_t D2 = 4;

    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[$1, $2, $3]{2, 1, 0} parameter(0)

        ROOT %ret = $0[$3, $1, $2]{0, 2, 1} transpose(%x), dimensions={2, 0, 1}
      }
    )",
                                 x_typename_, D0, D1, D2);

    SetUpTransposeHelper(D0, D1, D2);
    literals_[0] = literals_[0].Relayout(LayoutUtil::MakeLayout({2, 1, 0}));
    expected_literal_ =
        expected_literal_.Relayout(LayoutUtil::MakeLayout({0, 2, 1}));
  }

  void SetUpTransposeWithEmptyDimensions() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)

        ROOT %ret = $0[] transpose(%x), dimensions={}
      }
    )",
                                 x_typename_);

    literals_.push_back(
        LiteralUtil::CreateR0<T>(BaseIntTest<T>::GetRandomValue()));
    expected_literal_ = literals_[0].Clone();
  }

  void SetUpWhile() {
    hlo_text_ = absl::Substitute(R"(
      %condition {
        %param = (u32[], u32[], $0[], $0[]) parameter(0)
        %i = u32[] get-tuple-element(%param), index=0
        %n = u32[] get-tuple-element(%param), index=1

        ROOT ret = pred[] compare(%i, %n), direction=LT
      }

      %body {
        %param = (u32[], u32[], $0[], $0[]) parameter(0)
        %i = u32[] get-tuple-element(%param), index=0
        %n = u32[] get-tuple-element(%param), index=1
        %acc = $0[] get-tuple-element(%param), index=2
        %x = $0[] get-tuple-element(%param), index=3

        %one = u32[] constant(1)
        %next_i = u32[] add(%i, %one)
        %new_acc = $0[] add(%acc, %x)
        ROOT %ret = (u32[], u32[], $0[], $0[]) tuple(%next_i, %n, %new_acc, %x)
      }

      ENTRY %main {
        %zero = u32[] constant(0)
        %init = u32[] constant(0)
        %n = u32[] parameter(0)
        %x = $0[] parameter(1)

        %while.tuple = (u32[], u32[], $0[], $0[]) tuple(%zero, %n, %init, %x)
        %result = (u32[], u32[], $0[], $0[]) while(%while.tuple), condition=%condition, body=%body
        ROOT %ret = $0[] get-tuple-element(%result), index=2
      }
    )",
                                 x_typename_);

    auto x = BaseIntTest<T>::GetRandomValue();
    auto n = base::Uniform<uint32_t>() % 10;
    literals_.push_back(LiteralUtil::CreateR0<uint32_t>(n));
    literals_.push_back(LiteralUtil::CreateR0<T>(x));
    expected_literal_ = LiteralUtil::CreateR0<T>(n * x);
  }

 private:
  void SetUpBroadcastTensorR1ToR3Helper(
      int64_t m, int64_t d,
      std::function<Literal(const std::vector<T>&)> callback) {
    hlo_text_ = absl::Substitute(R"(
    ENTRY %main {
      %x = $0[$1] parameter(0)

      ROOT %ret = $0[$1, $1, $1] broadcast(%x), dimensions={$2}
    }
  )",
                                 x_typename_, m, d);

    auto x = base::CreateVector(
        m, []() { return BaseIntTest<T>::GetRandomValue(); });
    literals_.push_back(LiteralUtil::CreateR1<T>(x));
    expected_literal_ = callback(x);
  }

  void SetUpReverseHelper(int64_t d0, int64_t d1, int64_t d2, bool reverse_d0,
                          bool reverse_d1, bool reverse_d2) {
    Array3D<T> x_array(d0, d1, d2);
    for (int64_t i = 0; i < d0; ++i) {
      for (int64_t j = 0; j < d1; ++j) {
        for (int64_t k = 0; k < d2; ++k) {
          x_array({i, j, k}) = BaseIntTest<T>::GetRandomValue();
        }
      }
    }
    Array3D<T> expected_array(d0, d1, d2);
    for (int64_t i = 0; i < d0; ++i) {
      for (int64_t j = 0; j < d1; ++j) {
        for (int64_t k = 0; k < d2; ++k) {
          int64_t ri = reverse_d0 ? d0 - i - 1 : i;
          int64_t rj = reverse_d1 ? d1 - j - 1 : j;
          int64_t rk = reverse_d2 ? d2 - k - 1 : k;
          expected_array({i, j, k}) = x_array({ri, rj, rk});
        }
      }
    }
    literals_.push_back(LiteralUtil::CreateR3FromArray3D<T>(x_array));
    expected_literal_ = LiteralUtil::CreateR3FromArray3D<T>(expected_array);
  }

  void SetUpTransposeHelper(int64_t d0, int64_t d1, int64_t d2) {
    Array3D<T> x_array(d0, d1, d2);
    for (int64_t i = 0; i < d0; ++i) {
      for (int64_t j = 0; j < d1; ++j) {
        for (int64_t k = 0; k < d2; ++k) {
          x_array({i, j, k}) = BaseIntTest<T>::GetRandomValue();
        }
      }
    }
    literals_.push_back(LiteralUtil::CreateR3FromArray3D<T>(x_array));
    expected_literal_ = literals_[0].Transpose({2, 0, 1});
  }

  void SetUpIotaHelper(int64_t iota_dimension) {
    constexpr static int64_t D0 = 2;
    constexpr static int64_t D1 = 3;

    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        ROOT %ret = $0[$1, $2] iota(), iota_dimension=$3
      }
    )",
                                 x_typename_, D0, D1, iota_dimension);
    Shape iota_shape(ShapeUtil::MakeShape(
        primitive_util::NativeToPrimitiveType<T>(), {D0, D1}));
    Literal result(iota_shape);
    ShapeUtil::ForEachIndexNoStatus(
        iota_shape, [&](absl::Span<const int64_t> idx) {
          result.Set(idx, static_cast<T>(idx[iota_dimension]));
          return true;
        });
    expected_literal_ = std::move(result);
  }
};

}  // namespace zkx::cpu

#endif  // ZKX_BACKENDS_CPU_CODEGEN_INT_TEST_H_
