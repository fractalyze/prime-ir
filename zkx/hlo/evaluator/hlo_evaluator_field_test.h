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

#ifndef ZKX_HLO_EVALUATOR_HLO_EVALUATOR_FIELD_TEST_H_
#define ZKX_HLO_EVALUATOR_HLO_EVALUATOR_FIELD_TEST_H_

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"
#include "gtest/gtest.h"

#include "xla/tsl/platform/statusor.h"
#include "zkx/array2d.h"
#include "zkx/base/containers/container_util.h"
#include "zkx/hlo/evaluator/hlo_evaluator.h"
#include "zkx/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "zkx/literal.h"
#include "zkx/literal_util.h"
#include "zkx/primitive_util.h"
#include "zkx/tests/literal_test_util.h"

namespace zkx {

// Base class for HloEvaluator field tests.
template <typename F>
class HloEvaluatorFieldTestBase : public HloHardwareIndependentTestBase {
 public:
  void SetUp() override {
    HloHardwareIndependentTestBase::SetUp();
    typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<F>());
  }

 protected:
  void RunAndVerify() {
    TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_));
    HloEvaluator evaluator;

    std::vector<const Literal*> literal_ptrs;
    literal_ptrs.reserve(literals_.size());
    for (const auto& literal : literals_) {
      literal_ptrs.push_back(&literal);
    }

    TF_ASSERT_OK_AND_ASSIGN(
        Literal result,
        evaluator.Evaluate(*module->entry_computation(), literal_ptrs));

    EXPECT_TRUE(LiteralTestUtil::Equal(expected_, result));
  }

  std::string typename_;
  std::string hlo_;
  std::vector<Literal> literals_;
  Literal expected_;
};

// Tests for binary operations on field scalars.
template <typename F>
class FieldScalarBinaryTest : public HloEvaluatorFieldTestBase<F> {
 public:
  void SetUp() override {
    HloEvaluatorFieldTestBase<F>::SetUp();
    x_ = F::Random();
    y_ = F::Random();
    this->literals_.push_back(LiteralUtil::CreateR0<F>(x_));
    this->literals_.push_back(LiteralUtil::CreateR0<F>(y_));
  }

 protected:
  void SetUpAdd() {
    this->hlo_ = absl::Substitute(R"(
      HloModule add_field
      ENTRY main {
        x = $0[] parameter(0)
        y = $0[] parameter(1)
        ROOT ret = $0[] add(x, y)
      }
    )",
                                  this->typename_);
    this->expected_ = LiteralUtil::CreateR0<F>(x_ + y_);
  }

  void SetUpSubtract() {
    this->hlo_ = absl::Substitute(R"(
      HloModule subtract_field
      ENTRY main {
        x = $0[] parameter(0)
        y = $0[] parameter(1)
        ROOT ret = $0[] subtract(x, y)
      }
    )",
                                  this->typename_);
    this->expected_ = LiteralUtil::CreateR0<F>(x_ - y_);
  }

  void SetUpMultiply() {
    this->hlo_ = absl::Substitute(R"(
      HloModule multiply_field
      ENTRY main {
        x = $0[] parameter(0)
        y = $0[] parameter(1)
        ROOT ret = $0[] multiply(x, y)
      }
    )",
                                  this->typename_);
    this->expected_ = LiteralUtil::CreateR0<F>(x_ * y_);
  }

  void SetUpDivide() {
    // Ensure y is non-zero for division
    while (y_.IsZero()) {
      y_ = F::Random();
    }
    this->literals_[1] = LiteralUtil::CreateR0<F>(y_);

    this->hlo_ = absl::Substitute(R"(
      HloModule divide_field
      ENTRY main {
        x = $0[] parameter(0)
        y = $0[] parameter(1)
        ROOT ret = $0[] divide(x, y)
      }
    )",
                                  this->typename_);
    this->expected_ = LiteralUtil::CreateR0<F>(x_ / y_);
  }

  void SetUpMaximum() {
    this->hlo_ = absl::Substitute(R"(
      HloModule maximum_field
      ENTRY main {
        x = $0[] parameter(0)
        y = $0[] parameter(1)
        ROOT ret = $0[] maximum(x, y)
      }
    )",
                                  this->typename_);
    this->expected_ = LiteralUtil::CreateR0<F>(std::max(x_, y_));
  }

  void SetUpMinimum() {
    this->hlo_ = absl::Substitute(R"(
      HloModule minimum_field
      ENTRY main {
        x = $0[] parameter(0)
        y = $0[] parameter(1)
        ROOT ret = $0[] minimum(x, y)
      }
    )",
                                  this->typename_);
    this->expected_ = LiteralUtil::CreateR0<F>(std::min(x_, y_));
  }

  void SetUpCompareEq() {
    this->hlo_ = absl::Substitute(R"(
      HloModule compare_eq_field
      ENTRY main {
        x = $0[] parameter(0)
        y = $0[] parameter(1)
        ROOT ret = pred[] compare(x, y), direction=EQ
      }
    )",
                                  this->typename_);
    this->expected_ = LiteralUtil::CreateR0<bool>(x_ == y_);
  }

  void SetUpCompareNe() {
    this->hlo_ = absl::Substitute(R"(
      HloModule compare_ne_field
      ENTRY main {
        x = $0[] parameter(0)
        y = $0[] parameter(1)
        ROOT ret = pred[] compare(x, y), direction=NE
      }
    )",
                                  this->typename_);
    this->expected_ = LiteralUtil::CreateR0<bool>(x_ != y_);
  }

  void SetUpCompareLt() {
    this->hlo_ = absl::Substitute(R"(
      HloModule compare_lt_field
      ENTRY main {
        x = $0[] parameter(0)
        y = $0[] parameter(1)
        ROOT ret = pred[] compare(x, y), direction=LT
      }
    )",
                                  this->typename_);
    this->expected_ = LiteralUtil::CreateR0<bool>(x_ < y_);
  }

  void SetUpCompareLe() {
    this->hlo_ = absl::Substitute(R"(
      HloModule compare_le_field
      ENTRY main {
        x = $0[] parameter(0)
        y = $0[] parameter(1)
        ROOT ret = pred[] compare(x, y), direction=LE
      }
    )",
                                  this->typename_);
    this->expected_ = LiteralUtil::CreateR0<bool>(x_ <= y_);
  }

  void SetUpCompareGt() {
    this->hlo_ = absl::Substitute(R"(
      HloModule compare_gt_field
      ENTRY main {
        x = $0[] parameter(0)
        y = $0[] parameter(1)
        ROOT ret = pred[] compare(x, y), direction=GT
      }
    )",
                                  this->typename_);
    this->expected_ = LiteralUtil::CreateR0<bool>(x_ > y_);
  }

  void SetUpCompareGe() {
    this->hlo_ = absl::Substitute(R"(
      HloModule compare_ge_field
      ENTRY main {
        x = $0[] parameter(0)
        y = $0[] parameter(1)
        ROOT ret = pred[] compare(x, y), direction=GE
      }
    )",
                                  this->typename_);
    this->expected_ = LiteralUtil::CreateR0<bool>(x_ >= y_);
  }

 private:
  F x_;
  F y_;
};

// Tests for ternary operations on field scalars.
template <typename F>
class FieldScalarTernaryTest : public HloEvaluatorFieldTestBase<F> {
 public:
  void SetUp() override {
    HloEvaluatorFieldTestBase<F>::SetUp();
    x_ = F::Random();
    y_ = F::Random();
    z_ = F::Random();
    this->literals_.push_back(LiteralUtil::CreateR0<F>(x_));
    this->literals_.push_back(LiteralUtil::CreateR0<F>(y_));
    this->literals_.push_back(LiteralUtil::CreateR0<F>(z_));
  }

 protected:
  void SetUpClamp() {
    // Ensure low <= high for clamp
    F low = x_;
    F high = z_;
    if (low > high) {
      std::swap(low, high);
      std::swap(this->literals_[0], this->literals_[2]);
    }

    this->hlo_ = absl::Substitute(R"(
      HloModule clamp_field
      ENTRY main {
        low = $0[] parameter(0)
        value = $0[] parameter(1)
        high = $0[] parameter(2)
        ROOT ret = $0[] clamp(low, value, high)
      }
    )",
                                  this->typename_);
    this->expected_ = LiteralUtil::CreateR0<F>(std::clamp(y_, low, high));
  }

  void SetUpSelect() {
    bool cond = static_cast<uint64_t>(x_.IsZero()) % 2 == 0;
    this->literals_[0] = LiteralUtil::CreateR0<bool>(cond);

    this->hlo_ = absl::Substitute(R"(
      HloModule select_field
      ENTRY main {
        cond = pred[] parameter(0)
        on_true = $0[] parameter(1)
        on_false = $0[] parameter(2)
        ROOT ret = $0[] select(cond, on_true, on_false)
      }
    )",
                                  this->typename_);
    this->expected_ = LiteralUtil::CreateR0<F>(cond ? y_ : z_);
  }

 private:
  F x_;
  F y_;
  F z_;
};

// Tests for operations on field tensors (R1).
template <typename F>
class FieldR1TensorTest : public HloEvaluatorFieldTestBase<F> {
 public:
  static constexpr int64_t kSize = 4;

  void SetUp() override {
    HloEvaluatorFieldTestBase<F>::SetUp();
    x_ = base::CreateVector(kSize, []() { return F::Random(); });
    this->literals_.push_back(LiteralUtil::CreateR1<F>(x_));
  }

 protected:
  void SetUpBroadcast() {
    F scalar = F::Random();
    this->literals_.clear();
    this->literals_.push_back(LiteralUtil::CreateR0<F>(scalar));

    this->hlo_ = absl::Substitute(R"(
      HloModule broadcast_field
      ENTRY main {
        x = $0[] parameter(0)
        ROOT ret = $0[$1] broadcast(x), dimensions={}
      }
    )",
                                  this->typename_, kSize);
    std::vector<F> expected(kSize, scalar);
    this->expected_ = LiteralUtil::CreateR1<F>(expected);
  }

  void SetUpSlice() {
    this->hlo_ = absl::Substitute(R"(
      HloModule slice_field
      ENTRY main {
        x = $0[$1] parameter(0)
        ROOT ret = $0[2] slice(x), slice={[1:3]}
      }
    )",
                                  this->typename_, kSize);
    std::vector<F> expected = {x_[1], x_[2]};
    this->expected_ = LiteralUtil::CreateR1<F>(expected);
  }

  void SetUpConcatenate() {
    std::vector<F> y = base::CreateVector(kSize, []() { return F::Random(); });
    this->literals_.push_back(LiteralUtil::CreateR1<F>(y));

    this->hlo_ = absl::Substitute(R"(
      HloModule concatenate_field
      ENTRY main {
        x = $0[$1] parameter(0)
        y = $0[$1] parameter(1)
        ROOT ret = $0[$2] concatenate(x, y), dimensions={0}
      }
    )",
                                  this->typename_, kSize, kSize * 2);
    std::vector<F> expected;
    expected.reserve(kSize * 2);
    expected.insert(expected.end(), x_.begin(), x_.end());
    expected.insert(expected.end(), y.begin(), y.end());
    this->expected_ = LiteralUtil::CreateR1<F>(expected);
  }

  void SetUpReverse() {
    this->hlo_ = absl::Substitute(R"(
      HloModule reverse_field
      ENTRY main {
        x = $0[$1] parameter(0)
        ROOT ret = $0[$1] reverse(x), dimensions={0}
      }
    )",
                                  this->typename_, kSize);
    std::vector<F> expected(x_.rbegin(), x_.rend());
    this->expected_ = LiteralUtil::CreateR1<F>(expected);
  }

  void SetUpReduceSum() {
    this->hlo_ = absl::Substitute(R"(
      HloModule reduce_sum_field
      add {
        lhs = $0[] parameter(0)
        rhs = $0[] parameter(1)
        ROOT sum = $0[] add(lhs, rhs)
      }
      ENTRY main {
        x = $0[$1] parameter(0)
        zero = $0[] constant(0)
        ROOT ret = $0[] reduce(x, zero), dimensions={0}, to_apply=add
      }
    )",
                                  this->typename_, kSize);
    F expected = F(0);
    for (int64_t i = 0; i < kSize; ++i) {
      expected = expected + x_[i];
    }
    this->expected_ = LiteralUtil::CreateR0<F>(expected);
  }

  void SetUpReduceProduct() {
    this->hlo_ = absl::Substitute(R"(
      HloModule reduce_product_field
      mul {
        lhs = $0[] parameter(0)
        rhs = $0[] parameter(1)
        ROOT product = $0[] multiply(lhs, rhs)
      }
      ENTRY main {
        x = $0[$1] parameter(0)
        one = $0[] constant(1)
        ROOT ret = $0[] reduce(x, one), dimensions={0}, to_apply=mul
      }
    )",
                                  this->typename_, kSize);
    F expected = F(1);
    for (int64_t i = 0; i < kSize; ++i) {
      expected = expected * x_[i];
    }
    this->expected_ = LiteralUtil::CreateR0<F>(expected);
  }

  void SetUpReduceMax() {
    this->hlo_ = absl::Substitute(R"(
      HloModule reduce_max_field
      max {
        lhs = $0[] parameter(0)
        rhs = $0[] parameter(1)
        ROOT result = $0[] maximum(lhs, rhs)
      }
      ENTRY main {
        x = $0[$1] parameter(0)
        init = $0[] constant(0)
        ROOT ret = $0[] reduce(x, init), dimensions={0}, to_apply=max
      }
    )",
                                  this->typename_, kSize);
    F expected = F(0);
    for (int64_t i = 0; i < kSize; ++i) {
      expected = std::max(expected, x_[i]);
    }
    this->expected_ = LiteralUtil::CreateR0<F>(expected);
  }

  void SetUpDynamicSlice() {
    this->hlo_ = absl::Substitute(R"(
      HloModule dynamic_slice_field
      ENTRY main {
        x = $0[$1] parameter(0)
        start = s32[] parameter(1)
        ROOT ret = $0[2] dynamic-slice(x, start), dynamic_slice_sizes={2}
      }
    )",
                                  this->typename_, kSize);
    int32_t start = 1;
    this->literals_.push_back(LiteralUtil::CreateR0<int32_t>(start));
    std::vector<F> expected = {x_[start], x_[start + 1]};
    this->expected_ = LiteralUtil::CreateR1<F>(expected);
  }

  void SetUpDynamicUpdateSlice() {
    this->hlo_ = absl::Substitute(R"(
      HloModule dynamic_update_slice_field
      ENTRY main {
        x = $0[$1] parameter(0)
        update = $0[2] parameter(1)
        start = s32[] parameter(2)
        ROOT ret = $0[$1] dynamic-update-slice(x, update, start)
      }
    )",
                                  this->typename_, kSize);
    std::vector<F> update = {F::Random(), F::Random()};
    int32_t start = 1;
    this->literals_.push_back(LiteralUtil::CreateR1<F>(update));
    this->literals_.push_back(LiteralUtil::CreateR0<int32_t>(start));
    std::vector<F> expected = x_;
    expected[start] = update[0];
    expected[start + 1] = update[1];
    this->expected_ = LiteralUtil::CreateR1<F>(expected);
  }

 private:
  std::vector<F> x_;
};

// Tests for operations on field tensors (R2).
template <typename F>
class FieldR2TensorTest : public HloEvaluatorFieldTestBase<F> {
 public:
  static constexpr int64_t kRows = 2;
  static constexpr int64_t kCols = 3;

  void SetUp() override {
    HloEvaluatorFieldTestBase<F>::SetUp();
    Array2D<F> x(kRows, kCols);
    for (int64_t i = 0; i < kRows; ++i) {
      for (int64_t j = 0; j < kCols; ++j) {
        x(i, j) = F::Random();
        x_[i][j] = x(i, j);
      }
    }
    this->literals_.push_back(LiteralUtil::CreateR2FromArray2D(x));
  }

 protected:
  void SetUpTranspose() {
    this->hlo_ = absl::Substitute(R"(
      HloModule transpose_field
      ENTRY main {
        x = $0[$1,$2] parameter(0)
        ROOT ret = $0[$2,$1] transpose(x), dimensions={1,0}
      }
    )",
                                  this->typename_, kRows, kCols);
    Array2D<F> expected(kCols, kRows);
    for (int64_t i = 0; i < kRows; ++i) {
      for (int64_t j = 0; j < kCols; ++j) {
        expected(j, i) = x_[i][j];
      }
    }
    this->expected_ = LiteralUtil::CreateR2FromArray2D(expected);
  }

  void SetUpReshape() {
    this->hlo_ = absl::Substitute(R"(
      HloModule reshape_field
      ENTRY main {
        x = $0[$1,$2] parameter(0)
        ROOT ret = $0[$3] reshape(x)
      }
    )",
                                  this->typename_, kRows, kCols, kRows * kCols);
    std::vector<F> expected;
    expected.reserve(kRows * kCols);
    for (int64_t i = 0; i < kRows; ++i) {
      for (int64_t j = 0; j < kCols; ++j) {
        expected.push_back(x_[i][j]);
      }
    }
    this->expected_ = LiteralUtil::CreateR1<F>(expected);
  }

 private:
  F x_[kRows][kCols];
};

// Tests for dot product operations on fields.
template <typename F>
class FieldDotTest : public HloEvaluatorFieldTestBase<F> {
 public:
  static constexpr int64_t kM = 2;
  static constexpr int64_t kK = 3;
  static constexpr int64_t kN = 4;

  void SetUp() override { HloEvaluatorFieldTestBase<F>::SetUp(); }

 protected:
  void SetUpDotVectorVector() {
    std::vector<F> x = base::CreateVector(kK, []() { return F::Random(); });
    std::vector<F> y = base::CreateVector(kK, []() { return F::Random(); });
    this->literals_.push_back(LiteralUtil::CreateR1<F>(x));
    this->literals_.push_back(LiteralUtil::CreateR1<F>(y));

    this->hlo_ = absl::Substitute(R"(
      HloModule dot_vv_field
      ENTRY main {
        x = $0[$1] parameter(0)
        y = $0[$1] parameter(1)
        ROOT ret = $0[] dot(x, y), lhs_contracting_dims={0}, rhs_contracting_dims={0}
      }
    )",
                                  this->typename_, kK);
    F expected = F(0);
    for (int64_t i = 0; i < kK; ++i) {
      expected = expected + x[i] * y[i];
    }
    this->expected_ = LiteralUtil::CreateR0<F>(expected);
  }

  void SetUpDotMatrixVector() {
    Array2D<F> a(kM, kK);
    std::vector<F> x = base::CreateVector(kK, []() { return F::Random(); });
    for (int64_t i = 0; i < kM; ++i) {
      for (int64_t j = 0; j < kK; ++j) {
        a(i, j) = F::Random();
      }
    }
    this->literals_.push_back(LiteralUtil::CreateR2FromArray2D(a));
    this->literals_.push_back(LiteralUtil::CreateR1<F>(x));

    this->hlo_ = absl::Substitute(R"(
      HloModule dot_mv_field
      ENTRY main {
        a = $0[$1,$2] parameter(0)
        x = $0[$2] parameter(1)
        ROOT ret = $0[$1] dot(a, x), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      }
    )",
                                  this->typename_, kM, kK);
    std::vector<F> expected(kM, F(0));
    for (int64_t i = 0; i < kM; ++i) {
      for (int64_t j = 0; j < kK; ++j) {
        expected[i] = expected[i] + a(i, j) * x[j];
      }
    }
    this->expected_ = LiteralUtil::CreateR1<F>(expected);
  }

  void SetUpDotMatrixMatrix() {
    Array2D<F> a(kM, kK);
    Array2D<F> b(kK, kN);
    for (int64_t i = 0; i < kM; ++i) {
      for (int64_t j = 0; j < kK; ++j) {
        a(i, j) = F::Random();
      }
    }
    for (int64_t i = 0; i < kK; ++i) {
      for (int64_t j = 0; j < kN; ++j) {
        b(i, j) = F::Random();
      }
    }
    this->literals_.push_back(LiteralUtil::CreateR2FromArray2D(a));
    this->literals_.push_back(LiteralUtil::CreateR2FromArray2D(b));

    this->hlo_ = absl::Substitute(R"(
      HloModule dot_mm_field
      ENTRY main {
        a = $0[$1,$2] parameter(0)
        b = $0[$2,$3] parameter(1)
        ROOT ret = $0[$1,$3] dot(a, b), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      }
    )",
                                  this->typename_, kM, kK, kN);
    Array2D<F> expected(kM, kN);
    for (int64_t i = 0; i < kM; ++i) {
      for (int64_t j = 0; j < kN; ++j) {
        expected(i, j) = F(0);
        for (int64_t k = 0; k < kK; ++k) {
          expected(i, j) = expected(i, j) + a(i, k) * b(k, j);
        }
      }
    }
    this->expected_ = LiteralUtil::CreateR2FromArray2D(expected);
  }
};

}  // namespace zkx

#endif  // ZKX_HLO_EVALUATOR_HLO_EVALUATOR_FIELD_TEST_H_
