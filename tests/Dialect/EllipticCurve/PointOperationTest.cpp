/* Copyright 2026 The PrimeIR Authors.

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

#include "prime_ir/Dialect/EllipticCurve/IR/PointOperation.h"

#include "gtest/gtest.h"
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveDialect.h"
#include "zk_dtypes/include/elliptic_curve/bn/bn254/g1.h"
#include "zk_dtypes/include/elliptic_curve/bn/bn254/g2.h"

namespace mlir::prime_ir::elliptic_curve {
namespace {

template <typename Point>
class PointOperationTest : public testing::Test {
public:
  constexpr static PointKind Kind = getPointKind<Point>();

  static void SetUpTestSuite() { context.loadDialect<EllipticCurveDialect>(); }

  template <typename R>
  void runBinaryOperationTest(
      std::function<R(const Point &, const Point &)> p_operation,
      std::function<PointOperationBase<getPointKind<R>()>(
          const PointOperationBase<Kind> &, const PointOperationBase<Kind> &)>
          pt_operation) {
    auto a = Point::Random();
    auto b = Point::Random();
    runBinaryOperationTest<R>(p_operation, pt_operation, a, b);
  }

  template <typename R>
  void runBinaryOperationTest(
      std::function<R(const Point &, const Point &)> p_operation,
      std::function<PointOperationBase<getPointKind<R>()>(
          const PointOperationBase<Kind> &, const PointOperationBase<Kind> &)>
          pt_operation,
      const Point &a, const Point &b) {
    constexpr PointKind ResultKind = getPointKind<R>();
    auto ptA = PointOperationBase<Kind>::fromZkDtype(&context, a);
    auto ptB = PointOperationBase<Kind>::fromZkDtype(&context, b);
    EXPECT_EQ(PointOperationBase<ResultKind>::fromZkDtype(&context,
                                                          p_operation(a, b)),
              pt_operation(ptA, ptB));
  }

  template <typename R>
  void
  runUnaryOperationTest(std::function<R(const Point &)> p_operation,
                        std::function<PointOperationBase<getPointKind<R>()>(
                            const PointOperationBase<Kind> &)>
                            pt_operation) {
    auto a = Point::Random();
    runUnaryOperationTest<R>(p_operation, pt_operation, a);
  }

  template <typename R>
  void
  runUnaryOperationTest(std::function<R(const Point &)> p_operation,
                        std::function<PointOperationBase<getPointKind<R>()>(
                            const PointOperationBase<Kind> &)>
                            pt_operation,
                        const Point &a) {
    constexpr PointKind ResultKind = getPointKind<R>();
    auto ptA = PointOperationBase<Kind>::fromZkDtype(&context, a);
    EXPECT_EQ(
        PointOperationBase<ResultKind>::fromZkDtype(&context, p_operation(a)),
        pt_operation(ptA));
  }

  static MLIRContext context;
};

template <typename F>
MLIRContext PointOperationTest<F>::context;

using PointTypes = testing::Types<
    zk_dtypes::bn254::G1AffinePoint, zk_dtypes::bn254::G1JacobianPoint,
    zk_dtypes::bn254::G1PointXyzz, zk_dtypes::bn254::G2AffinePoint>;
TYPED_TEST_SUITE(PointOperationTest, PointTypes);

//===----------------------------------------------------------------------===//
// Binary Operations
//===----------------------------------------------------------------------===//

TYPED_TEST(PointOperationTest, Add) {
  using PointType = TypeParam;

  if constexpr (zk_dtypes::IsAffinePoint<PointType>) {
    using Curve = typename PointType::Curve;
    this->template runBinaryOperationTest<zk_dtypes::JacobianPoint<Curve>>(
        [](const PointType &a, const PointType &b) { return a + b; },
        [](const auto &a, const auto &b) { return a + b; });
  } else {
    this->template runBinaryOperationTest<PointType>(
        [](const PointType &a, const PointType &b) { return a + b; },
        [](const auto &a, const auto &b) { return a + b; });
  }
}

TYPED_TEST(PointOperationTest, Sub) {
  using PointType = TypeParam;

  if constexpr (zk_dtypes::IsAffinePoint<PointType>) {
    using Curve = typename PointType::Curve;

    this->template runBinaryOperationTest<zk_dtypes::JacobianPoint<Curve>>(
        [](const PointType &a, const PointType &b) { return a - b; },
        [](const auto &a, const auto &b) { return a - b; });
  } else {
    this->template runBinaryOperationTest<PointType>(
        [](const PointType &a, const PointType &b) { return a - b; },
        [](const auto &a, const auto &b) { return a - b; });
  }
}

TYPED_TEST(PointOperationTest, Negate) {
  using PointType = TypeParam;

  this->template runUnaryOperationTest<PointType>(
      [](const PointType &a) { return -a; }, [](const auto &a) { return -a; });
}

TYPED_TEST(PointOperationTest, Double) {
  using PointType = TypeParam;

  if constexpr (zk_dtypes::IsAffinePoint<PointType>) {
    this->template runUnaryOperationTest<
        zk_dtypes::JacobianPoint<typename PointType::Curve>>(
        [](const PointType &a) { return a.Double(); },
        [](const auto &a) { return a.dbl(); });
  } else {
    this->template runUnaryOperationTest<PointType>(
        [](const PointType &a) { return a.Double(); },
        [](const auto &a) { return a.dbl(); });
  }
}

TYPED_TEST(PointOperationTest, Convert) {
  using PointType = TypeParam;
  using Curve = typename PointType::Curve;

  if constexpr (!zk_dtypes::IsAffinePoint<PointType>) {
    this->template runUnaryOperationTest<zk_dtypes::AffinePoint<Curve>>(
        [](const PointType &a) { return a.ToAffine(); },
        [](const auto &a) { return a.template convert<PointKind::kAffine>(); });
  }
  if constexpr (!zk_dtypes::IsJacobianPoint<PointType>) {
    this->template runUnaryOperationTest<zk_dtypes::JacobianPoint<Curve>>(
        [](const PointType &a) { return a.ToJacobian(); },
        [](const auto &a) {
          return a.template convert<PointKind::kJacobian>();
        });
  }
  if constexpr (!zk_dtypes::IsPointXyzz<PointType>) {
    this->template runUnaryOperationTest<zk_dtypes::PointXyzz<Curve>>(
        [](const PointType &a) { return a.ToXyzz(); },
        [](const auto &a) { return a.template convert<PointKind::kXYZZ>(); });
  }
}

} // namespace
} // namespace mlir::prime_ir::elliptic_curve
