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

#include "gtest/gtest.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveDialect.h"
#include "prime_ir/Dialect/EllipticCurve/IR/PointOperation.h"
#include "zk_dtypes/include/elliptic_curve/bn/bn254/g1.h"
#include "zk_dtypes/include/elliptic_curve/bn/bn254/g2.h"

namespace mlir::prime_ir::elliptic_curve {
namespace {

class ECRoundTripTest : public testing::Test {
protected:
  static void SetUpTestSuite() {
    context.loadDialect<EllipticCurveDialect>();
    context.loadDialect<func::FuncDialect>();
  }

  /// parse → print → re-parse → re-print → assert equal
  void verifyRoundTrip(StringRef mlirSource) {
    auto module1 =
        parseSourceString<ModuleOp>(mlirSource, ParserConfig(&context));
    ASSERT_TRUE(module1) << "first parse failed";

    std::string printed1;
    {
      llvm::raw_string_ostream os(printed1);
      module1->print(os);
    }

    auto module2 =
        parseSourceString<ModuleOp>(printed1, ParserConfig(&context));
    ASSERT_TRUE(module2) << "second parse failed";

    std::string printed2;
    {
      llvm::raw_string_ostream os(printed2);
      module2->print(os);
    }

    EXPECT_EQ(printed1, printed2);
  }

  /// Wrap a type in a func.func identity function for round-trip testing.
  static std::string wrapInFunc(StringRef funcName, StringRef typeStr) {
    return ("func.func @" + funcName + "(%arg0: " + typeStr + ") -> " +
            typeStr + " {\n  return %arg0 : " + typeStr + "\n}\n")
        .str();
  }

  /// Get the MLIR textual representation of a point type from a zk_dtypes
  /// point type.
  template <typename ZkDtypePoint>
  static std::string pointTypeStr() {
    Type ty = AffinePointOperation::getPointType<ZkDtypePoint>(&context);
    std::string str;
    llvm::raw_string_ostream os(str);
    ty.print(os);
    return str;
  }

  static MLIRContext context;
};

MLIRContext ECRoundTripTest::context;

//===----------------------------------------------------------------------===//
// G1 point type round-trips (standard base field)
//===----------------------------------------------------------------------===//

TEST_F(ECRoundTripTest, G1AffineStd) {
  verifyRoundTrip(wrapInFunc("g1_affine_std",
                             pointTypeStr<zk_dtypes::bn254::G1AffinePoint>()));
}

TEST_F(ECRoundTripTest, G1JacobianStd) {
  verifyRoundTrip(wrapInFunc(
      "g1_jacobian_std", pointTypeStr<zk_dtypes::bn254::G1JacobianPoint>()));
}

TEST_F(ECRoundTripTest, G1XYZZStd) {
  verifyRoundTrip(
      wrapInFunc("g1_xyzz_std", pointTypeStr<zk_dtypes::bn254::G1PointXyzz>()));
}

//===----------------------------------------------------------------------===//
// G1 point type round-trips (Montgomery base field)
//===----------------------------------------------------------------------===//

TEST_F(ECRoundTripTest, G1AffineMont) {
  verifyRoundTrip(wrapInFunc(
      "g1_affine_mont", pointTypeStr<zk_dtypes::bn254::G1AffinePointMont>()));
}

TEST_F(ECRoundTripTest, G1XYZZMont) {
  verifyRoundTrip(wrapInFunc(
      "g1_xyzz_mont", pointTypeStr<zk_dtypes::bn254::G1PointXyzzMont>()));
}

//===----------------------------------------------------------------------===//
// G2 point type round-trips
//===----------------------------------------------------------------------===//

TEST_F(ECRoundTripTest, G2Affine) {
  verifyRoundTrip(
      wrapInFunc("g2_affine", pointTypeStr<zk_dtypes::bn254::G2AffinePoint>()));
}

TEST_F(ECRoundTripTest, G2XYZZ) {
  verifyRoundTrip(
      wrapInFunc("g2_xyzz", pointTypeStr<zk_dtypes::bn254::G2PointXyzz>()));
}

//===----------------------------------------------------------------------===//
// EC operations round-trip
//===----------------------------------------------------------------------===//

TEST_F(ECRoundTripTest, ECOperations) {
  std::string affine = pointTypeStr<zk_dtypes::bn254::G1AffinePointMont>();
  std::string jacobian = pointTypeStr<zk_dtypes::bn254::G1JacobianPointMont>();

  std::string source =
      ("func.func @ec_ops(%a: " + affine + ", %b: " + affine + ") -> " +
       affine +
       " {\n"
       "  %sum = elliptic_curve.add %a, %b : " +
       affine + ", " + affine + " -> " + jacobian +
       "\n"
       "  %dbl = elliptic_curve.double %sum : " +
       jacobian + " -> " + jacobian +
       "\n"
       "  %neg = elliptic_curve.negate %dbl : " +
       jacobian +
       "\n"
       "  %result = elliptic_curve.convert_point_type %neg : " +
       jacobian + " -> " + affine +
       "\n"
       "  return %result : " +
       affine + "\n}\n");

  verifyRoundTrip(source);
}

TEST_F(ECRoundTripTest, XYZZOperations) {
  std::string xyzz = pointTypeStr<zk_dtypes::bn254::G1PointXyzzMont>();

  std::string source =
      ("func.func @xyzz_ops(%a: " + xyzz + ", %b: " + xyzz + ") -> " + xyzz +
       " {\n"
       "  %sum = elliptic_curve.add %a, %b : " +
       xyzz + ", " + xyzz + " -> " + xyzz +
       "\n"
       "  %dbl = elliptic_curve.double %sum : " +
       xyzz + " -> " + xyzz +
       "\n"
       "  return %dbl : " +
       xyzz + "\n}\n");

  verifyRoundTrip(source);
}

} // namespace
} // namespace mlir::prime_ir::elliptic_curve
