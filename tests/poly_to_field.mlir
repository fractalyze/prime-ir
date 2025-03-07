// RUN: tools/zkir-opt -poly-to-field --split-input-file %s | FileCheck %s --enable-var-scope

!PF1 = !field.pf<7:i255>
!poly_ty1 = !poly.polynomial<!PF1, 3>
!poly_ty2 = !poly.polynomial<!PF1, 4>

// FIXME(batzor): without this line, the test will fail with the following error:
// LLVM ERROR: can't create Attribute 'mlir::polynomial::IntPolynomialAttr' because storage uniquer isn't initialized: the dialect was likely not loaded, or the attribute wasn't added with addAttributes<...>() in the Dialect::initialize() method.
#uni_poly = #poly.univariate_polynomial<x**6 + 1> : !poly_ty2

// CHECK-LABEL: @test_lower_constant
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_lower_constant() -> !poly_ty1 {
  // CHECK-NOT: poly.constant
  // CHECK: %[[CVAL:.*]] = arith.constant dense<[1, 0, 0, 1]> : [[TINT:.*]]
  // CHECK: %[[RES:.*]] = field.pf.encapsulate %[[CVAL]] : [[TINT]] -> [[T]]
  // CHECK: return %[[RES]] : [[T]]
  %res = poly.constant<x**3 + 1>:  !poly_ty1
  return %res: !poly_ty1
}

// CHECK-LABEL: @test_lower_add
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_add(%lhs : !poly_ty1, %rhs : !poly_ty1) -> !poly_ty1 {
  // CHECK-NOT: poly.add
  // CHECK: %[[RES:.*]] = field.pf.add %[[LHS]], %[[RHS]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  %res = poly.add %lhs, %rhs : !poly_ty1
  return %res : !poly_ty1
}

// CHECK-LABEL: @test_lower_sub
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_sub(%lhs : !poly_ty1, %rhs : !poly_ty1) -> !poly_ty1 {
  // CHECK-NOT: poly.sub
  // CHECK: %[[RES:.*]] = field.pf.sub %[[LHS]], %[[RHS]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  %res = poly.sub %lhs, %rhs : !poly_ty1
  return %res : !poly_ty1
}
