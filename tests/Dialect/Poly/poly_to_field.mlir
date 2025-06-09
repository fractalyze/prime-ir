// RUN: zkir-opt -poly-to-field --split-input-file %s | FileCheck %s --enable-var-scope

!PF1 = !field.pf<7:i255>
!poly_ty1 = !poly.polynomial<!PF1, 3>
!poly_ty2 = !poly.polynomial<!PF1, 4>
#elem = #field.pf.elem<6:i255>  : !PF1
#root_of_unity = #field.root_of_unity<#elem, 2:i255>
#root = #poly.primitive_root<#root_of_unity>

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
  // CHECK: %[[RES:.*]] = field.add %[[LHS]], %[[RHS]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  %res = poly.add %lhs, %rhs : !poly_ty1
  return %res : !poly_ty1
}

// CHECK-LABEL: @test_lower_sub
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_sub(%lhs : !poly_ty1, %rhs : !poly_ty1) -> !poly_ty1 {
  // CHECK-NOT: poly.sub
  // CHECK: %[[RES:.*]] = field.sub %[[LHS]], %[[RHS]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  %res = poly.sub %lhs, %rhs : !poly_ty1
  return %res : !poly_ty1
}

// FIXME(batzor): lowering doesn't work if I try `to_tensor` on the argument polynomial
// CHECK-LABEL: @test_lower_to_tensor
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_lower_to_tensor() -> tensor<4x!PF1> {
  // CHECK-NOT: poly.constant
  // CHECK: %[[CVAL:.*]] = arith.constant dense<[1, 0, 0, 1]> : [[TINT:.*]]
  // CHECK: %[[TVAL:.*]] = field.pf.encapsulate %[[CVAL]] : [[TINT]] -> [[T]]
  %0 = poly.constant<x**3 + 1> : !poly_ty1
  // CHECK-NOT: poly.to_tensor
  // CHECK: %[[RES:.*]] = field.add %[[TVAL]], %[[TVAL]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  %1 = poly.to_tensor %0 : !poly_ty1 -> tensor<4x!PF1>
  %res = field.add %1, %1 : tensor<4x!PF1>
  return %res : tensor<4x!PF1>
}

// CHECK-LABEL: @test_lower_from_tensor
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_lower_from_tensor(%t : tensor<4x!PF1>) -> !poly_ty1 {
  // CHECK-NOT: poly.from_tensor
  // CHECK: return %[[LHS]] : [[T]]
  %res = poly.from_tensor %t : tensor<4x!PF1> -> !poly_ty1
  return %res : !poly_ty1
}

// CHECK-LABEL: @test_lower_ntt
// CHECK-SAME: (%[[INPUT:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_lower_ntt(%input : tensor<2x!PF1>) -> tensor<2x!PF1> {
  // CHECK-NOT: poly.ntt
  %res = poly.ntt %input {root=#root} : tensor<2x!PF1>
  return %res: tensor<2x!PF1>
}

// CHECK-LABEL: @test_lower_intt
// CHECK-SAME: (%[[INPUT:.*]]: [[T:.*]]) -> [[P:.*]] {
func.func @test_lower_intt(%input : tensor<2x!PF1>) -> tensor<2x!PF1> {
  // CHECK-NOT: poly.intt
  %res = poly.intt %input {root=#root} : tensor<2x!PF1>
  return %res: tensor<2x!PF1>
}
