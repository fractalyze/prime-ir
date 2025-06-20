// RUN: zkir-opt -canonicalize -field-to-mod-arith -mod-arith-to-arith -canonicalize %s | FileCheck %s -enable-var-scope

!PF17 = !field.pf<17:i32>

// CHECK-LABEL: @test_constant_folding_scalar_mul
// CHECK-SAME: (%arg0: [[T:.*]]) -> [[T]] {
func.func @test_constant_folding_scalar_mul(%arg0: tensor<8x!PF17>) -> tensor<8x!PF17> {
  // CHECK: arith.constant dense<[1, 5, 3, 7, 2, 6, 4, 8]> : [[C:.*]]
  // CHECK-NOT: tensor_ext.bit_reverse
  %const = arith.constant dense<[1, 2, 3, 4, 5, 6, 7, 8]> : tensor<8xi32>
  %twiddles = field.pf.encapsulate %const : tensor<8xi32> -> tensor<8x!PF17>
  %arg0_rev = tensor_ext.bit_reverse %arg0 : tensor<8x!PF17>
  %product_rev = field.mul %arg0_rev, %twiddles : tensor<8x!PF17>
  %product = tensor_ext.bit_reverse %product_rev : tensor<8x!PF17>
  return %product : tensor<8x!PF17>
}
