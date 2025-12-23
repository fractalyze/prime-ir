
!Zp = !mod_arith.int<37 : i32>
!Zpm = !mod_arith.int<37 : i32, true>

// CHECK-LABEL: @test_cmp_splat_tensor_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_cmp_splat_tensor_fold() -> tensor<2xi1> {
  // CHECK: %[[C:.*]] = arith.constant dense<true> : [[T]]
  %0 = mod_arith.constant dense<2> : tensor<2x!Zp>
  %1 = mod_arith.constant dense<1> : tensor<2x!Zp>
  %2 = mod_arith.cmp ugt, %0, %1 : tensor<2x!Zp>
  return %2 : tensor<2xi1>
}
