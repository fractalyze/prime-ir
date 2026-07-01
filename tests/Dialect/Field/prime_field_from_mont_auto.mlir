// RUN: prime-ir-opt -field-to-mod-arith='lowering-mode=auto' %s \
// RUN:   | FileCheck %s -enable-var-scope

!PF1 = !field.pf<97:i32>
!PF1m = !field.pf<97:i32, true>
!PFmv = tensor<4x!PF1m>
!PFv = tensor<4x!PF1>

// Under 'auto' (the CPU trio's lowering mode) a multi-element prime-field
// from_mont scalarizes via tensor.generate -- a scalar mod_arith.from_mont per
// element -- so the CPU hero-emitter pipeline can lower it. A whole-tensor
// mod_arith.from_mont would become whole-tensor i256 arith it cannot scalarize.
// CHECK-LABEL: @test_from_mont_batch_auto
func.func @test_from_mont_batch_auto(%arg0: !PFmv) -> !PFv {
  // CHECK: tensor.generate
  // CHECK: mod_arith.from_mont
  %res = field.from_mont %arg0 : !PFv
  return %res : !PFv
}
