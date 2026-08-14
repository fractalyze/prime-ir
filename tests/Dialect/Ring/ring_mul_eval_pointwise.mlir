// RUN: prime-ir-opt %s -ring-to-mod-arith \
// RUN:   | FileCheck %s --implicit-check-not=poly. --implicit-check-not=field.

// The evaluation basis is what the transform above this dialect hands back. In
// it CRT has already diagonalised the ring, so the product is one mod_arith.mul
// per limb and the lowering emits no transform of its own — the implicit
// check-nots forbid any poly or field op from reappearing here.

!Rq_eval = !ring.rq<[12289, 40961], 8 : i32, eval>

// CHECK-LABEL: func.func @mul_in_eval_basis_is_pointwise
// CHECK: mod_arith.mul
func.func @mul_in_eval_basis_is_pointwise(%a: !Rq_eval, %b: !Rq_eval) -> !Rq_eval {
  %c = ring.mul %a, %b : !Rq_eval
  return %c : !Rq_eval
}
