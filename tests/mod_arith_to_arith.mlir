!Zp = !mod_arith.int<65537 : i32>

func.func @test_lower_add(%lhs : !Zp, %rhs : !Zp) -> !Zp {
  %res = mod_arith.add %lhs, %rhs : !Zp
  return %res : !Zp
}
