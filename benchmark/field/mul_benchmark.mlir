!mod = !mod_arith.int<21888242871839275222246405745257275088548364400416034343698204186575808495617 : i256>
!F = !field.pf<21888242871839275222246405745257275088548364400416034343698204186575808495617 : i256>
#mont = #mod_arith.montgomery<!mod>

func.func @mul(%arg0 : i256) -> i256 attributes { llvm.emit_c_interface } {
  %0 = field.pf.encapsulate %arg0 : i256 -> !F
  %1 = field.pf.mul %0, %0 : !F
  %2 = field.pf.mul %0, %1 : !F
  %3 = field.pf.extract %2 : !F -> i256
  return %3 : i256
}

func.func @mont_mul(%arg0 : i256) -> i256 attributes { llvm.emit_c_interface } {
  %0 = field.pf.encapsulate %arg0 : i256 -> !F
  %1 = field.pf.mont_mul %0, %0 {montgomery = #mont} : !F
  %2 = field.pf.mont_mul %0, %1 {montgomery = #mont} : !F
  %3 = field.pf.extract %2 : !F -> i256
  return %3 : i256
}
