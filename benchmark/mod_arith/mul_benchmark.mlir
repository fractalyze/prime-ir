!mod = !mod_arith.int<21888242871839275222246405745257275088548364400416034343698204186575808495617 : i256>
#mont = #mod_arith.montgomery<!mod>

func.func @mul(%arg0 : i256) -> i256 attributes { llvm.emit_c_interface } {
  %0 = mod_arith.encapsulate %arg0 : i256 -> !mod
  %1 = mod_arith.mul %0, %0 : !mod
  %2 = mod_arith.mul %0, %1 : !mod
  %3 = mod_arith.extract %2 : !mod -> i256
  return %3 : i256
}

func.func @mont_mul(%arg0 : i256) -> i256 attributes { llvm.emit_c_interface } {
  %0 = mod_arith.encapsulate %arg0 : i256 -> !mod
  %1 = mod_arith.mont_mul %0, %0 {montgomery = #mont} : !mod
  %2 = mod_arith.mont_mul %0, %1 {montgomery = #mont} : !mod
  %3 = mod_arith.extract %2 : !mod -> i256
  return %3 : i256
}
