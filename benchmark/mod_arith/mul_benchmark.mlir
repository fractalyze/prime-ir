!mod = !mod_arith.int<21888242871839275222246405745257275088548364400416034343698204186575808495617 : i256>
!modm = !mod_arith.int<21888242871839275222246405745257275088548364400416034343698204186575808495617 : i256, true>

func.func @mul(%arg0: memref<1048576x!mod>, %arg1: memref<1048576x!mod>, %out: memref<!mod>) attributes { llvm.emit_c_interface } {
  linalg.dot ins(%arg0, %arg1: memref<1048576x!mod>, memref<1048576x!mod>) outs(%out: memref<!mod>)
  return
}

func.func @mont_mul(%arg0 : memref<1048576x!modm>, %arg1 : memref<1048576x!modm>, %out: memref<!modm>) attributes { llvm.emit_c_interface } {
  linalg.dot ins(%arg0, %arg1: memref<1048576x!modm>, memref<1048576x!modm>) outs(%out: memref<!modm>)
  return
}
