!coeff_ty = !field.pf<21888242871839275222246405745257275088548364400416034343698204186575808495617 : i256>
!poly_ty = !poly.polynomial<!coeff_ty, 1048575>
!coefft_ty = tensor<1048576x!coeff_ty>
!intt_ty = tensor<1048576xi256>

#root_elem = #field.pf_elem<17220337697351015657950521176323262483320249231368149235373741788599650842711:i256> : !coeff_ty
#root = #poly.primitive_root<root=#root_elem, degree=1048576:i256>

!mod = !mod_arith.int<21888242871839275222246405745257275088548364400416034343698204186575808495617 : i256>
#mont = #mod_arith.montgomery<!mod>
#root_mont = #poly.primitive_root<root=#root_elem, degree=1048576:i256, montgomery=#mont>

func.func @input_generation() -> !poly_ty attributes { llvm.emit_c_interface } {
  %c42 = arith.constant 6420 : i256
  %full = tensor.splat %c42 : !intt_ty
  %coeffs = field.pf.encapsulate %full : !intt_ty -> !coefft_ty
  %poly = poly.from_tensor %coeffs : !coefft_ty -> !poly_ty
  return %poly : !poly_ty
}

func.func @ntt(%arg0 : !poly_ty) -> !intt_ty attributes { llvm.emit_c_interface } {
  %0 = poly.ntt %arg0 {root=#root} : !poly_ty -> !coefft_ty
  %1 = field.pf.extract %0 : !coefft_ty -> !intt_ty
  return %1 : !intt_ty
}

func.func @intt(%arg0 : !intt_ty) -> !poly_ty attributes { llvm.emit_c_interface } {
  %0 = field.pf.encapsulate %arg0 : !intt_ty -> !coefft_ty
  %1 = poly.intt %0 {root=#root} : !coefft_ty -> !poly_ty
  return %1 :!poly_ty
}

func.func @ntt_mont(%arg0 : !poly_ty) -> !intt_ty attributes { llvm.emit_c_interface } {
  %0 = poly.ntt %arg0 {root=#root_mont} : !poly_ty -> !coefft_ty
  %1 = field.pf.extract %0 : !coefft_ty -> !intt_ty
  return %1 : !intt_ty
}

func.func @intt_mont(%arg0 : !intt_ty) -> !poly_ty attributes { llvm.emit_c_interface } {
  %0 = field.pf.encapsulate %arg0 : !intt_ty -> !coefft_ty
  %1 = poly.intt %0 {root=#root_mont} : !coefft_ty -> !poly_ty
  return %1 :!poly_ty
}
