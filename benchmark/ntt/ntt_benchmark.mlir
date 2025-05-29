!coeff_ty = !field.pf<21888242871839275222246405745257275088548364400416034343698204186575808495617 : i256>
!coefft_ty = tensor<1048576x!coeff_ty>
!memref_ty = memref<1048576x!coeff_ty>

#root_elem = #field.pf.elem<17220337697351015657950521176323262483320249231368149235373741788599650842711:i256> : !coeff_ty
#root_of_unity = #field.root_of_unity<#root_elem, 1048576:i256>
#root = #poly.primitive_root<root_of_unity=#root_of_unity>

!mod = !mod_arith.int<21888242871839275222246405745257275088548364400416034343698204186575808495617 : i256>
#mont = #mod_arith.montgomery<!mod>
#root_mont = #poly.primitive_root<root_of_unity=#root_of_unity, montgomery=#mont>

func.func @ntt(%arg0 : !memref_ty) attributes { llvm.emit_c_interface } {
  %t = bufferization.to_tensor %arg0 restrict writable : !memref_ty to !coefft_ty
  poly.ntt %t {root=#root} : !coefft_ty
  return
}

func.func @intt(%arg0 : !memref_ty) attributes { llvm.emit_c_interface } {
  %t = bufferization.to_tensor %arg0 restrict writable : !memref_ty to !coefft_ty
  poly.intt %t {root=#root} : !coefft_ty
  return
}

func.func @ntt_mont(%arg0 : !memref_ty) attributes { llvm.emit_c_interface } {
  %t = bufferization.to_tensor %arg0 restrict writable : !memref_ty to !coefft_ty
  poly.ntt %t {root=#root_mont} : !coefft_ty
  return
}

func.func @intt_mont(%arg0 : !memref_ty) attributes { llvm.emit_c_interface } {
  %t = bufferization.to_tensor %arg0 restrict writable : !memref_ty to !coefft_ty
  poly.intt %t {root=#root_mont} : !coefft_ty
  return
}
