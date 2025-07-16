!coeff_ty = !field.pf<21888242871839275222246405745257275088548364400416034343698204186575808495617 : i256>
!coefft_ty = tensor<1048576x!coeff_ty>
!memref_ty = memref<1048576x!coeff_ty>

!coeff_ty_mont = !field.pf<21888242871839275222246405745257275088548364400416034343698204186575808495617 : i256, true>
!coefft_ty_mont = tensor<1048576x!coeff_ty_mont>
!memref_ty_mont = memref<1048576x!coeff_ty_mont>

#root_elem = #field.pf.elem<17220337697351015657950521176323262483320249231368149235373741788599650842711:i256> : !coeff_ty
#root_of_unity = #field.root_of_unity<#root_elem, 1048576:i256>

func.func @ntt(%arg0 : !memref_ty) attributes { llvm.emit_c_interface } {
  %t = bufferization.to_tensor %arg0 restrict writable : !memref_ty to !coefft_ty
  poly.ntt %t into %t {root=#root_of_unity} : !coefft_ty
  return
}

func.func @intt(%arg0 : !memref_ty) attributes { llvm.emit_c_interface } {
  %t = bufferization.to_tensor %arg0 restrict writable : !memref_ty to !coefft_ty
  poly.ntt %t into %t {root=#root_of_unity} inverse=true : !coefft_ty
  return
}

func.func @ntt_mont(%arg0 : !memref_ty_mont) attributes { llvm.emit_c_interface } {
  %t = bufferization.to_tensor %arg0 restrict writable : !memref_ty_mont to !coefft_ty_mont
  poly.ntt %t into %t {root=#root_of_unity} : !coefft_ty_mont
  return
}

func.func @intt_mont(%arg0 : !memref_ty_mont) attributes { llvm.emit_c_interface } {
  %t = bufferization.to_tensor %arg0 restrict writable : !memref_ty_mont to !coefft_ty_mont
  poly.ntt %t into %t {root=#root_of_unity} inverse=true : !coefft_ty_mont
  return
}
