!PF = !field.pf<21888242871839275222246405745257275088548364400416034343698204186575808495617 : i256>
!PFm = !field.pf<21888242871839275222246405745257275088548364400416034343698204186575808495617 : i256, true>

#root_elem = #field.pf.elem<17220337697351015657950521176323262483320249231368149235373741788599650842711:i256> : !PF
#root_of_unity = #field.root_of_unity<#root_elem, 1048576:i256>

func.func @ntt_cpu(%arg0 : memref<1048576x!PFm>) attributes { llvm.emit_c_interface } {
  %t = bufferization.to_tensor %arg0 restrict writable : memref<1048576x!PFm> to tensor<1048576x!PFm>
  %res = poly.ntt %t into %t {root=#root_of_unity} : tensor<1048576x!PFm>
  bufferization.materialize_in_destination %res in writable %arg0 : (tensor<1048576x!PFm>, memref<1048576x!PFm>) -> ()
  return
}

func.func @intt_cpu(%arg0 : memref<1048576x!PFm>) attributes { llvm.emit_c_interface } {
  %t = bufferization.to_tensor %arg0 restrict writable : memref<1048576x!PFm> to tensor<1048576x!PFm>
  %res = poly.ntt %t into %t {root=#root_of_unity} inverse=true : tensor<1048576x!PFm>
  bufferization.materialize_in_destination %res in writable %arg0 : (tensor<1048576x!PFm>, memref<1048576x!PFm>) -> ()
  return
}
