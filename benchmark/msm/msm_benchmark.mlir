!PFm = !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256, true>
!SFm = !field.pf<21888242871839275222246405745257275088548364400416034343698204186575808495617:i256, true>

#a = #field.pf.elem<0:i256> : !PFm
#b = #field.pf.elem<3:i256> : !PFm
#1 = #field.pf.elem<1:i256> : !PFm
#2 = #field.pf.elem<2:i256> : !PFm

#curve = #elliptic_curve.sw<#a, #b, (#1, #2)>
!affine = !elliptic_curve.affine<#curve>
!jacobian = !elliptic_curve.jacobian<#curve>
!xyzz = !elliptic_curve.xyzz<#curve>

!SFm_T = tensor<1048576x!SFm>
!SFm_M = memref<1048576x!SFm>
!points_T = tensor<1048576x!affine>
!points_M = memref<1048576x!affine>

func.func @msm(%scalars : !SFm_M, %points : !points_M) attributes { llvm.emit_c_interface } {
  %s = bufferization.to_tensor %scalars restrict writable : !SFm_M to !SFm_T
  %p = bufferization.to_tensor %points restrict writable : !points_M to !points_T
  %f = elliptic_curve.msm %s, %p : !SFm_T, !points_T -> !xyzz
  %res = elliptic_curve.convert_point_type %f : !xyzz -> !jacobian
  return
}
