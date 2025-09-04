#a = #field.pf.elem<0:i256> : !PF
#b = #field.pf.elem<3:i256> : !PF
#1 = #field.pf.elem<1:i256> : !PF
#2 = #field.pf.elem<2:i256> : !PF

#curve = #elliptic_curve.sw<#a, #b, (#1, #2)>
!affine = !elliptic_curve.affine<#curve>
!jacobian = !elliptic_curve.jacobian<#curve>
!xyzz = !elliptic_curve.xyzz<#curve>

#f2_elem = #field.f2.elem<#1, #2> : !QF
#g2curve = #elliptic_curve.sw<#f2_elem, #f2_elem, (#f2_elem, #f2_elem)>
!g2affine = !elliptic_curve.affine<#g2curve>
!g2jacobian = !elliptic_curve.jacobian<#g2curve>
!g2xyzz = !elliptic_curve.xyzz<#g2curve>
