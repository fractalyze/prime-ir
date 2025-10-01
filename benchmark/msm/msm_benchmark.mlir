!PF = !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
!SF = !field.pf<21888242871839275222246405745257275088548364400416034343698204186575808495617:i256>
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

func.func private @getGeneratorMultiple(%k: !SF) -> !affine {
  %onePF = field.constant 1 : !PF
  %twoPF = field.constant 2 : !PF
  %onePFm = field.to_mont %onePF : !PFm
  %twoPFm = field.to_mont %twoPF : !PFm
  %g = elliptic_curve.point %onePFm, %twoPFm : (!PFm, !PFm) -> !affine
  %g_multiple = elliptic_curve.scalar_mul %k, %g : !SF, !affine -> !jacobian
  %g_multiple_affine = elliptic_curve.convert_point_type %g_multiple : !jacobian -> !affine
  return %g_multiple_affine : !affine
}

func.func private @generate_points(%rand_scalar : memref<!SF>, %points : !points_M) attributes { llvm.emit_c_interface } {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %rand_scalar_extracted = memref.load %rand_scalar[] : memref<!SF>
  %first_point = func.call @getGeneratorMultiple(%rand_scalar_extracted) : (!SF) -> !affine

  %num_scalar_muls = memref.dim %points, %c0 : !points_M
  scf.for %i = %c0 to %num_scalar_muls step %c1 iter_args(%sum_iter = %first_point) -> (!affine){
    %point = elliptic_curve.double %sum_iter : !affine -> !jacobian
    %point_affine = elliptic_curve.convert_point_type %point : !jacobian -> !affine
    memref.store %point_affine, %points[%i] : !points_M
    scf.yield %point_affine : !affine
  }
  return
}

func.func @msm_serial(%scalars : !SFm_M, %points : !points_M) attributes { llvm.emit_c_interface } {
  %s = bufferization.to_tensor %scalars restrict writable : !SFm_M to !SFm_T
  %p = bufferization.to_tensor %points restrict writable : !points_M to !points_T
  %f = elliptic_curve.msm %s, %p degree=20 : !SFm_T, !points_T -> !xyzz
  %res = elliptic_curve.convert_point_type %f : !xyzz -> !jacobian
  return
}

func.func @msm_parallel(%scalars : !SFm_M, %points : !points_M) attributes { llvm.emit_c_interface } {
  %s = bufferization.to_tensor %scalars restrict writable : !SFm_M to !SFm_T
  %p = bufferization.to_tensor %points restrict writable : !points_M to !points_T
  %f = elliptic_curve.msm %s, %p degree=20 parallel : !SFm_T, !points_T -> !xyzz
  %res = elliptic_curve.convert_point_type %f : !xyzz -> !jacobian
  return
}
