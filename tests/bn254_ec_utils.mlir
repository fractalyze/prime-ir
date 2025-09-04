#a = #field.pf.elem<0:i256> : !PFm
#b = #field.pf.elem<3:i256> : !PFm
#1 = #field.pf.elem<1:i256> : !PFm
#2 = #field.pf.elem<2:i256> : !PFm

#curve = #elliptic_curve.sw<#a, #b, (#1, #2)>
!affine = !elliptic_curve.affine<#curve>
!jacobian = !elliptic_curve.jacobian<#curve>
!xyzz = !elliptic_curve.xyzz<#curve>

func.func @printAffine(%affine: !affine) {
  %x, %y = elliptic_curve.extract %affine : !affine -> !PFm, !PFm
  %point = tensor.from_elements %x, %y : tensor<2x!PFm>
  %point_standard = field.from_mont %point : tensor<2x!PF>
  %point_native = field.extract %point_standard : tensor<2x!PF> -> tensor<2xi256>
  %mem = bufferization.to_buffer %point_native : tensor<2xi256> to memref<2xi256>
  %mem_cast = memref.cast %mem : memref<2xi256> to memref<*xi256>
  func.call @printMemrefI256(%mem_cast) : (memref<*xi256>) -> ()
  return
}

func.func @printAffineFromJacobian(%jacobian: !jacobian) {
  %affine = elliptic_curve.convert_point_type %jacobian : !jacobian -> !affine
  func.call @printAffine(%affine) : (!affine) -> ()
  return
}

func.func @printAffineFromXYZZ(%xyzz: !xyzz) {
  %affine = elliptic_curve.convert_point_type %xyzz : !xyzz -> !affine
  func.call @printAffine(%affine) : (!affine) -> ()
  return
}

// assumes standard form scalar input and outputs affine with montgomery form coordinates
func.func @getGeneratorMultiple(%k: !SF) -> !affine {
  %onePF = field.constant 1 : !PF
  %twoPF = field.constant 2 : !PF
  %onePFm = field.to_mont %onePF : !PFm
  %twoPFm = field.to_mont %twoPF : !PFm
  %g = elliptic_curve.point %onePFm, %twoPFm : !affine
  %k_sf = field.to_mont %k : !SFm
  %g_multiple = elliptic_curve.scalar_mul %k_sf, %g : !SFm, !affine -> !jacobian
  %g_multiple_affine = elliptic_curve.convert_point_type %g_multiple : !jacobian -> !affine
  return %g_multiple_affine : !affine
}
