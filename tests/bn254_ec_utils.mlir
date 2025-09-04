func.func @printAffine(%affine: !affine) {
  %x, %y = elliptic_curve.extract %affine : !affine -> !PF, !PF
  %point = tensor.from_elements %x, %y : tensor<2x!PF>
  %point_native = field.extract %point : tensor<2x!PF> -> tensor<2xi256>
  %mem = bufferization.to_buffer %point_native : tensor<2xi256> to memref<2xi256>
  %mem_cast = memref.cast %mem : memref<2xi256> to memref<*xi256>
  func.call @printMemrefI256(%mem_cast) : (memref<*xi256>) -> ()
  return
}

func.func @printJacobian(%jacobian: !jacobian) {
  %x, %y, %z = elliptic_curve.extract %jacobian : !jacobian -> !PF, !PF, !PF
  %point = tensor.from_elements %x, %y, %z : tensor<3x!PF>
  %point_native = field.extract %point : tensor<3x!PF> -> tensor<3xi256>
  %mem = bufferization.to_buffer %point_native : tensor<3xi256> to memref<3xi256>
  %mem_cast = memref.cast %mem : memref<3xi256> to memref<*xi256>
  func.call @printMemrefI256(%mem_cast) : (memref<*xi256>) -> ()
  return
}

func.func @printXYZZ(%xyzz: !xyzz) {
  %x, %y, %zz, %zzz = elliptic_curve.extract %xyzz : !xyzz -> !PF, !PF, !PF, !PF
  %point = tensor.from_elements %x, %y, %zz, %zzz : tensor<4x!PF>
  %point_native = field.extract %point : tensor<4x!PF> -> tensor<4xi256>
  %mem = bufferization.to_buffer %point_native : tensor<4xi256> to memref<4xi256>
  %mem_cast = memref.cast %mem : memref<4xi256> to memref<*xi256>
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

// assumes standard form scalar input and outputs affine with standard form coordinates
func.func @getGeneratorMultiple(%k: !SF) -> !affine {
  %onePF = field.constant 1 : !PF
  %twoPF = field.constant 2 : !PF
  %g = elliptic_curve.point %onePF, %twoPF : !affine
  %g_multiple = elliptic_curve.scalar_mul %k, %g : !SF, !affine -> !jacobian
  %g_multiple_affine = elliptic_curve.convert_point_type %g_multiple : !jacobian -> !affine
  return %g_multiple_affine : !affine
}
