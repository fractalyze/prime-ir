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

// BENCH: CHANGE TO DESIRED DEGREE (current is 2²⁰ = 1048576)
!SFm_T = tensor<1048576x!SFm>
!SFm_M = memref<1048576x!SFm>
!points_T = tensor<1048576x!affine>
!points_M = memref<1048576x!affine>

// BENCH: FOR ALL DESIRED BITS PER WINDOW TESTS, CHANGE TO DESIRED DEGREE (current is 2²⁰)
func.func @msm_3(%scalars : !SFm_M, %points : !points_M) attributes { llvm.emit_c_interface } {
  %s = bufferization.to_tensor %scalars restrict writable : !SFm_M to !SFm_T
  %p = bufferization.to_tensor %points restrict writable : !points_M to !points_T
  %f = elliptic_curve.msm %s, %p degree=20 window_bits=3 parallel: !SFm_T, !points_T -> !xyzz
  %res = elliptic_curve.convert_point_type %f : !xyzz -> !jacobian
  return
}

func.func @msm_4(%scalars : !SFm_M, %points : !points_M) attributes { llvm.emit_c_interface } {
  %s = bufferization.to_tensor %scalars restrict writable : !SFm_M to !SFm_T
  %p = bufferization.to_tensor %points restrict writable : !points_M to !points_T
  %f = elliptic_curve.msm %s, %p degree=20 window_bits=4 parallel: !SFm_T, !points_T -> !xyzz
  %res = elliptic_curve.convert_point_type %f : !xyzz -> !jacobian
  return
}

func.func @msm_5(%scalars : !SFm_M, %points : !points_M) attributes { llvm.emit_c_interface } {
  %s = bufferization.to_tensor %scalars restrict writable : !SFm_M to !SFm_T
  %p = bufferization.to_tensor %points restrict writable : !points_M to !points_T
  %f = elliptic_curve.msm %s, %p degree=20 window_bits=5 parallel: !SFm_T, !points_T -> !xyzz
  %res = elliptic_curve.convert_point_type %f : !xyzz -> !jacobian
  return
}

func.func @msm_6(%scalars : !SFm_M, %points : !points_M) attributes { llvm.emit_c_interface } {
  %s = bufferization.to_tensor %scalars restrict writable : !SFm_M to !SFm_T
  %p = bufferization.to_tensor %points restrict writable : !points_M to !points_T
  %f = elliptic_curve.msm %s, %p degree=20 window_bits=6 parallel: !SFm_T, !points_T -> !xyzz
  %res = elliptic_curve.convert_point_type %f : !xyzz -> !jacobian
  return
}

func.func @msm_7(%scalars : !SFm_M, %points : !points_M) attributes { llvm.emit_c_interface } {
  %s = bufferization.to_tensor %scalars restrict writable : !SFm_M to !SFm_T
  %p = bufferization.to_tensor %points restrict writable : !points_M to !points_T
  %f = elliptic_curve.msm %s, %p degree=20 window_bits=7 parallel: !SFm_T, !points_T -> !xyzz
  %res = elliptic_curve.convert_point_type %f : !xyzz -> !jacobian
  return
}

func.func @msm_8(%scalars : !SFm_M, %points : !points_M) attributes { llvm.emit_c_interface } {
  %s = bufferization.to_tensor %scalars restrict writable : !SFm_M to !SFm_T
  %p = bufferization.to_tensor %points restrict writable : !points_M to !points_T
  %f = elliptic_curve.msm %s, %p degree=20 window_bits=8 parallel: !SFm_T, !points_T -> !xyzz
  %res = elliptic_curve.convert_point_type %f : !xyzz -> !jacobian
  return
}

func.func @msm_9(%scalars : !SFm_M, %points : !points_M) attributes { llvm.emit_c_interface } {
  %s = bufferization.to_tensor %scalars restrict writable : !SFm_M to !SFm_T
  %p = bufferization.to_tensor %points restrict writable : !points_M to !points_T
  %f = elliptic_curve.msm %s, %p degree=20 window_bits=9 parallel: !SFm_T, !points_T -> !xyzz
  %res = elliptic_curve.convert_point_type %f : !xyzz -> !jacobian
  return
}

func.func @msm_10(%scalars : !SFm_M, %points : !points_M) attributes { llvm.emit_c_interface } {
  %s = bufferization.to_tensor %scalars restrict writable : !SFm_M to !SFm_T
  %p = bufferization.to_tensor %points restrict writable : !points_M to !points_T
  %f = elliptic_curve.msm %s, %p degree=20 window_bits=10 parallel: !SFm_T, !points_T -> !xyzz
  %res = elliptic_curve.convert_point_type %f : !xyzz -> !jacobian
  return
}

func.func @msm_11(%scalars : !SFm_M, %points : !points_M) attributes { llvm.emit_c_interface } {
  %s = bufferization.to_tensor %scalars restrict writable : !SFm_M to !SFm_T
  %p = bufferization.to_tensor %points restrict writable : !points_M to !points_T
  %f = elliptic_curve.msm %s, %p degree=20 window_bits=11 parallel: !SFm_T, !points_T -> !xyzz
  %res = elliptic_curve.convert_point_type %f : !xyzz -> !jacobian
  return
}

func.func @msm_12(%scalars : !SFm_M, %points : !points_M) attributes { llvm.emit_c_interface } {
  %s = bufferization.to_tensor %scalars restrict writable : !SFm_M to !SFm_T
  %p = bufferization.to_tensor %points restrict writable : !points_M to !points_T
  %f = elliptic_curve.msm %s, %p degree=20 window_bits=12 parallel: !SFm_T, !points_T -> !xyzz
  %res = elliptic_curve.convert_point_type %f : !xyzz -> !jacobian
  return
}

func.func @msm_13(%scalars : !SFm_M, %points : !points_M) attributes { llvm.emit_c_interface } {
  %s = bufferization.to_tensor %scalars restrict writable : !SFm_M to !SFm_T
  %p = bufferization.to_tensor %points restrict writable : !points_M to !points_T
  %f = elliptic_curve.msm %s, %p degree=20 window_bits=13 parallel: !SFm_T, !points_T -> !xyzz
  %res = elliptic_curve.convert_point_type %f : !xyzz -> !jacobian
  return
}

func.func @msm_14(%scalars : !SFm_M, %points : !points_M) attributes { llvm.emit_c_interface } {
  %s = bufferization.to_tensor %scalars restrict writable : !SFm_M to !SFm_T
  %p = bufferization.to_tensor %points restrict writable : !points_M to !points_T
  %f = elliptic_curve.msm %s, %p degree=20 window_bits=14 parallel: !SFm_T, !points_T -> !xyzz
  %res = elliptic_curve.convert_point_type %f : !xyzz -> !jacobian
  return
}

func.func @msm_15(%scalars : !SFm_M, %points : !points_M) attributes { llvm.emit_c_interface } {
  %s = bufferization.to_tensor %scalars restrict writable : !SFm_M to !SFm_T
  %p = bufferization.to_tensor %points restrict writable : !points_M to !points_T
  %f = elliptic_curve.msm %s, %p degree=20 window_bits=15 parallel: !SFm_T, !points_T -> !xyzz
  %res = elliptic_curve.convert_point_type %f : !xyzz -> !jacobian
  return
}

func.func @msm_16(%scalars : !SFm_M, %points : !points_M) attributes { llvm.emit_c_interface } {
  %s = bufferization.to_tensor %scalars restrict writable : !SFm_M to !SFm_T
  %p = bufferization.to_tensor %points restrict writable : !points_M to !points_T
  %f = elliptic_curve.msm %s, %p degree=20 window_bits=16 parallel: !SFm_T, !points_T -> !xyzz
  %res = elliptic_curve.convert_point_type %f : !xyzz -> !jacobian
  return
}

func.func @msm_17(%scalars : !SFm_M, %points : !points_M) attributes { llvm.emit_c_interface } {
  %s = bufferization.to_tensor %scalars restrict writable : !SFm_M to !SFm_T
  %p = bufferization.to_tensor %points restrict writable : !points_M to !points_T
  %f = elliptic_curve.msm %s, %p degree=20 window_bits=17 parallel: !SFm_T, !points_T -> !xyzz
  %res = elliptic_curve.convert_point_type %f : !xyzz -> !jacobian
  return
}

func.func @msm_18(%scalars : !SFm_M, %points : !points_M) attributes { llvm.emit_c_interface } {
  %s = bufferization.to_tensor %scalars restrict writable : !SFm_M to !SFm_T
  %p = bufferization.to_tensor %points restrict writable : !points_M to !points_T
  %f = elliptic_curve.msm %s, %p degree=20 window_bits=18 parallel: !SFm_T, !points_T -> !xyzz
  %res = elliptic_curve.convert_point_type %f : !xyzz -> !jacobian
  return
}
func.func @msm_19(%scalars : !SFm_M, %points : !points_M) attributes { llvm.emit_c_interface } {
  %s = bufferization.to_tensor %scalars restrict writable : !SFm_M to !SFm_T
  %p = bufferization.to_tensor %points restrict writable : !points_M to !points_T
  %f = elliptic_curve.msm %s, %p degree=20 window_bits=19 parallel: !SFm_T, !points_T -> !xyzz
  %res = elliptic_curve.convert_point_type %f : !xyzz -> !jacobian
  return
}

func.func @msm_20(%scalars : !SFm_M, %points : !points_M) attributes { llvm.emit_c_interface } {
  %s = bufferization.to_tensor %scalars restrict writable : !SFm_M to !SFm_T
  %p = bufferization.to_tensor %points restrict writable : !points_M to !points_T
  %f = elliptic_curve.msm %s, %p degree=20 window_bits=20 parallel: !SFm_T, !points_T -> !xyzz
  %res = elliptic_curve.convert_point_type %f : !xyzz -> !jacobian
  return
}

func.func @msm_21(%scalars : !SFm_M, %points : !points_M) attributes { llvm.emit_c_interface } {
  %s = bufferization.to_tensor %scalars restrict writable : !SFm_M to !SFm_T
  %p = bufferization.to_tensor %points restrict writable : !points_M to !points_T
  %f = elliptic_curve.msm %s, %p degree=20 window_bits=21 parallel: !SFm_T, !points_T -> !xyzz
  %res = elliptic_curve.convert_point_type %f : !xyzz -> !jacobian
  return
}

func.func @msm_22(%scalars : !SFm_M, %points : !points_M) attributes { llvm.emit_c_interface } {
  %s = bufferization.to_tensor %scalars restrict writable : !SFm_M to !SFm_T
  %p = bufferization.to_tensor %points restrict writable : !points_M to !points_T
  %f = elliptic_curve.msm %s, %p degree=20 window_bits=22 parallel: !SFm_T, !points_T -> !xyzz
  %res = elliptic_curve.convert_point_type %f : !xyzz -> !jacobian
  return
}

func.func @msm_23(%scalars : !SFm_M, %points : !points_M) attributes { llvm.emit_c_interface } {
  %s = bufferization.to_tensor %scalars restrict writable : !SFm_M to !SFm_T
  %p = bufferization.to_tensor %points restrict writable : !points_M to !points_T
  %f = elliptic_curve.msm %s, %p degree=20 window_bits=23 parallel: !SFm_T, !points_T -> !xyzz
  %res = elliptic_curve.convert_point_type %f : !xyzz -> !jacobian
  return
}

func.func @msm_24(%scalars : !SFm_M, %points : !points_M) attributes { llvm.emit_c_interface } {
  %s = bufferization.to_tensor %scalars restrict writable : !SFm_M to !SFm_T
  %p = bufferization.to_tensor %points restrict writable : !points_M to !points_T
  %f = elliptic_curve.msm %s, %p degree=20 window_bits=24 parallel: !SFm_T, !points_T -> !xyzz
  %res = elliptic_curve.convert_point_type %f : !xyzz -> !jacobian
  return
}

func.func @msm_25(%scalars : !SFm_M, %points : !points_M) attributes { llvm.emit_c_interface } {
  %s = bufferization.to_tensor %scalars restrict writable : !SFm_M to !SFm_T
  %p = bufferization.to_tensor %points restrict writable : !points_M to !points_T
  %f = elliptic_curve.msm %s, %p degree=20 window_bits=25 parallel: !SFm_T, !points_T -> !xyzz
  %res = elliptic_curve.convert_point_type %f : !xyzz -> !jacobian
  return
}

func.func @msm_26(%scalars : !SFm_M, %points : !points_M) attributes { llvm.emit_c_interface } {
  %s = bufferization.to_tensor %scalars restrict writable : !SFm_M to !SFm_T
  %p = bufferization.to_tensor %points restrict writable : !points_M to !points_T
  %f = elliptic_curve.msm %s, %p degree=20 window_bits=26 parallel: !SFm_T, !points_T -> !xyzz
  %res = elliptic_curve.convert_point_type %f : !xyzz -> !jacobian
  return
}

func.func @msm_27(%scalars : !SFm_M, %points : !points_M) attributes { llvm.emit_c_interface } {
  %s = bufferization.to_tensor %scalars restrict writable : !SFm_M to !SFm_T
  %p = bufferization.to_tensor %points restrict writable : !points_M to !points_T
  %f = elliptic_curve.msm %s, %p degree=20 window_bits=27 parallel: !SFm_T, !points_T -> !xyzz
  %res = elliptic_curve.convert_point_type %f : !xyzz -> !jacobian
  return
}

func.func @msm_28(%scalars : !SFm_M, %points : !points_M) attributes { llvm.emit_c_interface } {
  %s = bufferization.to_tensor %scalars restrict writable : !SFm_M to !SFm_T
  %p = bufferization.to_tensor %points restrict writable : !points_M to !points_T
  %f = elliptic_curve.msm %s, %p degree=20 window_bits=28 parallel: !SFm_T, !points_T -> !xyzz
  %res = elliptic_curve.convert_point_type %f : !xyzz -> !jacobian
  return
}

func.func @msm_29(%scalars : !SFm_M, %points : !points_M) attributes { llvm.emit_c_interface } {
  %s = bufferization.to_tensor %scalars restrict writable : !SFm_M to !SFm_T
  %p = bufferization.to_tensor %points restrict writable : !points_M to !points_T
  %f = elliptic_curve.msm %s, %p degree=20 window_bits=29 parallel: !SFm_T, !points_T -> !xyzz
  %res = elliptic_curve.convert_point_type %f : !xyzz -> !jacobian
  return
}

func.func @msm_30(%scalars : !SFm_M, %points : !points_M) attributes { llvm.emit_c_interface } {
  %s = bufferization.to_tensor %scalars restrict writable : !SFm_M to !SFm_T
  %p = bufferization.to_tensor %points restrict writable : !points_M to !points_T
  %f = elliptic_curve.msm %s, %p degree=20 window_bits=30 parallel: !SFm_T, !points_T -> !xyzz
  %res = elliptic_curve.convert_point_type %f : !xyzz -> !jacobian
  return
}
