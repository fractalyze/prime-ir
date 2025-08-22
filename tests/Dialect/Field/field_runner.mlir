// RUN: zkir-opt %s --field-to-llvm \
// RUN:   | mlir-runner -e test_power -entry-point-result=void \
// RUN:      --shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s --check-prefix=CHECK_TEST_POWER < %t

#mont = #mod_arith.montgomery<7:i32>
!PF = !field.pf<7:i32>
!PFm = !field.pf<7:i32, true>
!PF_exp = !field.pf<7:i64>

#beta = #field.pf.elem<6:i32> : !PF
#beta_mont = #field.pf.elem<3:i32> : !PFm
!QF = !field.f2<!PF, #beta>
!QFm = !field.f2<!PFm, #beta_mont>
#ef = #field.f2.elem<#beta, #beta> : !QF

func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

func.func @test_power() {
  %exp = arith.constant 51 : i64
  %exp_pf = field.encapsulate %exp : i64 -> !PF_exp
  %base = arith.constant 3 : i32
  %base_pf = field.encapsulate %base: i32 -> !PF

  %res1 = field.powui %base_pf, %exp : !PF, i64
  %1 = field.extract %res1 : !PF -> i32
  %2 = tensor.from_elements %1 : tensor<1xi32>
  %3 = bufferization.to_buffer %2 : tensor<1xi32> to memref<1xi32>
  %U1 = memref.cast %3 : memref<1xi32> to memref<*xi32>
  func.call @printMemrefI32(%U1) : (memref<*xi32>) -> ()

  %base_pf_mont = field.to_mont %base_pf : !PFm
  %res1_mont = field.powui %base_pf_mont, %exp : !PFm, i64
  %res1_standard = field.from_mont %res1_mont : !PF
  %4 = field.extract %res1_standard : !PF -> i32
  %5 = tensor.from_elements %4 : tensor<1xi32>
  %6 = bufferization.to_buffer %5 : tensor<1xi32> to memref<1xi32>
  %U2 = memref.cast %6 : memref<1xi32> to memref<*xi32>
  func.call @printMemrefI32(%U2) : (memref<*xi32>) -> ()

  %base_f2 = field.encapsulate %base, %base : i32, i32 -> !QF
  %res2 = field.powui %base_f2, %exp : !QF, i64
  %9, %10 = field.extract %res2 : !QF -> i32, i32
  %11 = tensor.from_elements %9, %10 : tensor<2xi32>
  %12 = bufferization.to_buffer %11 : tensor<2xi32> to memref<2xi32>
  %U3 = memref.cast %12 : memref<2xi32> to memref<*xi32>
  func.call @printMemrefI32(%U3) : (memref<*xi32>) -> ()

  %base_f2_mont = field.to_mont %base_f2 : !QFm
  %res2_mont = field.powui %base_f2_mont, %exp : !QFm, i64
  %res2_standard = field.from_mont %res2_mont : !QF
  %13, %14 = field.extract %res2_standard : !QF -> i32, i32
  %15 = tensor.from_elements %13, %14 : tensor<2xi32>
  %16 = bufferization.to_buffer %15 : tensor<2xi32> to memref<2xi32>
  %U4 = memref.cast %16 : memref<2xi32> to memref<*xi32>
  func.call @printMemrefI32(%U4) : (memref<*xi32>) -> ()

  %res1_pf = field.powpf %base_pf, %exp_pf : !PF, !PF_exp
  %17 = field.extract %res1_pf : !PF -> i32
  %18 = tensor.from_elements %17 : tensor<1xi32>
  %19 = bufferization.to_buffer %18 : tensor<1xi32> to memref<1xi32>
  %U5 = memref.cast %19 : memref<1xi32> to memref<*xi32>
  func.call @printMemrefI32(%U5) : (memref<*xi32>) -> ()

  %res2_pf_mont = field.powpf %base_f2_mont, %exp_pf : !QFm, !PF_exp
  %res2_pf_standard = field.from_mont %res2_pf_mont : !QF
  %20, %21 = field.extract %res2_pf_standard : !QF -> i32, i32
  %22 = tensor.from_elements %20, %21 : tensor<2xi32>
  %23 = bufferization.to_buffer %22 : tensor<2xi32> to memref<2xi32>
  %U6 = memref.cast %23 : memref<2xi32> to memref<*xi32>
  func.call @printMemrefI32(%U6) : (memref<*xi32>) -> ()

  return
}

// CHECK_TEST_POWER: [6]
// CHECK_TEST_POWER: [6]
// CHECK_TEST_POWER: [2, 5]
// CHECK_TEST_POWER: [2, 5]
// CHECK_TEST_POWER: [6]
// CHECK_TEST_POWER: [2, 5]
