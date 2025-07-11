// RUN: zkir-opt %s --field-to-llvm \
// RUN:   | mlir-runner -e test_power -entry-point-result=void \
// RUN:      --shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s --check-prefix=CHECK_TEST_POWER < %t

#mont = #mod_arith.montgomery<7:i32>
!PF = !field.pf<7:i32>
!PFm = !field.pf<7:i32, true>

#beta = #field.pf.elem<6:i32> : !PF
#beta_mont = #field.pf.elem<3:i32> : !PFm
!QF = !field.f2<!PF, #beta>
!QFm = !field.f2<!PFm, #beta_mont>
#ef = #field.f2.elem<#beta, #beta> : !QF

func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

func.func @test_power() {
  %exp = arith.constant 51 : i64
  %base = arith.constant 3 : i32
  %base_pf = field.encapsulate %base: i32 -> !PF

  %res1 = field.pow %base_pf, %exp : !PF, i64
  %1 = field.extract %res1 : !PF -> i32
  %2 = tensor.from_elements %1 : tensor<1xi32>
  %3 = bufferization.to_memref %2 : tensor<1xi32> to memref<1xi32>
  %U1 = memref.cast %3 : memref<1xi32> to memref<*xi32>
  func.call @printMemrefI32(%U1) : (memref<*xi32>) -> ()

  %base_pf_mont = field.to_mont %base_pf : !PFm
  %res1_mont = field.pow %base_pf_mont, %exp : !PFm, i64
  %res1_standard = field.from_mont %res1_mont : !PF
  %4 = field.extract %res1_standard : !PF -> i32
  %5 = tensor.from_elements %4 : tensor<1xi32>
  %6 = bufferization.to_memref %5 : tensor<1xi32> to memref<1xi32>
  %U2 = memref.cast %6 : memref<1xi32> to memref<*xi32>
  func.call @printMemrefI32(%U2) : (memref<*xi32>) -> ()

  %base_f2 = field.encapsulate %base, %base : i32, i32 -> !QF
  %res2 = field.pow %base_f2, %exp : !QF, i64
  %9, %10 = field.extract %res2 : !QF -> i32, i32
  %11 = tensor.from_elements %9, %10 : tensor<2xi32>
  %12 = bufferization.to_memref %11 : tensor<2xi32> to memref<2xi32>
  %U3 = memref.cast %12 : memref<2xi32> to memref<*xi32>
  func.call @printMemrefI32(%U3) : (memref<*xi32>) -> ()

  %base_f2_mont = field.to_mont %base_f2 : !QFm
  %res2_mont = field.pow %base_f2_mont, %exp : !QFm, i64
  %res2_standard = field.from_mont %res2_mont : !QF
  %13, %14 = field.extract %res2_standard : !QF -> i32, i32
  %15 = tensor.from_elements %13, %14 : tensor<2xi32>
  %16 = bufferization.to_memref %15 : tensor<2xi32> to memref<2xi32>
  %U4 = memref.cast %16 : memref<2xi32> to memref<*xi32>
  func.call @printMemrefI32(%U4) : (memref<*xi32>) -> ()

  return
}

// CHECK_TEST_POWER: [6]
// CHECK_TEST_POWER: [6]
// CHECK_TEST_POWER: [2, 5]
// CHECK_TEST_POWER: [2, 5]
