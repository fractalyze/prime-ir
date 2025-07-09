// RUN: zkir-opt %s -mod-arith-to-arith -convert-elementwise-to-linalg -one-shot-bufferize -convert-linalg-to-parallel-loops -convert-scf-to-cf -convert-cf-to-llvm -convert-to-llvm -convert-vector-to-llvm \
// RUN:   | mlir-runner -e test_lower_inverse -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_TEST_INVERSE < %t

// RUN: zkir-opt %s -mod-arith-to-arith -convert-elementwise-to-linalg -one-shot-bufferize -convert-linalg-to-parallel-loops -convert-scf-to-cf -convert-cf-to-llvm -convert-to-llvm -convert-vector-to-llvm \
// RUN:   | mlir-runner -e test_lower_inverse_tensor -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_TEST_INVERSE_TENSOR < %t

// RUN: zkir-opt %s -mod-arith-to-arith -convert-elementwise-to-linalg -one-shot-bufferize  -convert-linalg-to-parallel-loops -convert-scf-to-cf -convert-cf-to-llvm -convert-to-llvm -convert-vector-to-llvm \
// RUN:   | mlir-runner -e test_lower_mont_reduce -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_TEST_MONT_REDUCE < %t

// RUN: zkir-opt %s -mod-arith-to-arith -convert-elementwise-to-linalg -one-shot-bufferize -convert-linalg-to-parallel-loops -convert-scf-to-cf -convert-cf-to-llvm -convert-to-llvm -convert-vector-to-llvm \
// RUN:   | mlir-runner -e test_lower_mont_mul -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_TEST_MONT_MUL < %t

// RUN: zkir-opt %s -mod-arith-to-arith -convert-elementwise-to-linalg -one-shot-bufferize -convert-linalg-to-parallel-loops -convert-scf-to-cf -convert-cf-to-llvm -convert-to-llvm -convert-vector-to-llvm \
// RUN:   | mlir-runner -e test_lower_mont_square -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_TEST_MONT_SQUARE < %t


func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

!Fq = !mod_arith.int<21888242871839275222246405745257275088548364400416034343698204186575808495617 : i256>
!Fqm = !mod_arith.int<21888242871839275222246405745257275088548364400416034343698204186575808495617 : i256, true>
#Fq_mont = #mod_arith.montgomery<!Fq>
!Fr = !mod_arith.int<2147483647:i32>
!Frm = !mod_arith.int<2147483647:i32, true>

func.func @test_lower_inverse() {
  %p = mod_arith.constant 3723 : !Fq
  %p_mont = mod_arith.to_mont %p : !Fqm
  %p_inv = mod_arith.mont_inverse %p_mont : !Fqm
  %mul = mod_arith.mul %p_inv, %p_mont : !Fqm
  %from_mont = mod_arith.from_mont %mul : !Fq
  %2 = mod_arith.extract %from_mont : !Fq -> i256
  %3 = arith.trunci %2 : i256 to i32
  %4 = tensor.from_elements %3 : tensor<1xi32>
  %5 = bufferization.to_memref %4 : tensor<1xi32> to memref<1xi32>
  %U = memref.cast %5 : memref<1xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()

  %inv = mod_arith.inverse %p : !Fq
  %mul2 = mod_arith.mul %inv, %p : !Fq
  %6 = mod_arith.extract %mul2 : !Fq -> i256
  %7 = arith.trunci %6 : i256 to i32
  %8 = tensor.from_elements %7 : tensor<1xi32>
  %9 = bufferization.to_memref %8 : tensor<1xi32> to memref<1xi32>
  %10 = memref.cast %9 : memref<1xi32> to memref<*xi32>
  func.call @printMemrefI32(%10) : (memref<*xi32>) -> ()

  %p_r = mod_arith.constant 3724 : !Fr
  %p_r_mont = mod_arith.to_mont %p_r : !Frm
  %p_r_inv = mod_arith.mont_inverse %p_r_mont : !Frm
  %mul_r = mod_arith.mul %p_r_inv, %p_r_mont : !Frm
  %from_mont_r = mod_arith.from_mont %mul_r : !Fr
  %r_ext = mod_arith.extract %from_mont_r : !Fr -> i32
  %r_tensor = tensor.from_elements %r_ext : tensor<1xi32>
  %r_mem = bufferization.to_memref %r_tensor : tensor<1xi32> to memref<1xi32>
  %r_mem_cast = memref.cast %r_mem : memref<1xi32> to memref<*xi32>
  func.call @printMemrefI32(%r_mem_cast) : (memref<*xi32>) -> ()
  return
}

// CHECK_TEST_INVERSE: data =
// CHECK_TEST_INVERSE-NEXT: [1]
// CHECK_TEST_INVERSE: data =
// CHECK_TEST_INVERSE-NEXT: [1]
// CHECK_TEST_INVERSE: data =
// CHECK_TEST_INVERSE-NEXT: [1]

func.func @test_lower_inverse_tensor() {
  %p1 = arith.constant 3723 : i256
  %p2 = arith.constant 3724 : i256
  %p3 = arith.constant 3725 : i256
  %tensor1 = tensor.from_elements %p1, %p2, %p3 : tensor<3xi256>
  %tensor2 = mod_arith.encapsulate %tensor1 : tensor<3xi256> -> tensor<3x!Fq>
  %inv = mod_arith.inverse %tensor2 : tensor<3x!Fq>
  %mul = mod_arith.mul %inv, %tensor2 : tensor<3x!Fq>
  %ext = mod_arith.extract %mul : tensor<3x!Fq> -> tensor<3xi256>
  %trunc = arith.trunci %ext : tensor<3xi256> to tensor<3xi32>
  %1 = bufferization.to_memref %trunc : tensor<3xi32> to memref<3xi32>
  %U1 = memref.cast %1 : memref<3xi32> to memref<*xi32>
  func.call @printMemrefI32(%U1) : (memref<*xi32>) -> ()

  %tensor_mont = mod_arith.to_mont %tensor2 : tensor<3x!Fqm>
  %inv_mont = mod_arith.inverse %tensor_mont : tensor<3x!Fqm>
  %mul_mont = mod_arith.mul %inv_mont, %tensor_mont : tensor<3x!Fqm>
  %from_mont = mod_arith.from_mont %mul_mont : tensor<3x!Fq>
  %ext2 = mod_arith.extract %from_mont : tensor<3x!Fq> -> tensor<3xi256>
  %trunc2 = arith.trunci %ext2 : tensor<3xi256> to tensor<3xi32>
  %2 = bufferization.to_memref %trunc2 : tensor<3xi32> to memref<3xi32>
  %U2 = memref.cast %2 : memref<3xi32> to memref<*xi32>
  func.call @printMemrefI32(%U2) : (memref<*xi32>) -> ()
  return
}

// CHECK_TEST_INVERSE_TENSOR: [1, 1, 1]
// CHECK_TEST_INVERSE_TENSOR: [1, 1, 1]

func.func @test_lower_mont_reduce() {
  %p = arith.constant 2188824287183927522224640574525727508854836440041603434369820418657580849561 : i256
  %zero = arith.constant 0 : i256
  // `pR` is `p` << 256 so just give `p` as `high` and set `low` to 0
  %p_mont = mod_arith.mont_reduce %zero, %p : i256 -> !Fq

  %2 = mod_arith.extract %p_mont : !Fq -> i256
  // check if mod_arith.mont_reduce(pR) == p
  %true = arith.cmpi eq, %2, %p : i256
  %trueExt = arith.extui %true : i1 to i32
  %3 = vector.from_elements %trueExt : vector<1xi32>
  %mem = memref.alloc() : memref<1xi32>
  %idx_0 = arith.constant 0 : index
  vector.store %3, %mem[%idx_0] : memref<1xi32>, vector<1xi32>

  %U = memref.cast %mem : memref<1xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()
  return
}

// CHECK_TEST_MONT_REDUCE: data =
// CHECK_TEST_MONT_REDUCE-NEXT: [1]

func.func @test_lower_mont_mul() {
  %p = mod_arith.constant 17221657567640823606390383439573883756117969501024189775361 : !Fq
  %p_mont = mod_arith.to_mont %p : !Fqm
  %p_mont_sq = mod_arith.mont_mul %p_mont, %p_mont : !Fqm
  %p_sq = mod_arith.from_mont %p_mont_sq : !Fq

  %2 = mod_arith.extract %p_sq : !Fq -> i256
  %3 = vector.from_elements %2 : vector<1xi256>
  %4 = vector.bitcast %3 : vector<1xi256> to vector<8xi32>
  %mem = memref.alloc() : memref<8xi32>
  %idx_0 = arith.constant 0 : index
  vector.store %4, %mem[%idx_0] : memref<8xi32>, vector<8xi32>

  %U = memref.cast %mem : memref<8xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()
  return
}

// CHECK_TEST_MONT_MUL: [-1717936988, -857005375, 1976922116, -1939796685, 1587159113, 557631023, 126776667, 742573744]

func.func @test_lower_mont_square() {
  %p = mod_arith.constant 17221657567640823606390383439573883756117969501024189775361 : !Fq
  %p_mont = mod_arith.to_mont %p : !Fqm
  %p_mont_sq = mod_arith.mont_square %p_mont : !Fqm
  %p_sq = mod_arith.from_mont %p_mont_sq : !Fq

  %2 = mod_arith.extract %p_sq : !Fq -> i256
  %3 = vector.from_elements %2 : vector<1xi256>
  %4 = vector.bitcast %3 : vector<1xi256> to vector<8xi32>
  %mem = memref.alloc() : memref<8xi32>
  %idx_0 = arith.constant 0 : index
  vector.store %4, %mem[%idx_0] : memref<8xi32>, vector<8xi32>

  %U = memref.cast %mem : memref<8xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()
  return
}

// CHECK_TEST_MONT_SQUARE: [-1717936988, -857005375, 1976922116, -1939796685, 1587159113, 557631023, 126776667, 742573744]
