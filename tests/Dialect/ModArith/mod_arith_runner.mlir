// RUN: zkir-opt %s --mod-arith-to-arith -convert-elementwise-to-linalg --one-shot-bufferize --convert-scf-to-cf --convert-cf-to-llvm --convert-to-llvm --convert-vector-to-llvm \
// RUN:   | mlir-runner -e test_lower_inverse -entry-point-result=void \
// RUN:      --shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s --check-prefix=CHECK_TEST_INVERSE < %t

// RUN: zkir-opt %s --mod-arith-to-arith -convert-elementwise-to-linalg --one-shot-bufferize --convert-scf-to-cf --convert-cf-to-llvm --convert-to-llvm --convert-vector-to-llvm \
// RUN:   | mlir-runner -e test_lower_mont_reduce -entry-point-result=void \
// RUN:      --shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s --check-prefix=CHECK_TEST_MONT_REDUCE < %t

// RUN: zkir-opt %s --mod-arith-to-arith -convert-elementwise-to-linalg --one-shot-bufferize --convert-scf-to-cf --convert-cf-to-llvm --convert-to-llvm --convert-vector-to-llvm \
// RUN:   | mlir-runner -e test_lower_mont_mul -entry-point-result=void \
// RUN:      --shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s --check-prefix=CHECK_TEST_MONT_MUL < %t

!Fr = !mod_arith.int<2147483647:i32>

func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

func.func @test_lower_inverse() {
  %p = mod_arith.constant 3723 : !Fr
  %1 = mod_arith.inverse %p : !Fr
  %2 = mod_arith.extract %1 : !Fr -> i32
  %3 = tensor.from_elements %2 : tensor<1xi32>

  %4 = bufferization.to_memref %3 : tensor<1xi32> to memref<1xi32>
  %U = memref.cast %4 : memref<1xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()
  return
}

// CHECK_TEST_INVERSE: [1324944920]

!Fq = !mod_arith.int<21888242871839275222246405745257275088548364400416034343698204186575808495617 : i256>
#Fq_mont = #mod_arith.montgomery<!Fq>

func.func @test_lower_mont_reduce() {
  %p = arith.constant 2188824287183927522224640574525727508854836440041603434369820418657580849561 : i256
  %zero = arith.constant 0 : i256
  // `pR` is `p` << 256 so just give `p` as `high` and set `low` to 0
  %p_mont = mod_arith.mont_reduce %zero, %p {montgomery=#Fq_mont} : i256 -> !Fq

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

// CHECK_TEST_MONT_REDUCE: [1]

func.func @test_lower_mont_mul() {
  %p = mod_arith.constant 17221657567640823606390383439573883756117969501024189775361 : !Fq
  %p_mont = mod_arith.to_mont %p {montgomery=#Fq_mont} : !Fq
  %p_mont_sq = mod_arith.mont_mul %p_mont, %p_mont {montgomery=#Fq_mont} : !Fq
  %p_sq = mod_arith.from_mont %p_mont_sq {montgomery=#Fq_mont} : !Fq

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
