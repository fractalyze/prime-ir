// Copyright 2025 The ZKIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

// RUN: zkir-opt %s -specialize-arith-to-avx | FileCheck %s --check-prefix=CHECK-LOWERING

// RUN: zkir-opt %s -specialize-arith-to-avx -convert-to-llvm \
// RUN:   | mlir-runner -e main -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s < %t


func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

func.func @packed_mului_extended() {
  %a = arith.constant dense<[1,10,100,1000,10000,100000,100000,10000000,1,10,100,1000,10000,100000,100000,10000000]> : vector<16xi32>
  %b = arith.constant dense<[1,10,100,1000,10000,100000,100000,10000000,1,10,100,1000,10000,100000,100000,10000000]> : vector<16xi32>

  // CHECK-LOWERING-NOT: arith.mului_extended
  %c:2 = arith.mului_extended %a, %b : vector<16xi32>

  %mem = memref.alloc() : memref<32xi32>
  %idx_low = arith.constant 0 : index
  %idx_high = arith.constant 16 : index
  vector.store %c#0, %mem[%idx_low] : memref<32xi32>, vector<16xi32>
  vector.store %c#1, %mem[%idx_high] : memref<32xi32>, vector<16xi32>

  %U = memref.cast %mem : memref<32xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()
  return
}

func.func @packed_mulsi_extended() {
  %a = arith.constant dense<[1,10,100,1000,10000,100000,100000,10000000,-1,-10,-100,-1000,-10000,-100000,-100000,-10000000]> : vector<16xi32>
  %b = arith.constant dense<[1,10,100,1000,10000,100000,100000,10000000,1,10,100,1000,10000,100000,100000,10000000]> : vector<16xi32>

  // CHECK-LOWERING-NOT: arith.mulsi_extended
  %c:2 = arith.mulsi_extended %a, %b : vector<16xi32>

  %mem = memref.alloc() : memref<32xi32>
  %idx_low = arith.constant 0 : index
  %idx_high = arith.constant 16 : index
  vector.store %c#0, %mem[%idx_low] : memref<32xi32>, vector<16xi32>
  vector.store %c#1, %mem[%idx_high] : memref<32xi32>, vector<16xi32>

  %U = memref.cast %mem : memref<32xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()
  return
}

func.func @packed_muli() {
  // Use large numbers to test overflow
  %a = arith.constant dense<[2147483647, 2000000000, 1500000000, 1000000000, 500000000, 2147483647, 2000000000, 1500000000, 1000000000, 500000000, 2147483647, 2000000000, 1500000000, 1000000000, 500000000, 2147483647]> : vector<16xi32>
  %b = arith.constant dense<[2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 2]> : vector<16xi32>

  // CHECK-LOWERING-NOT: arith.muli
  %c = arith.muli %a, %b : vector<16xi32>

  %mem = memref.alloc() : memref<16xi32>
  %idx = arith.constant 0 : index
  vector.store %c, %mem[%idx] : memref<16xi32>, vector<16xi32>

  %U = memref.cast %mem : memref<16xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()
  return
}

func.func @packed_addi() {
  // Chain addi operations with extended multiplication results
  // This creates a sequence where addi works on gather results
  %a = arith.constant dense<[1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000, 9000000, 10000000, 11000000, 12000000, 13000000, 14000000, 15000000, 16000000]> : vector<16xi32>
  %b = arith.constant dense<[100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000, 1100000, 1200000, 1300000, 1400000, 1500000, 1600000]> : vector<16xi32>

  // CHECK-LOWERING-NOT: arith.mului_extended
  %mul:2 = arith.mului_extended %a, %b : vector<16xi32>
  %mul1:2 = arith.mului_extended %mul#0, %mul#1 : vector<16xi32>

  // Chain addi operations on the low results (gather low path)
  // CHECK-LOWERING-NOT: arith.addi
  %low = arith.addi %mul#0, %mul1#0 : vector<16xi32>
  %high = arith.addi %mul#1, %mul1#1 : vector<16xi32>


  // CHECK-LOWERING-NOT: arith.mului_extended
  %mul2:2 = arith.mului_extended %low, %high : vector<16xi32>

  %mem = memref.alloc() : memref<32xi32>
  %idx_low = arith.constant 0 : index
  %idx_high = arith.constant 16 : index
  vector.store %mul2#0, %mem[%idx_low] : memref<32xi32>, vector<16xi32>
  vector.store %mul2#1, %mem[%idx_high] : memref<32xi32>, vector<16xi32>

  %U = memref.cast %mem : memref<32xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()
  return
}

func.func @packed_subi() {
  // Chain subi operations with extended multiplication results
  // This creates a sequence where subi works on gather results
  %a = arith.constant dense<[2000000000, 1900000000, 1800000000, 1700000000, 1600000000, 1500000000, 1400000000, 1300000000, 1200000000, 1100000000, 1000000000, 900000000, 800000000, 700000000, 600000000, 500000000]> : vector<16xi32>
  %b = arith.constant dense<[100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000, 1100000, 1200000, 1300000, 1400000, 1500000, 1600000]> : vector<16xi32>

  // CHECK-LOWERING-NOT: arith.mului_extended
  %mul:2 = arith.mului_extended %a, %b : vector<16xi32>
  %mul1:2 = arith.mului_extended %mul#0, %mul#1 : vector<16xi32>

  // CHECK-LOWERING-NOT: arith.addi
  %low = arith.subi %mul#0, %mul1#0 : vector<16xi32>
  %high = arith.subi %mul#1, %mul1#1 : vector<16xi32>


  // CHECK-LOWERING-NOT: arith.mului_extended
  %mul2:2 = arith.mulsi_extended %low, %high : vector<16xi32>

  %mem = memref.alloc() : memref<32xi32>
  %idx_low = arith.constant 0 : index
  %idx_high = arith.constant 16 : index
  vector.store %mul2#0, %mem[%idx_low] : memref<32xi32>, vector<16xi32>
  vector.store %mul2#1, %mem[%idx_high] : memref<32xi32>, vector<16xi32>

  %U = memref.cast %mem : memref<32xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()
  return
}

func.func @main() {
  func.call @packed_mului_extended() : () -> ()
  func.call @packed_mulsi_extended() : () -> ()
  func.call @packed_muli() : () -> ()
  func.call @packed_addi() : () -> ()
  func.call @packed_subi() : () -> ()
  return
}

// CHECK: [1, 100, 10000, 1000000, 100000000, 1410065408, 1410065408, 276447232, 1, 100, 10000, 1000000, 100000000, 1410065408, 1410065408, 276447232, 0, 0, 0, 0, 0, 2, 2, 23283, 0, 0, 0, 0, 0, 2, 2, 23283]
// CHECK: [1, 100, 10000, 1000000, 100000000, 1410065408, 1410065408, 276447232, -1, -100, -10000, -1000000, -100000000, -1410065408, -1410065408, -276447232, 0, 0, 0, 0, 0, 2, 2, 23283, -1, -1, -1, -1, -1, -3, -3, -23284]
// CHECK: [-2, 1705032704, 1705032704, 705032704, -1294967296, -2, 1705032704, 1705032704, 705032704, -1294967296, -2, 1705032704, 1705032704, 705032704, -1294967296, -2]
// CHECK: [54968320, 1637007360, 1301966848, 1629224960, 500674560, 274325504, -72935424, -659423232, -848732160, 1780326400, -1201016832, 1053392896, -460959744, 1955725312, -53716992, 1405091840, 23, 45, 320, 186, 415, 674, 1801, 222, 2219, 2100, 2540, 3150, 353, 3578, 1745, 6551]
// CHECK: [647626752, -1145110528, -868925440, -857636864, -1325268992, -542474240, -1870118912, 1091567616, 696254464, -1803812864, -1803812864, 696254464, 1091567616, -1870118912, -542474240, -1325268992, -14139, -10074, 15822, 6581, -10895, 36639, -76454, -8245, -17665, -17039, -17039, -17665, -8245, -76454, 36639, -10895]
