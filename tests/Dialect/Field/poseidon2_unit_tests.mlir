// Copyright 2025 The PrimeIR Authors.
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

// RUN: cat %S/../../poseidon2.mlir %S/../../poseidon2_packed.mlir %s \
// RUN:   | prime-ir-opt --field-to-llvm=bufferize-function-boundaries -convert-vector-to-llvm \
// RUN:   | mlir-runner -e main -entry-point-result=void \
// RUN:      --shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s < %t

// Unit tests for Poseidon2 individual functions
// Reference: https://github.com/Plonky3/Plonky3/blob/main/baby-bear/src/poseidon2.rs


// Test: apply_mat4 function with Plonky3 test case
// Input: [x0, x1, x2, x3] with seeded random values
// Expected: Matrix multiplication with [2 3 1 1; 1 2 3 1; 1 1 2 3; 3 1 1 2]
func.func @test_apply_mat4() {
  %input_int = arith.constant dense<[1983708094, 1477844074, 1638775686, 98517138]> : tensor<4xi32>
  %input_std = field.bitcast %input_int : tensor<4xi32> -> tensor<4x!pf_std>
  %input = field.to_mont %input_std : tensor<4x!pf>

  %input_memref = bufferization.to_buffer %input : tensor<4x!pf> to memref<4x!pf, strided<[1], offset: ?>>
  func.call @apply_mat4(%input_memref) : (memref<4x!pf, strided<[1], offset: ?>>) -> ()
  %result = bufferization.to_tensor %input_memref restrict : memref<4x!pf, strided<[1], offset: ?>> to tensor<4x!pf>

  %result_std = field.from_mont %result : tensor<4x!pf_std>
  %result_i32 = field.bitcast %result_std : tensor<4x!pf_std> -> tensor<4xi32>

  %mem = bufferization.to_buffer %result_i32 : tensor<4xi32> to memref<4xi32>
  %mem_cast = memref.cast %mem : memref<4xi32> to memref<*xi32>
  func.call @printMemrefI32(%mem_cast) : (memref<*xi32>) -> ()

  return
}
// Manual calculation for BabyBear field (modulus 2013265921):
// x0 = 1983708094, x1 = 1477844074, x2 = 1638775686, x3 = 98517138
// y0 = 2*x0 + 3*x1 + 1*x2 + 1*x3 = 2*1983708094 + 3*1477844074 + 1638775686 + 98517138
// y1 = 1*x0 + 2*x1 + 3*x2 + 1*x3 = 1983708094 + 2*1477844074 + 3*1638775686 + 98517138
// y2 = 1*x0 + 1*x1 + 2*x2 + 3*x3 = 1983708094 + 1477844074 + 2*1638775686 + 3*98517138
// y3 = 3*x0 + 1*x1 + 1*x2 + 2*x3 = 3*1983708094 + 1477844074 + 1638775686 + 2*98517138
// CHECK: [71911629, 1901176754, 994857191, 1211714634]

func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

// Test: Poseidon2 permutation on state [0, 1, ..., 15] (Plonky3 test vector)
func.func @test_poseidon2_permute() {
  // Prepare input tensor [0, 1, ..., 15]
  %input_int = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : tensor<16xi32>
  %input_std = field.bitcast %input_int : tensor<16xi32> -> tensor<16x!pf_std>
  %input = field.to_mont %input_std : tensor<16x!pf>
  // Run Poseidon2 permutation
  %state = bufferization.to_buffer %input : tensor<16x!pf> to !state
  func.call @poseidon2_permute(%state) : (!state) -> ()
  %output = bufferization.to_tensor %state restrict : !state to tensor<16x!pf>
  %output_std = field.from_mont %output : tensor<16x!pf_std>
  %output_i32 = field.bitcast %output_std : tensor<16x!pf_std> -> tensor<16xi32>
  // Print output
  %mem = bufferization.to_buffer %output_i32 : tensor<16xi32> to memref<16xi32>
  %mem_cast = memref.cast %mem : memref<16xi32> to memref<*xi32>
  func.call @printMemrefI32(%mem_cast) : (memref<*xi32>) -> ()
  return
}

// expected output for input [0..15]:
// CHECK: [1906786279, 1737026427, 1959749225, 700325316, 1638050605, 1021608788, 1726691001, 1761127344, 1552405120, 417318995, 36799261, 1215172152, 614923223, 1300746575, 957311597, 304856115]

func.func @test_packed_poseidon2_permute() {
  // Prepare input tensor [0, 1, ..., 15]
  %state = memref.alloca() : !packed_state
  // Use a for loop to fill each slot with vector<j> (where j = i), bitcast to field, convert to mont
  %zero = arith.constant 0 : index
  %sixteen = arith.constant 16 : index
  %one = arith.constant 1 : index
  scf.for %i = %zero to %sixteen step %one {
    %val = arith.index_cast %i : index to i32
    %v_i32 = vector.splat %val : vector<16xi32>
    %v_std = field.bitcast %v_i32 : vector<16xi32> -> !packed_std
    %v_mont = field.to_mont %v_std : !packed
    memref.store %v_mont, %state[%i] : !packed_state
  }

  // Run Poseidon2 permutation
  func.call @packed_poseidon2_permute(%state) : (!packed_state) -> ()
  // Load diagonal entries from %state (memref<16x!packed>) and print them.
  // Allocate output: memref<16x!pf>
  %diag_mem = memref.alloca() : memref<16xi32>
  scf.for %i = %zero to %sixteen step %one {
    %vec = memref.load %state[%i] : memref<16x!packed>
    %std = field.from_mont %vec : !packed_std
    %intvec = field.bitcast %std : !packed_std -> vector<16xi32>
    %val = vector.extract %intvec[%i] : i32 from vector<16xi32>
    memref.store %val, %diag_mem[%i] : memref<16xi32>
  }
  %mem_cast = memref.cast %diag_mem : memref<16xi32> to memref<*xi32>
  func.call @printMemrefI32(%mem_cast) : (memref<*xi32>) -> ()
  return
}

// expected output for input [0..15]:
// CHECK: [1906786279, 1737026427, 1959749225, 700325316, 1638050605, 1021608788, 1726691001, 1761127344, 1552405120, 417318995, 36799261, 1215172152, 614923223, 1300746575, 957311597, 304856115]


func.func @main() {
  func.call @test_apply_mat4() : () -> ()
  func.call @test_poseidon2_permute() : () -> ()
  func.call @test_packed_poseidon2_permute() : () -> ()
  return
}
