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

// Poseidon2 utility functions for BabyBear field
// Based on Plonky3 implementation: https://github.com/Plonky3/Plonky3

!pf = !field.pf<2013265921 : i32, true>
!pf_std = !field.pf<2013265921 : i32>
!state = memref<16x!pf>
!state_std = memref<16x!pf_std>

func.func @add_rc_and_sbox(%var: !pf, %c: !pf) -> !pf {
  %c7 = arith.constant 7 : i32
  %sum = field.add %var, %c : !pf
  %sum_exp7 = field.powui %sum, %c7 : !pf, i32
  return %sum_exp7 : !pf
}

// In-place version of apply_mat4 using memref
// Optimally, we just want to do matmul which then lowers to the following
// sequence but at this moment, it seems hard to achieve. Therefore, we just use field addition instead of matrix multiplication.
func.func @apply_mat4(%state: memref<4x!pf, strided<[1], offset: ?>>) {
  // Load the 4x4 MDS matrix (no changes here)
  %matrix = arith.constant dense<[
    [2, 3, 1, 1],
    [1, 2, 3, 1],
    [1, 1, 2, 3],
    [3, 1, 1, 2]
  ]> : tensor<4x4xi32>

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index

  // Compute the sum of all 4 elements
  %x0 = memref.load %state[%c0] : memref<4x!pf, strided<[1], offset: ?>>
  %x1 = memref.load %state[%c1] : memref<4x!pf, strided<[1], offset: ?>>
  %x2 = memref.load %state[%c2] : memref<4x!pf, strided<[1], offset: ?>>
  %x3 = memref.load %state[%c3] : memref<4x!pf, strided<[1], offset: ?>>

  %x01 = field.add %x0, %x1 : !pf
  %x23 = field.add %x2, %x3 : !pf
  %x0123 = field.add %x01, %x23 : !pf
  %x01123 = field.add %x0123, %x1 : !pf
  %x01233 = field.add %x0123, %x3 : !pf

  %x00 = field.double %x0 : !pf
  %x22 = field.double %x2 : !pf


  // x[0] = x01123 + x01
  %x0_new = field.add %x01123, %x01 : !pf
  // x[1] = x01123 + 2*x[2]
  %x1_new = field.add %x01123, %x22 : !pf
  // x[2] = x01233 + x23
  %x2_new = field.add %x01233, %x23 : !pf
  // x[3] = x01233 + 2*x[0]
  %x3_new = field.add %x01233, %x00 : !pf

  // Store the sum in all output positions
  memref.store %x0_new, %state[%c0] : memref<4x!pf, strided<[1], offset: ?>>
  memref.store %x1_new, %state[%c1] : memref<4x!pf, strided<[1], offset: ?>>
  memref.store %x2_new, %state[%c2] : memref<4x!pf, strided<[1], offset: ?>>
  memref.store %x3_new, %state[%c3] : memref<4x!pf, strided<[1], offset: ?>>
  return
}

func.func @mds_light_permutation(%state: !state) {
  // First, apply M_4 to each consecutive four elements of the state
  // This replaces each x_i with x_i'
  affine.for %chunk_idx = 0 to 4 {
    // Calculate offset for this chunk
    %x0 = affine.load %state[%chunk_idx * 4] : !state
    %x1 = affine.load %state[%chunk_idx * 4 + 1] : !state
    %x01 = field.add %x0, %x1 : !pf
    %x2 = affine.load %state[%chunk_idx * 4 + 2] : !state
    %x3 = affine.load %state[%chunk_idx * 4 + 3] : !state
    %x23 = field.add %x2, %x3 : !pf
    %x0123 = field.add %x01, %x23 : !pf
    %x01123 = field.add %x0123, %x1 : !pf
    %x01233 = field.add %x0123, %x3 : !pf

    %x00 = field.double %x0 : !pf
    %x22 = field.double %x2 : !pf

    // x[0] = x01123 + x01
    %x0_new = field.add %x01123, %x01 : !pf
    // x[1] = x01123 + 2*x[2]
    %x1_new = field.add %x01123, %x22 : !pf
    // x[2] = x01233 + x23
    %x2_new = field.add %x01233, %x23 : !pf
    // x[3] = x01233 + 2*x[0]
    %x3_new = field.add %x01233, %x00 : !pf

    // Store the sum in all output positions
    affine.store %x0_new, %state[%chunk_idx * 4] : !state
    affine.store %x1_new, %state[%chunk_idx * 4 + 1] : !state
    affine.store %x2_new, %state[%chunk_idx * 4 + 2] : !state
    affine.store %x3_new, %state[%chunk_idx * 4 + 3] : !state
  }

  // Now apply the outer circulant matrix
  // Precompute the four sums of every four elements
  // Compute sums: sums[k] = sum of state[j + k] for j = 0, 4, 8, 12
  %sums = memref.alloca() : memref<4x!pf>
  affine.for %k = 0 to 4 {
    %val0 = affine.load %state[%k] : !state
    %val1 = affine.load %state[%k + 4] : !state
    %val2 = affine.load %state[%k + 8] : !state
    %val3 = affine.load %state[%k + 12] : !state
    %sum01 = field.add %val0, %val1 : !pf
    %sum23 = field.add %val2, %val3 : !pf
    %new_sum = field.add %sum01, %sum23 : !pf
    affine.store %new_sum, %sums[%k] : memref<4x!pf>
  }

  // Apply the formula: y_i = x_i' + sums[i % 4]
  affine.for %i = 0 to 4 {
    %val0 = affine.load %state[%i] : !state
    %val1 = affine.load %state[%i + 4] : !state
    %val2 = affine.load %state[%i + 8] : !state
    %val3 = affine.load %state[%i + 12] : !state
    %sum = affine.load %sums[%i] : memref<4x!pf>
    %sum0 = field.add %val0, %sum : !pf
    %sum1 = field.add %val1, %sum : !pf
    %sum2 = field.add %val2, %sum : !pf
    %sum3 = field.add %val3, %sum : !pf
    affine.store %sum0, %state[%i] : !state
    affine.store %sum1, %state[%i + 4] : !state
    affine.store %sum2, %state[%i + 8] : !state
    affine.store %sum3, %state[%i + 12] : !state
  }
  return
}

// Internal layer matrix multiplication
func.func @internal_layer_mat_mul(%state: !state, %sum: !pf) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index
  %c8 = arith.constant 8 : index
  %c9 = arith.constant 9 : index
  %c10 = arith.constant 10 : index
  %c11 = arith.constant 11 : index
  %c12 = arith.constant 12 : index
  %c13 = arith.constant 13 : index
  %c14 = arith.constant 14 : index
  %c15 = arith.constant 15 : index

  // Precompute powers of 2 inverses using powui
  %two_std = field.constant 2 : !pf_std
  %two = field.to_mont %two_std : !pf
  %inv_two = field.inverse %two : !pf
  %exp_2 = arith.constant 2 : i32
  %exp_3 = arith.constant 3 : i32
  %exp_4 = arith.constant 4 : i32
  %exp_8 = arith.constant 8 : i32
  %exp_27 = arith.constant 27 : i32
  %inv_four = field.powui %inv_two, %exp_2 : !pf, i32
  %inv_eight = field.powui %inv_two, %exp_3 : !pf, i32
  %inv_sixteen = field.powui %inv_two, %exp_4 : !pf, i32
  %inv_256 = field.square %inv_sixteen : !pf
  %inv_2_27 = field.powui %inv_two, %exp_27 : !pf, i32

  // state[1] += sum
  %s1 = memref.load %state[%c1] : !state
  %new_s1 = field.add %s1, %sum : !pf
  memref.store %new_s1, %state[%c1] : !state

  // state[2] = state[2].double() + sum
  %s2 = memref.load %state[%c2] : !state
  %s2_double = field.double %s2 : !pf
  %new_s2 = field.add %s2_double, %sum : !pf
  memref.store %new_s2, %state[%c2] : !state

  // state[3] = state[3].halve() + sum
  %s3 = memref.load %state[%c3] : !state
  %s3_halve = field.mul %s3, %inv_two : !pf
  %new_s3 = field.add %s3_halve, %sum : !pf
  memref.store %new_s3, %state[%c3] : !state

  // state[4] = sum + state[4].double() + state[4]
  %s4 = memref.load %state[%c4] : !state
  %s4_double = field.double %s4 : !pf
  %s4_sum = field.add %s4_double, %s4 : !pf
  %new_s4 = field.add %sum, %s4_sum : !pf
  memref.store %new_s4, %state[%c4] : !state

  // state[5] = sum + state[5].double().double()
  %s5 = memref.load %state[%c5] : !state
  %s5_double = field.double %s5 : !pf
  %s5_double_double = field.double %s5_double : !pf
  %new_s5 = field.add %sum, %s5_double_double : !pf
  memref.store %new_s5, %state[%c5] : !state

  // state[6] = sum - state[6].halve()
  %s6 = memref.load %state[%c6] : !state
  %s6_halve = field.mul %s6, %inv_two : !pf
  %new_s6 = field.sub %sum, %s6_halve : !pf
  memref.store %new_s6, %state[%c6] : !state

  // state[7] = sum - (state[7].double() + state[7])
  %s7 = memref.load %state[%c7] : !state
  %s7_double = field.double %s7 : !pf
  %s7_sum = field.add %s7_double, %s7 : !pf
  %new_s7 = field.sub %sum, %s7_sum : !pf
  memref.store %new_s7, %state[%c7] : !state

  // state[8] = sum - state[8].double().double()
  %s8 = memref.load %state[%c8] : !state
  %s8_double = field.double %s8 : !pf
  %s8_double_double = field.double %s8_double : !pf
  %new_s8 = field.sub %sum, %s8_double_double : !pf
  memref.store %new_s8, %state[%c8] : !state

  // state[9] = state[9] * inv_256 + sum
  %s9 = memref.load %state[%c9] : !state
  %s9_div_256 = field.mul %s9, %inv_256 : !pf
  %new_s9 = field.add %s9_div_256, %sum : !pf
  memref.store %new_s9, %state[%c9] : !state

  // state[10] = state[10] * inv_four + sum
  %s10 = memref.load %state[%c10] : !state
  %s10_div_4 = field.mul %s10, %inv_four : !pf
  %new_s10 = field.add %s10_div_4, %sum : !pf
  memref.store %new_s10, %state[%c10] : !state

  // state[11] = state[11] * inv_eight + sum
  %s11 = memref.load %state[%c11] : !state
  %s11_div_8 = field.mul %s11, %inv_eight : !pf
  %new_s11 = field.add %s11_div_8, %sum : !pf
  memref.store %new_s11, %state[%c11] : !state

  // state[12] = state[12] * inv_2_27 + sum
  %s12 = memref.load %state[%c12] : !state
  %s12_div_27 = field.mul %s12, %inv_2_27 : !pf
  %new_s12 = field.add %s12_div_27, %sum : !pf
  memref.store %new_s12, %state[%c12] : !state

  // state[13] = sum - state[13] * inv_256
  %s13 = memref.load %state[%c13] : !state
  %s13_div_256 = field.mul %s13, %inv_256 : !pf
  %new_s13 = field.sub %sum, %s13_div_256 : !pf
  memref.store %new_s13, %state[%c13] : !state

  // state[14] = sum - state[14] * inv_sixteen
  %s14 = memref.load %state[%c14] : !state
  %s14_div_16 = field.mul %s14, %inv_sixteen : !pf
  %new_s14 = field.sub %sum, %s14_div_16 : !pf
  memref.store %new_s14, %state[%c14] : !state

  // state[15] = sum - state[15] * inv_2_27
  %s15 = memref.load %state[%c15] : !state
  %s15_div_27 = field.mul %s15, %inv_2_27 : !pf
  %new_s15 = field.sub %sum, %s15_div_27 : !pf
  memref.store %new_s15, %state[%c15] : !state

  return
}

// Internal layer: permutation (add RC to first element, S-box first, internal diffusion)
func.func @permute_state(%state: !state) {
  // Convert to memref for in-place operations
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index
  %c8 = arith.constant 8 : index
  %c9 = arith.constant 9 : index
  %c10 = arith.constant 10 : index
  %c11 = arith.constant 11 : index
  %c12 = arith.constant 12 : index
  %c13 = arith.constant 13 : index
  %c14 = arith.constant 14 : index
  %c15 = arith.constant 15 : index

  // BABYBEAR_RC16_INTERNAL (13 scalar constants)
  %rc_internal = arith.constant dense<[0x5a8053c0, 0x693be639, 0x3858867d, 0x19334f6b, 0x128f0fd8, 0x4e2b1ccb, 0x61210ce0, 0x3c318939, 0x0b5b2f22, 0x2edb11d5, 0x213effdf, 0x0cac4606, 0x241af16d]> : tensor<13xi32>
  %rc_internal_std = field.bitcast %rc_internal : tensor<13xi32> -> tensor<13x!pf_std>
  %rc_internal_mont = field.to_mont %rc_internal_std : tensor<13x!pf>

  // For each internal constant: add RC and S-box to first element, then apply matrix multiplication
  affine.for %round = 0 to 13 {
    // Get current round constant via tensor.extract
    %rc = tensor.extract %rc_internal_mont[%round] : tensor<13x!pf>

    // Add RC and apply S-box to first element
    %s0 = memref.load %state[%c0] : !state
    %elem0 = func.call @add_rc_and_sbox(%s0, %rc) : (!pf, !pf) -> !pf

    // Compute sum of all elements using affine.for
    // NOTE: this is extremely slow, so we manually add them.
    // %zero = field.constant 0 : !pf
    // %sum = affine.for %i = 0 to 16 iter_args(%acc = %zero) -> (!pf) {
    //   %elem = tensor.extract %t[%i] : tensor<16x!pf>
    //   %new_acc = field.add %acc, %elem : !pf
    //   affine.yield %new_acc : !pf
    // }
    %elem1  = memref.load %state[%c1]  : memref<16x!pf>
    %elem2  = memref.load %state[%c2]  : memref<16x!pf>
    %elem3  = memref.load %state[%c3]  : memref<16x!pf>
    %elem4  = memref.load %state[%c4]  : memref<16x!pf>
    %elem5  = memref.load %state[%c5]  : memref<16x!pf>
    %elem6  = memref.load %state[%c6]  : memref<16x!pf>
    %elem7  = memref.load %state[%c7]  : memref<16x!pf>
    %elem8  = memref.load %state[%c8]  : memref<16x!pf>
    %elem9  = memref.load %state[%c9]  : memref<16x!pf>
    %elem10 = memref.load %state[%c10] : memref<16x!pf>
    %elem11 = memref.load %state[%c11] : memref<16x!pf>
    %elem12 = memref.load %state[%c12] : memref<16x!pf>
    %elem13 = memref.load %state[%c13] : memref<16x!pf>
    %elem14 = memref.load %state[%c14] : memref<16x!pf>
    %elem15 = memref.load %state[%c15] : memref<16x!pf>

    // --- Step 2: Sum the elements using a reduction tree ---
    // This structure allows for maximum parallel execution by the CPU.

    // Level 1 (8 parallel additions)
    %sum2_3   = field.add %elem2,  %elem3  : !pf
    %sum4_5   = field.add %elem4,  %elem5  : !pf
    %sum6_7   = field.add %elem6,  %elem7  : !pf
    %sum8_9   = field.add %elem8,  %elem9  : !pf
    %sum10_11 = field.add %elem10, %elem11 : !pf
    %sum12_13 = field.add %elem12, %elem13 : !pf
    %sum14_15 = field.add %elem14, %elem15 : !pf

    // Level 2 (4 parallel additions)
    %sum1_3   = field.add %elem1,   %sum2_3   : !pf
    %sum4_7   = field.add %sum4_5,   %sum6_7   : !pf
    %sum8_11  = field.add %sum8_9,   %sum10_11 : !pf
    %sum12_15 = field.add %sum12_13, %sum14_15 : !pf

    // Level 3 (2 parallel additions)
    %sum1_7   = field.add %sum1_3,   %sum4_7   : !pf
    %sum8_15  = field.add %sum8_11,  %sum12_15 : !pf

    // Level 4 (Partial sum)
    %partial_sum = field.add %sum1_7, %sum8_15 : !pf

    %total_sum = field.add %partial_sum,  %elem0  : !pf
    %new_s0 = field.sub %partial_sum, %elem0 : !pf
    memref.store %new_s0, %state[%c0] : !state

    // Apply internal layer matrix multiplication
    func.call @internal_layer_mat_mul(%state, %total_sum) : (!state, !pf) -> ()
  }
  return
}

// External layer: terminal permutation (4 rounds: add RC, S-box, MDS)
func.func @permute_state_terminal(%state: !state) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index

  // BABYBEAR_RC16_EXTERNAL_FINAL (4 rounds x 16 constants)
  %rc_external_const = arith.constant dense<[
    [0x7290a80d, 0x6f7e5329, 0x598ec8a8, 0x76a859a0, 0x6559e868, 0x657b83af, 0x13271d3f, 0x1f876063, 0x0aeeae37, 0x706e9ca6, 0x46400cee, 0x72a05c26, 0x2c589c9e, 0x20bd37a7, 0x6a2d3d10, 0x20523767],
    [0x5b8fe9c4, 0x2aa501d6, 0x1e01ac3e, 0x1448bc54, 0x5ce5ad1c, 0x4918a14d, 0x2c46a83f, 0x4fcf6876, 0x61d8d5c8, 0x6ddf4ff9, 0x11fda4d3, 0x02933a8f, 0x170eaf81, 0x5a9c314f, 0x49a12590, 0x35ec52a1],
    [0x58eb1611, 0x5e481e65, 0x367125c9, 0x0eba33ba, 0x1fc28ded, 0x066399ad, 0x0cbec0ea, 0x75fd1af0, 0x50f5bf4e, 0x643d5f41, 0x6f4fe718, 0x5b3cbbde, 0x1e3afb3e, 0x296fb027, 0x45e1547b, 0x4a8db2ab],
    [0x59986d19, 0x30bcdfa3, 0x1db63932, 0x1d7c2824, 0x53b33681, 0x0673b747, 0x038a98a3, 0x2c5bce60, 0x351979cd, 0x5008fb73, 0x547bca78, 0x711af481, 0x3f93bf64, 0x644d987b, 0x3c8bcd87, 0x608758b8]
  ]> : tensor<4x16xi32>

  %rc_external_std = field.bitcast %rc_external_const : tensor<4x16xi32> -> tensor<4x16x!pf_std>
  %rc_external_final = field.to_mont %rc_external_std : tensor<4x16x!pf>
  %state_tensor = bufferization.to_tensor %state restrict : memref<16x!pf> to tensor<16x!pf>

  // Loop through 4 rounds of external terminal permutation
  affine.for %round = 0 to 4 {
    affine.for %i = 0 to 16 {
      %s = tensor.extract %state_tensor[%i] : tensor<16x!pf>
      %c = tensor.extract %rc_external_final[%round, %i] : tensor<4x16x!pf>
      %sbox = func.call @add_rc_and_sbox(%s, %c) : (!pf, !pf) -> !pf
      affine.store %sbox, %state[%i] : !state
    }

    // Apply MDS light permutation (in-place)
    func.call @mds_light_permutation(%state) : (!state) -> ()
  }

  return
}

// External layer: initial permutation (MDS light + terminal permutation)
func.func @permute_state_initial(%state: !state) {
  // First apply MDS light permutation
  func.call @mds_light_permutation(%state) : (!state) -> ()

  // Round constants for 16-width Poseidon2 on BabyBear
  // BABYBEAR_RC16_EXTERNAL_INITIAL (4 rounds x 16 constants)
  %rc_external_const = arith.constant dense<[
    [0x69cbb6af, 0x46ad93f9, 0x60a00f4e, 0x6b1297cd, 0x23189afe, 0x732e7bef, 0x72c246de, 0x2c941900, 0x0557eede, 0x1580496f, 0x3a3ea77b, 0x54f3f271, 0x0f49b029, 0x47872fe1, 0x221e2e36, 0x1ab7202e],
    [0x487779a6, 0x3851c9d8, 0x38dc17c0, 0x209f8849, 0x268dcee8, 0x350c48da, 0x5b9ad32e, 0x0523272b, 0x3f89055b, 0x01e894b2, 0x13ddedde, 0x1b2ef334, 0x7507d8b4, 0x6ceeb94e, 0x52eb6ba2, 0x50642905],
    [0x05453f3f, 0x06349efc, 0x6922787c, 0x04bfff9c, 0x768c714a, 0x3e9ff21a, 0x15737c9c, 0x2229c807, 0x0d47f88c, 0x097e0ecc, 0x27eadba0, 0x2d7d29e4, 0x3502aaa0, 0x0f475fd7, 0x29fbda49, 0x018afffd],
    [0x0315b618, 0x6d4497d1, 0x1b171d9e, 0x52861abd, 0x2e5d0501, 0x3ec8646c, 0x6e5f250a, 0x148ae8e6, 0x17f5fa4a, 0x3e66d284, 0x0051aa3b, 0x483f7913, 0x2cfe5f15, 0x023427ca, 0x2cc78315, 0x1e36ea47]
  ]> : tensor<4x16xi32>

  %rc_external_std = field.bitcast %rc_external_const : tensor<4x16xi32> -> tensor<4x16x!pf_std>
  %rc_external_final = field.to_mont %rc_external_std : tensor<4x16x!pf>
  %state_tensor = bufferization.to_tensor %state restrict : memref<16x!pf> to tensor<16x!pf>

  // Then apply terminal permutation with initial external constants
  // Loop through 4 rounds of external terminal permutation
  affine.for %round = 0 to 4 {
    affine.for %i = 0 to 16 {
      %s = tensor.extract %state_tensor[%i] : tensor<16x!pf>
      %c = tensor.extract %rc_external_final[%round, %i] : tensor<4x16x!pf>
      %sbox = func.call @add_rc_and_sbox(%s, %c) : (!pf, !pf) -> !pf
      affine.store %sbox, %state[%i] : !state
    }

    // Apply MDS light permutation (in-place)
    func.call @mds_light_permutation(%state) : (!state) -> ()
  }
  return
}

// Complete Poseidon2 permutation
func.func @poseidon2_permute(%state: !state) {
  func.call @permute_state_initial(%state) : (!state) -> ()
  func.call @permute_state(%state) : (!state) -> ()
  func.call @permute_state_terminal(%state) : (!state) -> ()
  return
}

func.func @permute_10000(%state : !state) attributes { llvm.emit_c_interface } {
  affine.for %i = 0 to 10000 {
    func.call @poseidon2_permute(%state) : (!state) -> ()
  }
  return
}
