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

// Poseidon2 utility functions for BabyBear field
// Based on Plonky3 implementation: https://github.com/Plonky3/Plonky3

!scalar = !field.pf<2013265921 : i32, true>
!packed = vector<16x!field.pf<2013265921 : i32, true>>
!packed_std = vector<16x!field.pf<2013265921 : i32>>
!packed_state = memref<16x!packed>
!packed_state_std = memref<16x!packed_std>

func.func @packed_add_rc_and_sbox(%var: !packed, %c: !packed) -> !packed {
  %c7 = arith.constant 7 : i32
  %sum = field.add %var, %c : !packed
  %sum_exp7 = field.powui %sum, %c7 : !packed, i32
  return %sum_exp7 : !packed
}

// In-place version of apply_mat4 using memref
// Optimally, we just want to do matmul which then lowers to the following
// sequence but at this moment, it seems hard to achieve. Therefore, we just use field addition instead of matrix multiplication.
func.func @packed_apply_mat4(%state: memref<4x!packed, strided<[1], offset: ?>>) {
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
  %x0 = memref.load %state[%c0] : memref<4x!packed, strided<[1], offset: ?>>
  %x1 = memref.load %state[%c1] : memref<4x!packed, strided<[1], offset: ?>>
  %x2 = memref.load %state[%c2] : memref<4x!packed, strided<[1], offset: ?>>
  %x3 = memref.load %state[%c3] : memref<4x!packed, strided<[1], offset: ?>>

  %x01 = field.add %x0, %x1 : !packed
  %x23 = field.add %x2, %x3 : !packed
  %x0123 = field.add %x01, %x23 : !packed
  %x01123 = field.add %x0123, %x1 : !packed
  %x01233 = field.add %x0123, %x3 : !packed

  %x00 = field.double %x0 : !packed
  %x22 = field.double %x2 : !packed


  // x[0] = x01123 + x01
  %x0_new = field.add %x01123, %x01 : !packed
  // x[1] = x01123 + 2*x[2]
  %x1_new = field.add %x01123, %x22 : !packed
  // x[2] = x01233 + x23
  %x2_new = field.add %x01233, %x23 : !packed
  // x[3] = x01233 + 2*x[0]
  %x3_new = field.add %x01233, %x00 : !packed

  // Store the sum in all output positions
  memref.store %x0_new, %state[%c0] : memref<4x!packed, strided<[1], offset: ?>>
  memref.store %x1_new, %state[%c1] : memref<4x!packed, strided<[1], offset: ?>>
  memref.store %x2_new, %state[%c2] : memref<4x!packed, strided<[1], offset: ?>>
  memref.store %x3_new, %state[%c3] : memref<4x!packed, strided<[1], offset: ?>>
  return
}

func.func @packed_mds_light_permutation(%state: !packed_state) {
  // First, apply M_4 to each consecutive four elements of the state
  // This replaces each x_i with x_i'
  affine.for %chunk_idx = 0 to 4 {
    // Calculate offset for this chunk
    %x0 = affine.load %state[%chunk_idx * 4] : !packed_state
    %x1 = affine.load %state[%chunk_idx * 4 + 1] : !packed_state
    %x2 = affine.load %state[%chunk_idx * 4 + 2] : !packed_state
    %x3 = affine.load %state[%chunk_idx * 4 + 3] : !packed_state

    %x01 = field.add %x0, %x1 : !packed
    %x23 = field.add %x2, %x3 : !packed
    %x0123 = field.add %x01, %x23 : !packed
    %x01123 = field.add %x0123, %x1 : !packed
    %x01233 = field.add %x0123, %x3 : !packed

    %x00 = field.double %x0 : !packed
    %x22 = field.double %x2 : !packed

    // x[0] = x01123 + x01
    %x0_new = field.add %x01123, %x01 : !packed
    // x[1] = x01123 + 2*x[2]
    %x1_new = field.add %x01123, %x22 : !packed
    // x[2] = x01233 + x23
    %x2_new = field.add %x01233, %x23 : !packed
    // x[3] = x01233 + 2*x[0]
    %x3_new = field.add %x01233, %x00 : !packed

    // Store the sum in all output positions
    affine.store %x0_new, %state[%chunk_idx * 4] : !packed_state
    affine.store %x1_new, %state[%chunk_idx * 4 + 1] : !packed_state
    affine.store %x2_new, %state[%chunk_idx * 4 + 2] : !packed_state
    affine.store %x3_new, %state[%chunk_idx * 4 + 3] : !packed_state
  }

  // Now apply the outer circulant matrix
  // Precompute the four sums of every four elements
  // Compute sums: sums[k] = sum of state[j + k] for j = 0, 4, 8, 12
  %sums = memref.alloca() : memref<4x!packed>
  affine.for %k = 0 to 4 {
    %val0 = affine.load %state[%k] : !packed_state
    %val1 = affine.load %state[%k + 4] : !packed_state
    %val2 = affine.load %state[%k + 8] : !packed_state
    %val3 = affine.load %state[%k + 12] : !packed_state
    %sum01 = field.add %val0, %val1 : !packed
    %sum23 = field.add %val2, %val3 : !packed
    %new_sum = field.add %sum01, %sum23 : !packed
    affine.store %new_sum, %sums[%k] : memref<4x!packed>
  }

  // Apply the formula: y_i = x_i' + sums[i % 4]
  affine.for %i = 0 to 4 {
    %val0 = affine.load %state[%i] : !packed_state
    %val1 = affine.load %state[%i + 4] : !packed_state
    %val2 = affine.load %state[%i + 8] : !packed_state
    %val3 = affine.load %state[%i + 12] : !packed_state
    %sum = affine.load %sums[%i] : memref<4x!packed>
    %sum0 = field.add %val0, %sum : !packed
    %sum1 = field.add %val1, %sum : !packed
    %sum2 = field.add %val2, %sum : !packed
    %sum3 = field.add %val3, %sum : !packed
    affine.store %sum0, %state[%i] : !packed_state
    affine.store %sum1, %state[%i + 4] : !packed_state
    affine.store %sum2, %state[%i + 8] : !packed_state
    affine.store %sum3, %state[%i + 12] : !packed_state
  }

  return
}

// Internal layer matrix multiplication
func.func @packed_internal_layer_mat_mul(%state: !packed_state, %sum : !packed) {
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
  // [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, 1/4, 1/8, 1/2^27, -1/2^8, -1/16, -1/2^27]
  %inv_two_scalar = arith.constant 134217727 : i32
  %inv_two_vec = vector.splat %inv_two_scalar : vector<16xi32>
  %inv_two = field.bitcast %inv_two_vec : vector<16xi32> -> !packed

  %inv_four_scalar = arith.constant 1073741824 : i32
  %inv_four_vec = vector.splat %inv_four_scalar : vector<16xi32>
  %inv_four = field.bitcast %inv_four_vec : vector<16xi32> -> !packed

  %inv_eight_scalar = arith.constant 536870912 : i32
  %inv_eight_vec = vector.splat %inv_eight_scalar : vector<16xi32>
  %inv_eight = field.bitcast %inv_eight_vec : vector<16xi32> -> !packed

  %inv_sixteen_scalar = arith.constant 268435456 : i32
  %inv_sixteen_vec = vector.splat %inv_sixteen_scalar : vector<16xi32>
  %inv_sixteen = field.bitcast %inv_sixteen_vec : vector<16xi32> -> !packed

  %inv_256_scalar = arith.constant 16777216 : i32
  %inv_256_vec = vector.splat %inv_256_scalar : vector<16xi32>
  %inv_256 = field.bitcast %inv_256_vec : vector<16xi32> -> !packed

  %inv_2_27_scalar = arith.constant 32 : i32
  %inv_2_27_vec = vector.splat %inv_2_27_scalar : vector<16xi32>
  %inv_2_27 = field.bitcast %inv_2_27_vec : vector<16xi32> -> !packed

  // state[1] += sum
  %s1 = memref.load %state[%c1] : !packed_state
  %new_s1 = field.add %s1, %sum : !packed
  memref.store %new_s1, %state[%c1] : !packed_state

  // state[2] = state[2].double() + sum
  %s2 = memref.load %state[%c2] : !packed_state
  %s2_double = field.double %s2 : !packed
  %new_s2 = field.add %s2_double, %sum : !packed
  memref.store %new_s2, %state[%c2] : !packed_state

  // state[3] = state[3].halve() + sum
  %s3 = memref.load %state[%c3] : !packed_state
  %s3_halve = field.mul %s3, %inv_two : !packed
  %new_s3 = field.add %s3_halve, %sum : !packed
  memref.store %new_s3, %state[%c3] : !packed_state

  // state[4] = sum + state[4].double() + state[4]
  %s4 = memref.load %state[%c4] : !packed_state
  %s4_double = field.double %s4 : !packed
  %s4_sum = field.add %s4_double, %s4 : !packed
  %new_s4 = field.add %sum, %s4_sum : !packed
  memref.store %new_s4, %state[%c4] : !packed_state

  // state[5] = sum + state[5].double().double()
  %s5 = memref.load %state[%c5] : !packed_state
  %s5_double = field.double %s5 : !packed
  %s5_double_double = field.double %s5_double : !packed
  %new_s5 = field.add %sum, %s5_double_double : !packed
  memref.store %new_s5, %state[%c5] : !packed_state

  // state[6] = sum - state[6].halve()
  %s6 = memref.load %state[%c6] : !packed_state
  %s6_halve = field.mul %s6, %inv_two : !packed
  %new_s6 = field.sub %sum, %s6_halve : !packed
  memref.store %new_s6, %state[%c6] : !packed_state

  // state[7] = sum - (state[7].double() + state[7])
  %s7 = memref.load %state[%c7] : !packed_state
  %s7_double = field.double %s7 : !packed
  %s7_sum = field.add %s7_double, %s7 : !packed
  %new_s7 = field.sub %sum, %s7_sum : !packed
  memref.store %new_s7, %state[%c7] : !packed_state

  // state[8] = sum - state[8].double().double()
  %s8 = memref.load %state[%c8] : !packed_state
  %s8_double = field.double %s8 : !packed
  %s8_double_double = field.double %s8_double : !packed
  %new_s8 = field.sub %sum, %s8_double_double : !packed
  memref.store %new_s8, %state[%c8] : !packed_state

  // state[9] = state[9] * inv_256 + sum
  %s9 = memref.load %state[%c9] : !packed_state
  %s9_div_256 = field.mul %s9, %inv_256 : !packed
  %new_s9 = field.add %s9_div_256, %sum : !packed
  memref.store %new_s9, %state[%c9] : !packed_state

  // state[10] = state[10] * inv_four + sum
  %s10 = memref.load %state[%c10] : !packed_state
  %s10_div_4 = field.mul %s10, %inv_four : !packed
  %new_s10 = field.add %s10_div_4, %sum : !packed
  memref.store %new_s10, %state[%c10] : !packed_state

  // state[11] = state[11] * inv_eight + sum
  %s11 = memref.load %state[%c11] : !packed_state
  %s11_div_8 = field.mul %s11, %inv_eight : !packed
  %new_s11 = field.add %s11_div_8, %sum : !packed
  memref.store %new_s11, %state[%c11] : !packed_state

  // state[12] = state[12] * inv_2_27 + sum
  %s12 = memref.load %state[%c12] : !packed_state
  %s12_div_27 = field.mul %s12, %inv_2_27 : !packed
  %new_s12 = field.add %s12_div_27, %sum : !packed
  memref.store %new_s12, %state[%c12] : !packed_state

  // state[13] = sum - state[13] * inv_256
  %s13 = memref.load %state[%c13] : !packed_state
  %s13_div_256 = field.mul %s13, %inv_256 : !packed
  %new_s13 = field.sub %sum, %s13_div_256 : !packed
  memref.store %new_s13, %state[%c13] : !packed_state

  // state[14] = sum - state[14] * inv_sixteen
  %s14 = memref.load %state[%c14] : !packed_state
  %s14_div_16 = field.mul %s14, %inv_sixteen : !packed
  %new_s14 = field.sub %sum, %s14_div_16 : !packed
  memref.store %new_s14, %state[%c14] : !packed_state

  // state[15] = sum - state[15] * inv_2_27
  %s15 = memref.load %state[%c15] : !packed_state
  %s15_div_27 = field.mul %s15, %inv_2_27 : !packed
  %new_s15 = field.sub %sum, %s15_div_27 : !packed
  memref.store %new_s15, %state[%c15] : !packed_state

  return
}

// Internal layer: permutation (add RC to first element, S-box first, internal diffusion)
func.func @packed_permute_state(%state: !packed_state) {
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
  %rc_internal = arith.constant dense<[250494022, 528496384, 1472966118, 977089650, 1885890237, 1094557811, 147492661, 664163003, 398852570, 336233633, 1628648315, 888594966, 586791090]> : tensor<13xi32>
  %rc_internal_mont = field.bitcast %rc_internal : tensor<13xi32> -> tensor<13x!scalar>

  // For each internal constant: add RC and S-box to first element, then apply matrix multiplication
  affine.for %round = 0 to 13 {
    // Get current round constant via tensor.extract
    %rc_scalar = tensor.extract %rc_internal_mont[%round] : tensor<13x!scalar>
    %rc = vector.splat %rc_scalar : !packed

    // Add RC and apply S-box to first element
    %s0 = memref.load %state[%c0] : !packed_state
    %elem0 = func.call @packed_add_rc_and_sbox(%s0, %rc) : (!packed, !packed) -> !packed

    // Compute sum of all elements using affine.for
    // NOTE: this is extremely slow, so we manually add them.
    // %zero = field.constant 0 : !packed
    // %sum = affine.for %i = 0 to 16 iter_args(%acc = %zero) -> (!packed) {
    //   %elem = tensor.extract %t[%i] : tensor<16x!packed>
    //   %new_acc = field.add %acc, %elem : !packed
    //   affine.yield %new_acc : !packed
    // }
    %elem1  = memref.load %state[%c1]  : memref<16x!packed>
    %elem2  = memref.load %state[%c2]  : memref<16x!packed>
    %elem3  = memref.load %state[%c3]  : memref<16x!packed>
    %elem4  = memref.load %state[%c4]  : memref<16x!packed>
    %elem5  = memref.load %state[%c5]  : memref<16x!packed>
    %elem6  = memref.load %state[%c6]  : memref<16x!packed>
    %elem7  = memref.load %state[%c7]  : memref<16x!packed>
    %elem8  = memref.load %state[%c8]  : memref<16x!packed>
    %elem9  = memref.load %state[%c9]  : memref<16x!packed>
    %elem10 = memref.load %state[%c10] : memref<16x!packed>
    %elem11 = memref.load %state[%c11] : memref<16x!packed>
    %elem12 = memref.load %state[%c12] : memref<16x!packed>
    %elem13 = memref.load %state[%c13] : memref<16x!packed>
    %elem14 = memref.load %state[%c14] : memref<16x!packed>
    %elem15 = memref.load %state[%c15] : memref<16x!packed>

    // This structure allows for maximum parallel execution by the CPU.

    // Level 1 (8 parallel additions)
    %sum2_3   = field.add %elem2,  %elem3  : !packed
    %sum4_5   = field.add %elem4,  %elem5  : !packed
    %sum6_7   = field.add %elem6,  %elem7  : !packed
    %sum8_9   = field.add %elem8,  %elem9  : !packed
    %sum10_11 = field.add %elem10, %elem11 : !packed
    %sum12_13 = field.add %elem12, %elem13 : !packed
    %sum14_15 = field.add %elem14, %elem15 : !packed

    // Level 2 (4 parallel additions)
    %sum1_3   = field.add %elem1,   %sum2_3   : !packed
    %sum4_7   = field.add %sum4_5,   %sum6_7   : !packed
    %sum8_11  = field.add %sum8_9,   %sum10_11 : !packed
    %sum12_15 = field.add %sum12_13, %sum14_15 : !packed

    // Level 3 (2 parallel additions)
    %sum1_7   = field.add %sum1_3,   %sum4_7   : !packed
    %sum8_15  = field.add %sum8_11,  %sum12_15 : !packed

    // Level 4 (Partial sum)
    %partial_sum = field.add %sum1_7, %sum8_15 : !packed

    %total_sum = field.add %partial_sum,  %elem0  : !packed
    %new_s0 = field.sub %partial_sum, %elem0 : !packed
    memref.store %new_s0, %state[%c0] : !packed_state

    // Apply internal layer matrix multiplication
    func.call @packed_internal_layer_mat_mul(%state, %total_sum) : (!packed_state, !packed) -> ()
  }
  return
}

// External layer: terminal permutation (4 rounds: add RC, S-box, MDS)
// External layer: terminal permutation (4 rounds: add RC, S-box, MDS)
func.func @packed_permute_state_terminal(%state: !packed_state) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index

  // BABYBEAR_RC16_EXTERNAL_FINAL (4 rounds x 16 constants)
  %rc_external_const = arith.constant dense<[
    [999830298, 304461056, 552699684, 450698925, 667466464, 1736509752, 1327760865, 1153241151, 816675655, 1076172858, 1914832527, 1668723429, 1365579850, 975704528, 1031625628, 1393317533],
    [1554700828, 1023828605, 1610378860, 347744760, 1909572073, 739227895, 428565985, 633143046, 121797685, 94048546, 1369350241, 1250010422, 114268841, 515033604, 49052844, 1962329907],
    [1380892638, 1860017417, 64711457, 9758460, 1681838395, 710850601, 1020228997, 1414164790, 1531515535, 36158805, 713604525, 89935127, 1870801994, 395985906, 1122769045, 1760811055],
    [819787042, 134654834, 1755145179, 18433016, 1701878989, 1782339297, 1483861396, 962480061, 1857590724, 222440409, 63223417, 515206622, 1348364213, 973414686, 1591066884, 705852913]
  ]> : tensor<4x16xi32>

  %rc_external_final = field.bitcast %rc_external_const : tensor<4x16xi32> -> tensor<4x16x!scalar>
  %state_tensor = bufferization.to_tensor %state restrict : memref<16x!packed> to tensor<16x!packed>

  // Loop through 4 rounds of external terminal permutation
  affine.for %round = 0 to 4 {
    affine.for %i = 0 to 16 {
      %s = tensor.extract %state_tensor[%i] : tensor<16x!packed>
      %c_scalar = tensor.extract %rc_external_final[%round, %i] : tensor<4x16x!scalar>
      %c = vector.splat %c_scalar : !packed

      %sbox = func.call @packed_add_rc_and_sbox(%s, %c) : (!packed, !packed) -> !packed
      affine.store %sbox, %state[%i] : !packed_state
    }

    // Apply MDS light permutation (in-place)
    func.call @packed_mds_light_permutation(%state) : (!packed_state) -> ()
  }

  return
}

// External layer: initial permutation (MDS light + terminal permutation)
func.func @packed_permute_state_initial(%state: !packed_state) {
  // First apply MDS light permutation
  func.call @packed_mds_light_permutation(%state) : (!packed_state) -> ()

  // Round constants for 16-width Poseidon2 on BabyBear
  // BABYBEAR_RC16_EXTERNAL_INITIAL (4 rounds x 16 constants)
  %rc_external_const = arith.constant dense<[
    [1582131512, 1899519471, 1641921850, 462688640, 1293997949, 1380417575, 1932416963, 283521298, 1016708647, 35751290, 1270782647, 851730739, 795004022, 929571430, 523703523, 1593957757],
    [895976710, 1742343460, 917700746, 1516725708, 1170237629, 785693164, 613651155, 352999196, 678775274, 1005433272, 1704854670, 1174551920, 508930349, 530338447, 1327158816, 1417652352],
    [1153538870, 583201050, 397833841, 1440603828, 454600685, 174490638, 171758601, 1998476616, 1403697810, 1807736944, 450348306, 1458895865, 787037868, 1063762964, 1987002214, 481645916],
    [1231767638, 1323639433, 238360103, 2012412459, 1024945356, 1108359895, 1284135849, 606928406, 1021455954, 719347978, 659671051, 769588663, 805534062, 592213995, 1752728055, 663410947]
  ]> : tensor<4x16xi32>

  %rc_external_final = field.bitcast %rc_external_const : tensor<4x16xi32> -> tensor<4x16x!scalar>
  %state_tensor = bufferization.to_tensor %state restrict : memref<16x!packed> to tensor<16x!packed>

  // Then apply terminal permutation with initial external constants
  // Loop through 4 rounds of external terminal permutation
  affine.for %round = 0 to 4 {
    affine.for %i = 0 to 16 {
      %s = tensor.extract %state_tensor[%i] : tensor<16x!packed>
      %c_scalar = tensor.extract %rc_external_final[%round, %i] : tensor<4x16x!scalar>
      %c = vector.splat %c_scalar : !packed
      %sbox = func.call @packed_add_rc_and_sbox(%s, %c) : (!packed, !packed) -> !packed
      affine.store %sbox, %state[%i] : !packed_state
    }

    // Apply MDS light permutation (in-place)
    func.call @packed_mds_light_permutation(%state) : (!packed_state) -> ()
  }
  return
}

// Complete Poseidon2 permutation
func.func @packed_poseidon2_permute(%state: !packed_state) {
  func.call @packed_permute_state_initial(%state) : (!packed_state) -> ()
  func.call @packed_permute_state(%state) : (!packed_state) -> ()
  func.call @packed_permute_state_terminal(%state) : (!packed_state) -> ()
  return
}

func.func @packed_permute_10000(%state : !packed_state) attributes { llvm.emit_c_interface } {
  affine.for %i = 0 to 10000 {
    func.call @packed_poseidon2_permute(%state) : (!packed_state) -> ()
  }
  return
}
