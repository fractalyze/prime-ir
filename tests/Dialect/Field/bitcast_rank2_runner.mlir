// Copyright 2026 The PrimeIR Authors.
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

// Tensor reinterpret bitcast on a rank-2 operand. The verifier admits it — it
// compares total bitwidth via getNumElements(), which is rank-agnostic — and a
// row-major EF2 tensor is byte-identical to the PF tensor with the trailing dim
// doubled, so the reinterpret is meaningful at any rank.
//
// The descriptor rebuild in ExtFieldToLLVM only runs when the shapes differ
// (equal shapes forward the input descriptor untouched), so rank > 1 combined
// with a shape change is the one path that exercises the stride computation.
//
// Each bitcast sits in its own function so the canonicalizer cannot fold
// bitcast(bitcast(x)) -> x and skip the lowering entirely.

// RUN: prime-ir-opt %s --field-to-llvm='bufferize-function-boundaries=true' \
// RUN:   | mlir-runner -e test_bitcast_rank2 -entry-point-result=void \
// RUN:      --shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s < %t

!PF = !field.pf<7:i32>
!EF2 = !field.ef<2x!PF, 6:i32>

func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

// EF2[2][3] -> PF[2][6]: each EF2 is two PF coefficients, so only the trailing
// dimension changes. Rows must stay row-major, i.e. strides [6, 1].
func.func @bitcast_ef2_to_pf_rank2(%src: tensor<2x3x!EF2>) -> tensor<2x6x!PF> {
  %pf = field.bitcast %src : tensor<2x3x!EF2> -> tensor<2x6x!PF>
  return %pf : tensor<2x6x!PF>
}

// PF[2][6] -> EF2[2][3], the same reinterpret in the widening direction.
func.func @bitcast_pf_to_ef2_rank2(%src: tensor<2x6x!PF>) -> tensor<2x3x!EF2> {
  %ef = field.bitcast %src : tensor<2x6x!PF> -> tensor<2x3x!EF2>
  return %ef : tensor<2x3x!EF2>
}

func.func @test_bitcast_rank2() {
  // Rows are reversals of each other, so a row that reads from the wrong base
  // offset prints a visibly different sequence rather than a plausible one.
  %ef = field.constant dense<[[[1, 2], [3, 4], [5, 6]],
                              [[6, 5], [4, 3], [2, 1]]]> : tensor<2x3x!EF2>

  // Narrowing: EF2[2][3] -> PF[2][6].
  %pf = func.call @bitcast_ef2_to_pf_rank2(%ef)
      : (tensor<2x3x!EF2>) -> tensor<2x6x!PF>
  %i_a = field.bitcast %pf : tensor<2x6x!PF> -> tensor<2x6xi32>
  %m_a = bufferization.to_buffer %i_a : tensor<2x6xi32> to memref<2x6xi32>
  %u_a = memref.cast %m_a : memref<2x6xi32> to memref<*xi32>
  func.call @printMemrefI32(%u_a) : (memref<*xi32>) -> ()
  // CHECK: {{\[}}[1, 2, 3, 4, 5, 6]
  // CHECK: [6, 5, 4, 3, 2, 1]

  // Widening: PF[2][6] -> EF2[2][3], back to the original bytes.
  %pf_src = field.constant dense<[[1, 2, 3, 4, 5, 6],
                                  [6, 5, 4, 3, 2, 1]]> : tensor<2x6x!PF>
  %ef_b = func.call @bitcast_pf_to_ef2_rank2(%pf_src)
      : (tensor<2x6x!PF>) -> tensor<2x3x!EF2>
  %i_b = field.bitcast %ef_b : tensor<2x3x!EF2> -> tensor<2x6xi32>
  %m_b = bufferization.to_buffer %i_b : tensor<2x6xi32> to memref<2x6xi32>
  %u_b = memref.cast %m_b : memref<2x6xi32> to memref<*xi32>
  func.call @printMemrefI32(%u_b) : (memref<*xi32>) -> ()
  // CHECK: {{\[}}[1, 2, 3, 4, 5, 6]
  // CHECK: [6, 5, 4, 3, 2, 1]

  return
}
