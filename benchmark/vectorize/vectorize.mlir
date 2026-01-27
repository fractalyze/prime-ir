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

// Extension field operations benchmark
// Note: Extension fields currently cannot be vectorized due to a bug in field-to-llvm

!pf = !field.pf<2013265921 : i32, true>

// Extension fields for benchmarking
// Quadratic extension: degree 2, non-residue 11
!qf = !field.ef<2x!pf, 11:i32>
// Quartic extension: degree 4, non-residue 11
!qrf = !field.ef<4x!pf, 11:i32>

// ==============================================================================
// Extension Field Benchmarks (scalar only)
// ==============================================================================

// Quadratic extension field square (262144 elements, 1000 iterations)
// Each element has 2 coefficients, so memory footprint is similar to 524288 prime field elements
func.func @ext2_square_buffer(%buffer: memref<262144x!qf>) attributes { llvm.emit_c_interface } {
  affine.for %iter = 0 to 1000 {
    affine.for %i = 0 to 262144 {
      %val = affine.load %buffer[%i] : memref<262144x!qf>
      %sq = field.square %val : !qf
      affine.store %sq, %buffer[%i] : memref<262144x!qf>
    }
  }
  return
}

// Quadratic extension field multiply
func.func @ext2_mul_buffers(%a: memref<262144x!qf>, %b: memref<262144x!qf>, %c: memref<262144x!qf>) attributes { llvm.emit_c_interface } {
  affine.for %iter = 0 to 1000 {
    affine.for %i = 0 to 262144 {
      %va = affine.load %a[%i] : memref<262144x!qf>
      %vb = affine.load %b[%i] : memref<262144x!qf>
      %prod = field.mul %va, %vb : !qf
      affine.store %prod, %c[%i] : memref<262144x!qf>
    }
  }
  return
}

// Quartic extension field square (65536 elements, 100 iterations)
// Each element has 4 coefficients, so memory footprint is similar to 262144 prime field elements
// This operation generates significant code (Toom-Cook algorithm)
func.func @ext4_square_buffer(%buffer: memref<65536x!qrf>) attributes { llvm.emit_c_interface } {
  affine.for %iter = 0 to 100 {
    affine.for %i = 0 to 65536 {
      %val = affine.load %buffer[%i] : memref<65536x!qrf>
      %sq = field.square %val : !qrf
      affine.store %sq, %buffer[%i] : memref<65536x!qrf>
    }
  }
  return
}

// Quartic extension field multiply
// This is the most complex operation - candidate for intrinsic function mode
func.func @ext4_mul_buffers(%a: memref<65536x!qrf>, %b: memref<65536x!qrf>, %c: memref<65536x!qrf>) attributes { llvm.emit_c_interface } {
  affine.for %iter = 0 to 100 {
    affine.for %i = 0 to 65536 {
      %va = affine.load %a[%i] : memref<65536x!qrf>
      %vb = affine.load %b[%i] : memref<65536x!qrf>
      %prod = field.mul %va, %vb : !qrf
      affine.store %prod, %c[%i] : memref<65536x!qrf>
    }
  }
  return
}
