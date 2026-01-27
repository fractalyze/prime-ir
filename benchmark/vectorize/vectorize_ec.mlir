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

// Elliptic curve operations benchmark
// Tests EC add, double, and scalar_mul operations on BN254 G1 curve

// BN254 G1 curve definitions
!PFm = !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256, true>
!SFm = !field.pf<21888242871839275222246405745257275088548364400416034343698204186575808495617:i256, true>

#curve = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !PFm
!affine = !elliptic_curve.affine<#curve>
!jacobian = !elliptic_curve.jacobian<#curve>

// Buffer sizes chosen to balance memory footprint and benchmark time
// Jacobian point = 3 x 256 bits = 96 bytes per point
// 16384 points = 1.5 MB

// EC point doubling benchmark (16384 points, 100 iterations)
// Each iteration doubles all points in the buffer
func.func @ec_double_buffer(%points: memref<16384x!jacobian>) attributes { llvm.emit_c_interface } {
  affine.for %iter = 0 to 100 {
    affine.for %i = 0 to 16384 {
      %p = affine.load %points[%i] : memref<16384x!jacobian>
      %doubled = elliptic_curve.double %p : !jacobian -> !jacobian
      affine.store %doubled, %points[%i] : memref<16384x!jacobian>
    }
  }
  return
}

// EC point addition benchmark (Jacobian + Affine -> Jacobian)
// Mixed addition is more efficient than Jacobian + Jacobian
func.func @ec_add_buffer(%jac: memref<16384x!jacobian>, %aff: memref<16384x!affine>,
                         %out: memref<16384x!jacobian>) attributes { llvm.emit_c_interface } {
  affine.for %iter = 0 to 100 {
    affine.for %i = 0 to 16384 {
      %pj = affine.load %jac[%i] : memref<16384x!jacobian>
      %pa = affine.load %aff[%i] : memref<16384x!affine>
      %sum = elliptic_curve.add %pj, %pa : !jacobian, !affine -> !jacobian
      affine.store %sum, %out[%i] : memref<16384x!jacobian>
    }
  }
  return
}

// EC scalar multiplication benchmark (this generates significant code)
// This is the primary candidate for intrinsic function mode
// 1024 points, 10 iterations (scalar_mul is very expensive)
func.func @ec_scalar_mul_buffer(%scalars: memref<1024x!SFm>, %points: memref<1024x!affine>,
                                %out: memref<1024x!jacobian>) attributes { llvm.emit_c_interface } {
  affine.for %iter = 0 to 10 {
    affine.for %i = 0 to 1024 {
      %s = affine.load %scalars[%i] : memref<1024x!SFm>
      %p = affine.load %points[%i] : memref<1024x!affine>
      %result = elliptic_curve.scalar_mul %s, %p : !SFm, !affine -> !jacobian
      affine.store %result, %out[%i] : memref<1024x!jacobian>
    }
  }
  return
}
