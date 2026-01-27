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

// Prime field operations benchmark to test affine-super-vectorize
// This file contains only prime field ops which can be safely vectorized

!pf = !field.pf<2013265921 : i32, true>

// Square all elements in a buffer (1M elements, 1000 iterations)
func.func @square_buffer(%buffer: memref<1048576x!pf>) attributes { llvm.emit_c_interface } {
  affine.for %iter = 0 to 1000 {
    affine.for %i = 0 to 1048576 {
      %val = affine.load %buffer[%i] : memref<1048576x!pf>
      %sq = field.square %val : !pf
      affine.store %sq, %buffer[%i] : memref<1048576x!pf>
    }
  }
  return
}

// Add two buffers element-wise
func.func @add_buffers(%a: memref<1048576x!pf>, %b: memref<1048576x!pf>, %c: memref<1048576x!pf>) attributes { llvm.emit_c_interface } {
  affine.for %iter = 0 to 1000 {
    affine.for %i = 0 to 1048576 {
      %va = affine.load %a[%i] : memref<1048576x!pf>
      %vb = affine.load %b[%i] : memref<1048576x!pf>
      %sum = field.add %va, %vb : !pf
      affine.store %sum, %c[%i] : memref<1048576x!pf>
    }
  }
  return
}

// Multiply-accumulate: c[i] = a[i] * b[i] + c[i]
func.func @mul_add_buffers(%a: memref<1048576x!pf>, %b: memref<1048576x!pf>, %c: memref<1048576x!pf>) attributes { llvm.emit_c_interface } {
  affine.for %iter = 0 to 1000 {
    affine.for %i = 0 to 1048576 {
      %va = affine.load %a[%i] : memref<1048576x!pf>
      %vb = affine.load %b[%i] : memref<1048576x!pf>
      %vc = affine.load %c[%i] : memref<1048576x!pf>
      %prod = field.mul %va, %vb : !pf
      %sum = field.add %prod, %vc : !pf
      affine.store %sum, %c[%i] : memref<1048576x!pf>
    }
  }
  return
}
