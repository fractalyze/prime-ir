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

// Binary field benchmark - BF8 x86 GFNI specialization
//
// Packed 16x BF8 multiplication using GFNI (128-bit SSE).
// Uses memref interface for passing SIMD vectors through C FFI.

!BF8 = !field.bf<3>  // GF(2⁸) tower field

// Packed 16x BF8 multiplication using GFNI
// C interface: void bf8x16_mul_gfni(MemRefDescriptor* a, MemRefDescriptor* b, MemRefDescriptor* c)
func.func @bf8x16_mul_gfni(%a_ptr: memref<16xi8>, %b_ptr: memref<16xi8>, %c_ptr: memref<16xi8>)
    attributes { llvm.emit_c_interface } {
  // Load vectors
  %c0 = arith.constant 0 : index
  %a_i8 = vector.load %a_ptr[%c0] : memref<16xi8>, vector<16xi8>
  %b_i8 = vector.load %b_ptr[%c0] : memref<16xi8>, vector<16xi8>

  // Cast to bf<3> for field.mul
  %a = builtin.unrealized_conversion_cast %a_i8 : vector<16xi8> to vector<16x!BF8>
  %b = builtin.unrealized_conversion_cast %b_i8 : vector<16xi8> to vector<16x!BF8>

  // Multiply (will be specialized to GFNI)
  %c = field.mul %a, %b : vector<16x!BF8>

  // Cast back and store
  %c_i8 = builtin.unrealized_conversion_cast %c : vector<16x!BF8> to vector<16xi8>
  vector.store %c_i8, %c_ptr[%c0] : memref<16xi8>, vector<16xi8>
  return
}
