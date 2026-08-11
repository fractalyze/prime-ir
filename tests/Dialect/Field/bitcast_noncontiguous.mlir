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

// The descriptor rebuild in the LLVM lowering carries the input offset across
// and derives sizes and strides from the result shape, which only describes the
// same bytes when the source is packed. The verifier is where a gapped source
// gets refused: a diagnostic raised from a conversion pattern is rolled back
// with the failing pattern and never reaches the user.

// RUN: prime-ir-opt --split-input-file --verify-diagnostics %s

!PF = !field.pf<7:i256>
!EF2 = !field.ef<2x!PF, 6:i256>

// Stride 2 on the innermost dimension: every other element belongs to someone
// else, so the 12 elements are not 12 consecutive ones.
func.func @gapped_source(%m: memref<12x!PF, strided<[2]>>) {
  // expected-error @+1 {{requires a contiguous input buffer}}
  %r = field.bitcast %m : memref<12x!PF, strided<[2]>> -> memref<6x!EF2>
  return
}

// -----

!PF = !field.pf<7:i256>
!EF2 = !field.ef<2x!PF, 6:i256>

// A dynamic stride cannot be shown to be either, and bufferizing function
// boundaries produces exactly this shape, so it has to keep working.
func.func @dynamic_stride_is_trusted(
    %m: memref<12x!PF, strided<[?], offset: ?>>) {
  %r = field.bitcast %m : memref<12x!PF, strided<[?], offset: ?>> -> memref<6x!EF2>
  return
}
