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

// The LLVM lowering rebuilds the descriptor from the result shape and carries
// the input offset across, dividing it by the coordinates per point. Both of
// the assumptions that makes -- a packed source and a static result -- have to
// be refused here: a diagnostic raised from a conversion pattern is rolled back
// with the failing pattern and never reaches the user.

// RUN: prime-ir-opt --split-input-file --verify-diagnostics %s

!PF = !field.pf<7:i256>
#curve = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !PF
!jacobian = !elliptic_curve.jacobian<#curve>

func.func @gapped_source(%m: memref<12x!PF, strided<[2]>>) {
  // expected-error @+1 {{requires a contiguous input buffer}}
  %p = elliptic_curve.bitcast %m
      : memref<12x!PF, strided<[2]>> -> memref<4x!jacobian>
  return
}

// -----

!PF = !field.pf<7:i256>
#curve = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !PF
!jacobian = !elliptic_curve.jacobian<#curve>

// A jacobian point spans three field elements, so an offset of 4 names a
// position inside a point rather than the start of one.
func.func @offset_lands_inside_a_point(
    %m: memref<12x!PF, strided<[1], offset: 4>>) {
  // expected-error @+1 {{input offset does not start on a point boundary}}
  %p = elliptic_curve.bitcast %m
      : memref<12x!PF, strided<[1], offset: 4>> -> memref<4x!jacobian>
  return
}

// -----

!PF = !field.pf<7:i256>
#curve = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !PF
!jacobian = !elliptic_curve.jacobian<#curve>

// The rebuild stamps the result extents in as constants, so a dynamic result
// would become a descriptor full of the dynamic sentinel.
func.func @dynamic_result_extent(%m: memref<12x!PF>) {
  // expected-error @+1 {{cannot rebuild a descriptor for a dynamically shaped result}}
  %p = elliptic_curve.bitcast %m : memref<12x!PF> -> memref<?x!jacobian>
  return
}
