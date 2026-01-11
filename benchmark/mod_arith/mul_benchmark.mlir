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

!mod = !mod_arith.int<21888242871839275222246405745257275088548364400416034343698204186575808495617 : i256>
!modm = !mod_arith.int<21888242871839275222246405745257275088548364400416034343698204186575808495617 : i256, true>

func.func @mul(%arg0: memref<1048576x!mod>, %arg1: memref<1048576x!mod>, %out: memref<!mod>) attributes { llvm.emit_c_interface } {
  linalg.dot ins(%arg0, %arg1: memref<1048576x!mod>, memref<1048576x!mod>) outs(%out: memref<!mod>)
  return
}

func.func @mont_mul(%arg0 : memref<1048576x!modm>, %arg1 : memref<1048576x!modm>, %out: memref<!modm>) attributes { llvm.emit_c_interface } {
  linalg.dot ins(%arg0, %arg1: memref<1048576x!modm>, memref<1048576x!modm>) outs(%out: memref<!modm>)
  return
}
