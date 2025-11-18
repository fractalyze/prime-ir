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

!PF = !field.pf<9223372036836950017 : i64>

func.func @matvec_gpu(%arg0: memref<1048576x100x!PF>, %arg1: memref<100x!PF>, %arg2: memref<1048576x!PF>) attributes {llvm.emit_c_interface}  {
  linalg.matvec ins(%arg0, %arg1: memref<1048576x100x!PF>, memref<100x!PF>) outs(%arg2: memref<1048576x!PF>)
  return
}
