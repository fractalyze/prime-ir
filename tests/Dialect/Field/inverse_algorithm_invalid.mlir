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

// An unrecognized inverse-algorithm value fails the pass rather than silently
// falling back to a default.
// RUN: not prime-ir-opt -field-to-mod-arith="inverse-algorithm=fermt" %s 2>&1 | FileCheck %s --check-prefix=BADINV
// BADINV: invalid inverse-algorithm option: 'fermt'

!PF = !field.pf<97:i32, true>

func.func @inv(%arg0: !PF) -> !PF {
  %inv = field.inverse %arg0 : !PF
  return %inv : !PF
}
