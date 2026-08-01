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

// Round-trip and rejection for the binary-field flat-basis syntax: `flat` is
// defined at levels 4/5 only (moduli in TowerFlatBasis.h); 3/7 keep their
// named bases and every other level has no flat basis.

// RUN: prime-ir-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// CHECK-LABEL: @roundtrip
// CHECK-SAME: (%{{.*}}: !field.bf<4, flat>, %{{.*}}: !field.bf<5, flat>)
func.func @roundtrip(%a: !field.bf<4, flat>, %b: !field.bf<5, flat>) {
  return
}

// -----

// expected-error @+1 {{the flat basis is defined at tower level 4 or 5}}
func.func @flat_level6(%a: !field.bf<6, flat>) {
  return
}
