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

// `field.powui`'s `unroll` attribute defaults to `true`, so the assembly has
// exactly two spellings: absent (or explicit `true`, which the printer elides)
// and `unroll = false`. Piping through prime-ir-opt twice pins that the elided
// form re-parses to the same op rather than drifting on each round trip.
// The lowering these spellings select is covered by
// bit_serial_constant_unrolling.mlir.

// RUN: prime-ir-opt %s | prime-ir-opt | FileCheck %s -enable-var-scope

!GL = !field.pf<18446744069414584321:i64>

// CHECK-LABEL: @powui_unroll_absent
// CHECK-SAME: (%[[BASE:.*]]: [[T:.*]], %[[EXP:.*]]: i32)
// CHECK-NEXT: %[[RES:.*]] = field.powui %[[BASE]], %[[EXP]] : [[T]], i32
// CHECK-NEXT: return %[[RES]] : [[T]]
func.func @powui_unroll_absent(%base: !GL, %exp: i32) -> !GL {
  %res = field.powui %base, %exp : !GL, i32
  return %res : !GL
}

// An explicit `true` is the default, so it prints elided — byte for byte the
// same line as the absent case above. That is what keeps the attribute's
// arrival off every existing .mlir file and every FileCheck pattern written
// before it.
// CHECK-LABEL: @powui_unroll_true
// CHECK-SAME: (%[[BASE:.*]]: [[T:.*]], %[[EXP:.*]]: i32)
// CHECK-NEXT: %[[RES:.*]] = field.powui %[[BASE]], %[[EXP]] : [[T]], i32
// CHECK-NEXT: return %[[RES]] : [[T]]
func.func @powui_unroll_true(%base: !GL, %exp: i32) -> !GL {
  %res = field.powui %base, %exp {unroll = true} : !GL, i32
  return %res : !GL
}

// `false` is the only value that has to survive in the text.
// CHECK-LABEL: @powui_unroll_false
// CHECK-SAME: (%[[BASE:.*]]: [[T:.*]], %[[EXP:.*]]: i32)
// CHECK-NEXT: %[[RES:.*]] = field.powui %[[BASE]], %[[EXP]] {unroll = false} : [[T]], i32
// CHECK-NEXT: return %[[RES]] : [[T]]
func.func @powui_unroll_false(%base: !GL, %exp: i32) -> !GL {
  %res = field.powui %base, %exp {unroll = false} : !GL, i32
  return %res : !GL
}
