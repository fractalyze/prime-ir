/* Copyright 2017 The OpenXLA Authors.
Copyright 2026 The ZKX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef ZKX_SERVICE_CPU_DYNAMIC_UPDATE_SLICE_UTIL_H_
#define ZKX_SERVICE_CPU_DYNAMIC_UPDATE_SLICE_UTIL_H_

#include "zkx/hlo/ir/hlo_instruction.h"

namespace zkx::llvm_ir {

// Determines whether the given instruction might be implemented as an
// in-place dynamic-update-slice after we have a buffer assignment.
//
// If this returns false, then CanUpdateDynamicSliceInPlace and
// CanEmitFusedDynamicUpdateSliceInPlace will also return false.
//
// This is useful if you want to check whether an instruction might be an
// in-place DUS during an HLO pass, at which point you don't have a buffer
// assignment.
//
// Note that simplifications to the HLO graph might change this function from
// returning false to returning true.  Specifically, simplifying the contents of
// fusion nodes might cause a false->true transition.  In general this isn't a
// problem by the time you're calling this function, but beware.
bool MayBeImplementedAsInPlaceDynamicUpdateSlice(const HloInstruction* instr);

}  // namespace zkx::llvm_ir

#endif  // ZKX_SERVICE_CPU_DYNAMIC_UPDATE_SLICE_UTIL_H_
