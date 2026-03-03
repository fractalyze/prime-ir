/* Copyright 2024 The OpenXLA Authors.
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

#ifndef ZKX_HLO_IR_HLO_INSTRUCTION_UTILS_H_
#define ZKX_HLO_IR_HLO_INSTRUCTION_UTILS_H_

#include "zkx/hlo/ir/hlo_instruction.h"

namespace zkx {
namespace hlo_instruction_utils {

// Returns true if the given HLO is a slice operation which has a unit stride in
// all dimensions.
bool IsUnstridedSlice(const HloInstruction* hlo);

}  // namespace hlo_instruction_utils
}  // namespace zkx

#endif  // ZKX_HLO_IR_HLO_INSTRUCTION_UTILS_H_
