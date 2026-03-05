/* Copyright 2024 The OpenXLA Authors. All Rights Reserved.
Copyright 2025 The ZKX Authors.

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

#ifndef ZKX_PYTHON_AGGREGATE_PROFILE_H_
#define ZKX_PYTHON_AGGREGATE_PROFILE_H_

#include "absl/types/span.h"

#include "xla/tsl/profiler/protobuf/profiled_instructions.pb.h"

namespace zkx {

// Aggregates and gets given percentile of multiple ProfiledInstructionsProtos
// into one ProfiledInstructionsProto.
void AggregateProfiledInstructionsProto(
    absl::Span<const tensorflow::profiler::ProfiledInstructionsProto> profiles,
    int percentile,
    tensorflow::profiler::ProfiledInstructionsProto *result_profile);

}  // namespace zkx

#endif  // ZKX_PYTHON_AGGREGATE_PROFILE_H_
