/* Copyright 2018 The OpenXLA Authors.
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

#ifndef ZKX_SERVICE_SOURCE_MAP_UTIL_H_
#define ZKX_SERVICE_SOURCE_MAP_UTIL_H_

#include "absl/status/status.h"
#include "absl/strings/str_format.h"

#include "zkx/service/executable.h"

namespace zkx {
namespace source_map_util {

// Creates an INVALID_ARGUMENT status with the given format string.
template <typename... Args>
absl::Status InvalidParameterArgument(const OpMetadata& op_metadata,
                                      const absl::FormatSpec<Args...>& format,
                                      const Args&... args) {
  std::string message = absl::StrFormat(format, args...);
  if (!op_metadata.source_file().empty()) {
    absl::StrAppendFormat(&message, " (%s:%d)", op_metadata.source_file(),
                          op_metadata.source_line());
  }
  return absl::InvalidArgumentError(message);
}

// Creates an INVALID_ARGUMENT status with the given format string.
//
// Also, attempts to extract the OpMetadata for parameter_number on executable
// and append it to the status message for source mapping to user code.
//
// executable may be nullptr, but parameter_number should not be out of bounds
// or a CHECK-failure may occur.
template <typename... Args>
absl::Status InvalidParameterArgument(Executable* executable,
                                      int parameter_number,
                                      const absl::FormatSpec<Args...>& format,
                                      const Args&... args) {
  if (executable != nullptr && executable->has_module()) {
    const HloModule& module = executable->module();
    const HloComputation& computation = *module.entry_computation();
    HloInstruction* param = computation.parameter_instruction(parameter_number);
    const OpMetadata& metadata = param->metadata();
    return InvalidParameterArgument(metadata, format, args...);
  }
  return absl::InvalidArgumentError(absl::StrFormat(format, args...));
}

}  // namespace source_map_util
}  // namespace zkx

#endif  // ZKX_SERVICE_SOURCE_MAP_UTIL_H_
