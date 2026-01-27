/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
#ifndef XLA_TSL_PROFILER_LIB_TRACEME_ENCODE_H_
#define XLA_TSL_PROFILER_LIB_TRACEME_ENCODE_H_

#include <string.h>

#include <initializer_list>
#include <string>

#include "absl/base/attributes.h"
#include "absl/base/optimization.h"
#include "absl/log/absl_check.h"
#include "absl/log/check.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"

namespace tsl::profiler {

// An argument passed to TraceMeEncode.
struct TraceMeArg {
  // String conversions of value types are supported via AlphaNum. We keep a
  // reference to the AlphaNum's internal buffer here, so it must remain valid
  // for the lifetime of this object. We cannot store it by value because it is
  // not safe to construct an AlphaNum as a member of a class, particularly when
  // AbslStringify is being used (it may reference default arguments that are on
  // the caller's stack, if we constructed it here those default arguments would
  // be destroyed before they are used).
  TraceMeArg(std::string_view k,
             const absl::AlphaNum& v ABSL_ATTRIBUTE_LIFETIME_BOUND)
      : key(k), value(v.Piece()) {}

  TraceMeArg(const TraceMeArg&) = delete;
  void operator=(const TraceMeArg&) = delete;

  std::string_view key;
  std::string_view value;
};

namespace traceme_internal {

// Copies the contents of str to the address pointed by out.
// Returns the address after the copy.
// REQUIRED: The address range [out, out + str.size()] must have been allocated.
ABSL_ATTRIBUTE_ALWAYS_INLINE inline char* Append(char* out,
                                                 std::string_view str) {
  ABSL_DCHECK(!absl::StrContains(str, '#'))
      << "'#' is not a valid character in TraceMeEncode";
  const size_t str_size = str.size();
  if (ABSL_PREDICT_TRUE(str_size > 0)) {
    memcpy(out, str.data(), str_size);
    out += str_size;
  }
  return out;
}

// Appends args encoded as TraceMe metadata to name.
ABSL_ATTRIBUTE_ALWAYS_INLINE inline std::string AppendArgs(
    std::string name, std::initializer_list<TraceMeArg> args) {
  if (ABSL_PREDICT_TRUE(args.size() > 0)) {
    const auto old_size = name.size();
    auto new_size = old_size + args.size() * 2 + 1;
    for (const auto& arg : args) {
      new_size += arg.key.size() + arg.value.size();
    }
    name.resize(new_size);
    char* const begin = &name[0];
    char* out = begin + old_size;
    *out++ = '#';
    for (const auto& arg : args) {
      out = Append(out, arg.key);
      *out++ = '=';
      out = Append(out, arg.value);
      *out++ = ',';
    }
    *(out - 1) = '#';
    ABSL_DCHECK_EQ(out, begin + new_size);
  }
  return name;
}

// Appends new_metadata to the metadata part of name.
ABSL_ATTRIBUTE_ALWAYS_INLINE inline void AppendMetadata(
    std::string* name, std::string_view new_metadata) {
  if (!ABSL_PREDICT_FALSE(new_metadata.empty())) {
    if (!name->empty() && name->back() == '#') {  // name already has metadata
      name->back() = ',';
      if (ABSL_PREDICT_TRUE(new_metadata.front() == '#')) {
        new_metadata.remove_prefix(1);
      }
    }
    name->append(new_metadata.data(), new_metadata.size());
  }
}

}  // namespace traceme_internal

// Encodes an event name and arguments into TraceMe metadata.
// Use within a lambda to avoid expensive operations when tracing is disabled.
// Example Usage:
//   TraceMe trace_me([value1]() {
//     return TraceMeEncode("my_trace", {{"key1", value1}, {"key2", 42}});
//   });
ABSL_ATTRIBUTE_ALWAYS_INLINE inline std::string TraceMeEncode(
    std::string name, std::initializer_list<TraceMeArg> args) {
  return traceme_internal::AppendArgs(std::move(name), args);
}
ABSL_ATTRIBUTE_ALWAYS_INLINE inline std::string TraceMeEncode(
    std::string_view name, std::initializer_list<TraceMeArg> args) {
  return traceme_internal::AppendArgs(std::string(name), args);
}
ABSL_ATTRIBUTE_ALWAYS_INLINE inline std::string TraceMeEncode(
    const char* name, std::initializer_list<TraceMeArg> args) {
  return traceme_internal::AppendArgs(std::string(name), args);
}

// Encodes arguments into TraceMe metadata.
// Use within a lambda to avoid expensive operations when tracing is disabled.
// Example Usage:
//   TraceMe trace_me("my_trace");
//   ...
//   trace_me.AppendMetadata([value1]() {
//     return TraceMeEncode({{"key1", value1}, {"key2", 42}});
//   });
ABSL_ATTRIBUTE_ALWAYS_INLINE inline std::string TraceMeEncode(
    std::initializer_list<TraceMeArg> args) {
  return traceme_internal::AppendArgs(std::string(), args);
}

// Concatenates op_name and op_type.
ABSL_ATTRIBUTE_ALWAYS_INLINE inline std::string TraceMeOp(
    std::string_view op_name, std::string_view op_type) {
  return absl::StrCat(op_name, ":", op_type);
}

ABSL_ATTRIBUTE_ALWAYS_INLINE inline std::string TraceMeOp(const char* op_name,
                                                          const char* op_type) {
  return absl::StrCat(op_name, ":", op_type);
}

ABSL_ATTRIBUTE_ALWAYS_INLINE inline std::string TraceMeOp(
    std::string&& op_name, std::string_view op_type) {
  absl::StrAppend(&op_name, ":", op_type);
  return op_name;
}

// Concatenates op_name and op_type.
ABSL_ATTRIBUTE_ALWAYS_INLINE inline std::string TraceMeOpOverride(
    std::string_view op_name, std::string_view op_type) {
  return absl::StrCat("#zkx_op=", op_name, ":", op_type, "#");
}

ABSL_ATTRIBUTE_ALWAYS_INLINE inline std::string TraceMeOpOverride(
    const char* op_name, const char* op_type) {
  return absl::StrCat("#zkx_op=", op_name, ":", op_type, "#");
}

}  // namespace tsl::profiler

#endif  // XLA_TSL_PROFILER_LIB_TRACEME_ENCODE_H_
