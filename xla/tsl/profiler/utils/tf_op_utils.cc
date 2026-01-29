/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

#include "xla/tsl/profiler/utils/tf_op_utils.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "re2/re2.h"

namespace tsl {
namespace profiler {
namespace {

const std::string_view kIterator = "Iterator";
const std::string_view kSeparator = "::";
constexpr char kNameScopeSeparator = '/';
constexpr char kOpNameSuffixSeparator = '_';

bool IsInteger(std::string_view str) {
  int64_t unused;
  return absl::SimpleAtoi(str, &unused);
}

// Returns an op type derived from an op name.
std::string_view DeriveOpType(std::string_view full_op_name) {
  // Use the op name without name scopes and suffix as an op type. A full op
  // name consists of name scopes, an op type, and optionally a numeric suffix
  // (e.g., model/layer/MatMul_1).
  std::vector<std::string_view> name_scopes_and_op_name =
      absl::StrSplit(full_op_name, kNameScopeSeparator);
  std::string_view op_name = name_scopes_and_op_name.back();
  std::vector<std::string_view> op_type_and_maybe_suffix =
      absl::StrSplit(op_name, kOpNameSuffixSeparator);
  std::string_view maybe_suffix = op_type_and_maybe_suffix.back();
  std::string_view op_type = op_name;
  if (IsInteger(maybe_suffix)) {
    // NOTE: assuming a numeric suffix is not part of an op type while
    // technically it is allowed.
    op_type = op_name.substr(0, op_name.size() - maybe_suffix.size() - 1);
  }
  return op_type;
}

// TODO(xprof-devs): Include the corresponding Ops on TPU.
std::optional<TfOp> GetMemcpyOp(std::string_view tf_op_fullname) {
  TfOp tf_op;
  tf_op.name = tf_op_fullname;
  if (absl::StartsWithIgnoreCase(tf_op_fullname, "MEMCPYHToD")) {
    tf_op.category = Category::kMemcpyHToD;
    tf_op.type = kMemcpyHToDOp;
    return tf_op;
  }
  if (absl::StartsWithIgnoreCase(tf_op_fullname, "MEMCPYDToH")) {
    tf_op.category = Category::kMemcpyDToH;
    tf_op.type = kMemcpyDToHOp;
    return tf_op;
  }
  if (absl::StartsWithIgnoreCase(tf_op_fullname, "MEMCPYDToD")) {
    tf_op.category = Category::kMemcpyDToD;
    tf_op.type = kMemcpyDToDOp;
    return tf_op;
  } else if (absl::StartsWithIgnoreCase(tf_op_fullname, "MEMCPYHToH")) {
    tf_op.category = Category::kMemcpyHToH;
    tf_op.type = kMemcpyHToHOp;
    return tf_op;
  }
  return std::nullopt;
}

}  // namespace

const std::string_view kUnknownOp = "";  // op types are non-empty strings
const std::string_view kDatasetOp = "Dataset";
const std::string_view kMemcpyHToDOp = "MemcpyHToD";
const std::string_view kMemcpyDToHOp = "MemcpyDToH";
const std::string_view kMemcpyDToDOp = "MemcpyDToD";
const std::string_view kMemcpyHToHOp = "MemcpyHToH";

// Example inputs: "MyOpName", "MyNamespace>MyOpName"
bool IsTfOpName(std::string_view op_name) {
  // TODO(b/177602927): Confirm the naming convention with the TF team.
  static const LazyRE2 kTfOpNameRegEx = {"[A-Za-z0-9.][A-Za-z0-9_.\\/>-]*"};
  return RE2::FullMatch(op_name, *kTfOpNameRegEx);
}

// Example inputs: "MyType", "_MyInternalType"
bool IsTfOpType(std::string_view op_type) {
  static const LazyRE2 kTfOpTypeRegEx = {"[A-Z_][a-zA-Z0-9_]*"};
  return RE2::FullMatch(op_type, *kTfOpTypeRegEx);
}

bool IsJaxOpType(std::string_view op_type) {
  // Jax op type should start with lowercase character or underscore.
  // If it contains '[]', it must end with ']' and whatever chars inside
  // it are considered as a match.
  static const LazyRE2 kJaxOpTypeRegEx = {"[a-z_][a-z0-9_]*(\\[.*\\])?"};
  return RE2::FullMatch(op_type, *kJaxOpTypeRegEx);
}

bool IsJaxOpNameAndType(std::string_view op_name, std::string_view op_type) {
  if (op_name.empty() || !IsJaxOpType(op_type)) return false;
  std::vector<std::string_view> split_result =
      absl::StrSplit(op_name, kNameScopeSeparator);
  return absl::StrContains(split_result.back(), op_type);
}

TfOp ParseTfOpFullname(std::string_view tf_op_fullname) {
  // For op types below, they all have the format "<op_name>:<op_type>", though
  // op_type could be empty.
  TfOp tf_op = {Category::kUnknown, tf_op_fullname, kUnknownOp};
  std::vector<std::string_view> parts =
      absl::StrSplit(tf_op_fullname, absl::MaxSplits(':', 1));

  if (parts.size() != 2) {
    // Two possibilities here: GPU memcpy op or invalid op.
    if (std::optional<TfOp> tfop = GetMemcpyOp(parts[0]); tfop.has_value()) {
      return *tfop;
    }
    return tf_op;
  }

  // Check for a Dataset op.
  if (parts[0] == kIterator) {
    // Dataset Op names (e.g., Iterator::Batch::Map::TFRecord) do not follow the
    // format of TF Op names. But we still want to capture them for
    // input-pipeline analysis.
    tf_op.category = Category::kTfData;
    tf_op.type = kDatasetOp;
    return tf_op;
  }

  // Check for Tensorflow Op.
  if (IsTfOpName(parts[0]) && IsTfOpType(parts[1])) {
    tf_op.category = Category::kTensorFlow;
    tf_op.name = parts[0];
    tf_op.type = parts[1];
    return tf_op;
  }

  // Check for JAX op.
  std::string_view op_type =
      parts[1].empty() ? DeriveOpType(parts[0]) : parts[1];
  if (IsJaxOpType(op_type)) {
    // JAX category introduces op_type with '[]' including unnecessary details
    // to represent a group of ops.
    // We need to striping the brackets and contents inside. Based on our
    // analysis, all the op_type ends with a closing ']' if it contains
    // brackets. It's safe to remove all the characters starting with the
    // position of '['.
    // Example:
    //    "transpose[permutation=(0, 3, 1, 2)]"  =>  "transpose"
    // See: go/xprof-jax-op-type
    tf_op.category = Category::kJax;
    tf_op.name = parts[0];
    tf_op.type = op_type.substr(0, op_type.find('['));
    return tf_op;
  }

  if (parts[1].empty()) {
    tf_op.category = Category::kTensorFlow;
    tf_op.name = parts[0];
    tf_op.type = op_type;
    return tf_op;
  }

  return tf_op;
}

std::vector<std::string_view> ParseTfNameScopes(std::string_view tf_op_name) {
  std::vector<std::string_view> name_scopes =
      absl::StrSplit(tf_op_name, kNameScopeSeparator);
  // The last element is an op name not TF name scope.
  if (!name_scopes.empty()) name_scopes.pop_back();
  return name_scopes;
}

std::vector<std::string_view> ParseTfNameScopes(const TfOp& tf_op) {
  return ParseTfNameScopes(tf_op.name);
}

std::string TfOpEventName(const TfOp& tf_op) {
  std::string event_name;
  if (tf_op.category == Category::kUnknown) {
    // Some TraceMe names contain trailing whitespace, remove it.
    event_name = std::string(absl::StripTrailingAsciiWhitespace(tf_op.name));
  } else if (tf_op.category == Category::kTfData) {
    event_name = DatasetOpEventName(tf_op.name);
  } else {
    event_name = std::string(tf_op.type);
  }
  return event_name;
}

std::string TfOpEventName(std::string_view tf_op_fullname) {
  return TfOpEventName(ParseTfOpFullname(tf_op_fullname));
}

std::string DatasetOpEventName(std::string_view full_name) {
  std::vector<std::string_view> split_result =
      absl::StrSplit(full_name, kSeparator);
  return absl::StrCat(kIterator, kSeparator, split_result.back());
}

std::string IteratorName(std::string_view full_name) {
  std::vector<std::string_view> split_result =
      absl::StrSplit(full_name, kSeparator);
  return std::string(split_result.back());
}

std::vector<std::string_view> ParseTensorShapes(
    std::string_view tensor_shapes) {
  absl::ConsumePrefix(&tensor_shapes, "(");
  absl::ConsumeSuffix(&tensor_shapes, ")");
  return absl::StrSplit(tensor_shapes, ';');
}

}  // namespace profiler
}  // namespace tsl
