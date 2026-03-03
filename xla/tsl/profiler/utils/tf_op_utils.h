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

#ifndef XLA_TSL_PROFILER_UTILS_TF_OP_UTILS_H_
#define XLA_TSL_PROFILER_UTILS_TF_OP_UTILS_H_

#include <string>
#include <string_view>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/strings/match.h"

namespace tsl {
namespace profiler {

// Special op types.
ABSL_CONST_INIT extern const std::string_view kUnknownOp;
ABSL_CONST_INIT extern const std::string_view kDatasetOp;
ABSL_CONST_INIT extern const std::string_view kMemcpyHToDOp;
ABSL_CONST_INIT extern const std::string_view kMemcpyDToHOp;
ABSL_CONST_INIT extern const std::string_view kMemcpyDToDOp;
ABSL_CONST_INIT extern const std::string_view kMemcpyHToHOp;

enum class Category {
  kUnknown,
  kTensorFlow,
  kJax,
  kTfData,
  kMemcpyHToD,
  kMemcpyDToH,
  kMemcpyDToD,
  kMemcpyHToH,
};

// Breaks a TensorFlow op fullname into name and type.
struct TfOp {
  Category category = Category::kUnknown;
  std::string_view name;
  std::string_view type;
};
TfOp ParseTfOpFullname(std::string_view tf_op_fullname);

// Returns a vector of TF name scopes extracted from a TF op name.
std::vector<std::string_view> ParseTfNameScopes(std::string_view tf_op_name);
std::vector<std::string_view> ParseTfNameScopes(const TfOp& tf_op);

// Trace event name for TF ops is the op type so they have the same color in
// trace viewer.
std::string TfOpEventName(const TfOp& tf_op);
std::string TfOpEventName(std::string_view tf_op_fullname);

// Trace event name for dataset ops.
std::string DatasetOpEventName(std::string_view full_name);

// Returns the iterator name without prefix and parent iterator names.
std::string IteratorName(std::string_view full_name);

// Returns true if the given name is a TensorFlow Dataset Op.
inline bool IsDatasetOp(std::string_view tf_op_type) {
  return tf_op_type == kDatasetOp;
}
inline bool IsDatasetOp(const TfOp& tf_op) {
  return tf_op.category == Category::kTfData;
}

// Returns true if the given name is a TensorFlow Infeed Enqueue Op.
// See: tensorflow/tsl/tpu/kernels/infeed_ops.h
inline bool IsInfeedEnqueueOp(std::string_view tf_op_type) {
  return absl::StartsWith(tf_op_type, "InfeedEnqueue");
}
inline bool IsInfeedEnqueueOp(const TfOp& tf_op) {
  return tf_op.category == Category::kTensorFlow &&
         IsInfeedEnqueueOp(tf_op.type);
}

// Returns true if the given op has ZkxSendToHost/ZkxRecvFromHost in fullname.
inline bool IsOutsideCompilationOp(std::string_view tf_op_fullname) {
  if (absl::EndsWith(tf_op_fullname, ":ZkxSendToHost")) return true;
  if (absl::EndsWith(tf_op_fullname, ":ZkxRecvFromHost")) return true;
  return false;
}

// Returns true if the given op is for outside compilation.
inline bool IsOutsideCompilationOp(std::string_view tf_op_fullname,
                                   std::string_view hlo_expression) {
  if (IsOutsideCompilationOp(tf_op_fullname)) return true;
  if (absl::StrContains(hlo_expression, "send-done") &&
      absl::StrContains(hlo_expression, "is_host_transfer=true"))
    return true;
  return false;
}

// Returns true if the given name is a TensorFlow embedding op.
inline bool IsEmbeddingOp(std::string_view tf_op_fullname) {
  return absl::StrContains(tf_op_fullname, "Embedding");
}

// Returns true if the given op is for copying data from host to device.
inline bool IsMemcpyHToDOp(std::string_view tf_op_type) {
  return tf_op_type == kMemcpyHToDOp;
}
inline bool IsMemcpyHToDOp(const TfOp& tf_op) {
  return tf_op.category == Category::kMemcpyHToD;
}

// Returns true if the given op is for copying data from device to host.
inline bool IsMemcpyDToHOp(const TfOp& tf_op) {
  return tf_op.category == Category::kMemcpyDToH;
}

// Returns true if the given op is for copying data from device to device.
inline bool IsMemcpyDToDOp(const TfOp& tf_op) {
  return tf_op.category == Category::kMemcpyDToD;
}

// Returns true if the given op is for copying data from host to host.
inline bool IsMemcpyHToHOp(const TfOp& tf_op) {
  return tf_op.category == Category::kMemcpyHToH;
}

// Splits a string of tensor shapes in "(shape1;shape2;...)" format, i.e.,
// delimited by '(' and ')' and separated by ';', into the individual shapes.
std::vector<std::string_view> ParseTensorShapes(std::string_view tensor_shapes);

// Returns true if the given string matches OpDef.name pattern.
bool IsTfOpName(std::string_view op_name);

// Returns true if the given string matches NodeDef.name pattern.
bool IsTfOpType(std::string_view op_type);

// Returns true if the given string matches JAX pattern.
bool IsJaxOpType(std::string_view op_type);

// Returns true if the given strings match JAX pattern.
bool IsJaxOpNameAndType(std::string_view op_name, std::string_view op_type);

}  // namespace profiler
}  // namespace tsl

#endif  // XLA_TSL_PROFILER_UTILS_TF_OP_UTILS_H_
