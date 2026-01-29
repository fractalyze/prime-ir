/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
#ifndef XLA_TSL_PROFILER_CONVERT_ZKX_OP_UTILS_H_
#define XLA_TSL_PROFILER_CONVERT_ZKX_OP_UTILS_H_

#include <string>
#include <string_view>

#include "absl/base/attributes.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"

namespace tsl::profiler {

// HLO categories used for analysis.
inline constexpr std::string_view kHloInfeed = "infeed";
inline constexpr std::string_view kHloOutfeed = "outfeed";
inline constexpr std::string_view kHloAllReduce = "all-reduce";
inline constexpr std::string_view kHloAllToAll = "all-to-all";
inline constexpr std::string_view kHloSend = "send";
inline constexpr std::string_view kHloSendDone = "send-done";
inline constexpr std::string_view kHloRecv = "recv";
inline constexpr std::string_view kHloRecvDone = "recv-done";
inline constexpr std::string_view kHloHostSend = "host send";
inline constexpr std::string_view kHloHostSendDone = "host send-done";
inline constexpr std::string_view kHloHostRecv = "host recv";
inline constexpr std::string_view kHloHostRecvDone = "host recv-done";
inline constexpr std::string_view kHloCall = "call";
inline constexpr std::string_view kHloConditional = "conditional";
inline constexpr std::string_view kHloWhile = "while";
inline constexpr std::string_view kHloConvolution = "convolution";
inline constexpr std::string_view kHloConvolutionBaseDilated =
    "convolution base-dilated";
inline constexpr std::string_view kHloConvolutionWindowDilated =
    "convolution window-dilated";
inline constexpr std::string_view kHloOutputFusion = "output fusion";
inline constexpr std::string_view kHloConvolutionFusion = "convolution fusion";
inline constexpr std::string_view kHloCustomFusion = "custom fusion";
inline constexpr std::string_view kHloAllReduceFusion = "all-reduce fusion";
inline constexpr std::string_view kHloAllGatherFusion = "all-gather fusion";
inline constexpr std::string_view kHloAllReduceScatterFusion =
    "all-reduce-scatter fusion";
inline constexpr std::string_view kHloGatherFusion = "gather fusion";
inline constexpr std::string_view kHloScatterFusion = "scatter fusion";
inline constexpr std::string_view kHloMegacoreFusion = "megacore fusion";
inline constexpr std::string_view kHloCopy = "copy";
inline constexpr std::string_view kHloCopyStart = "copy-start";
inline constexpr std::string_view kHloCopyDone = "copy-done";
inline constexpr std::string_view kHloCollectivePermute = "collective-permute";
inline constexpr std::string_view kHloCollectivePermuteStart =
    "collective-permute-start";
inline constexpr std::string_view kHloCollectivePermuteDone =
    "collective-permute-done";
inline constexpr std::string_view kHloAllGatherStart = "all-gather-start";
inline constexpr std::string_view kHloAllGatherDone = "all-gather-done";
inline constexpr std::string_view kHloAfterAll = "after-all";
inline constexpr std::string_view kHloAllGather = "all-gather";
inline constexpr std::string_view kHloAllReduceStart = "all-reduce-start";
inline constexpr std::string_view kHloAllReduceDone = "all-reduce-done";
inline constexpr std::string_view kHloAsyncStart = "async-start";
inline constexpr std::string_view kHloAsyncUpdate = "async-update";
inline constexpr std::string_view kHloAsyncDone = "async-done";
inline constexpr std::string_view kHloReshape = "reshape";
inline constexpr std::string_view kHloTranspose = "transpose";

// SparseCore V0 sub-categories.
ABSL_CONST_INIT extern const std::string_view kHloSparseCoreV0Infeed;
ABSL_CONST_INIT extern const std::string_view kHloSparseCoreV0Outfeed;
ABSL_CONST_INIT extern const std::string_view kHloSparseCoreV0InfeedWait;
ABSL_CONST_INIT extern const std::string_view kHloSparseCoreV0InfeedTransform;

// Return if a category is fusion.
inline bool IsFusion(std::string_view category) {
  return absl::EndsWith(category, " fusion");
}

// Return a concatenation of the program name with program id.
inline std::string HloModuleNameWithProgramId(std::string_view hlo_module_name,
                                              uint64_t program_id) {
  return absl::StrCat(hlo_module_name, "(", program_id, ")");
}

inline bool IsHloRematerialization(std::string_view hlo_expression) {
  auto pos = hlo_expression.find_first_of('=');
  if (pos != std::string_view::npos) {
    hlo_expression.remove_suffix(hlo_expression.size() - pos);
  }
  return absl::StrContains(hlo_expression, ".remat");
}

// Return true if framework_op is a remat.
inline bool IsFrameworkRematerialization(std::string_view framework_op_name) {
  return absl::StrContains(framework_op_name, "/rematted_computation/");
}

// Return true if hlo_expression is a remat.
inline bool IsRematerialization(std::string_view hlo_expression,
                                std::string_view framework_op_name) {
  return IsHloRematerialization(hlo_expression) ||
         IsFrameworkRematerialization(framework_op_name);
}

inline bool IsInfeedOrOutfeed(std::string_view category) {
  return category == kHloInfeed || category == kHloOutfeed ||
         absl::StrContains(category, kHloInfeed) ||
         absl::StrContains(category, kHloOutfeed);
}

inline bool IsHostOrSparseCoreV0Infeed(std::string_view category) {
  return category == tsl::profiler::kHloInfeed ||
         category == tsl::profiler::kHloSparseCoreV0Infeed;
}

inline bool MayHaveInnerOps(std::string_view category) {
  return category == kHloCall || category == kHloConditional ||
         category == kHloWhile || category == kHloMegacoreFusion;
}

inline bool IsOffDutyOp(std::string_view category) {
  return (
      category == tsl::profiler::kHloInfeed ||
      category == tsl::profiler::kHloOutfeed ||
      category == tsl::profiler::kHloHostSend ||
      category == tsl::profiler::kHloHostSendDone ||
      category == tsl::profiler::kHloHostRecv ||
      category == tsl::profiler::kHloHostRecvDone ||
      category ==
          tsl::profiler::kHloMegacoreFusion);  // Only self-time in megacore
                                               // fusion is off-duty. The op
                                               // time of children is on-duty.
}

// File and line that the framework op corresponding to an HLO op is associated
// to in a user's program; e.g. it could be the file and line of user code that
// generated the op.
struct OpSourceInfo {
  std::string_view source_file;
  int32_t source_line = -1;
  std::string stack_frame;

  std::string GetSourceTopLine() const {
    if (source_file.empty()) return "";
    return absl::StrCat(source_file, ":", source_line);
  }

  std::string GetSourceStack() const { return stack_frame; }
};

}  // namespace tsl::profiler

#endif  // XLA_TSL_PROFILER_CONVERT_ZKX_OP_UTILS_H_
