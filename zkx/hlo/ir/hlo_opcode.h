/* Copyright 2017 The OpenXLA Authors.
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

#ifndef ZKX_HLO_IR_HLO_OPCODE_H_
#define ZKX_HLO_IR_HLO_OPCODE_H_

#include <stdint.h>

#include <optional>
#include <ostream>
#include <string_view>

#include "absl/status/statusor.h"

namespace zkx {

// High-level optimizer instruction opcodes -- these are linear-algebra level
// opcodes. They are a flattened form of the UnaryOp, BinaryOp, ... opcodes
// present in the ZKX service protobuf.
//
// See the ZKX documentation for the semantics of each opcode.
//
// Each entry has the format:
// (enum_name, opcode_name, arity)
//
// Note: Do not use ':' in opcode names. It is used as a special character
// in these places:
// - In extended opcode strings (HloInstruction::ExtendedOpcodeString()), to
//   separate the opcode from the fusion kind
// - In fully qualified names (HloInstruction::FullyQualifiedName()), to
//   separate the qualifiers (name of the computation and potentially the
//   fusion instruction) from the name
//
// If you change one of these opcodes, please make the corresponding change to
// the MHLO opset to keep both opsets synchronized.

// NOTE(chokobole): If you add a new opcode, please update the following:
// - HloInstruction::IsOpElementwise
// - HloInstruction::IdenticalSlowPath
// - HloInstruction::MightHaveCalledComputations
// - HloInstruction::HasSideEffectNoRecurse
// - HloInstruction::CloneWithNewOperands
// - HloInstruction::CreateUnary
// - HloInstruction::CreateBinary
// - HloInstruction::has_to_apply
// - HloInstruction::CreateFromProto
// - GetInstructionCallContext
// - GatherComputationsByAllocationType
// - ZKX_UNOP_PATTERN
// - ZKX_BINOP_PATTERN
// - ZKX_COMMUTATIVE_BINOP_PATTERN
// - ZKX_NULLOP_PATTERN
// - ZKX_TERNOP_PATTERN
// - ZKX_VARIADIC_OP_PATTERN
// - MatchReductionInstruction
#define HLO_OPCODE_LIST(V)                                                     \
  /* go/keep-sorted start */                                                   \
  V(kAbs, "abs", 1)                                                            \
  V(kAdd, "add", 2)                                                            \
  V(kAddDependency, "add-dependency", 2)                                       \
  V(kAfterAll, "after-all", kHloOpcodeIsVariadic)                              \
  V(kAllGather, "all-gather", kHloOpcodeIsVariadic)                            \
  V(kAllGatherDone, "all-gather-done", 1)                                      \
  V(kAllGatherStart, "all-gather-start", kHloOpcodeIsVariadic)                 \
  V(kAllReduce, "all-reduce", kHloOpcodeIsVariadic)                            \
  V(kAllReduceDone, "all-reduce-done", 1)                                      \
  V(kAllReduceStart, "all-reduce-start", kHloOpcodeIsVariadic)                 \
  V(kAllToAll, "all-to-all", kHloOpcodeIsVariadic)                             \
  V(kAnd, "and", 2)                                                            \
  V(kAsyncDone, "async-done", 1)                                               \
  V(kAsyncStart, "async-start", kHloOpcodeIsVariadic)                          \
  V(kAsyncUpdate, "async-update", 1)                                           \
  V(kBitcast, "bitcast", 1)                                                    \
  V(kBitcastConvert, "bitcast-convert", 1)                                     \
  V(kBitReverse, "bit-reverse", 1)                                             \
  V(kBroadcast, "broadcast", 1)                                                \
  V(kCall, "call", kHloOpcodeIsVariadic)                                       \
  V(kClamp, "clamp", 3)                                                        \
  V(kClz, "count-leading-zeros", 1)                                            \
  V(kCollectiveBroadcast, "collective-broadcast", kHloOpcodeIsVariadic)        \
  V(kCollectivePermute, "collective-permute", kHloOpcodeIsVariadic)            \
  V(kCollectivePermuteDone, "collective-permute-done", 1)                      \
  V(kCollectivePermuteStart, "collective-permute-start", kHloOpcodeIsVariadic) \
  V(kCompare, "compare", 2)                                                    \
  V(kConcatenate, "concatenate", kHloOpcodeIsVariadic)                         \
  V(kConditional, "conditional", kHloOpcodeIsVariadic)                         \
  V(kConstant, "constant", 0)                                                  \
  V(kConvert, "convert", 1)                                                    \
  V(kCopy, "copy", 1)                                                          \
  V(kCopyDone, "copy-done", 1)                                                 \
  V(kCopyStart, "copy-start", 1)                                               \
  V(kCustomCall, "custom-call", kHloOpcodeIsVariadic)                          \
  V(kDivide, "divide", 2)                                                      \
  V(kDomain, "domain", 1)                                                      \
  V(kDot, "dot", kHloOpcodeIsVariadic)                                         \
  V(kDynamicReshape, "dynamic-reshape", kHloOpcodeIsVariadic)                  \
  V(kDynamicSlice, "dynamic-slice", kHloOpcodeIsVariadic)                      \
  V(kDynamicUpdateSlice, "dynamic-update-slice", kHloOpcodeIsVariadic)         \
  V(kFusion, "fusion", kHloOpcodeIsVariadic)                                   \
  V(kFft, "fft", kHloOpcodeIsVariadic)                                         \
  V(kGather, "gather", 2)                                                      \
  V(kGetDimensionSize, "get-dimension-size", 1)                                \
  V(kGetTupleElement, "get-tuple-element", 1)                                  \
  V(kInfeed, "infeed", 1)                                                      \
  V(kIota, "iota", 0)                                                          \
  V(kInverse, "inverse", 1)                                                    \
  V(kMap, "map", kHloOpcodeIsVariadic)                                         \
  V(kMaximum, "maximum", 2)                                                    \
  V(kMinimum, "minimum", 2)                                                    \
  V(kMsm, "msm", 2)                                                            \
  V(kMultiply, "multiply", 2)                                                  \
  V(kNegate, "negate", 1)                                                      \
  V(kNot, "not", 1)                                                            \
  V(kOptimizationBarrier, "opt-barrier", 1)                                    \
  V(kOr, "or", 2)                                                              \
  V(kOutfeed, "outfeed", 2)                                                    \
  V(kPad, "pad", 2)                                                            \
  V(kParameter, "parameter", 0)                                                \
  V(kPartitionId, "partition-id", 0)                                           \
  V(kPopulationCount, "popcnt", 1)                                             \
  V(kPower, "power", 2)                                                        \
  V(kRaggedAllToAll, "ragged-all-to-all", 6)                                   \
  V(kRaggedDot, "ragged-dot", 3)                                               \
  V(kRecv, "recv", 1)                                                          \
  V(kRecvDone, "recv-done", 1)                                                 \
  V(kReduce, "reduce", kHloOpcodeIsVariadic)                                   \
  V(kReduceScatter, "reduce-scatter", kHloOpcodeIsVariadic)                    \
  V(kReduceWindow, "reduce-window", kHloOpcodeIsVariadic)                      \
  V(kRemainder, "remainder", 2)                                                \
  V(kReplicaId, "replica-id", 0)                                               \
  V(kReshape, "reshape", 1)                                                    \
  V(kReverse, "reverse", 1)                                                    \
  V(kScatter, "scatter", kHloOpcodeIsVariadic)                                 \
  V(kSelect, "select", 3)                                                      \
  V(kSend, "send", 2)                                                          \
  V(kSendDone, "send-done", 1)                                                 \
  V(kSetDimensionSize, "set-dimension-size", 2)                                \
  V(kShiftLeft, "shift-left", 2)                                               \
  V(kShiftRightArithmetic, "shift-right-arithmetic", 2)                        \
  V(kShiftRightLogical, "shift-right-logical", 2)                              \
  V(kSign, "sign", 1)                                                          \
  V(kSlice, "slice", 1)                                                        \
  V(kSort, "sort", kHloOpcodeIsVariadic)                                       \
  V(kSubtract, "subtract", 2)                                                  \
  V(kTranspose, "transpose", 1)                                                \
  V(kTuple, "tuple", kHloOpcodeIsVariadic)                                     \
  V(kWhile, "while", 1)                                                        \
  V(kXor, "xor", 2)                                                            \
  /* go/keep-sorted end */

// Upto 256 opcodes. Increase the base type if/when needed.
enum class HloOpcode : uint8_t {
#define DECLARE_ENUM(enum_name, opcode_name, ...) enum_name,
  HLO_OPCODE_LIST(DECLARE_ENUM)
#undef DECLARE_ENUM
};

// Arity value that denotes that an operator is variadic.
enum {
  kHloOpcodeIsVariadic = -1,
};

// Returns a string representation of the opcode.
std::string_view HloOpcodeString(HloOpcode opcode);

// Retrieves the opcode enum by name if the opcode exists.
absl::StatusOr<HloOpcode> StringToHloOpcode(std::string_view opcode_name);

inline std::ostream& operator<<(std::ostream& os, HloOpcode opcode) {
  return os << HloOpcodeString(opcode);
}

// Returns the arity of opcode or nullopt for variadic opcodes.
std::optional<int8_t> HloOpcodeArity(HloOpcode opcode);

// Returns true for kAsyncStart, kAsyncUpdate, kAsyncDone.
inline bool HloOpcodeIsAsync(HloOpcode opcode) {
  switch (opcode) {
    case HloOpcode::kAsyncStart:
    case HloOpcode::kAsyncUpdate:
    case HloOpcode::kAsyncDone:
      return true;
    default:
      return false;
  }
}

// True if the op takes two arguments and order doesn't matter.
inline bool HloOpcodeIsBinaryCommutative(HloOpcode opcode) {
  switch (opcode) {
    case HloOpcode::kAdd:
    case HloOpcode::kAnd:
    case HloOpcode::kMultiply:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kOr:
    case HloOpcode::kXor:
      return true;
    default:
      return false;
  }
}

// Returns the number of HloOpcode values.
inline constexpr uint32_t HloOpcodeCount() {
#define HLO_COUNT_ONE(...) +1
#define HLO_XLIST_LENGTH(list) list(HLO_COUNT_ONE)
  return HLO_XLIST_LENGTH(HLO_OPCODE_LIST);
}
static_assert(HloOpcodeCount() < 256,
              "HloOpcode is a uint8_t. You need to increase its size before "
              "adding new op codes.");

}  // namespace zkx

#endif  // ZKX_HLO_IR_HLO_OPCODE_H_
