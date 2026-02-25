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

#ifndef ZKX_HLO_UTILS_HLO_MATCHERS_H_
#define ZKX_HLO_UTILS_HLO_MATCHERS_H_

#include <optional>
#include <string>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/parser/hlo_parser.h"
#include "zkx/zkx_data.pb.h"

namespace zkx {
namespace testing {

class HloMatcher : public ::testing::MatcherInterface<const HloInstruction*> {
 public:
  HloMatcher(HloOpcode opcode,
             std::vector<::testing::Matcher<const HloInstruction*>> operands)
      : opcode_(opcode), operands_(operands) {}

  bool MatchAndExplain(const HloInstruction* instruction,
                       ::testing::MatchResultListener* listener) const override;

  void DescribeTo(::std::ostream* os) const override;

 private:
  HloOpcode opcode_;
  std::vector<::testing::Matcher<const HloInstruction*>> operands_;
};

// Custom matcher for parameters, which accepts a parameter number.
class HloParameterMatcher : public HloMatcher {
 public:
  explicit HloParameterMatcher(int64_t parameter_number)
      : HloMatcher(HloOpcode::kParameter, /*operands=*/{}),
        parameter_number_(parameter_number) {}

  bool MatchAndExplain(const HloInstruction* instruction,
                       ::testing::MatchResultListener* listener) const override;

 private:
  int64_t parameter_number_;
};

// Custom matcher for comparisons, which accepts a comparison direction.
class HloComparisonMatcher : public HloMatcher {
 public:
  explicit HloComparisonMatcher(
      ComparisonDirection direction,
      std::vector<::testing::Matcher<const HloInstruction*>> operands)
      : HloMatcher(HloOpcode::kCompare, operands), direction_(direction) {}

  bool MatchAndExplain(const HloInstruction* instruction,
                       ::testing::MatchResultListener* listener) const override;

 private:
  ComparisonDirection direction_;
};

// Custom matcher for get-tuple-element instructions, which accepts a tuple
// index to match.
class HloGetTupleElementMatcher : public HloMatcher {
 public:
  HloGetTupleElementMatcher(::testing::Matcher<const HloInstruction*> operand,
                            int64_t tuple_index)
      : HloMatcher(HloOpcode::kGetTupleElement, /*operands=*/{operand}),
        tuple_index_(tuple_index) {}

  bool MatchAndExplain(const HloInstruction* instruction,
                       ::testing::MatchResultListener* listener) const override;

 private:
  int64_t tuple_index_;
};

// Custom matcher for custom-call instructions, which accepts a matcher for its
// call target.
class HloCustomCallMatcher : public HloMatcher {
 public:
  HloCustomCallMatcher(
      ::testing::Matcher<std::string> call_target_matcher,
      std::vector<::testing::Matcher<const HloInstruction*>> operands)
      : HloMatcher(HloOpcode::kCustomCall, operands),
        call_target_matcher_(call_target_matcher) {}

  bool MatchAndExplain(const HloInstruction* instruction,
                       ::testing::MatchResultListener* listener) const override;
  void DescribeTo(std::ostream* os) const override;

 private:
  ::testing::Matcher<std::string> call_target_matcher_;
};

class HloShapeMatcher
    : public ::testing::MatcherInterface<const HloInstruction*> {
 public:
  explicit HloShapeMatcher(const Shape& shape) : shape_(shape) {}

  bool MatchAndExplain(const HloInstruction* instruction,
                       ::testing::MatchResultListener* listener) const override;
  void DescribeTo(std::ostream* os) const override;

 private:
  Shape shape_;
};

class HloShapeAndLayoutMatcher
    : public ::testing::MatcherInterface<const HloInstruction*> {
 public:
  explicit HloShapeAndLayoutMatcher(const Shape& shape,
                                    bool minor_to_major_only = false)
      : shape_(shape), minor_to_major_only_(minor_to_major_only) {}

  bool MatchAndExplain(const HloInstruction* instruction,
                       ::testing::MatchResultListener* listener) const override;
  void DescribeTo(std::ostream* os) const override;

 private:
  Shape shape_;
  bool minor_to_major_only_;
};

// Verify the sharding of an instruction against the provided HloSharding. If a
// nullopt is provided for the expected sharding then it checks that no sharding
// is present for an instruction.
class HloShardingMatcher
    : public ::testing::MatcherInterface<const HloInstruction*> {
 public:
  explicit HloShardingMatcher(const std::optional<HloSharding>& sharding)
      : sharding_(sharding) {}

  bool MatchAndExplain(const HloInstruction* instruction,
                       ::testing::MatchResultListener* listener) const override;
  void DescribeTo(std::ostream* os) const override;

 private:
  std::optional<HloSharding> sharding_;
};

// Matches a Dot HLO instruction with specific LHS and RHS contracting
// dimensions.
class HloDotWithContractingDimsMatcher : public HloMatcher {
 public:
  explicit HloDotWithContractingDimsMatcher(
      ::testing::Matcher<const HloInstruction*> lhs,
      ::testing::Matcher<const HloInstruction*> rhs,
      int64_t lhs_contracting_dim, int64_t rhs_contracting_dim)
      : HloMatcher(HloOpcode::kDot, /*operands=*/{lhs, rhs}),
        lhs_contracting_dim_(lhs_contracting_dim),
        rhs_contracting_dim_(rhs_contracting_dim) {}

  bool MatchAndExplain(const HloInstruction* instruction,
                       ::testing::MatchResultListener* listener) const override;
  void DescribeTo(std::ostream* os) const override;

 private:
  int64_t lhs_contracting_dim_;
  int64_t rhs_contracting_dim_;
};

// Custom matcher for asynchronous copy (CopyStart/CopyDone pair) with specified
// source and destination memory spaces.
class HloAsyncCopyMatcher : public HloMatcher {
 public:
  HloAsyncCopyMatcher(int64_t to_space, int64_t from_space,
                      ::testing::Matcher<const HloInstruction*> operand)
      : HloMatcher(HloOpcode::kCopyDone,
                   {::testing::MakeMatcher(
                       new HloMatcher(HloOpcode::kCopyStart, {operand}))}),
        to_space_(to_space),
        from_space_(from_space) {}

  bool MatchAndExplain(const HloInstruction* instruction,
                       ::testing::MatchResultListener* listener) const override;
  void DescribeTo(std::ostream* os) const override;

 private:
  int64_t to_space_;
  int64_t from_space_;
};

class HloConstantMatcher : public HloMatcher {
 public:
  explicit HloConstantMatcher(Literal literal)
      : HloMatcher(HloOpcode::kConstant, /*operands=*/{}),
        literal_(std::move(literal)) {}
  bool MatchAndExplain(const HloInstruction* instruction,
                       ::testing::MatchResultListener* listener) const override;
  void DescribeTo(std::ostream* os) const override;

 private:
  Literal literal_;
};

class HloReplicaGroupsMatcher
    : public ::testing::MatcherInterface<const HloInstruction*> {
 public:
  explicit HloReplicaGroupsMatcher(
      std::vector<std::vector<int64_t>> replica_groups)
      : replica_groups_(std::move(replica_groups)) {}

  bool MatchAndExplain(const HloInstruction* instruction,
                       ::testing::MatchResultListener* listener) const override;
  void DescribeTo(std::ostream* os) const override;

 private:
  std::vector<std::vector<int64_t>> replica_groups_;
};

class HloSourceTargetPairsMatcher
    : public ::testing::MatcherInterface<const HloInstruction*> {
 public:
  explicit HloSourceTargetPairsMatcher(
      std::vector<std::pair<int64_t, int64_t>> source_target_pairs)
      : source_target_pairs_(std::move(source_target_pairs)) {}

  bool MatchAndExplain(const HloInstruction* instruction,
                       ::testing::MatchResultListener* listener) const override;
  void DescribeTo(std::ostream* os) const override;

 private:
  std::vector<std::pair<int64_t, int64_t>> source_target_pairs_;
};

class HloMetadataMatcher
    : public ::testing::MatcherInterface<const HloInstruction*> {
 public:
  explicit HloMetadataMatcher(OpMetadata metadata)
      : metadata_(std::move(metadata)) {}

  bool MatchAndExplain(const HloInstruction* instruction,
                       ::testing::MatchResultListener* listener) const override;
  void DescribeTo(std::ostream* os) const override;

 private:
  OpMetadata metadata_;
};

// HloInstruction* matchers for opcode and operands. Example:
//   namespace op = zkx::opcode_matchers;
//   EXPECT_THAT(instruction,
//               op::Add(op::Reshape(), op::Add(op::Reshape(), _)));
namespace opcode_matchers {
#define HLO_MATCHER(opcode)                                                \
  template <typename... M>                                                 \
  ::testing::Matcher<const ::zkx::HloInstruction*> opcode(M... operands) { \
    return ::testing::MakeMatcher(new ::zkx::testing::HloMatcher(          \
        ::zkx::HloOpcode::k##opcode, {operands...}));                      \
  }
HLO_MATCHER(Abs);
HLO_MATCHER(Add);
HLO_MATCHER(AddDependency);
HLO_MATCHER(AfterAll);
HLO_MATCHER(AsyncStart);
HLO_MATCHER(AsyncUpdate);
HLO_MATCHER(AsyncDone);
HLO_MATCHER(AllGather);
HLO_MATCHER(AllGatherStart);
HLO_MATCHER(AllGatherDone);
HLO_MATCHER(AllReduce);
HLO_MATCHER(AllReduceStart);
HLO_MATCHER(AllReduceDone);
HLO_MATCHER(AllToAll);
HLO_MATCHER(And);
HLO_MATCHER(Bitcast);
HLO_MATCHER(BitcastConvert);
HLO_MATCHER(Broadcast);
HLO_MATCHER(Call);
HLO_MATCHER(Clamp);
HLO_MATCHER(CollectiveBroadcast);
HLO_MATCHER(CollectivePermute);
HLO_MATCHER(CollectivePermuteStart);
HLO_MATCHER(CollectivePermuteDone);
HLO_MATCHER(Compare);
HLO_MATCHER(Concatenate);
HLO_MATCHER(Conditional);
HLO_MATCHER(Convert);
HLO_MATCHER(Copy);
HLO_MATCHER(CopyDone);
HLO_MATCHER(CopyStart);
HLO_MATCHER(Divide);
HLO_MATCHER(Domain);
HLO_MATCHER(DynamicSlice);
HLO_MATCHER(DynamicUpdateSlice);
HLO_MATCHER(Fft);
HLO_MATCHER(Fusion);
HLO_MATCHER(Gather);
HLO_MATCHER(GetDimensionSize);
HLO_MATCHER(Infeed);
HLO_MATCHER(Iota);
HLO_MATCHER(Map);
HLO_MATCHER(Maximum);
HLO_MATCHER(Minimum);
HLO_MATCHER(Multiply);
HLO_MATCHER(Negate);
HLO_MATCHER(Not);
HLO_MATCHER(Or);
HLO_MATCHER(Outfeed);
HLO_MATCHER(Pad);
HLO_MATCHER(PartitionId);
HLO_MATCHER(Power);
HLO_MATCHER(RaggedAllToAll);
HLO_MATCHER(Recv);
HLO_MATCHER(RecvDone);
HLO_MATCHER(Reduce);
HLO_MATCHER(ReduceScatter);
HLO_MATCHER(Remainder);
HLO_MATCHER(ReplicaId);
HLO_MATCHER(Reshape);
HLO_MATCHER(Reverse);
HLO_MATCHER(Scatter);
HLO_MATCHER(Select);
HLO_MATCHER(Send);
HLO_MATCHER(SendDone);
HLO_MATCHER(SetDimensionSize);
HLO_MATCHER(ShiftLeft);
HLO_MATCHER(ShiftRightArithmetic);
HLO_MATCHER(ShiftRightLogical);
HLO_MATCHER(Sign);
HLO_MATCHER(Slice);
HLO_MATCHER(Sort);
HLO_MATCHER(Subtract);
HLO_MATCHER(Transpose);
HLO_MATCHER(Tuple);
HLO_MATCHER(While);
HLO_MATCHER(Xor);
HLO_MATCHER(OptimizationBarrier);

#define HLO_MATCHER_VECTOR_OPERANDS(opcode)                              \
  template <>                                                            \
  inline ::testing::Matcher<const ::zkx::HloInstruction*> opcode(        \
      std::vector<::testing::Matcher<const HloInstruction*>> operands) { \
    return ::testing::MakeMatcher(new ::zkx::testing::HloMatcher(        \
        ::zkx::HloOpcode::k##opcode, operands));                         \
  }

HLO_MATCHER_VECTOR_OPERANDS(DynamicSlice);

// The special cases below let you check additional information about the
// HloInstruction, beyond just its opcode and operands.  In all cases you can
// still use the generic matcher which doesn't check this info.
//
// Feel free to add additional custom matchers below.

//  - Parameter(N) matches parameter number N.
//  - Parameter() matches any parameter.
inline ::testing::Matcher<const ::zkx::HloInstruction*> Parameter(
    int64_t parameter_number) {
  return ::testing::MakeMatcher(
      new ::zkx::testing::HloParameterMatcher(parameter_number));
}
inline ::testing::Matcher<const ::zkx::HloInstruction*> Parameter() {
  return ::testing::MakeMatcher(
      new ::zkx::testing::HloMatcher(HloOpcode::kParameter, {}));
}

// Comparison matchers below do not require any additional arguments.
template <typename... M>
inline ::testing::Matcher<const ::zkx::HloInstruction*> Eq(M... operands) {
  return ::testing::MakeMatcher(new ::zkx::testing::HloComparisonMatcher(
      ComparisonDirection::kEq, {operands...}));
}
template <typename... M>
inline ::testing::Matcher<const ::zkx::HloInstruction*> Ne(M... operands) {
  return ::testing::MakeMatcher(new ::zkx::testing::HloComparisonMatcher(
      ComparisonDirection::kNe, {operands...}));
}
template <typename... M>
inline ::testing::Matcher<const ::zkx::HloInstruction*> Ge(M... operands) {
  return ::testing::MakeMatcher(new ::zkx::testing::HloComparisonMatcher(
      ComparisonDirection::kGe, {operands...}));
}
template <typename... M>
inline ::testing::Matcher<const ::zkx::HloInstruction*> Gt(M... operands) {
  return ::testing::MakeMatcher(new ::zkx::testing::HloComparisonMatcher(
      ComparisonDirection::kGt, {operands...}));
}
template <typename... M>
inline ::testing::Matcher<const ::zkx::HloInstruction*> Le(M... operands) {
  return ::testing::MakeMatcher(new ::zkx::testing::HloComparisonMatcher(
      ComparisonDirection::kLe, {operands...}));
}
template <typename... M>
inline ::testing::Matcher<const ::zkx::HloInstruction*> Lt(M... operands) {
  return ::testing::MakeMatcher(new ::zkx::testing::HloComparisonMatcher(
      ComparisonDirection::kLt, {operands...}));
}

// GetTupleElement(operand, N) matches a GTE instruction which gets the N'th
// tuple element of operand, while GetTupleElement(operand) matches any GTE
// operation on operand, and GetTupleElement() matches any GTE operation at all.
inline ::testing::Matcher<const ::zkx::HloInstruction*> GetTupleElement(
    ::testing::Matcher<const HloInstruction*> operand, int64_t tuple_index) {
  return ::testing::MakeMatcher(
      new ::zkx::testing::HloGetTupleElementMatcher(operand, tuple_index));
}
inline ::testing::Matcher<const ::zkx::HloInstruction*> GetTupleElement(
    ::testing::Matcher<const HloInstruction*> operand) {
  return ::testing::MakeMatcher(
      new ::zkx::testing::HloMatcher(HloOpcode::kGetTupleElement, {operand}));
}
inline ::testing::Matcher<const ::zkx::HloInstruction*> GetTupleElement() {
  return ::testing::MakeMatcher(
      new ::zkx::testing::HloMatcher(HloOpcode::kGetTupleElement, {}));
}

// - CustomCall(T, operand1, ..., operandN) matches a CustomCall with call
//   target T and the given operands.
//
// - CustomCall(operand1, ..., operandN) matches any CustomCall HLO with the
//   given operands.
//
// - CustomCall() matches any CustomCall HLO at all.
template <typename... M>
inline ::testing::Matcher<const ::zkx::HloInstruction*> CustomCall(
    ::testing::Matcher<std::string> call_target_matcher, M... operands) {
  return ::testing::MakeMatcher(new ::zkx::testing::HloCustomCallMatcher(
      call_target_matcher, {operands...}));
}
// This overload of CustomCall(A, B, C, ...) exists iff A is not convertible to
// ::testing::Matcher<std::string>.  In that case, we want to prefer the
// overload above.
template <
    typename FirstM, typename... M,
    typename Dummy = typename std::enable_if<
        !std::is_convertible<FirstM, ::testing::Matcher<std::string>>::value,
        void>::type*>
inline ::testing::Matcher<const ::zkx::HloInstruction*> CustomCall(
    FirstM operands_first, M... operands_rest) {
  return ::testing::MakeMatcher(new ::zkx::testing::HloMatcher(
      HloOpcode::kCustomCall, {operands_first, operands_rest...}));
}
inline ::testing::Matcher<const ::zkx::HloInstruction*> CustomCall() {
  return ::testing::MakeMatcher(
      new ::zkx::testing::HloMatcher(HloOpcode::kCustomCall, {}));
}

// Verifies the shape or the shape and the layout of an HLO instruction against
// the provided shape object.
inline ::testing::Matcher<const ::zkx::HloInstruction*> Shape(
    const class Shape& shape) {
  return ::testing::MakeMatcher(new ::zkx::testing::HloShapeMatcher(shape));
}
inline ::testing::Matcher<const ::zkx::HloInstruction*> Shape(
    std::string_view shape) {
  return ::testing::MakeMatcher(
      new ::zkx::testing::HloShapeMatcher(ParseShape(shape).value()));
}
inline ::testing::Matcher<const ::zkx::HloInstruction*> ShapeWithLayout(
    const class Shape& shape) {
  return ::testing::MakeMatcher(
      new ::zkx::testing::HloShapeAndLayoutMatcher(shape));
}
inline ::testing::Matcher<const ::zkx::HloInstruction*> ShapeWithLayout(
    std::string_view shape, bool minor_to_major_only = false) {
  return ::testing::MakeMatcher(new ::zkx::testing::HloShapeAndLayoutMatcher(
      ParseShape(shape).value(), minor_to_major_only));
}

// Verifies the value of the HloSharing against the provided sharding object.
inline ::testing::Matcher<const ::zkx::HloInstruction*> Sharding(
    const HloSharding& sharding) {
  return ::testing::MakeMatcher(
      new ::zkx::testing::HloShardingMatcher(sharding));
}
// Matcher for Sharding from sharding string
inline ::testing::Matcher<const ::zkx::HloInstruction*> Sharding(
    std::string_view sharding) {
  return ::testing::MakeMatcher(
      new ::zkx::testing::HloShardingMatcher(ParseSharding(sharding).value()));
}
// Verifies that no HloSharding is set for an HLO instruction.
inline ::testing::Matcher<const ::zkx::HloInstruction*> NoSharding() {
  return ::testing::MakeMatcher(
      new ::zkx::testing::HloShardingMatcher(std::nullopt));
}

inline ::testing::Matcher<const ::zkx::HloInstruction*> Dot() {
  return ::testing::MakeMatcher(
      new ::zkx::testing::HloMatcher(::zkx::HloOpcode::kDot, {}));
}

inline ::testing::Matcher<const ::zkx::HloInstruction*> Dot(
    ::testing::Matcher<const HloInstruction*> lhs_matcher,
    ::testing::Matcher<const HloInstruction*> rhs_matcher) {
  return ::testing::MakeMatcher(new ::zkx::testing::HloMatcher(
      ::zkx::HloOpcode::kDot, {lhs_matcher, rhs_matcher}));
}

// Matches a Dot HLO instruction if it has exactly one lhs contracting dimension
// equal to `lhs_contracting_dim` and exactly one rhs contracting dimension
// equal to `rhs_contracting_dim`.
//
// Currently the HLO verifier rejects Dot operations with more than one
// contracting dimension (even though we can represent these in the
// DotDimensionNumbers proto) so there is no need to generalize this to support
// multiple contracting dimensions.
inline ::testing::Matcher<const ::zkx::HloInstruction*> Dot(
    ::testing::Matcher<const HloInstruction*> lhs_matcher,
    ::testing::Matcher<const HloInstruction*> rhs_matcher,
    int64_t lhs_contracting_dim, int64_t rhs_contracting_dim) {
  return ::testing::MakeMatcher(
      new ::zkx::testing::HloDotWithContractingDimsMatcher(
          lhs_matcher, rhs_matcher, lhs_contracting_dim, rhs_contracting_dim));
}

// Matcher for asynchronous copies from one memory space to another. Implies
// CopyDone(CopyStart(...)) where from_space and to_space is the source and
// destination memory spaces, respectively.
inline ::testing::Matcher<const ::zkx::HloInstruction*> AsyncCopy(
    int64_t to_space, int64_t from_space,
    ::testing::Matcher<const HloInstruction*> operand_matcher) {
  return ::testing::MakeMatcher(new ::zkx::testing::HloAsyncCopyMatcher(
      to_space, from_space, operand_matcher));
}

//  - Constant() matches any constant.
//  - Constant(V) matches a constant with the given value.
inline ::testing::Matcher<const ::zkx::HloInstruction*> Constant() {
  return ::testing::MakeMatcher(
      new ::zkx::testing::HloMatcher(HloOpcode::kConstant, {}));
}
inline ::testing::Matcher<const ::zkx::HloInstruction*> Constant(
    Literal value) {
  return ::testing::MakeMatcher(
      new ::zkx::testing::HloConstantMatcher(std::move(value)));
}

inline ::testing::Matcher<const ::zkx::HloInstruction*> ReplicaGroups(
    std::vector<std::vector<int64_t>> replica_groups) {
  return ::testing::MakeMatcher(
      new ::zkx::testing::HloReplicaGroupsMatcher(std::move(replica_groups)));
}

inline ::testing::Matcher<const ::zkx::HloInstruction*> SourceTargetPairs(
    std::vector<std::pair<int64_t, int64_t>> source_target_pairs) {
  return ::testing::MakeMatcher(new ::zkx::testing::HloSourceTargetPairsMatcher(
      std::move(source_target_pairs)));
}

inline ::testing::Matcher<const ::zkx::HloInstruction*> Metadata(
    OpMetadata metadata) {
  return ::testing::MakeMatcher(
      new ::zkx::testing::HloMetadataMatcher(std::move(metadata)));
}

#undef HLO_MATCHER
}  // namespace opcode_matchers

// Helper to convert smart to raw pointers for matching.
template <typename Container>
std::vector<const HloInstruction*> Pointers(const Container& container) {
  std::vector<const HloInstruction*> result;
  result.reserve(container.size());
  for (const auto& entry : container) result.push_back(entry.get());
  return result;
}

}  // namespace testing

// Tell GMock to print HloInstruction* by value, so error messages are nice.
// Has to be in the same namespace as 'HloInstruction'.
void PrintTo(const HloInstruction* inst, ::std::ostream* os);

}  // namespace zkx

#endif  // ZKX_HLO_UTILS_HLO_MATCHERS_H_
