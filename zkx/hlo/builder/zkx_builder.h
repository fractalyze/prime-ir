/* Copyright 2018 The OpenXLA Authors.
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

#ifndef ZKX_HLO_BUILDER_ZKX_BUILDER_H_
#define ZKX_HLO_BUILDER_ZKX_BUILDER_H_

#include <cstdint>
#include <deque>
#include <initializer_list>
#include <map>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"

#include "xla/tsl/lib/core/bitmap.h"
#include "xla/tsl/platform/errors.h"
#include "zkx/array.h"
#include "zkx/array2d.h"
#include "zkx/array3d.h"
#include "zkx/comparison_util.h"
#include "zkx/hlo/builder/zkx_computation.h"
#include "zkx/hlo/ir/hlo_input_output_alias_config.h"
#include "zkx/hlo/ir/hlo_opcode.h"
#include "zkx/layout.h"
#include "zkx/literal.h"
#include "zkx/literal_util.h"
#include "zkx/service/hlo.pb.h"
#include "zkx/shape.h"
#include "zkx/shape_util.h"
#include "zkx/zkx_data.pb.h"

namespace zkx {

class ZkxBuilder;
class ZkxOp;
class HloInstruction;

namespace internal {

struct ZkxBuilderFriend {
  static ZkxOp BuildBitcast(ZkxBuilder* builder, ZkxOp operand,
                            const Shape& shape);

  static HloInstructionProto* GetInstruction(ZkxOp op);
  static HloInstructionProto* GetInstructionByHandle(ZkxBuilder* builder,
                                                     int64_t handle);
};

}  // namespace internal

// This represents an instruction that has been enqueued using the ZkxBuilder.
// This is used to pass to subsequent computations that depends upon the
// instruction as an operand.
class ZkxOp {
 public:
  ZkxOp() : handle_(-1), builder_(nullptr) {
    static_assert(std::is_trivially_destructible<ZkxOp>::value,
                  "ZkxOp should be trivially destructible");
  }
  ~ZkxOp() = default;

  ZkxOp(const ZkxOp& other) = default;
  ZkxOp& operator=(const ZkxOp& other) = default;

  // Precondition: !IsUninitialized().
  //
  // It's very common to do foo.builder()->bar(). Without this precondition, if
  // foo.builder() is null, the call to bar will segfault at some point possibly
  // deep in the callstack when we finally dereference `this`. The precondition
  // lets us avoid this tricky-to-debug problem.
  ZkxBuilder* builder() const {
    CHECK_NE(builder_, nullptr);
    return builder_;
  }

  // Returns true if the ZkxOp represents valid, non-erroneous value.
  bool valid() const { return handle_ >= 0; }

  // Returns true if the ZkxOp was created by the ZkxOp() constructor and
  // not returned by a builder.
  bool IsUninitialized() const { return builder_ == nullptr; }

  bool IsIdenticalTo(ZkxOp rhs) const {
    return handle_ == rhs.handle_ && builder_ == rhs.builder_;
  }

  friend std::ostream& operator<<(std::ostream& out, ZkxOp op) {
    out << op.handle();
    return out;
  }

 private:
  explicit ZkxOp(ZkxBuilder* builder) : handle_(-1), builder_(builder) {}
  ZkxOp(int64_t handle, ZkxBuilder* builder)
      : handle_(handle), builder_(builder) {}

  int64_t handle() const { return handle_; }

  friend class ZkxBuilder;
  friend class ValueInference;
  friend struct internal::ZkxBuilderFriend;

  // < 0 means "invalid handle".
  int64_t handle_;

  // Not owned. Non-null for any handle returned by ZkxBuilder, even if the
  // handle is invalid.
  ZkxBuilder* builder_;
};

// Arithmetic operator overloads for the ZkxOp type.
ZkxOp operator-(ZkxOp x);
ZkxOp operator+(ZkxOp x, ZkxOp y);
ZkxOp operator-(ZkxOp x, ZkxOp y);
ZkxOp operator*(ZkxOp x, ZkxOp y);
ZkxOp operator/(ZkxOp x, ZkxOp y);
ZkxOp operator%(ZkxOp x, ZkxOp y);

// Bitwise operator overloads for the ZkxOp type.
ZkxOp operator~(ZkxOp x);
ZkxOp operator&(ZkxOp x, ZkxOp y);
ZkxOp operator|(ZkxOp x, ZkxOp y);
ZkxOp operator^(ZkxOp x, ZkxOp y);
ZkxOp operator<<(ZkxOp x, ZkxOp y);
// Performs a right arithmetic shift if 'x' is a signed type, otherwise performs
// a right logical shift.
ZkxOp operator>>(ZkxOp x, ZkxOp y);

// We don't overload the relational operators (==, !=, <, <=, >, >=) because the
// semantics might be surprising since their result types are usually 'bool'.
// Further programmers may expect == to be a structural equality.
// We also choose not to overload any of the mutating operators (e.g., +=, -=)
// because the semantics might be misleading — ZKX computations are immutable.

// A convenient interface for building up computations.
//
// Thread-compatible.
class ZkxBuilder {
 public:
  // computation_name: name to use for the built computation.
  explicit ZkxBuilder(const std::string& computation_name);

  ZkxBuilder(const ZkxBuilder&) = delete;
  ZkxBuilder& operator=(const ZkxBuilder&) = delete;

  virtual ~ZkxBuilder();

  // Returns the computation name.
  const std::string& name() const { return name_; }

  // Sets OpMetadata that will be added to all instructions until cleared.
  //
  // OpMetadata is often applied to a series of ZKX HLO instructions. As a
  // result, OpMetadata is set on the computation builder. All subsequent
  // instructions generated via this computation builder will have the same
  // OpMetadata attached until a call to ClearOpMetadata.
  void SetOpMetadata(OpMetadata metadata) { metadata_ = std::move(metadata); }

  // Swaps the passed op metadata with the ones currently set.
  //
  // Returns the old op metadata.
  OpMetadata SwapOpMetadata(OpMetadata metadata) {
    OpMetadata old_metadata = std::move(metadata_);
    metadata_ = std::move(metadata);
    return old_metadata;
  }

  // Similar to SetOpMetadata, but only set the metadata for the next op.
  void SetOneShotOpMetadata(OpMetadata metadata) {
    one_shot_metadata_ = std::move(metadata);
  }

  // Clears the HloMetadata state.
  void ClearOpMetadata() { metadata_.Clear(); }

  // Sets an OpSharding that will be attached to all instructions until cleared.
  void SetSharding(const OpSharding& sharding) { sharding_ = sharding; }

  // Sets the FrontendAttributes that will be added to all instructions until
  // cleared.
  //
  // FrontendAttributes are often applied to a series of ZKX HLO instructions.
  // As a result they are set on the computation builder and all the
  // instructions generated via the computation builder will have the same
  // frontend attributes attached to them.
  virtual void SetFrontendAttributes(
      const FrontendAttributes& frontend_attributes) {
    frontend_attributes_ = frontend_attributes;
  }

  // Swap the passed FrontendAttributes with the ones currently set.
  //
  // Return the old attributes.
  FrontendAttributes SwapFrontendAttributes(
      const FrontendAttributes& frontend_attributes) {
    FrontendAttributes old_attributes = std::move(frontend_attributes_);
    frontend_attributes_ = frontend_attributes;
    return old_attributes;
  }

  // Returns the FrontendAttributes that will be attached to all instructions.
  const FrontendAttributes& frontend_attributes() const {
    return frontend_attributes_;
  }

  // Clears all the frontend attributes.
  void ClearFrontendAttributes() { frontend_attributes_.Clear(); }

  // Clears the sharding. Ops will be sharded according to the default placement
  // policy.
  void ClearSharding() { sharding_ = std::nullopt; }

  // Returns the OpSharding that will be attached to all instructions.
  const std::optional<OpSharding>& sharding() const { return sharding_; }

  // Sets the builder to a mode where it will die immediately when an error is
  // encountered, rather than producing it in a deferred fashion when Build() is
  // called (which is the default).
  void set_die_immediately_on_error(bool enabled) {
    die_immediately_on_error_ = enabled;
  }

  // Returns a new ZkxBuilder whose resultant Computation is used only by this
  // ZkxBuilder. The sub-ZkxBuilder has the same die_immediately_on_error
  // behavior as the parent.
  std::unique_ptr<ZkxBuilder> CreateSubBuilder(
      const std::string& computation_name);

  // Builds the computation with the requested operations, or returns a non-ok
  // status. Note that all ops that have been enqueued will be moved to the
  // computation being returned. The root of the computation will be the last
  // added operation.
  //
  // `remove_dynamic_dimensions` tells the builder whether to remove the
  // dynamic dimensions information in all ops.
  //
  // TODO(b/121223198): Delete `remove_dynamic_dimensions` and keeps the
  // dynamic dimensions information when ZKX backend can handle dynamic
  // dimensions.
  absl::StatusOr<ZkxComputation> Build(bool remove_dynamic_dimensions = false);

  // Overload of Build which specifies a particular root instruction for the
  // computation.
  absl::StatusOr<ZkxComputation> Build(ZkxOp root,
                                       bool remove_dynamic_dimensions = false);

  // Builds the computation with the requested operations, or notes an error in
  // the parent ZkxBuilder and returns an empty computation if building failed.
  // This function is intended to be used where the returned ZkxComputation is
  // only used by the parent ZkxBuilder and hence further operation on the
  // returned ZkxComputation will simply be error'ed out if an error occurred
  // while building this computation. If the built computation is to be used by
  // a ZkxBuilder other than the parent ZkxBuilder then Build() should be used
  // instead.
  ZkxComputation BuildAndNoteError();

  // Returns a subgraph that roots on the given root. If the root is not a
  // compile-time constant (see `IsConstant`), returns an error.
  //
  // This will copy the needed ops/computations to the subgraph.
  absl::StatusOr<ZkxComputation> BuildConstantSubGraph(
      ZkxOp root_op, bool dynamic_dimension_is_minus_one = false);

  // Returns the first error that was encountered while building the
  // computation. When an error is encountered, by default we return a vacuous
  // ZkxOp and inform the user of the error that occurred while
  // building the computation when they make a final call to Build().
  //
  // See also set_die_immediately_on_error().
  absl::Status first_error() const { return first_error_; }

  // Returns the current status of the builder, complete with the stack trace
  // information.
  absl::Status GetCurrentStatus() const;

  // Returns the shape of the given op.
  absl::StatusOr<Shape> GetShape(ZkxOp op) const;

  // Returns the shape of the given op.
  virtual absl::StatusOr<const Shape*> GetShapePtr(ZkxOp op) const;

  // Returns the OpSharding of the given op. If "op" has no sharding, return
  // std::nullopt.
  absl::StatusOr<std::optional<OpSharding>> GetOpSharding(ZkxOp op) const;

  // Returns the (inferred) result for the current computation's shape. This
  // assumes the root instruction is the last added instruction.
  absl::StatusOr<ProgramShape> GetProgramShape() const;

  // Returns the (inferred) result for the current computation's shape using the
  // given operation as the root.
  absl::StatusOr<ProgramShape> GetProgramShape(ZkxOp root) const;

  // Reports an error to the builder, by
  // * storing it internally and capturing a backtrace if it's the first error
  //   (this deferred value will be produced on the call to
  //    Build()/GetShape()/...)
  // * dying if die_immediately_on_error_ is true.
  // Returns an ZkxOp with an invalid handle but a valid builder. This value can
  // be returned in place of a value in APIs that return an ZkxOp.
  ZkxOp ReportError(const absl::Status& error);

  // A helper function that converts a absl::StatusOr<ZkxOp> into an ZkxOp.
  // If the absl::Status was an error, reports the error to builder and returns
  // an invalid ZkxOp handle.
  ZkxOp ReportErrorOrReturn(const absl::StatusOr<ZkxOp>& op);

  // A helper function that runs a function that returns a absl::StatusOr<ZkxOp>
  // and returns an ZkxOp.
  ZkxOp ReportErrorOrReturn(
      absl::FunctionRef<absl::StatusOr<ZkxOp>()> op_creator);

  // Returns true if 'operand' is a compile-time constant. A compile-time
  // constant does not depend on any parameters, or on stateful operators such
  // as `RngNormal` or `Infeed`.
  //
  // This tests whether a computation is a compile-time constant without
  // evaluating the computation.
  absl::StatusOr<bool> IsConstant(ZkxOp operand) const;

  // Adds a new input/output alias. Since the input/output shape information are
  // not available until the computation is built, any eventual error in the
  // arguments of this API will be detected only at computation Build() time.
  //
  // Note: Except when 'must-alias' is true, alias is assumed to be 'may-alias'
  // and only donated buffer at runtime will be aliased with output. If a buffer
  // is not donated at runtime, a copy will be inserted by ZKX to prevent buffer
  // clobbering.
  void SetUpAlias(const ShapeIndex& output_index, int64_t param_number,
                  const ShapeIndex& param_index,
                  HloInputOutputAliasConfig::AliasKind kind =
                      HloInputOutputAliasConfig::AliasKind::kMayAlias) {
    input_output_aliases_.push_back(
        {output_index, param_number, param_index, kind});
  }

  // Describes an input/output alias as inserted by the SetUpAlias() API.
  struct InputOutputAlias {
    // Specifies the index of the aliased buffer in the result tuple.
    ShapeIndex output_index;
    // Specifies the parameter containing the buffer to be aliased.
    int64_t param_number;
    // Specifies the index of the aliased buffer in the parameter.
    ShapeIndex param_index;
    // Specifies if the alias is a must alias or may alias.
    HloInputOutputAliasConfig::AliasKind kind;
  };

  // Adds a new buffer donor. The donated buffer may be paired with any valid
  // output. On the contrary, the buffer aliasing bonds the input output pair.
  // The input can only donate the buffer to the paired output.
  void AddBufferDonor(int64_t param_number, const ShapeIndex& param_index) {
    buffer_donors_.insert({param_number, param_index});
  }

  // Looks up the HloInstruction and sets the frontend attribute "attribute" to
  // "value". If the attribute already existed, then its value is updated.
  //
  // The attribute is only added to the HloInstruction, not to the builder.
  absl::Status SetInstructionFrontendAttribute(ZkxOp op, std::string attribute,
                                               std::string value);

  // Looks up the HloInstruction and sets the sharding. If the sharding already
  // existed, then its value is updated.
  //
  // The sharding is only added to the HloInstruction, not to the builder.
  absl::Status SetInstructionSharding(
      ZkxOp op, const std::optional<OpSharding>& sharding);

  // Returns shapes for the operands.
  absl::StatusOr<std::vector<Shape>> GetOperandShapes(
      absl::Span<const ZkxOp> operands) const;

  // Converts the op to string for the ease of debugging.
  std::string OpToString(ZkxOp op) const;

 private:
  void ToStringHelper(std::string* out, int ident, int64_t op_handle) const;

  // Build helper which takes the id of the root operation..
  absl::StatusOr<ZkxComputation> Build(int64_t root_id,
                                       bool remove_dynamic_dimensions);

  // Description for the methods below can be found in the corresponding public
  // functions section in this file.

  ZkxOp Parameter(int64_t parameter_number, const Shape& shape,
                  const std::string& name,
                  const std::vector<bool>& replicated_at_leaf_buffers);
  ZkxOp Parameter(int64_t parameter_number, const Shape& shape,
                  const std::string& name) {
    std::vector<bool> empty_bools;
    return Parameter(parameter_number, shape, name, empty_bools);
  }

  virtual ZkxOp ConstantLiteral(const LiteralSlice& literal);

  ZkxOp Broadcast(ZkxOp operand, absl::Span<const int64_t> broadcast_sizes);

  ZkxOp BroadcastInDim(ZkxOp operand, absl::Span<const int64_t> out_dim_size,
                       absl::Span<const int64_t> broadcast_dimensions);

  ZkxOp Pad(ZkxOp operand, ZkxOp padding_value,
            const PaddingConfig& padding_config);
  ZkxOp PadInDim(ZkxOp operand, ZkxOp padding_value, int64_t dimno,
                 int64_t pad_lo, int64_t pad_hi);

  virtual absl::StatusOr<ZkxOp> PadInternal(
      const Shape& shape, ZkxOp operand, ZkxOp padding_value,
      const PaddingConfig& padding_config);

  ZkxOp Reshape(ZkxOp operand, absl::Span<const int64_t> dimensions,
                absl::Span<const int64_t> new_sizes,
                int64_t inferred_dimension = -1);

  ZkxOp Reshape(ZkxOp operand, absl::Span<const int64_t> new_sizes,
                int64_t inferred_dimension = -1);

  ZkxOp Reshape(const Shape& shape, ZkxOp operand,
                int64_t inferred_dimension = -1);

  ZkxOp DynamicReshape(ZkxOp operand, absl::Span<const ZkxOp> dim_sizes,
                       absl::Span<const int64_t> new_size_bounds,
                       const std::vector<bool>& dims_are_dynamic);

  ZkxOp MhloDynamicReshape(ZkxOp operand, ZkxOp output_shape,
                           const Shape& shape);

  ZkxOp Collapse(ZkxOp operand, absl::Span<const int64_t> dimensions);

  ZkxOp Slice(ZkxOp operand, absl::Span<const int64_t> start_indices,
              absl::Span<const int64_t> limit_indices,
              absl::Span<const int64_t> strides);
  virtual absl::StatusOr<ZkxOp> SliceInternal(
      const Shape& shape, ZkxOp operand,
      absl::Span<const int64_t> start_indices,
      absl::Span<const int64_t> limit_indices,
      absl::Span<const int64_t> strides);
  virtual ZkxOp SliceInDim(ZkxOp operand, int64_t start_index,
                           int64_t limit_index, int64_t stride, int64_t dimno);

  ZkxOp DynamicSlice(ZkxOp operand, absl::Span<const ZkxOp> start_indices,
                     absl::Span<const int64_t> slice_sizes);
  virtual absl::StatusOr<ZkxOp> DynamicSliceInternal(
      const Shape& shape, ZkxOp operand, absl::Span<const ZkxOp> start_indices,
      absl::Span<const int64_t> slice_sizes);

  ZkxOp DynamicUpdateSlice(ZkxOp operand, ZkxOp update,
                           absl::Span<const ZkxOp> start_indices);
  virtual absl::StatusOr<ZkxOp> DynamicUpdateSliceInternal(
      const Shape& shape, ZkxOp operand, ZkxOp update,
      absl::Span<const ZkxOp> start_indices);

  ZkxOp ConcatInDim(absl::Span<const ZkxOp> operands, int64_t dimension);
  virtual absl::StatusOr<ZkxOp> ConcatInDimInternal(
      const Shape& shape, absl::Span<const ZkxOp> operands, int64_t dimension);

  ZkxOp Select(ZkxOp pred, ZkxOp on_true, ZkxOp on_false);

  ZkxOp Tuple(absl::Span<const ZkxOp> elements);
  virtual absl::StatusOr<ZkxOp> TupleInternal(const Shape& shape,
                                              absl::Span<const ZkxOp> elements);

  ZkxOp GetTupleElement(ZkxOp tuple_data, int64_t index);
  virtual absl::StatusOr<ZkxOp> GetTupleElementInternal(const Shape& shape,
                                                        ZkxOp tuple_data,
                                                        int64_t index);

  ZkxOp Call(const ZkxComputation& computation,
             absl::Span<const ZkxOp> operands);

  ZkxOp CompositeCall(const ZkxComputation& computation,
                      absl::Span<const ZkxOp> operands, const std::string& name,
                      std::optional<std::string_view> attributes = std::nullopt,
                      std::optional<int64_t> version = std::nullopt);

  ZkxOp Reduce(ZkxOp operand, ZkxOp init_value,
               const ZkxComputation& computation,
               absl::Span<const int64_t> dimensions_to_reduce);

  ZkxOp Reduce(absl::Span<const ZkxOp> operands,
               absl::Span<const ZkxOp> init_values,
               const ZkxComputation& computation,
               absl::Span<const int64_t> dimensions_to_reduce);

  virtual absl::StatusOr<ZkxOp> ReduceInternal(
      const Shape& shape, absl::Span<const ZkxOp> all_operands,
      const ZkxComputation& computation,
      absl::Span<const int64_t> dimensions_to_reduce);

  ZkxOp ReduceWindow(ZkxOp operand, ZkxOp init_value,
                     const ZkxComputation& computation, const Window& window);

  ZkxOp ReduceWindow(absl::Span<const ZkxOp> operands,
                     absl::Span<const ZkxOp> init_values,
                     const ZkxComputation& computation, const Window& window);

  virtual absl::StatusOr<ZkxOp> ReduceWindowInternal(
      const Shape& shape, absl::Span<const ZkxOp> all_operands,
      const ZkxComputation& computation, const Window& window);

  virtual ZkxOp Iota(const Shape& shape, int64_t iota_dimension);

  ZkxOp Iota(PrimitiveType type, int64_t size);

  ZkxOp ConvertElementType(ZkxOp operand, PrimitiveType new_element_type);

  ZkxOp BitcastConvertType(ZkxOp operand, PrimitiveType new_element_type);
  virtual absl::StatusOr<ZkxOp> BitcastConvertTypeInternal(const Shape& shape,
                                                           ZkxOp operand);

  ZkxOp Transpose(ZkxOp operand, absl::Span<const int64_t> permutation);
  virtual absl::StatusOr<ZkxOp> TransposeInternal(
      const Shape& shape, ZkxOp operand, absl::Span<const int64_t> permutation);

  ZkxOp Rev(ZkxOp operand, absl::Span<const int64_t> dimensions);
  virtual absl::StatusOr<ZkxOp> RevInternal(
      const Shape& shape, ZkxOp operand, absl::Span<const int64_t> dimensions);

  ZkxOp Sort(absl::Span<const ZkxOp> operands, const ZkxComputation& comparator,
             int64_t dimension = -1, bool is_stable = false);
  virtual absl::StatusOr<ZkxOp> SortInternal(const Shape& shape,
                                             absl::Span<const ZkxOp> operands,
                                             const ZkxComputation& comparator,
                                             int64_t dimension, bool is_stable);

  ZkxOp Clamp(ZkxOp min, ZkxOp operand, ZkxOp max);

  ZkxOp Map(absl::Span<const ZkxOp> operands, const ZkxComputation& computation,
            absl::Span<const int64_t> dimensions,
            absl::Span<const ZkxOp> static_operands = {});

  ZkxOp While(const ZkxComputation& condition, const ZkxComputation& body,
              ZkxOp init);
  virtual absl::StatusOr<ZkxOp> WhileInternal(const Shape& shape,
                                              const ZkxComputation& condition,
                                              const ZkxComputation& body,
                                              ZkxOp init);

  ZkxOp Conditional(ZkxOp predicate, ZkxOp true_operand,
                    const ZkxComputation& true_computation, ZkxOp false_operand,
                    const ZkxComputation& false_computation);

  ZkxOp Conditional(ZkxOp branch_index,
                    absl::Span<const ZkxComputation* const> branch_computations,
                    absl::Span<const ZkxOp> branch_operands);

  ZkxOp Scatter(ZkxOp input, ZkxOp scatter_indices, ZkxOp updates,
                const ZkxComputation& update_computation,
                const ScatterDimensionNumbers& dimension_numbers,
                bool indices_are_sorted = false, bool unique_indices = false);
  ZkxOp Scatter(absl::Span<const ZkxOp> inputs, ZkxOp scatter_indices,
                absl::Span<const ZkxOp> updates,
                const ZkxComputation& update_computation,
                const ScatterDimensionNumbers& dimension_numbers,
                bool indices_are_sorted = false, bool unique_indices = false);

  virtual absl::StatusOr<ZkxOp> ScatterInternal(
      const Shape& shape, absl::Span<const ZkxOp> inputs, ZkxOp scatter_indices,
      absl::Span<const ZkxOp> updates, const ZkxComputation& update_computation,
      const ScatterDimensionNumbers& dimension_numbers, bool indices_are_sorted,
      bool unique_indices);

  virtual ZkxOp CreateToken();

  ZkxOp GetDimensionSize(ZkxOp operand, int64_t dimension);

  ZkxOp SetDimensionSize(ZkxOp operand, ZkxOp val, int64_t dimension);

  virtual absl::StatusOr<ZkxOp> SetDimensionSizeInternal(const Shape& shape,
                                                         ZkxOp operand,
                                                         ZkxOp val,
                                                         int64_t dimension);

  ZkxOp RemoveDynamicDimension(ZkxOp operand, int64_t dimension);

  virtual absl::StatusOr<ZkxOp> AddInstruction(
      HloInstructionProto&& instr, HloOpcode opcode,
      absl::Span<const ZkxOp> operands);
  absl::StatusOr<ZkxOp> AddInstruction(HloInstructionProto&& instr,
                                       HloOpcode opcode) {
    return AddInstruction(std::move(instr), opcode, /*operands=*/{});
  }

  void AddCalledComputation(const ZkxComputation& computation,
                            HloInstructionProto* instr);

  absl::StatusOr<const HloInstructionProto*> LookUpInstruction(ZkxOp op) const;
  absl::StatusOr<const HloInstructionProto*> LookUpInstructionByHandle(
      int64_t handle) const;
  absl::StatusOr<HloInstructionProto*> LookUpMutableInstruction(ZkxOp op);
  absl::StatusOr<HloInstructionProto*> LookUpMutableInstructionByHandle(
      int64_t handle);

  // Internal helper method that does the building for an arbitrary unary op.
  virtual ZkxOp UnaryOp(HloOpcode unop, ZkxOp operand);

  // Internal helper method that does the building for an arbitrary binary op.
  // broadcast_dimensions specifies which dimensions to use for broadcasting
  // when the operation is between tensors of different ranks. The direction is
  // only used if opcode is kCompare.
  ZkxOp BinaryOp(HloOpcode binop, ZkxOp lhs, ZkxOp rhs,
                 absl::Span<const int64_t> broadcast_dimensions,
                 std::optional<ComparisonDirection> direction = std::nullopt);

  // Internal helper method for binary op compare without broadcast dimensions.
  virtual absl::StatusOr<ZkxOp> Compare(const Shape& shape, ZkxOp lhs,
                                        ZkxOp rhs,
                                        ComparisonDirection direction);

  // Internal helper method that does the building for an arbitrary binary op
  // with same ranked operands that doesn't broadcast.
  virtual ZkxOp BinaryOpNoBroadcast(HloOpcode binop, const Shape& shape,
                                    ZkxOp lhs, ZkxOp rhs);

  // Internal helper method that does the building for an arbitrary ternary op.
  ZkxOp TernaryOp(HloOpcode triop, ZkxOp lhs, ZkxOp rhs, ZkxOp ehs);

  virtual absl::StatusOr<ZkxOp> InDimBroadcast(
      const Shape& shape, ZkxOp operand,
      absl::Span<const int64_t> broadcast_dimensions);

  // Internal helper method that creates a sequence of instructions that
  // performs an explicit broadcast of the operand to the target shape.
  // All dimensions of the operand must either be equal to the corresponding
  // output shape dimension, or be exactly 1. (Such dimensions are the
  // degenerate dimensions.)
  absl::StatusOr<ZkxOp> AddBroadcastSequence(const Shape& output_shape,
                                             ZkxOp operand);

  // Internal helper method that broadcasts a scalar to the shape of the output.
  absl::StatusOr<ZkxOp> BroadcastScalarToOutputShape(ZkxOp scalar,
                                                     ZkxOp output);

  // Internal helper method for creating a Reshape op with the already inferred
  // shape.
  virtual absl::StatusOr<ZkxOp> ReshapeInternal(const Shape& shape,
                                                ZkxOp operand,
                                                int64_t inferred_dimension);

  // Returns the (inferred) result for the program shape using the given root.
  absl::StatusOr<ProgramShape> GetProgramShape(int64_t root_id) const;

  // A visitor which checks whether an operation is a compile-time constant,
  // meaning that it doesn't depend on any parameters, or on any stateful
  // operation such as `RngNormal` or `Infeed`. The visitor walks the
  // computation starting at a given operation and sets is_constant to false iff
  // a parameter or stateful operation is encountered.
  void IsConstantVisitor(int64_t op_handle, int depth,
                         absl::flat_hash_set<int64_t>* visited,
                         bool* is_constant) const;

  int64_t GetNextId() { return ++next_id_; }

  // Populates the module with the input/output alias information stored within
  // the input_output_aliases vector.
  static absl::Status PopulateInputOutputAliasAndBufferDonor(
      HloModuleProto* module, const ProgramShape& program_shape,
      const std::vector<InputOutputAlias>& input_output_aliases,
      const absl::flat_hash_set<HloBufferDonorConfig::BufferDonor>&
          buffer_donors);

  std::string name_;  // Name to use for the built computation.

  // The next sequential ID for every instruction/computation contained within
  // this computation.
  int64_t next_id_ = 0;

  // The first error encountered while building the computation.
  // This is OK until the first error is encountered.
  absl::Status first_error_;

  // The saved stack trace from the point at which the first error occurred.
  // TODO(chokobole): Uncomment this. Dependency: SavedStackTrace, How about
  // implementing this with abseil stack trace?
  // tsl::SavedStackTrace first_error_backtrace_;

  // The instructions of this computation.
  // Use a deque so pointers into this are stable, for example the return
  // value of LookUpInstructionByHandle().
  std::deque<HloInstructionProto> instructions_;
  // A cache for the HloInstructionProto shapes, to avoid recreating Shape
  // objects from protos and to support the GetShapePtr() API.
  std::vector<std::unique_ptr<Shape>> instruction_shapes_;

  // Holds the input/output alias information populated by the SetUpAlias() API.
  std::vector<InputOutputAlias> input_output_aliases_;

  // Holds the buffer donor information populated by the AddBufferDonor() API.
  absl::flat_hash_set<HloBufferDonorConfig::BufferDonor> buffer_donors_;

  // A map from ZkxOp::Handle to the index in the instructions_ vector where the
  // instruction is held.
  absl::flat_hash_map<int64_t, int64_t> handle_to_index_;

  // Track imported instructions by their computation id and the position in
  // their computation's instruction list.
  struct ImportedInstruction {
    int64_t computation_id;
    int64_t instruction_index;
  };

  absl::flat_hash_map<int64_t, ImportedInstruction> handle_to_imported_index_;

  // The embedded computations used by this computation. Each computation was
  // the entry computation of some ZkxComputation, the key is the unique id of
  // that ZkxComputation.
  std::map<int64_t, HloComputationProto> embedded_;

  // The unique parameter numbers.
  absl::flat_hash_set<int64_t> parameter_numbers_;

  // The metadata to attach to each op. This is structured as a "modal"-like
  // operation, in order to simplify client code (and not sprinkle this metadata
  // throughout the TensorFlow op kernel implementations).
  OpMetadata metadata_;

  // A temporary metadata that will only be applied to the next op created.
  std::optional<OpMetadata> one_shot_metadata_;

  // Sharding for this operator. This is structured as a "model"-like operation,
  // in order to simplify client code, similar to metadata_.
  std::optional<OpSharding> sharding_;

  // Mode bit that indicates whether to die when a first error is encountered.
  bool die_immediately_on_error_ = false;

  ZkxBuilder* parent_builder_{nullptr};

  FrontendAttributes frontend_attributes_;

  // If the user cannot provide a token for infeed/outfeed, assume they are
  // being added to the computation in the correct order. Implicitly reuse
  // the tokens from the previous op to guarantee the user intended ordering.
  ZkxOp infeed_token_;
  ZkxOp outfeed_token_;

  friend ZkxOp Parameter(ZkxBuilder* builder, int64_t parameter_number,
                         const Shape& shape, const std::string& name,
                         const std::vector<bool>& replicated_at_leaf_buffers);
  friend ZkxOp ConstantLiteral(ZkxBuilder* builder,
                               const LiteralSlice& literal);

  friend ZkxOp Broadcast(ZkxOp operand,
                         absl::Span<const int64_t> broadcast_sizes);

  friend ZkxOp BroadcastInDim(ZkxOp operand,
                              absl::Span<const int64_t> out_dim_size,
                              absl::Span<const int64_t> broadcast_dimensions);

  friend ZkxOp Copy(ZkxOp operand);

  friend ZkxOp Pad(ZkxOp operand, ZkxOp padding_value,
                   const PaddingConfig& padding_config);

  friend ZkxOp PadInDim(ZkxOp operand, ZkxOp padding_value, int64_t dimno,
                        int64_t pad_lo, int64_t pad_hi);

  friend ZkxOp Reshape(ZkxOp operand, absl::Span<const int64_t> dimensions,
                       absl::Span<const int64_t> new_sizes);

  friend ZkxOp Reshape(ZkxOp operand, absl::Span<const int64_t> new_sizes);

  friend ZkxOp Reshape(const Shape& shape, ZkxOp operand);

  friend ZkxOp DynamicReshape(ZkxOp operand, absl::Span<const ZkxOp> dim_sizes,
                              absl::Span<const int64_t> new_size_bounds,
                              const std::vector<bool>& dims_are_dynamic);

  friend ZkxOp ReshapeWithInferredDimension(ZkxOp operand,
                                            absl::Span<const int64_t> new_sizes,
                                            int64_t inferred_dimension);

  friend ZkxOp Collapse(ZkxOp operand, absl::Span<const int64_t> dimensions);

  friend ZkxOp Slice(ZkxOp operand, absl::Span<const int64_t> start_indices,
                     absl::Span<const int64_t> limit_indices,
                     absl::Span<const int64_t> strides);

  friend ZkxOp SliceInDim(ZkxOp operand, int64_t start_index,
                          int64_t limit_index, int64_t stride, int64_t dimno);

  friend ZkxOp DynamicSlice(ZkxOp operand,
                            absl::Span<const ZkxOp> start_indices,
                            absl::Span<const int64_t> slice_sizes);

  friend ZkxOp DynamicUpdateSlice(ZkxOp operand, ZkxOp update,
                                  absl::Span<const ZkxOp> start_indices);

  friend ZkxOp ConcatInDim(ZkxBuilder* builder,
                           absl::Span<const ZkxOp> operands, int64_t dimension);

  friend ZkxOp Select(ZkxOp pred, ZkxOp on_true, ZkxOp on_false);
  friend ZkxOp Tuple(ZkxBuilder* builder, absl::Span<const ZkxOp> elements);
  friend ZkxOp GetTupleElement(ZkxOp tuple_data, int64_t index);
  friend ZkxOp Compare(ZkxOp lhs, ZkxOp rhs,
                       absl::Span<const int64_t> broadcast_dimensions,
                       ComparisonDirection direction);

  friend ZkxOp Call(ZkxBuilder* builder, const ZkxComputation& computation,
                    absl::Span<const ZkxOp> operands);

  friend ZkxOp CompositeCall(ZkxBuilder* builder,
                             const ZkxComputation& computation,
                             absl::Span<const ZkxOp> operands,
                             const std::string& name,
                             std::optional<std::string_view> attributes,
                             std::optional<int64_t> version);

  friend ZkxOp Add(ZkxOp lhs, ZkxOp rhs,
                   absl::Span<const int64_t> broadcast_dimensions);
  friend ZkxOp Sub(ZkxOp lhs, ZkxOp rhs,
                   absl::Span<const int64_t> broadcast_dimensions);
  friend ZkxOp Mul(ZkxOp lhs, ZkxOp rhs,
                   absl::Span<const int64_t> broadcast_dimensions);
  friend ZkxOp Div(ZkxOp lhs, ZkxOp rhs,
                   absl::Span<const int64_t> broadcast_dimensions);
  friend ZkxOp Rem(ZkxOp lhs, ZkxOp rhs,
                   absl::Span<const int64_t> broadcast_dimensions);
  friend ZkxOp Max(ZkxOp lhs, ZkxOp rhs,
                   absl::Span<const int64_t> broadcast_dimensions);
  friend ZkxOp Min(ZkxOp lhs, ZkxOp rhs,
                   absl::Span<const int64_t> broadcast_dimensions);
  friend ZkxOp And(ZkxOp lhs, ZkxOp rhs,
                   absl::Span<const int64_t> broadcast_dimensions);
  friend ZkxOp Or(ZkxOp lhs, ZkxOp rhs,
                  absl::Span<const int64_t> broadcast_dimensions);
  friend ZkxOp Xor(ZkxOp lhs, ZkxOp rhs,
                   absl::Span<const int64_t> broadcast_dimensions);
  friend ZkxOp Not(ZkxOp operand);
  friend ZkxOp PopulationCount(ZkxOp operand);
  friend ZkxOp ShiftLeft(ZkxOp lhs, ZkxOp rhs,
                         absl::Span<const int64_t> broadcast_dimensions);
  friend ZkxOp ShiftRightArithmetic(
      ZkxOp lhs, ZkxOp rhs, absl::Span<const int64_t> broadcast_dimensions);
  friend ZkxOp ShiftRightLogical(
      ZkxOp lhs, ZkxOp rhs, absl::Span<const int64_t> broadcast_dimensions);
  friend ZkxOp Reduce(ZkxOp operand, ZkxOp init_value,
                      const ZkxComputation& computation,
                      absl::Span<const int64_t> dimensions_to_reduce);
  friend ZkxOp Reduce(ZkxBuilder* builder, absl::Span<const ZkxOp> operands,
                      absl::Span<const ZkxOp> init_values,
                      const ZkxComputation& computation,
                      absl::Span<const int64_t> dimensions_to_reduce);
  friend ZkxOp ReduceWindow(ZkxOp operand, ZkxOp init_value,
                            const ZkxComputation& computation,
                            const Window& window);
  friend ZkxOp ReduceWindow(ZkxBuilder* builder,
                            absl::Span<const ZkxOp> operands,
                            absl::Span<const ZkxOp> init_values,
                            const ZkxComputation& computation,
                            const Window& window);

  friend ZkxOp Abs(ZkxOp operand);
  friend ZkxOp Sign(ZkxOp operand);
  friend ZkxOp Clz(ZkxOp operand);
  friend ZkxOp Pow(ZkxOp lhs, ZkxOp rhs,
                   absl::Span<const int64_t> broadcast_dimensions);
  friend ZkxOp Iota(ZkxBuilder* builder, const Shape& shape,
                    int64_t iota_dimension);
  friend ZkxOp Iota(ZkxBuilder* builder, PrimitiveType type, int64_t size);
  friend ZkxOp ConvertElementType(ZkxOp operand,
                                  PrimitiveType new_element_type);
  friend ZkxOp BitcastConvertType(ZkxOp operand,
                                  PrimitiveType new_element_type);
  friend ZkxOp Neg(ZkxOp operand);
  friend ZkxOp Transpose(ZkxOp operand, absl::Span<const int64_t> permutation);
  friend ZkxOp Rev(ZkxOp operand, absl::Span<const int64_t> dimensions);
  friend ZkxOp Sort(absl::Span<const ZkxOp> operands,
                    const ZkxComputation& comparator, int64_t dimension,
                    bool is_stable);
  friend ZkxOp Clamp(ZkxOp min, ZkxOp operand, ZkxOp max);
  friend ZkxOp Map(ZkxBuilder* builder, absl::Span<const ZkxOp> operands,
                   const ZkxComputation& computation,
                   absl::Span<const int64_t> dimensions,
                   absl::Span<const ZkxOp> static_operands);
  friend ZkxOp While(const ZkxComputation& condition,
                     const ZkxComputation& body, ZkxOp init);
  friend ZkxOp Conditional(ZkxOp predicate, ZkxOp true_operand,
                           const ZkxComputation& true_computation,
                           ZkxOp false_operand,
                           const ZkxComputation& false_computation);
  friend ZkxOp Conditional(
      ZkxOp branch_index,
      absl::Span<const ZkxComputation* const> branch_computations,
      absl::Span<const ZkxOp> branch_operands);
  friend ZkxOp ConditionalImpl(
      ZkxOp branch_index,
      absl::Span<const ZkxComputation* const> branch_computations,
      absl::Span<const ZkxOp> branch_operands);
  friend ZkxOp Scatter(ZkxOp input, ZkxOp scatter_indices, ZkxOp updates,
                       const ZkxComputation& update_computation,
                       const ScatterDimensionNumbers& dimension_numbers,
                       bool indices_are_sorted, bool unique_indices);
  friend ZkxOp Scatter(absl::Span<const ZkxOp> inputs, ZkxOp scatter_indices,
                       absl::Span<const ZkxOp> updates,
                       const ZkxComputation& update_computation,
                       const ScatterDimensionNumbers& dimension_numbers,
                       bool indices_are_sorted, bool unique_indices);
  friend ZkxOp CreateToken(ZkxBuilder* builder);

  friend ZkxOp GetDimensionSize(ZkxOp operand, int64_t dimension);
  friend ZkxOp SetDimensionSize(ZkxOp operand, ZkxOp val, int64_t dimension);
  friend ZkxOp RemoveDynamicDimension(ZkxOp operand, int64_t dimension);

 protected:
  // Returns OK status if the given op was built using this builder. Otherwise,
  // returns an error.
  absl::Status CheckOpBuilder(ZkxOp op) const;

 private:
  ZkxOp ConditionalImpl(
      ZkxOp branch_index,
      absl::Span<const ZkxComputation* const> branch_computations,
      absl::Span<const ZkxOp> branch_operands);

  // Creates an op with the given opcode and the output shape.
  virtual absl::StatusOr<ZkxOp> AddOpWithShape(
      HloOpcode opcode, const Shape& shape, absl::Span<const ZkxOp> operands);

  // Here, InstructionType is either const HloInstructionProto* or non-const
  // HloInstructionProto*.
  template <typename InstructionType>
  absl::StatusOr<InstructionType> LookUpInstructionByHandleInternal(
      int64_t handle) const {
    auto it = handle_to_index_.find(handle);
    if (it == handle_to_index_.end()) {
      // Try look for the instruction in the imported instructions.
      auto imported_it = handle_to_imported_index_.find(handle);
      if (imported_it != handle_to_imported_index_.end()) {
        ImportedInstruction imported = imported_it->second;
        return const_cast<InstructionType>(
            &embedded_.at(imported.computation_id)
                 .instructions(imported.instruction_index));
      }
      return absl::InvalidArgumentError(
          absl::StrFormat("No ZkxOp with handle %d", handle));
    }
    return const_cast<InstructionType>(&instructions_.at(it->second));
  }

  // Here, InstructionType is either const HloInstructionProto* or non-const
  // HloInstructionProto*.
  //
  // TODO(hinsu): Return const pointer within absl::StatusOr and use
  // absl::implicit_cast at callsites. This requires implicit_cast support in
  // absl::StatusOr similar to absl::StatusOr.
  template <typename InstructionType>
  absl::StatusOr<InstructionType> LookUpInstructionInternal(ZkxOp op) const {
    TF_RETURN_IF_ERROR(CheckOpBuilder(op));
    return LookUpInstructionByHandleInternal<InstructionType>(op.handle());
  }

  friend struct internal::ZkxBuilderFriend;

  friend class ValueInference;
};

// RAII-style object: sets the current sharding assignment in builder on
// construction, and sets back to the previous assignment on destruction.
class ZkxScopedShardingAssignment {
 public:
  ZkxScopedShardingAssignment(ZkxBuilder* builder,
                              std::optional<OpSharding> sharding)
      : builder_(builder), prev_sharding_(builder->sharding()) {
    SetSharding(sharding);
  }

  ZkxScopedShardingAssignment(const ZkxScopedShardingAssignment&) = delete;
  ZkxScopedShardingAssignment& operator=(const ZkxScopedShardingAssignment&) =
      delete;

  ~ZkxScopedShardingAssignment() { SetSharding(prev_sharding_); }

 private:
  void SetSharding(const std::optional<OpSharding>& sharding) {
    if (sharding.has_value()) {
      builder_->SetSharding(sharding.value());
    } else {
      builder_->ClearSharding();
    }
  }

  ZkxBuilder* const builder_;
  std::optional<OpSharding> prev_sharding_;
};

// RAII-style object: save the current builder's frontend attributes, and merge
// them with the new ones on construction.
// Restore the original attributes on destruction.
class ZkxScopedFrontendAttributesAssignment {
 public:
  ZkxScopedFrontendAttributesAssignment(ZkxBuilder* builder,
                                        FrontendAttributes attributes)
      : builder_(builder) {
    saved_ = builder_->SwapFrontendAttributes(attributes);
  }

  ~ZkxScopedFrontendAttributesAssignment() {
    builder_->SetFrontendAttributes(saved_);
  }

 private:
  ZkxBuilder* const builder_;
  FrontendAttributes saved_;

  ZkxScopedFrontendAttributesAssignment(
      const ZkxScopedFrontendAttributesAssignment&) = delete;
  ZkxScopedFrontendAttributesAssignment& operator=(
      const ZkxScopedFrontendAttributesAssignment&) = delete;
};

// RAII-style object: sets the current op metadata in builder on construction,
// and sets back to the previous assignment on destruction.
class ZkxScopedOpMetadataAssignment {
 public:
  ZkxScopedOpMetadataAssignment(ZkxBuilder* builder, OpMetadata metadata)
      : builder_(builder) {
    saved_ = builder_->SwapOpMetadata(metadata);
  }

  ~ZkxScopedOpMetadataAssignment() { builder_->SwapOpMetadata(saved_); }

 private:
  ZkxBuilder* const builder_;
  OpMetadata saved_;

  ZkxScopedOpMetadataAssignment(const ZkxScopedOpMetadataAssignment&) = delete;
  ZkxScopedOpMetadataAssignment& operator=(
      const ZkxScopedOpMetadataAssignment&) = delete;
};

// Free functions for building ZkxOps. The intention is that these will
// become the public API for building ZkxOps rather than calling methods on
// ZkxBuilder directly.
//

// Enqueues a "retrieve parameter value" instruction for a parameter that was
// passed to the computation.
ZkxOp Parameter(ZkxBuilder* builder, int64_t parameter_number,
                const Shape& shape, const std::string& name);

// Same as above, but with leaf buffer replication annotation.
ZkxOp Parameter(ZkxBuilder* builder, int64_t parameter_number,
                const Shape& shape, const std::string& name,
                const std::vector<bool>& replicated_at_leaf_buffers);

// Enqueues a constant with the value of the given literal onto the
// computation.
ZkxOp ConstantLiteral(ZkxBuilder* builder, const LiteralSlice& literal);

// Enqueues a constant onto the computation. Methods are templated on the
// native host type (NativeT) which corresponds to a specific ZKX
// PrimitiveType as given in the following table:
//
//  Native Type   PrimitiveType
// -----------------------------
//   bool           PRED
//   int32_t        S32
//   int64_t        S64
//   uint32_t       U32
//   uint64_t       U64
//   float          F32
//   double         F64
//
// Note: not all primitive types defined in zkx_data.proto have a
// corresponding native type yet.
template <typename NativeT>
ZkxOp ConstantR0(ZkxBuilder* builder, NativeT value);
template <typename NativeT>
ZkxOp ConstantR1(ZkxBuilder* builder, absl::Span<const NativeT> values);
ZkxOp ConstantR1(ZkxBuilder* builder, const tsl::core::Bitmap& values);
template <typename NativeT>
ZkxOp ConstantR2(ZkxBuilder* builder,
                 std::initializer_list<std::initializer_list<NativeT>> values);
template <typename NativeT>
ZkxOp ConstantFromArrayWithLayout(ZkxBuilder* builder,
                                  const Array<NativeT>& values,
                                  const Layout& layout);
template <typename NativeT>
ZkxOp ConstantFromArray(ZkxBuilder* builder, const Array<NativeT>& values);
template <typename NativeT>
ZkxOp ConstantR2FromArray2DWithLayout(ZkxBuilder* builder,
                                      const Array2D<NativeT>& values,
                                      const Layout& layout);
template <typename NativeT>
ZkxOp ConstantR2FromArray2D(ZkxBuilder* builder,
                            const Array2D<NativeT>& values);
template <typename NativeT>
ZkxOp ConstantR3FromArray3DWithLayout(ZkxBuilder* builder,
                                      const Array3D<NativeT>& values,
                                      const Layout& layout);
template <typename NativeT>
ZkxOp ConstantR3FromArray3D(ZkxBuilder* builder,
                            const Array3D<NativeT>& values);

// Enqueues a rank one constant (ZkxBuilder* builder, vector) onto the
// computation. The vector has size 'length' and every element has the value
// 'value'.
template <typename NativeT>
ZkxOp ConstantR1(ZkxBuilder* builder, int64_t length, NativeT value);

// Adds dimensions to an array by duplicating the data in the array.
//
// The new dimensions are inserted on the left, i.e. if
// broadcast_sizes has values {a0, ..., aN} and the operand shape
// has dimensions {b0, ..., bM} then the shape of the output has
// dimensions {a0, ..., aN, b0, ..., bM}.
//
// The new dimensions index into copies of the operand, i.e.
//
//   output[i0, ..., iN, j0, ..., jM] = operand[j0, ..., jM]
ZkxOp Broadcast(ZkxOp operand, absl::Span<const int64_t> broadcast_sizes);

// This op broadcasts the `operand` to an output with the given `shape`.
// `broadcast_dimensions` are the dimensions to be broadcasting into, i.e., the
// i'th dimension of the operand is mapped to the broadcast_dimensions[i]'th
// dimension of the output. This also requires that the i'th input dimension is
// either 1 or is the same as the output dimension it's broadcasting into.
//
// For example, say operand = {1, 2}, i.e., a 1D tensor in shape s32[2]; the
// output shape is s32[2,2]:
// - Specifying {1} as broadcast_dimension will generate output
//   {{1, 2},
//    {1, 2}}
// - On the other hand, specifying {0} as broadcast_dimension
//   will generate output
//   {{1 , 1},
//    {2 , 2}}
ZkxOp BroadcastInDim(ZkxOp operand, absl::Span<const int64_t> out_dim_size,
                     absl::Span<const int64_t> broadcast_dimensions);

// Copies the input operand to the output. This operation is for internal
// purpose and is only used by the compiler for optimization purposes or to
// ensure correctness. The ZKX client should never have to generate this
// instruction.
//
// Copy has two potential use cases:
//
// * Create a copy of the operand with a new layout.
//
// * Create a copy of the operand in a separately allocated buffer. This is
//   necessary for some backends if the operand is a parameter or constant and
//   the operand is returned within a tuple. In this case, the lifetime of the
//   operand buffer must be the same as the lifetime of the output result.
//   However, the lifetimes of parameters and constants are managed separately
//   from the lifetime of the output result. Creating a separate copy of the
//   parameter or constant buffer resolves this issue.
ZkxOp Copy(ZkxOp operand);

// Enqueues a pad operation onto the computation that pads the given value on
// the edges as well as between the elements of the input. padding_config
// specifies the padding amount for each dimension.
ZkxOp Pad(ZkxOp operand, ZkxOp padding_value,
          const PaddingConfig& padding_config);

// Enqueues a pad operation in a given dimension, taking all other
// dimensions as they are.
ZkxOp PadInDim(ZkxOp operand, ZkxOp padding_value, int64_t dimno,
               int64_t pad_lo, int64_t pad_hi);

// Enqueues an operation onto the computation that flattens the operand based
// on the dimension order (major/slowest-varying to minor/fastest-varying)
// given, followed by reshaping it into the shape with the given dimension
// sizes (also major to minor). Conceptually, this is a limited form of
// "shape casting".
ZkxOp Reshape(ZkxOp operand, absl::Span<const int64_t> dimensions,
              absl::Span<const int64_t> new_sizes);

// Enqueues a dynamic reshape operation. The dynamic reshape takes additional
// ZkxOps as sizes for the result dimension. The result dim i is a dynamic
// dimension dimension if dims_are_dynamic[i] is true.
ZkxOp DynamicReshape(ZkxOp operand, absl::Span<const ZkxOp> dim_sizes,
                     absl::Span<const int64_t> new_size_bounds,
                     const std::vector<bool>& dims_are_dynamic);

// This is an experimental API for creating the mhlo.dynamic_reshape op from the
// ZkxBuilder. This is only intended for export to MHLO or StableHLO, and cannot
// be compiled.
ZkxOp MhloDynamicReshape(ZkxOp operand, ZkxOp output_shape, const Shape& shape);

// Enqueues an operation onto the computation that collapses the operand,
// from first to last dimension (C order), then reshapes it to the given
// dimension sizes. Conceptually, this is a limited form of "shape casting".
ZkxOp Reshape(ZkxOp operand, absl::Span<const int64_t> new_sizes);

// Enqueues a Reshape op that uses an explicit target shape.
ZkxOp Reshape(const Shape& shape, ZkxOp operand);

// `inferred_dimension` represents the output dimension that's inferred by
// upper-level framework by dividing the input element count by the known
// output element count. While an inferred_dimension can be static, if there
// is a dynamic dimension in the output, it must be the inferred dimension.
ZkxOp ReshapeWithInferredDimension(ZkxOp operand,
                                   absl::Span<const int64_t> new_sizes,
                                   int64_t inferred_dimension);

// Wrapper for Reshape.
// Enqueues an operation to collapse the provided dimensions; e.g. an
// operand with dimensions {x=256, y=2, z=2, p=32} can be collapsed to
// {x=1024, y=32} by collapsing dims {0, 1, 2}. Collapsing dimensions must
// be a consecutive, in-order subsequence of the operand dimensions.
//
// Note that collapsing a single dimension does nothing:
//
//    {256} collapsing {0} => {256}
//    {1} collapsing {0} => {1}
//
// Collapsing multiple dimensions produces a single result dimension:
//
//    {256, 2} collapsing {0,1} => {512}
//    {256, 2, 3} collapsing {0,1} => {512, 3}
//
// This could potentially cause data to be moved -- it provides a more
// structured form of reshaping than an arbitrary Reshape operation.
ZkxOp Collapse(ZkxOp operand, absl::Span<const int64_t> dimensions);

// Enqueues a slice operation onto the computation that slices the operand
// from the start indices to the limit indices; e.g.
//
//        x
//   [ 0 1 2 3 ]
// y [ 4 5 6 7 ] => slice(start={1, 1}, limit={2, 3}) => [ 5 6 ]
//   [ 8 9 a b ]
//
// Note that "limit" means up-to-but-not-including; i.e. [start, limit) in 1D
// range notation.
// The strides parameter determines the stride over the slice
ZkxOp Slice(ZkxOp operand, absl::Span<const int64_t> start_indices,
            absl::Span<const int64_t> limit_indices,
            absl::Span<const int64_t> strides);

// Enqueues a slice operation in a given dimension, taking all other
// dimensions as they are; e.g. if dimno is 1 from start_index 2 to
// limit_index 4 by 1, and the shape is f32[7,8,9], this call is short-hand
// for:
//
//  array[:, 2:4:1, :]
ZkxOp SliceInDim(ZkxOp operand, int64_t start_index, int64_t limit_index,
                 int64_t stride, int64_t dimno);

// Enqueues a slice operation onto the computation that slices the 'operand'
// from dynamic start indices which are passed in 'start_indices'.
// The size of the slice in each dimension is passed in 'slice_sizes',
// which specify the end point of exclusive slice intervals in each
// dimension [start, start + size).
// The shape of each element of 'start_indices' must be scalar, with the span
// size equal to the rank of the 'operand'. All elements of 'start_indices' must
// have the same shape.
// Slice index calculations are computed modulo input dimension sizes to
// prevent dynamic start indices from generating out-of-bound array accesses.
ZkxOp DynamicSlice(ZkxOp operand, absl::Span<const ZkxOp> start_indices,
                   absl::Span<const int64_t> slice_sizes);

// Enqueues a dynamic update slice operation onto the computation, which
// updates a slice of 'operand' with 'update' at dynamic 'start_indices'.
// The shape of 'update' determines the shape of the slice of 'operand'
// which is updated.
// The indices specified in 'start_indices' specify the offset of the slice
// of 'operand' which is updated.
//
//               update = {10, 11} // calculated at runtime.
//   [1 2 3]     start  = {1, 1}   // calculated at runtime.  [1 2  3 ]
//   [4 5 6]  => DynamicUpdateslice(data, update, start)   => [4 10 11]
//   [7 8 9]                                                  [7 8  9 ]
//
// The shape of each element of 'start_indices' must be scalar, with the span
// size equal to the rank of the 'operand'. All elements of 'start_indices' must
// have the same shape.
// Slice index calculations are computed modulo update dimension sizes to
// prevent dynamic start indices from generating out-of-bound array accesses.
ZkxOp DynamicUpdateSlice(ZkxOp operand, ZkxOp update,
                         absl::Span<const ZkxOp> start_indices);

// Enqueues a concatenate instruction onto the computation. 'operands' must
// have >= 1 entry.
ZkxOp ConcatInDim(ZkxBuilder* builder, absl::Span<const ZkxOp> operands,
                  int64_t dimension);

// Enqueues a conditional-move-like select operation onto the computation;
// predicated on pred, selects between on_true and on_false.
ZkxOp Select(ZkxOp pred, ZkxOp on_true, ZkxOp on_false);

// Enqueues a tuple-creation instruction onto the computation.
ZkxOp Tuple(ZkxBuilder* builder, absl::Span<const ZkxOp> elements);

// Enqueues a tuple-element-get instruction onto the computation.
ZkxOp GetTupleElement(ZkxOp tuple_data, int64_t index);

// Enqueues an equal-to comparison instruction onto the computation.
ZkxOp Eq(ZkxOp lhs, ZkxOp rhs,
         absl::Span<const int64_t> broadcast_dimensions = {});

// Enqueues a not-equal comparison instruction onto the computation.
ZkxOp Ne(ZkxOp lhs, ZkxOp rhs,
         absl::Span<const int64_t> broadcast_dimensions = {});

// Enqueues a greater-or-equal comparison instruction onto the computation.
ZkxOp Ge(ZkxOp lhs, ZkxOp rhs,
         absl::Span<const int64_t> broadcast_dimensions = {});

// Enqueues a greater-than comparison instruction onto the computation.
ZkxOp Gt(ZkxOp lhs, ZkxOp rhs,
         absl::Span<const int64_t> broadcast_dimensions = {});

// Enqueues a less-than comparison instruction onto the computation.
ZkxOp Lt(ZkxOp lhs, ZkxOp rhs,
         absl::Span<const int64_t> broadcast_dimensions = {});

// Enqueues a less-or-equal comparison instruction onto the computation.
ZkxOp Le(ZkxOp lhs, ZkxOp rhs,
         absl::Span<const int64_t> broadcast_dimensions = {});

// Enqueues a comparison instruction onto the computation (optionally without
// broadcast_dimensions for consistency with others).
ZkxOp Compare(ZkxOp lhs, ZkxOp rhs,
              absl::Span<const int64_t> broadcast_dimensions,
              ComparisonDirection direction);
ZkxOp Compare(ZkxOp lhs, ZkxOp rhs, ComparisonDirection direction);

// Enqueues a call instruction onto the computation.
ZkxOp Call(ZkxBuilder* builder, const ZkxComputation& computation,
           absl::Span<const ZkxOp> operands);

// Enqueues a composite call instruction onto the computation.
ZkxOp CompositeCall(ZkxBuilder* builder, const ZkxComputation& computation,
                    absl::Span<const ZkxOp> operands, const std::string& name,
                    std::optional<std::string_view> attributes = std::nullopt,
                    std::optional<int64_t> version = std::nullopt);

// Enqueues an add instruction onto the computation.
ZkxOp Add(ZkxOp lhs, ZkxOp rhs,
          absl::Span<const int64_t> broadcast_dimensions = {});

// Enqueues a subtract instruction onto the computation.
ZkxOp Sub(ZkxOp lhs, ZkxOp rhs,
          absl::Span<const int64_t> broadcast_dimensions = {});

// Enqueues a multiply instruction onto the computation.
ZkxOp Mul(ZkxOp lhs, ZkxOp rhs,
          absl::Span<const int64_t> broadcast_dimensions = {});

// Enqueues a divide instruction onto the computation.
ZkxOp Div(ZkxOp lhs, ZkxOp rhs,
          absl::Span<const int64_t> broadcast_dimensions = {});

// Enqueues a remainder instruction onto the computation.
ZkxOp Rem(ZkxOp lhs, ZkxOp rhs,
          absl::Span<const int64_t> broadcast_dimensions = {});

// Enqueues a max instruction onto the computation.
ZkxOp Max(ZkxOp lhs, ZkxOp rhs,
          absl::Span<const int64_t> broadcast_dimensions = {});

// Enqueues a min instruction onto the computation.
ZkxOp Min(ZkxOp lhs, ZkxOp rhs,
          absl::Span<const int64_t> broadcast_dimensions = {});

// Element-wise logical operators
ZkxOp And(ZkxOp lhs, ZkxOp rhs,
          absl::Span<const int64_t> broadcast_dimensions = {});

// Overload to call And with 3 or more operands. We need the following somewhat
// convoluted overload set to disambiguate with the overload that takes the
// `broadcast_dimensions` optional param.
inline ZkxOp And(ZkxOp op1, ZkxOp op2, ZkxOp op3) {
  return And(op1, And(op2, op3));
}
template <typename... ZkxOpTs>
ZkxOp And(ZkxOp op1, ZkxOp op2, ZkxOp op3, const ZkxOpTs&... operands) {
  return And(op1, And(op2, And(op3, operands...)));
}

ZkxOp Or(ZkxOp lhs, ZkxOp rhs,
         absl::Span<const int64_t> broadcast_dimensions = {});

// Overload to call Or with 3 or more operands. As with `And`, we need the
// following complicated overload set to handle the default arg in the `Or`
// overload above.
inline ZkxOp Or(ZkxOp op1, ZkxOp op2, ZkxOp op3) {
  return Or(op1, Or(op2, op3));
}
template <typename... ZkxOpTs>
ZkxOp Or(ZkxOp op1, ZkxOp op2, ZkxOp op3, const ZkxOpTs&... operands) {
  return Or(op1, Or(op2, Or(op3, operands...)));
}

ZkxOp Xor(ZkxOp lhs, ZkxOp rhs,
          absl::Span<const int64_t> broadcast_dimensions = {});

ZkxOp Not(ZkxOp operand);

ZkxOp PopulationCount(ZkxOp operand);

ZkxOp ShiftLeft(ZkxOp lhs, ZkxOp rhs,
                absl::Span<const int64_t> broadcast_dimensions = {});
ZkxOp ShiftRightArithmetic(ZkxOp lhs, ZkxOp rhs,
                           absl::Span<const int64_t> broadcast_dimensions = {});
ZkxOp ShiftRightLogical(ZkxOp lhs, ZkxOp rhs,
                        absl::Span<const int64_t> broadcast_dimensions = {});
// Reduces an array among the provided dimensions, given "computation" as a
// reduction operator.
ZkxOp Reduce(ZkxOp operand, ZkxOp init_value, const ZkxComputation& computation,
             absl::Span<const int64_t> dimensions_to_reduce);

// Reduces several arrays simultaneously among the provided dimensions, given
// "computation" as a reduction operator.
ZkxOp Reduce(ZkxBuilder* builder, absl::Span<const ZkxOp> operands,
             absl::Span<const ZkxOp> init_values,
             const ZkxComputation& computation,
             absl::Span<const int64_t> dimensions_to_reduce);

// Applies a reduction function to windows of the operand.
ZkxOp ReduceWindow(ZkxOp operand, ZkxOp init_value,
                   const ZkxComputation& computation, const Window& window);

// Reduces several arrays simultaneously over windows.
ZkxOp ReduceWindow(ZkxBuilder* builder, absl::Span<const ZkxOp> operands,
                   absl::Span<const ZkxOp> init_values,
                   const ZkxComputation& computation, const Window& window);

// Enqueues an abs instruction onto the computation.
ZkxOp Abs(ZkxOp operand);

// Enqueues a sign instruction onto the computation.
ZkxOp Sign(ZkxOp operand);

// Enqueues a count leading zeros instruction onto the computation.
ZkxOp Clz(ZkxOp operand);

// Enqueues a lhs^rhs computation onto the computation.
ZkxOp Pow(ZkxOp lhs, ZkxOp rhs,
          absl::Span<const int64_t> broadcast_dimensions = {});

// Enqueues an iota operation onto the computation.
ZkxOp Iota(ZkxBuilder* builder, const Shape& shape, int64_t iota_dimension);

// Enqueues a rank-1 iota operation onto the computation.
ZkxOp Iota(ZkxBuilder* builder, PrimitiveType type, int64_t size);

// Enqueues a convert instruction onto the computation that changes the
// element type of the operand array to primitive_type.
ZkxOp ConvertElementType(ZkxOp operand, PrimitiveType new_element_type);

// Enqueues a no-op instruction onto the computation that changes
// the element type of the operand array to primitive_type. The
// bit-widths of the source and destination element types must be
// identical.
ZkxOp BitcastConvertType(ZkxOp operand, PrimitiveType new_element_type);

// Enqueues a negate instruction onto the computation.
ZkxOp Neg(ZkxOp operand);

// Enqueues a transpose instruction onto the computation.
ZkxOp Transpose(ZkxOp operand, absl::Span<const int64_t> permutation);

// Enqueues a reverse instruction onto the computation. The order of the
// elements in the given dimensions is reversed (i.e., the element at index i
// is moved to index dimension_size - 1 - i).
ZkxOp Rev(ZkxOp operand, absl::Span<const int64_t> dimensions);

// Enqueues a sort instruction onto the computation, using 'comparator' for
// comparisons. 'comparator' needs to define a strict weak order. 'is_stable'
// determines whether the stable sorting should be used.
// If only one operand is provided:
// * If the operand is a rank-1 tensor (an array), the result is a sorted array.
//   The resulting sorting order has the property that for all index positions
//   i, j with i < j, either
//   comparator(value[i], value[j]) = comparator(value[j], value[i]) = false or
//   comparator(value[i], value[j]) = true.
// * If the operand has higher rank, the operand is sorted along the provided
//   dimension. For example, for a rank-2 tensor (a matrix), a dimension value
//   of 0 will independently sort every column, and a dimension value of 1 will
//   independently sort each row. If no dimension number is provided, then the
//   last dimension is chosen by default. For the dimension which is sorted, the
//   same sorting order applies as in the rank-1 case.
//
// If more than one operand is provided:
// * All operands must be tensors with the same dimensions. The element types of
//   the tensors may be different.
// * The result is a tuple that consists of the operands in sorted order (along
//   the provided dimension, as above). The same permutation as implied by the
//   comparison computation is applied to all operand tensors. When comparing
//   two index positions, 'comparator' is called with 2 * n scalar parameters,
//   where parameter 2 * i and 2 * i + 1 correspond to the value of operand i at
//   two index positions.
// Default comparator computations can be found in lib/comparators.h
ZkxOp Sort(absl::Span<const ZkxOp> operands, const ZkxComputation& comparator,
           int64_t dimension = -1, bool is_stable = false);

// Enqueues a clamp instruction onto the computation.
ZkxOp Clamp(ZkxOp min, ZkxOp operand, ZkxOp max);

// Enqueues a map instruction onto the computation.
ZkxOp Map(ZkxBuilder* builder, absl::Span<const ZkxOp> operands,
          const ZkxComputation& computation,
          absl::Span<const int64_t> dimensions,
          absl::Span<const ZkxOp> static_operands = {});

// Enqueues a while node onto the computation.
ZkxOp While(const ZkxComputation& condition, const ZkxComputation& body,
            ZkxOp init);

// Enqueues a conditional node onto the computation.
ZkxOp Conditional(ZkxOp predicate, ZkxOp true_operand,
                  const ZkxComputation& true_computation, ZkxOp false_operand,
                  const ZkxComputation& false_computation);

// Enqueues either a predicated (if/else) or indexed (switch/case/default)
// conditional node onto the computation. N >= 1 branch_computations and
// branch_operands are matched by index. branch_index selects the branch that
// will be executed. Out of range branch_index uses the N-1'th
// branch_computation as default.
ZkxOp Conditional(ZkxOp branch_index,
                  absl::Span<const ZkxComputation* const> branch_computations,
                  absl::Span<const ZkxOp> branch_operands);

// Enqueues a Scatter node onto the computation.
ZkxOp Scatter(ZkxOp input, ZkxOp scatter_indices, ZkxOp updates,
              const ZkxComputation& update_computation,
              const ScatterDimensionNumbers& dimension_numbers,
              bool indices_are_sorted = false, bool unique_indices = false);
ZkxOp Scatter(absl::Span<const ZkxOp> inputs, ZkxOp scatter_indices,
              absl::Span<const ZkxOp> updates,
              const ZkxComputation& update_computation,
              const ScatterDimensionNumbers& dimension_numbers,
              bool indices_are_sorted = false, bool unique_indices = false);

// Enqueues an operation (AfterAll) with no operands that produces a
// token-shaped value. Tokens are used for ordering side-effecting operations.
// This is a separate method from AfterAll to facility the removal of
// operand-less AfterAll instructions.
// TODO(b/110532604): Remove this function when all tokens are derived from a
// single token generated or passed into the entry computation.
ZkxOp CreateToken(ZkxBuilder* builder);
// Returns the size of the given dimension of the operand. The operand must be
// array shaped.
ZkxOp GetDimensionSize(ZkxOp operand, int64_t dimension);

// Sets the size of the given dimension of the operand. The operand must be
// array shaped. The result will have the same shape as the operand, but the
// given dimension will be dynamic (if not already).
ZkxOp SetDimensionSize(ZkxOp operand, ZkxOp val, int64_t dimension);

// Returns the same op but with dynamic dimension removed.
ZkxOp RemoveDynamicDimension(ZkxOp operand, int64_t dimension);

// Implementation details below this point.
//

// Free function template implementations.

template <typename NativeT>
ZkxOp ConstantR0(ZkxBuilder* builder, NativeT value) {
  return ConstantLiteral(builder, LiteralUtil::CreateR0<NativeT>(value));
}

template <typename NativeT>
ZkxOp ConstantR1(ZkxBuilder* builder, absl::Span<const NativeT> values) {
  BorrowingLiteral literal(
      reinterpret_cast<const char*>(values.begin()),
      ShapeUtil::MakeShape(primitive_util::NativeToPrimitiveType<NativeT>(),
                           {static_cast<int64_t>(values.size())}));
  return ConstantLiteral(builder, literal);
}

template <typename NativeT>
ZkxOp ConstantR1(ZkxBuilder* builder, int64_t length, NativeT value) {
  Literal literal(ShapeUtil::MakeShape(
      primitive_util::NativeToPrimitiveType<NativeT>(), {length}));
  literal.PopulateWithValue(value);
  return ConstantLiteral(builder, literal);
}

inline ZkxOp ConstantR1(ZkxBuilder* builder, const tsl::core::Bitmap& values) {
  return ConstantLiteral(builder, LiteralUtil::CreateR1(values));
}

template <typename NativeT>
ZkxOp ConstantR2(ZkxBuilder* builder,
                 std::initializer_list<std::initializer_list<NativeT>> values) {
  return ConstantLiteral(builder, LiteralUtil::CreateR2<NativeT>(values));
}

template <typename NativeT>
ZkxOp ConstantFromArrayWithLayout(ZkxBuilder* builder,
                                  const Array<NativeT>& values,
                                  const Layout& layout) {
  return ConstantLiteral(
      builder, LiteralUtil::CreateFromArrayWithLayout<NativeT>(values, layout));
}

template <typename NativeT>
ZkxOp ConstantFromArray(ZkxBuilder* builder, const Array<NativeT>& values) {
  return ConstantLiteral(builder,
                         LiteralUtil::CreateFromArray<NativeT>(values));
}

template <typename NativeT>
ZkxOp ConstantR2FromArray2DWithLayout(ZkxBuilder* builder,
                                      const Array2D<NativeT>& values,
                                      const Layout& layout) {
  return ConstantLiteral(
      builder, LiteralUtil::CreateFromArrayWithLayout<NativeT>(values, layout));
}

template <typename NativeT>
ZkxOp ConstantR2FromArray2D(ZkxBuilder* builder,
                            const Array2D<NativeT>& values) {
  return ConstantLiteral(builder,
                         LiteralUtil::CreateR2FromArray2D<NativeT>(values));
}

template <typename NativeT>
ZkxOp ConstantR3FromArray3DWithLayout(ZkxBuilder* builder,
                                      const Array3D<NativeT>& values,
                                      const Layout& layout) {
  return ConstantLiteral(
      builder,
      LiteralUtil::CreateR3FromArray3DWithLayout<NativeT>(values, layout));
}

template <typename NativeT>
ZkxOp ConstantR3FromArray3D(ZkxBuilder* builder,
                            const Array3D<NativeT>& values) {
  return ConstantFromArray(builder, values);
}

}  // namespace zkx

#endif  // ZKX_HLO_BUILDER_ZKX_BUILDER_H_
