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

#include "zkx/literal.h"

#include <string.h>

#include "absl/container/inlined_vector.h"
#include "absl/debugging/leak_check.h"
#include "absl/numeric/bits.h"

#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/mem.h"
#include "xla/tsl/util/byte_swap_array.h"
#include "zk_dtypes/include/all_types.h"
#include "zkx/permutation_util.h"

namespace zkx {

namespace {

constexpr bool kLittleEndian = absl::little_endian::IsLittleEndian();

// Converts between little and big endian.
//
// Precondition: size % 2 == 0 (elements in the array are 16 bits long)
void ConvertEndianShort(std::string* bytes) {
  CHECK_EQ(bytes->size() % 2, 0);
  for (int64_t i = 0, end = bytes->size(); i < end; i += 2) {
    std::swap((*bytes)[i], (*bytes)[i + 1]);
  }
}

void ConvertEndianShort(char* bytes, int64_t size) {
  CHECK_EQ(size % 2, 0);
  for (int64_t i = 0; i < size; i += 2) {
    std::swap(bytes[i], bytes[i + 1]);
  }
}

bool LiteralProtoHasValues(const LiteralProto& proto) {
  return !proto.s1s().empty() || !proto.s2s().empty() || !proto.s4s().empty() ||
         !proto.s8s().empty() || !proto.s16s().empty() || proto.s32s_size() ||
         proto.s64s_size() || !proto.u1s().empty() || !proto.u2s().empty() ||
         !proto.u4s().empty() || !proto.u8s().empty() ||
         !proto.u16s().empty() || proto.u32s_size() || proto.u64s_size() ||
         proto.preds_size() || proto.tuple_literals_size();
}

// TODO(chokobole): Update the comment below if we don't have HloEvaluator in
// the end.
// Lazy getter for the interned scalar shape in static storage. We reuse this
// shape pointer to when constructing scalar Literals, which can happen a lot
// when we are evaluating reduce-like ops in HloEvaluator, and copying the
// shape over and over again significantly slows down the evaluator.
template <PrimitiveType kType>
const Shape& ScalarShapeImpl() {
  static_assert(primitive_util::IsArrayType(kType),
                "Not a valid type for a scalar.");
  static const Shape* shape = [] {
    auto shape = absl::IgnoreLeak(new Shape(kType, {}, {}, {}));
    shape->mutable_layout();
    return shape;
  }();
  return *shape;
}

const Shape& ScalarShape(PrimitiveType type) {
  return primitive_util::ArrayTypeSwitch<const Shape&>(
      [&](auto primitive_type_constant) -> const Shape& {
        return ScalarShapeImpl<primitive_type_constant>();
      },
      type);
}

const Shape& NilShape() {
  static const Shape* shape = absl::IgnoreLeak(new Shape(TUPLE, {}, {}, {}));
  return *shape;
}

// Returns the interned shape pointer in static storage if it's a scalar shape
// or nil shape.
const Shape* TryInternShape(const Shape& shape) {
  if (shape.IsTuple() && shape.tuple_shapes_size() == 0) {
    return &NilShape();
  }
  if (shape.IsArray() && shape.dimensions_size() == 0 && shape.is_static() &&
      shape.has_layout() && shape.layout().tiles_size() == 0 &&
      shape.layout().memory_space() == 0 &&
      shape.layout().element_size_in_bits() == 0) {
    return &ScalarShape(shape.element_type());
  }
  return nullptr;
}

// Utility structure which is used to create the optimal configuration for
// a ShapeUtil::ForEachIndex() scan across two literals.
struct StrideConfig {
  StrideConfig(const Shape& source_shape, const Shape& dest_shape,
               absl::Span<const int64_t> dimensions);

  // The dimensions of the stride operation. Essentially every dimension
  // will be iterated from base[i] to base[i]+dimensions[i], in step[i]
  // steps.
  absl::Span<const int64_t> dimensions;
  DimensionVector base;
  DimensionVector step;
  int64_t minor_dimension = 0;
  // The size of the strides for source and destination. One of the two
  // (the one looping through its most minor dimension) will be 1, while
  // the other will be the stride size at the dimension matching the other
  // shape most minor dimension being scanned.
  int64_t dest_stride = 1;
  int64_t source_stride = 1;
  // The size of the inner loop on the most minor dimension.
  int64_t minor_loop_size = 1;
};

StrideConfig::StrideConfig(const Shape& source_shape, const Shape& dest_shape,
                           absl::Span<const int64_t> dimensions)
    : dimensions(dimensions),
      base(dimensions.size(), 0),
      step(dimensions.size(), 1) {
  if (!dimensions.empty()) {
    // Selects the shape with the largest minor dimension as the one upon
    // which to run the tight stride loop.
    if (dimensions[LayoutUtil::Minor(source_shape.layout(), 0)] >=
        dimensions[LayoutUtil::Minor(dest_shape.layout(), 0)]) {
      minor_dimension = LayoutUtil::Minor(source_shape.layout(), 0);
      dest_stride = IndexUtil::GetDimensionStride(dest_shape, minor_dimension);
    } else {
      minor_dimension = LayoutUtil::Minor(dest_shape.layout(), 0);
      source_stride =
          IndexUtil::GetDimensionStride(source_shape, minor_dimension);
    }
    minor_loop_size = dimensions[minor_dimension];
    step[minor_dimension] = minor_loop_size;
  }
}

}  // namespace

LiteralBase::~LiteralBase() = default;

namespace {

template <int64_t PRIMITIVE_SIZE>
absl::StatusOr<Literal> BroadcastHelper(const LiteralBase& src,
                                        const Shape& src_shape,
                                        const Shape& result_shape,
                                        absl::Span<const int64_t> dimensions) {
  for (int64_t i = 0, end = dimensions.size(); i < end; i++) {
    TF_RET_CHECK(src_shape.dimensions(i) ==
                 result_shape.dimensions(dimensions[i]));
  }

  TF_RET_CHECK(result_shape.element_type() == src_shape.element_type());
  Literal result(result_shape);
  if (src_shape.is_dynamic()) {
    for (int64_t i = 0; i < dimensions.size(); ++i) {
      if (src_shape.is_dynamic_dimension(i)) {
        // Set any dynamic sizes in the new literal.
        int64_t dynamic_size = src.GetDynamicSize(i);
        result.SetDynamicSize(dimensions[i], dynamic_size);
      }
    }
  }

  // scratch_source_index is temporary storage space for the computed index into
  // the input literal. We put it here to avoid allocating an std::vector in
  // every iteration of ShapeUtil::ForEachIndex.
  int src_shape_dims = src_shape.dimensions_size();
  std::vector<int64_t> scratch_source_index(src_shape_dims);
  // Make the span once outside the ForEachIndex... loop, pointing into
  // scratch_source_index
  absl::Span<int64_t> scratch_source_span(scratch_source_index);
  int64_t* scratch_source_array = scratch_source_span.data();

  const char* source_data = static_cast<const char*>(src.untyped_data());
  char* dest_data = static_cast<char*>(result.untyped_data());

  auto src_minor_to_major = LayoutUtil::MinorToMajor(src_shape);
  auto result_minor_to_major = LayoutUtil::MinorToMajor(result_shape);

  ShapeUtil::ForEachIndexNoStatus(
      result_shape, [&](absl::Span<const int64_t> output_index) {
        // Compute dest_index
        int64_t dest_index = IndexUtil::MultidimensionalIndexToLinearIndex(
            result_shape, result_minor_to_major, output_index);

        // Compute source_index
        int64_t source_index;
        for (int64_t i = 0, end = dimensions.size(); i < end; ++i) {
          scratch_source_array[i] = output_index[dimensions[i]];
        }
        if (src_shape_dims == 1) {
          // Fast path for this case
          source_index = scratch_source_array[0];
          DCHECK_EQ(source_index,
                    IndexUtil::MultidimensionalIndexToLinearIndex(
                        src_shape, src_minor_to_major, scratch_source_span));
        } else {
          source_index = IndexUtil::MultidimensionalIndexToLinearIndex(
              src_shape, src_minor_to_major, scratch_source_span);
        }
        // Move one element from source_index in source to dest_index in dest
        memcpy(dest_data + PRIMITIVE_SIZE * dest_index,
               source_data + PRIMITIVE_SIZE * source_index, PRIMITIVE_SIZE);
        return true;
      });

  return std::move(result);
}

}  // namespace

absl::StatusOr<Literal> LiteralBase::Broadcast(
    const Shape& result_shape, absl::Span<const int64_t> dimensions) const {
  const LiteralBase& src = *this;
  const Shape& src_shape = shape();
  if (!src_shape.IsArray()) {
    return absl::InvalidArgumentError("Broadcast only supports arrays.");
  }
  const int64_t primitive_size =
      ShapeUtil::ByteSizeOfPrimitiveType(src_shape.element_type());

  switch (primitive_size) {
    case 0:
      return BroadcastHelper<0>(src, src_shape, result_shape, dimensions);
    case 1:
      return BroadcastHelper<1>(src, src_shape, result_shape, dimensions);
    case 2:
      return BroadcastHelper<2>(src, src_shape, result_shape, dimensions);
    case 4:
      return BroadcastHelper<4>(src, src_shape, result_shape, dimensions);
    case 8:
      return BroadcastHelper<8>(src, src_shape, result_shape, dimensions);
    case 16:
      return BroadcastHelper<16>(src, src_shape, result_shape, dimensions);
    case 32:
      return BroadcastHelper<32>(src, src_shape, result_shape, dimensions);
    case 64:
      return BroadcastHelper<64>(src, src_shape, result_shape, dimensions);
    case 128:
      return BroadcastHelper<128>(src, src_shape, result_shape, dimensions);
    default:
      LOG(FATAL) << "Unhandled primitive size " << primitive_size;
      return absl::InvalidArgumentError("Unhandled primitive size");
  }
}

absl::StatusOr<Literal> LiteralBase::Reshape(
    absl::Span<const int64_t> dimensions) const {
  if (!LayoutUtil::IsDenseArray(shape())) {
    return absl::InvalidArgumentError(
        "Reshape is only supported for dense arrays.");
  }
  if (shape().is_dynamic()) {
    // TODO(b/243182930): We should consider supporting dynamic reshape.
    return absl::UnimplementedError("Dynamic reshape is not implemented.");
  }
  Literal output;
  if (!LayoutUtil::IsMonotonicWithDim0Major(shape().layout())) {
    output = Relayout(LayoutUtil::GetDefaultLayoutForRank(shape().rank()));
  } else {
    output = Clone();
  }
  // Because the layout is monotonic, we can simply reuse the same sequence of
  // values without changing their order.
  *output.mutable_shape_do_not_use() =
      ShapeUtil::MakeShape(shape().element_type(), dimensions);

  int64_t elements_before = ShapeUtil::ElementsIn(shape());
  int64_t elements_after = ShapeUtil::ElementsIn(output.shape());
  if (elements_before != elements_after) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Shapes before and after Literal::Reshape have different numbers "
        "of elements: %s vs %s.",
        ShapeUtil::HumanString(shape()),
        ShapeUtil::HumanString(output.shape())));
  }
  return std::move(output);
}

Literal LiteralBase::Transpose(absl::Span<const int64_t> permutation) const {
  CHECK(LayoutUtil::IsDenseArray(shape()))
      << __func__ << " is only supported for dense arrays: " << shape();
  CHECK(shape().rank() == permutation.size() && IsPermutation(permutation))
      << "Given permutation is not a permutation of dimension numbers";
  // To transpose the array, we just permute the dimensions and layout, and
  // do a straight memory copy of the raw data set.
  // This is considerably faster than iterating over every array element using
  // the EachCell<>() and Set<>() APIs.
  //
  // NOTE(chokobole): Deduplicate XLA logic by delegating to
  // ShapeUtil::PermuteDimensions(...) to compute the transposed shape *and*
  // its layout. When the input shape has a layout, PermuteDimensions updates
  // minor_to_major as `new = inv(P) ∘ old` and ensures the transpose is a
  // bitcast—exactly what the inlined code below used to do.
  //
  // References (pinned to the same commit):
  // - https://github.com/openxla/xla/blob/f281e568/xla/literal.cc#L1255-L1261
  // clang-format off
  // - https://github.com/openxla/xla/blob/f281e568/xla/shape_util.cc#L1307-L1313
  // clang-format on
  Shape owned_shape;
  if (!shape().has_layout()) {
    owned_shape = shape();
    LayoutUtil::SetToDefaultLayout(&owned_shape);
  }
  const Shape& shape_ref = shape().has_layout() ? shape() : owned_shape;
  Shape permuted_shape = ShapeUtil::PermuteDimensions(permutation, shape_ref);

  Literal new_literal(permuted_shape);
  if (shape().is_dynamic()) {
    std::vector<int64_t> inverse_permutation = InversePermutation(permutation);
    for (int64_t i = 0; i < shape().rank(); i++) {
      if (shape().is_dynamic_dimension(i)) {
        // Set the dynamic size of any dynamic dimension in the transposed
        // literal.
        new_literal.SetDynamicSize(inverse_permutation[i], GetDynamicSize(i));
      }
    }
  }
  DCHECK_EQ(ShapeUtil::ByteSizeOf(new_literal.shape()),
            ShapeUtil::ByteSizeOf(shape()));
  std::memcpy(new_literal.untyped_data(), untyped_data(), size_bytes());
  return new_literal;
}

namespace {

template <typename NativeT>
void SliceInternal(const LiteralBase& src_literal,
                   absl::Span<const int64_t> start_indices,
                   Literal& result_literal) {
  const Shape& result_shape = result_literal.shape();
  DimensionVector new_indices(result_shape.rank());
  CHECK_OK(
      result_literal.Populate<NativeT>([&](absl::Span<const int64_t> indices) {
        for (int64_t i = 0; i < result_shape.rank(); ++i) {
          new_indices[i] = indices[i] + start_indices[i];
        }
        return src_literal.Get<NativeT>(new_indices);
      }));
  for (int64_t dnum = 0; dnum < src_literal.shape().rank(); ++dnum) {
    if (src_literal.shape().is_dynamic_dimension(dnum)) {
      int64_t dynamic_size =
          src_literal.GetDynamicSize(dnum) - start_indices[dnum];
      CHECK_GE(dynamic_size, 0) << src_literal.GetDynamicSize(dnum);
      dynamic_size = std::min(dynamic_size, result_shape.dimensions(dnum));
      result_literal.SetDynamicSize(dnum, dynamic_size);
    }
  }
}

}  // namespace

Literal LiteralBase::Slice(absl::Span<const int64_t> start_indices,
                           absl::Span<const int64_t> limit_indices) const {
  CHECK(shape().IsArray()) << "tuple is not supported for slice";

  DimensionVector result_dimensions;
  for (int64_t dnum = 0; dnum < shape().rank(); ++dnum) {
    CHECK_GE(start_indices[dnum], 0);
    CHECK_LE(limit_indices[dnum], shape().dimensions(dnum))
        << "dnum = " << dnum;
    int64_t dimension = limit_indices[dnum] - start_indices[dnum];
    CHECK_GE(dimension, 0) << "dnum = " << dnum;
    result_dimensions.push_back(dimension);
  }
  auto result_shape = ShapeUtil::MakeShapeWithDenseLayout(
      shape().element_type(), result_dimensions,
      LayoutUtil::MinorToMajor(shape()));
  ShapeUtil::CopyDynamicDimensions(&result_shape, shape());
  Literal result_literal(result_shape);
  primitive_util::ArrayTypeSwitch<void>(
      [&](auto primitive_type_constant) -> void {
        using NativeT = primitive_util::NativeTypeOf<primitive_type_constant>;
        return SliceInternal<NativeT>(*this, start_indices, result_literal);
      },
      result_shape.element_type());
  return result_literal;
}

namespace {

// Computes the bit-reversed value for a given index.
// Uses compiler builtin when available for better performance.
uint64_t BitRev(uint64_t n) {
#if defined(__clang__) && __has_builtin(__builtin_bitreverse64)
  return __builtin_bitreverse64(n);
#else
  uint64_t rev = 0;
  for (int i = 0; i < 64; ++i) {
    rev <<= 1;
    if (n & 1) rev |= 1;
    n >>= 1;
  }
  return rev;
#endif
}

// Computes the bit-reversed index for a given index within a range of size n.
// n must be a power of 2.
int64_t BitReverseIndex(int64_t index, int64_t n) {
  CHECK_GT(n, 0);
  CHECK_EQ(n & (n - 1), 0) << "n must be a power of 2";
  int64_t log_n = absl::bit_width(static_cast<uint64_t>(n)) - 1;
  uint64_t reversed = BitRev(static_cast<uint64_t>(index));
  // Shift right to get only log_n bits
  return static_cast<int64_t>(reversed >> (64 - log_n));
}

template <typename NativeT>
void BitReverseInternal(const LiteralBase& src_literal,
                        absl::Span<const int64_t> dimensions,
                        Literal& result_literal) {
  const Shape& src_shape = src_literal.shape();
  DimensionVector src_indices(src_shape.rank());
  // Precompute dimension sizes for the dimensions to reverse
  std::vector<int64_t> dim_sizes;
  dim_sizes.reserve(dimensions.size());
  for (int64_t dim : dimensions) {
    dim_sizes.push_back(src_shape.dimensions(dim));
  }
  CHECK_OK(
      result_literal.Populate<NativeT>([&](absl::Span<const int64_t> indices) {
        for (int64_t i = 0; i < src_shape.rank(); ++i) {
          src_indices[i] = indices[i];
        }
        // Apply bit-reversal to each specified dimension
        for (size_t i = 0; i < dimensions.size(); ++i) {
          int64_t dim = dimensions[i];
          src_indices[dim] = BitReverseIndex(indices[dim], dim_sizes[i]);
        }
        return src_literal.Get<NativeT>(src_indices);
      }));
}

}  // namespace

Literal LiteralBase::BitReverse(absl::Span<const int64_t> dimensions) const {
  CHECK(shape().IsArray()) << "tuple is not supported for bit-reverse";
  for (int64_t dimension : dimensions) {
    CHECK_GE(dimension, 0);
    CHECK_LT(dimension, shape().rank());
    int64_t dim_size = shape().dimensions(dimension);
    CHECK_GT(dim_size, 0);
    CHECK_EQ(dim_size & (dim_size - 1), 0)
        << "dimension size must be a power of 2, got " << dim_size;
  }

  Literal result_literal(shape());
  primitive_util::ArrayTypeSwitch<void>(
      [&](auto primitive_type_constant) -> void {
        using NativeT = primitive_util::NativeTypeOf<primitive_type_constant>;
        return BitReverseInternal<NativeT>(*this, dimensions, result_literal);
      },
      shape().element_type());
  return result_literal;
}

Literal LiteralBase::Clone() const {
  Literal result(shape());
  CHECK_OK(result.CopyFrom(*this));
  return result;
}

std::unique_ptr<Literal> LiteralBase::CloneToUnique() const {
  auto result = std::make_unique<Literal>(shape());
  CHECK_OK(result->CopyFrom(*this));
  return result;
}

bool LiteralBase::IsDetermined(const ShapeIndex& shape_index) const {
  return piece(shape_index).IsDetermined();
}

bool LiteralBase::IsKnown(const ShapeIndex& shape_index) const {
  return piece(shape_index).IsKnown();
}

std::string LiteralBase::GetAsString(absl::Span<const int64_t> multi_index,
                                     const ShapeIndex& shape_index) const {
  const Shape& subshape = ShapeUtil::GetSubshape(shape(), shape_index);
  CHECK(LayoutUtil::IsDenseArray(subshape));
  return primitive_util::ArrayTypeSwitch<std::string>(
      [&](auto primitive_type_constant) -> std::string {
        using NativeT = primitive_util::NativeTypeOf<primitive_type_constant>;
        if constexpr (primitive_util::IsIntegralType(primitive_type_constant)) {
          // TODO(chokobole): XLA just uses absl::StrCat(). Maybe our absl is
          // too old...
          if constexpr (std::is_same_v<NativeT, u1> ||
                        std::is_same_v<NativeT, s1> ||
                        std::is_same_v<NativeT, u2> ||
                        std::is_same_v<NativeT, s2> ||
                        std::is_same_v<NativeT, u4> ||
                        std::is_same_v<NativeT, s4> ||
                        std::is_same_v<NativeT, u128> ||
                        std::is_same_v<NativeT, u256> ||
                        std::is_same_v<NativeT, s128> ||
                        std::is_same_v<NativeT, s256>) {
            return Get<NativeT>(multi_index, shape_index).ToString();
          } else {
            return absl::StrCat(Get<NativeT>(multi_index, shape_index));
          }
        }
        // TODO(chokobole): XLA just uses absl::StrCat(). Maybe our absl is
        // too old...
        if constexpr (primitive_util::IsFieldType(primitive_type_constant) ||
                      primitive_util::IsEcPointType(primitive_type_constant)) {
          return Get<NativeT>(multi_index, shape_index).ToString();
        }
        if constexpr (primitive_type_constant == PRED) {
          return Get<bool>(multi_index, shape_index) ? "true" : "false";
        }
        LOG(FATAL) << PrimitiveType_Name(subshape.element_type());
      },
      subshape.element_type());
}

std::optional<int64_t> LiteralBase::GetIntegralAsS64(
    absl::Span<const int64_t> multi_index) const {
  CHECK(LayoutUtil::IsDenseArray(shape()));
  return primitive_util::PrimitiveTypeSwitch<std::optional<int64_t>>(
      [&](auto primitive_type_constant) -> std::optional<int64_t> {
        if constexpr (primitive_util::IsIntegralType(primitive_type_constant) ||
                      primitive_type_constant == PRED) {
          using NativeT = primitive_util::NativeTypeOf<primitive_type_constant>;
          if constexpr (primitive_util::IsBigIntType(primitive_type_constant)) {
            auto value = Get<NativeT>(multi_index);
            // Check if value fits in int64_t (upper limbs must be zero)
            for (size_t i = 1; i < NativeT::kLimbNums; ++i) {
              if (value[i] != 0) {
                return std::nullopt;
              }
            }
            // Also check MSB of lowest limb for signed conversion
            uint64_t lower = static_cast<uint64_t>(value);
            if (lower >
                static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
              return std::nullopt;
            }
            return static_cast<int64_t>(lower);
          } else {
            return Get<NativeT>(multi_index);
          }
        }
        return std::nullopt;
      },
      shape().element_type());
}

namespace {

void PrintShape(bool print_layout, const Shape& shape, Printer* printer) {
  if (print_layout) {
    ShapeUtil::PrintHumanStringWithLayout(printer, shape);
  } else {
    ShapeUtil::PrintHumanString(printer, shape);
  }
}

void PrintHelper(const LiteralBase& literal, const ShapeIndex& shape_index,
                 bool print_shape, bool print_layout, bool oneline,
                 Printer* printer);

void TuplePrintHelper(const LiteralBase& literal, const ShapeIndex& shape_index,
                      bool print_shape, bool print_layout, bool oneline,
                      Printer* printer) {
  const Shape& subshape = ShapeUtil::GetSubshape(literal.shape(), shape_index);
  printer->Append(oneline ? "( " : "(\n");
  for (int i = 0; i < ShapeUtil::TupleElementCount(subshape); ++i) {
    ShapeIndex element_index = shape_index;
    element_index.push_back(i);
    if (i > 0) printer->Append(oneline ? ", " : ",\n");
    PrintHelper(literal, element_index, print_shape, print_layout, oneline,
                printer);
  }
  printer->Append(oneline ? " )" : "\n)");
}

void DenseArrayPrintHelper(const LiteralBase& literal,
                           const ShapeIndex& shape_index, bool print_shape,
                           bool print_layout, bool oneline, Printer* printer) {
  const Shape& subshape = ShapeUtil::GetSubshape(literal.shape(), shape_index);
  int64_t rank = subshape.rank();
  const std::string_view linebreak = oneline ? " " : "\n";

  std::function<void(absl::Span<const int64_t> dimensions,
                     std::vector<int64_t>*)>
      print_recursive = [&](absl::Span<const int64_t> dimensions,
                            std::vector<int64_t>* accum_indices) {
        // dimensions.size() decreases by 1 at each recursive call,
        // and accum_indices->size() increases by 1.
        // Their sum is equal to the rank of the tensor.
        CHECK_EQ(rank, dimensions.size() + accum_indices->size());

        auto brace_to_string = [&](std::string brace) -> std::string {
          // Handle 1D tensor
          if (rank == 1) {
            return brace;
          }
          // Handle the innermost tensor of a 2D+ tensor.
          if (dimensions.size() == 1 && brace == "{") {
            return absl::StrCat(oneline ? "" : "  ", brace,
                                dimensions[0] <= 1 ? "" : " ");
          }
          if (dimensions.size() == 1 && brace == "}") {
            return absl::StrCat(dimensions[0] <= 1 ? "" : " ", brace);
          }
          // Handle the non-innermost tensors of a 2D+ tensor.
          if (brace == "{") {
            const int64_t accum_indices_size = accum_indices->size();
            if (rank > 3 && !accum_indices->empty() &&
                accum_indices_size < rank) {
              int index = accum_indices->size() - 1;
              int value = accum_indices->back();
              int size = dimensions.front();
              return absl::StrCat(brace, " /*i", index, "=", value, "*/",
                                  size > 0 ? linebreak : "");
            }
            return absl::StrCat(brace, linebreak);
          }
          return absl::StrCat(linebreak, brace);
        };

        if (dimensions.empty()) {
          // Display predicates as 0s and 1s so that the string is more dense.
          std::string elem;
          if (subshape.element_type() == PRED && rank > 0) {
            elem = literal.Get<bool>(*accum_indices, shape_index) ? "1" : "0";
          } else {
            elem = literal.GetAsString(*accum_indices, shape_index);
          }
          printer->Append(elem);
        } else {
          printer->Append(brace_to_string("{"));
          for (int i = 0; i < dimensions[0]; ++i) {
            accum_indices->push_back(i);
            print_recursive(dimensions.subspan(1), accum_indices);
            accum_indices->pop_back();
            if (i < dimensions[0] - 1) {
              printer->Append(",");
              printer->Append(dimensions.size() > 1 ? linebreak : " ");
            }
          }
          printer->Append(brace_to_string("}"));
        }
      };

  if (print_shape) {
    PrintShape(print_layout, subshape, printer);
    if (subshape.is_dynamic()) {
      printer->Append("(");
      for (int64_t i = 0; i < subshape.dimensions_size(); ++i) {
        printer->Append(literal.GetDynamicSize(i, shape_index));
        if (i < subshape.dimensions_size() - 1) {
          printer->Append(",");
        }
      }
      printer->Append(")");
    }
    printer->Append(" ");
  }
  std::vector<int64_t> indices = {};
  std::vector<int64_t> dimensions;
  dimensions.reserve(subshape.rank());
  for (int64_t i = 0; i < subshape.rank(); ++i) {
    dimensions.push_back(literal.GetDynamicSize(i, shape_index));
  }
  print_recursive(dimensions, &indices);
}

void PrintHelper(const LiteralBase& literal, const ShapeIndex& shape_index,
                 bool print_shape, bool print_layout, bool oneline,
                 Printer* printer) {
  const Shape& subshape = ShapeUtil::GetSubshape(literal.shape(), shape_index);
  CHECK(LayoutUtil::HasLayout(literal.shape()));
  CHECK(LayoutUtil::HasLayout(subshape));
  if (subshape.IsTuple()) {
    TuplePrintHelper(literal, shape_index, print_shape, print_layout, oneline,
                     printer);
  } else if (subshape.IsToken()) {
    printer->Append("token");
  } else {
    CHECK(LayoutUtil::IsDenseArray(subshape));
    if (literal.IsKnown(shape_index)) {
      DenseArrayPrintHelper(literal, shape_index, print_shape, print_layout,
                            oneline, printer);
    } else {
      PrintShape(print_layout, subshape, printer);
      printer->Append(" ");
      if (literal.IsDetermined(shape_index)) {
        printer->Append("unknown");
      } else {
        printer->Append("undetermined");
      }
    }
  }
}

}  // namespace

void LiteralBase::Print(Printer* printer) const {
  CHECK(LayoutUtil::HasLayout(this->shape()));
  PrintHelper(*this, {}, /*print_shape=*/true, /*print_layout=*/false,
              /*oneline=*/false, printer);
}

void LiteralBase::PrintOneline(Printer* printer) const {
  CHECK(LayoutUtil::HasLayout(this->shape()));
  PrintHelper(*this, {}, /*print_shape=*/true, /*print_layout=*/false,
              /*oneline=*/true, printer);
}

void LiteralBase::PrintWithoutShape(Printer* printer) const {
  CHECK(LayoutUtil::HasLayout(this->shape()));
  PrintHelper(*this, {}, /*print_shape=*/false, /*print_layout=*/false,
              /*oneline=*/false, printer);
}

void LiteralBase::PrintWithoutShapeOneline(Printer* printer) const {
  CHECK(LayoutUtil::HasLayout(this->shape()));
  PrintHelper(*this, {}, /*print_shape=*/false, /*print_layout=*/false,
              /*oneline=*/true, printer);
}

void LiteralBase::PrintWithLayout(Printer* printer) const {
  CHECK(LayoutUtil::HasLayout(this->shape()));
  PrintHelper(*this, {}, /*print_shape=*/true, /*print_layout=*/true,
              /*oneline=*/false, printer);
}

void LiteralBase::PrintWithLayoutOneline(Printer* printer) const {
  CHECK(LayoutUtil::HasLayout(this->shape()));
  PrintHelper(*this, {}, /*print_shape=*/true, /*print_layout=*/true,
              /*oneline=*/true, printer);
}

std::string LiteralBase::ToString() const {
  StringPrinter printer;
  Print(&printer);
  return std::move(printer).ToString();
}

std::string LiteralBase::ToStringOneline() const {
  StringPrinter printer;
  PrintOneline(&printer);
  return std::move(printer).ToString();
}

std::string LiteralBase::ToStringWithoutShape() const {
  StringPrinter printer;
  PrintWithoutShape(&printer);
  return std::move(printer).ToString();
}

std::string LiteralBase::ToStringWithoutShapeOneline() const {
  StringPrinter printer;
  PrintWithoutShapeOneline(&printer);
  return std::move(printer).ToString();
}

std::string LiteralBase::ToStringWithLayout() const {
  StringPrinter printer;
  PrintWithLayout(&printer);
  return std::move(printer).ToString();
}

std::string LiteralBase::ToStringWithLayoutOneline() const {
  StringPrinter printer;
  PrintWithLayoutOneline(&printer);
  return std::move(printer).ToString();
}

namespace {

template <typename NativeSrcT, typename NativeDestT>
void ConvertBetweenNativeTypes(absl::Span<const NativeSrcT> src_data,
                               void* dst_base) {
  static_assert(!std::is_same_v<NativeSrcT, NativeDestT>);

  NativeDestT* dest_data = static_cast<NativeDestT*>(dst_base);
  for (const NativeSrcT& src : src_data) {
    if constexpr (is_big_int_v<NativeSrcT>) {
      if constexpr (is_big_int_v<NativeDestT>) {
        if constexpr (NativeDestT::kLimbNums >= NativeSrcT::kLimbNums) {
          // BigInt to larger/equal BigInt - direct conversion.
          *(dest_data++) = static_cast<NativeDestT>(src);
        } else {
          // BigInt to smaller BigInt - truncate
          *(dest_data++) = src.template Truncate<NativeDestT::kLimbNums>();
        }
      } else {
        // BigInt to non-BigInt - cast through uint64_t.
        *(dest_data++) = static_cast<NativeDestT>(static_cast<uint64_t>(src));
      }
    } else {
      *(dest_data++) = static_cast<NativeDestT>(src);
    }
  }
}

template <typename NativeSrcT, typename NativeDestT>
void CreateFromInteger(absl::Span<const NativeSrcT> src_data, void* dst_base) {
  static_assert(!std::is_same_v<NativeSrcT, NativeDestT>);

  NativeDestT* dest_data = static_cast<NativeDestT*>(dst_base);
  for (const NativeSrcT& src : src_data) {
    if constexpr (zk_dtypes::IsPrimeField<NativeDestT>) {
      // NOTE(chokobole): This assumes that if the target Prime Field uses
      // Montgomery form, the source integer has already been transformed
      // accordingly.
      //
      // In Zorch, for instance, a raw integer (e.g., '1') is typically
      // promoted to a field type (like babybear) that is already in
      // Montgomery form before reaching this conversion logic:
      //
      // ```python
      // def add(a):
      //     return a + 1  # '1' is promoted to babybear(1) in Montgomery form
      //
      // jax.jit(add)(babybear(1))
      // ```
      *(dest_data++) = NativeDestT::FromUnchecked(src);
    } else if constexpr (zk_dtypes::IsExtensionField<NativeDestT>) {
      using BasePrimeField = typename NativeDestT::BasePrimeField;
      if constexpr (sizeof(src) <= sizeof(BasePrimeField)) {
        *(dest_data++) = NativeDestT(BasePrimeField::FromUnchecked(src));
      } else {
        LOG(FATAL) << "Do we really need this case??";
      }
    } else {
      *(dest_data++) = NativeDestT(src);
    }
  }
}

/// Converts prime/extension field elements to integers by extracting raw
/// stored values. This is the inverse of CreateFromInteger — symmetric with
/// the FromUnchecked/value() pair.
template <typename NativeSrcT, typename NativeDestT>
void CreateToInteger(absl::Span<const NativeSrcT> src_data, void* dst_base) {
  static_assert(!std::is_same_v<NativeSrcT, NativeDestT>);

  NativeDestT* dest_data = static_cast<NativeDestT*>(dst_base);
  for (const NativeSrcT& src : src_data) {
    if constexpr (zk_dtypes::IsPrimeField<NativeSrcT>) {
      *(dest_data++) = static_cast<NativeDestT>(src.value());
    } else if constexpr (zk_dtypes::IsExtensionField<NativeSrcT>) {
      *(dest_data++) = static_cast<NativeDestT>(src.values()[0].value());
    }
  }
}

template <PrimitiveType kSrcType>
absl::Status ConvertIfDestTypeMatches(const LiteralBase& src_literal,
                                      MutableLiteralBase& dst_literal) {
  DCHECK(dst_literal.shape().IsArray());
  using NativeSrcT = primitive_util::NativeTypeOf<kSrcType>;
  // Pass raw data Span/pointers to called template methods to avoid duplicating
  // the Literal method calls to many time which hurts code size.
  auto src_data = src_literal.data<NativeSrcT>();
  void* dst_base = dst_literal.untyped_data();
  DCHECK_EQ(src_data.size(), dst_literal.element_count());
  return primitive_util::ArrayTypeSwitch<absl::Status>(
      [&](auto primitive_type_constant) -> absl::Status {
        using NativeDestT =
            primitive_util::NativeTypeOf<primitive_type_constant>;
        if constexpr (primitive_util::IsFieldType(primitive_type_constant) ||
                      primitive_util::IsFieldType(kSrcType) ||
                      primitive_util::IsEcPointType(primitive_type_constant) ||
                      primitive_util::IsEcPointType(kSrcType)) {
          if constexpr (primitive_util::IsIntegralType(kSrcType) &&
                        primitive_util::IsFieldType(primitive_type_constant)) {
            // NOTE(chokobole): This conversion is technically "unsafe" as it
            // may result in precision loss. For example, converting a u32 to
            // the BabyBear field is allowed here even though BabyBear's modulus
            // is only 31 bits.
            //
            // We permit this risk to improve usability on the Zorch side; since
            // 32-bit integers are the default integral type, disallowing this
            // would create significant friction for the user.
            if constexpr (primitive_util::ByteWidth(kSrcType) <=
                          primitive_util::ByteWidth(primitive_type_constant)) {
              CreateFromInteger<NativeSrcT, NativeDestT>(src_data, dst_base);
              return absl::OkStatus();
            } else {
              return absl::InvalidArgumentError(
                  "prime field conversion from a wider integer type is not "
                  "supported");
            }
            // NOLINTNEXTLINE(readability/braces)
          } else if constexpr (primitive_util::IsIntegralType(kSrcType) &&
                               primitive_util::IsEcPointType(
                                   primitive_type_constant)) {
            using ScalarField = typename NativeDestT::ScalarField;
            if constexpr (primitive_util::ByteWidth(kSrcType) <=
                          ScalarField::kByteWidth) {
              CreateFromInteger<NativeSrcT, NativeDestT>(src_data, dst_base);
              return absl::OkStatus();
            } else {
              return absl::InvalidArgumentError(
                  "ec point conversion from a wider integer type is not "
                  "supported");
            }
            // NOLINTNEXTLINE(readability/braces)
          } else if constexpr ((primitive_util::IsPrimeFieldType(kSrcType) ||
                                primitive_util::IsExtensionFieldType(
                                    kSrcType)) &&
                               primitive_util::IsIntegralType(
                                   primitive_type_constant)) {
            if constexpr (primitive_util::ByteWidth(kSrcType) <=
                          primitive_util::ByteWidth(primitive_type_constant)) {
              CreateToInteger<NativeSrcT, NativeDestT>(src_data, dst_base);
              return absl::OkStatus();
            } else {
              return absl::InvalidArgumentError(
                  "field to integer conversion to a narrower integer type is "
                  "not supported");
            }
          }
          return absl::UnimplementedError("Undefined conversion");
        } else if constexpr (kSrcType != primitive_type_constant) {
          ConvertBetweenNativeTypes<NativeSrcT, NativeDestT>(src_data,
                                                             dst_base);
        }
        return absl::OkStatus();
      },
      dst_literal.shape().element_type());
}

absl::StatusOr<Literal> ConvertSwitch(const LiteralBase& literal,
                                      PrimitiveType primitive_dest_type) {
  TF_RET_CHECK(LayoutUtil::IsDenseArray(literal.shape()));
  if (literal.shape().element_type() == primitive_dest_type) {
    return literal.Clone();
  }
  // Source Array type requirement is ensured by IsDenseArray before.
  if (!primitive_util::IsArrayType(primitive_dest_type) ||
      !primitive_util::IsArrayType(literal.shape().element_type())) {
    return absl::UnimplementedError(absl::StrFormat(
        "%s from type %s to type %s is not implemented.", "Converting",
        PrimitiveType_Name(literal.shape().element_type()),
        PrimitiveType_Name(primitive_dest_type)));
  }
  // At this point, we know both src & dst are array types, while src is not
  // complex type, so we can allocate the result literal here to avoid
  // duplicating it N^2 times in the conversion implementation.
  Literal result(
      ShapeUtil::ChangeElementType(literal.shape(), primitive_dest_type));
  TF_RETURN_IF_ERROR(primitive_util::ArrayTypeSwitch<absl::Status>(
      [&](auto primitive_type_constant) -> absl::Status {
        return ConvertIfDestTypeMatches<primitive_type_constant>(literal,
                                                                 result);
      },
      literal.shape().element_type()));
  return result;
}

}  // namespace

absl::StatusOr<Literal> LiteralBase::Convert(
    PrimitiveType primitive_dest_type) const {
  return ConvertSwitch(*this, primitive_dest_type);
}

absl::StatusOr<Literal> LiteralBase::BitcastConvert(
    const Shape& dest_shape) const {
  if (ShapeUtil::ByteSizeOf(dest_shape) != ShapeUtil::ByteSizeOf(shape())) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Can not bitcast-convert from shape %s to a shape of different size %s",
        shape().ToString(), dest_shape.ToString()));
  }
  if (dest_shape.IsTuple() || shape().IsTuple()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("bitcast-convert is not valid for tuple shapes %s->%s",
                        shape().ToString(), dest_shape.ToString()));
  }
  if (shape().is_dynamic() || dest_shape.is_dynamic()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("bitcast-convert is not valid for dynamic shape %s->%s",
                        shape().ToString(), dest_shape.ToString()));
  }

  Literal out(dest_shape);
  memcpy(out.root_piece_.buffer(), root_piece().buffer(),
         root_piece().size_bytes_dense());

  // Perform the reshape on little endian encoding even on big endian machines.
  if constexpr (!kLittleEndian) {
    // Swap byte ordering as per the input data type.
    size_t input_elem_size =
        ShapeUtil::ByteSizeOfPrimitiveType(shape().element_type());
    TF_RETURN_IF_ERROR(tsl::ByteSwapArray(
        const_cast<char*>(out.root_piece().buffer()), input_elem_size,
        out.root_piece().size_bytes_dense() / input_elem_size));
    // Swap byte ordering as per the output data type.
    size_t output_elem_size =
        ShapeUtil::ByteSizeOfPrimitiveType(dest_shape.element_type());
    TF_RETURN_IF_ERROR(tsl::ByteSwapArray(
        const_cast<char*>(out.root_piece().buffer()), output_elem_size,
        out.root_piece().size_bytes_dense() / output_elem_size));
  }

  return out;
}

absl::StatusOr<Literal> LiteralBase::ConvertToShape(
    const Shape& dest_shape) const {
  if (!dest_shape.IsTuple()) {
    return Convert(dest_shape.element_type());
  }
  std::vector<Literal> elements;
  const auto tuple_element_count = ShapeUtil::TupleElementCount(shape());
  elements.reserve(tuple_element_count);
  for (int i = 0; i < tuple_element_count; ++i) {
    auto element = LiteralSlice(*this, {i});
    TF_ASSIGN_OR_RETURN(
        auto new_element,
        element.ConvertToShape(ShapeUtil::GetSubshape(dest_shape, {i})));
    elements.push_back(std::move(new_element));
  }
  return MutableLiteralBase::MoveIntoTuple(absl::MakeSpan(elements));
}

// static
Literal MutableLiteralBase::MoveIntoTuple(absl::Span<Literal> elements) {
  std::vector<const Shape*> element_shapes;
  element_shapes.reserve(elements.size());
  for (const Literal& element : elements) {
    element_shapes.push_back(&element.shape());
  }
  Literal literal(ShapeUtil::MakeTupleShapeWithPtrs(element_shapes),
                  /*allocate_arrays=*/false);
  for (int i = 0, end = elements.size(); i < end; ++i) {
    CHECK_OK(
        literal.MoveFrom(std::move(elements[i]), /*dest_shape_index=*/{i}));
  }
  return literal;
}

template <typename NativeT>
bool LiteralBase::Piece::EqualElementsInternal(
    const LiteralBase::Piece& other, std::vector<int64_t>* multi_index) const {
  if (multi_index->size() == subshape().rank()) {
    return (Get<NativeT>(*multi_index) == other.Get<NativeT>(*multi_index));
  }
  for (int64_t i = 0; i < GetDynamicSize(multi_index->size()); ++i) {
    multi_index->push_back(i);
    if (!EqualElementsInternal<NativeT>(other, multi_index)) {
      return false;
    }
    multi_index->pop_back();
  }
  return true;
}

bool LiteralBase::Piece::EqualElements(const LiteralBase::Piece& other) const {
  if (subshape().is_static() &&
      ShapeUtil::Equal(subshape(), other.subshape()) && subshape().IsArray()) {
    CHECK(LayoutUtil::IsDenseArray(subshape()))
        << __func__ << " is only supported for dense arrays: " << subshape();
    CHECK_EQ(size_bytes_dense(), other.size_bytes_dense());
    if (primitive_util::IsSubByteNonPredType(subshape().element_type())) {
      auto one_array = buffer();
      auto two_array = other.buffer();
      const int bits_per_element =
          primitive_util::BitWidth(subshape().element_type());
      const uint8_t mask = LsbMask<uint8_t>(bits_per_element);
      for (int64_t i = 0; i < size_bytes_dense(); ++i) {
        if ((one_array[i] & mask) != (two_array[i] & mask)) return false;
      }
      return true;
    }
    if (primitive_util::IsEcPointType(subshape().element_type()) &&
#define ZK_DTYPES_CASE(unused, unused2, enum, unused3) \
  subshape().element_type() != enum&&
        ZK_DTYPES_PUBLIC_AFFINE_POINT_TYPE_LIST(ZK_DTYPES_CASE)
#undef ZK_DTYPES_CASE
            true) {
      // NOTE(chokobole): Non-affine elliptic-curve points must not be
      // compared byte-wise. Use the native type’s equality instead to avoid
      // false mismatches.
      std::vector<int64_t> multi_index;
      return primitive_util::EcPointTypeSwitch<bool>(
          [&](auto primitive_type_constant) -> bool {
            using NativeT =
                primitive_util::NativeTypeOf<primitive_type_constant>;
            return EqualElementsInternal<NativeT>(other, &multi_index);
          },
          subshape().element_type());
    } else {
      return memcmp(buffer(), other.buffer(), size_bytes_dense()) == 0;
    }
  }

  std::vector<int64_t> multi_index;
  return primitive_util::ArrayTypeSwitch<bool>(
      [&](auto primitive_type_constant) -> bool {
        using NativeSrcT =
            primitive_util::NativeTypeOf<primitive_type_constant>;
        return EqualElementsInternal<NativeSrcT>(other, &multi_index);
      },
      subshape().element_type());
}

bool LiteralBase::Equal(const LiteralBase& other, bool layout_sensitive) const {
  // Checking the structure of tuple literals. Checks for dense arrays are
  // performed below.
  if (!ShapeUtil::EqualStructure(shape(), other.shape())) {
    return false;
  }

  return root_piece().ForEachSubpieceWithBool([&](const ShapeIndex& index,
                                                  const Piece& piece) {
    const Piece& other_piece = other.piece(index);
    const Shape& subshape = piece.subshape();
    const Shape& other_subshape = other_piece.subshape();
    if (subshape.element_type() != other_subshape.element_type()) {
      return false;
    }
    if (!piece.subshape().IsArray()) {
      return true;
    }
    if (subshape.rank() != other_subshape.rank()) {
      return false;
    }
    if (layout_sensitive && (subshape.layout() != other_subshape.layout())) {
      return false;
    }

    for (int64_t i = 0; i < subshape.rank(); ++i) {
      if (piece.GetDynamicSize(i) != other_piece.GetDynamicSize(i)) {
        return false;
      }
    }

    if (!piece.EqualElements(other_piece)) {
      return false;
    }
    return true;
  });
}

namespace {

template <typename NativeT>
bool AllElementsEqualValue(absl::Span<const NativeT> data, NativeT value) {
  for (int64_t i = 0; i < data.size(); ++i) {
    if (memcmp(&data[i], &value, sizeof value)) {
      return false;
    }
  }
  return true;
}

}  // namespace

bool Literal::Piece::IsAll(const Literal& scalar) const {
  CHECK(ShapeUtil::IsScalar(scalar.shape())) << scalar.shape().ToString();
  if (!subshape().IsArray()) {
    return false;
  }

  CHECK(LayoutUtil::IsDenseArray(subshape()))
      << __func__ << " is only supported for dense arrays: " << subshape();
  CHECK_EQ(subshape().element_type(), scalar.shape().element_type());
  return primitive_util::ArrayTypeSwitch<bool>(
      [&](auto primitive_type_constant) -> bool {
        using NativeT = primitive_util::NativeTypeOf<primitive_type_constant>;
        return AllElementsEqualValue(this->data<NativeT>(),
                                     scalar.GetFirstElement<NativeT>());
      },
      subshape().element_type());
}

bool LiteralBase::IsAll(const Literal& scalar) const {
  return root_piece().IsAll(scalar);
}

bool LiteralBase::IsAll(int8_t value) const {
  if (!shape().IsArray()) {
    return false;
  }
  PrimitiveType ty = shape().element_type();
  if (primitive_util::IsUnsignedIntegralType(ty) && value < 0) {
    return false;
  }
  Literal scalar(ShapeUtil::MakeScalarShape(ty));
  return primitive_util::ArrayTypeSwitch<bool>(
      [&](auto primitive_type_constant) -> bool {
        if constexpr (primitive_util::IsBigIntType(primitive_type_constant) ||
                      primitive_util::IsFieldType(primitive_type_constant) ||
                      primitive_util::IsEcPointType(primitive_type_constant)) {
          return false;
        } else {
          using NativeT = primitive_util::NativeTypeOf<primitive_type_constant>;
          NativeT converted(value);
          if (static_cast<int8_t>(converted) != value) {
            return false;
          }
          scalar.Set<NativeT>({}, converted);
          return root_piece().IsAll(scalar);
        }
      },
      ty);
}

bool LiteralBase::IsAllFirst() const {
  if (!shape().IsArray()) {
    return false;
  }

  // Empty shapes are not all the first element since there is no first element.
  if (ShapeUtil::IsZeroElementArray(shape())) {
    return false;
  }

  absl::InlinedVector<int64_t, 4> start_indices(/*n=*/shape().rank(), 0);
  absl::InlinedVector<int64_t, 4> end_indices(/*n=*/shape().rank(), 1);
  Literal first = Slice(start_indices, end_indices);
  return IsAll(first.Reshape({}).value());
}

const Shape& LiteralBase::shape() const { return root_piece().subshape(); }

const char* LiteralBase::Piece::buffer() const {
  // std::visit is avoided here due to its code size issues.
  if (auto* r = std::get_if<DenseRep>(&rep_)) {
    return r->data;
  }
  if (auto* r = std::get_if<DenseInlinedRep>(&rep_)) {
    return r->data;
  }
  DCHECK(std::holds_alternative<TupleRep>(rep_) ||
         std::holds_alternative<Uninitialized>(rep_));
  return nullptr;
}

const LiteralBase::Piece& LiteralBase::piece(
    const ShapeIndex& shape_index) const {
  const Piece* piece = &root_piece();
  for (const auto i : shape_index) {
    DCHECK_GE(i, 0);
    DCHECK_LT(i, piece->children_size());
    piece = &piece->child(i);
  }
  return *piece;
}

std::ostream& operator<<(std::ostream& out, const Literal& literal) {
  out << literal.ToString();
  return out;
}

Shape* MutableLiteralBase::mutable_shape_do_not_use() {
  const Shape* const_shape = shape_.get();
  if (!shape_.OwnsPtr()) {
    shape_ = MaybeOwningShapePtr(std::make_unique<Shape>(*shape_));
  }
  Shape* shape = shape_.get_mutable();

  if (shape != const_shape) {
    std::function<void(const Shape&, Piece*)> set_piece_shapes =
        [&set_piece_shapes](const Shape& shape, Piece* piece) {
          piece->set_subshape(&shape);
          if (shape.IsTuple()) {
            for (int i = 0; i < ShapeUtil::TupleElementCount(shape); ++i) {
              const Shape& subshape = shape.tuple_shapes(i);
              set_piece_shapes(subshape, &piece->child(i));
            }
          }
        };
    set_piece_shapes(*shape, &mutable_root_piece());
  }
  return shape;
}

Literal::Literal() : Literal(NilShape()) {}

Literal::Literal(const Shape& shape)
    : Literal(shape, /*allocate_arrays=*/true) {}

void Literal::SetShape(const Shape& shape) {
  if (const Shape* intered_shape_ptr = TryInternShape(shape)) {
    shape_ = intered_shape_ptr;
    return;
  }
  auto owning_shape_ptr = std::make_unique<Shape>(shape);
  if (owning_shape_ptr->IsArray() && !owning_shape_ptr->has_layout()) {
    *owning_shape_ptr->mutable_layout() =
        LayoutUtil::GetDefaultLayoutForShape(*owning_shape_ptr);
  }
  if (owning_shape_ptr->IsArray() &&
      LayoutUtil::HasCustomElementSizeInBits(*owning_shape_ptr)) {
    owning_shape_ptr->mutable_layout()->set_element_size_in_bits(0);
  }
  shape_ = std::move(owning_shape_ptr);
}

void Literal::SetPiece(const Shape& shape, Piece* piece, bool allocate_arrays,
                       ArrayValueState leaf_array_value_state) {
  if (shape.IsTuple()) {
    for (const Shape& subshape : shape.tuple_shapes()) {
      Piece child_piece;
      child_piece.set_subshape(&subshape);

      SetPiece(subshape, &child_piece, allocate_arrays, leaf_array_value_state);

      piece->emplace_back(std::move(child_piece));
    }
  } else if (shape.IsArray()) {
    DCHECK(LayoutUtil::IsDenseArray(shape))
        << "literal array storage is currently only supported for dense "
           "arrays: "
        << shape;
    piece->set_array_value_state(leaf_array_value_state);
    if (leaf_array_value_state == LiteralBase::ArrayValueState::kKnown &&
        allocate_arrays) {
      piece->AllocateBuffers();
    }
  }
}

Literal::Literal(const Shape& shape, bool allocate_arrays,
                 ArrayValueState leaf_array_value_state) {
  SetShape(shape);
  CHECK(leaf_array_value_state != ArrayValueState::kKnown ||
        LayoutUtil::HasLayout(*shape_));
  root_piece_.set_subshape(shape_.get());
  CHECK(&root_piece_.subshape() == shape_.get());

  SetPiece(*shape_, &root_piece_, allocate_arrays, leaf_array_value_state);
}

Literal::~Literal() { DeallocateBuffers(); }

void Literal::DeallocateBuffers() {
  root_piece_.ForEachMutableSubpiece(
      [&](const ShapeIndex& index, Piece* piece) {
        piece->DeallocateBuffers();
      });
}

Literal::Literal(Literal&& other) { *this = std::move(other); }

Literal& Literal::operator=(Literal&& other) {
  DCHECK(&other.root_piece_.subshape() == other.shape_.get());
  using std::swap;
  swap(shape_, other.shape_);
  swap(root_piece_, other.root_piece_);
  DCHECK(&root_piece_.subshape() == shape_.get());

  return *this;
}

// static
Literal LiteralBase::CreateFromShape(const Shape& shape) {
  Literal literal(shape);
  literal.root_piece_.ForEachMutableSubpiece(
      [&](const ShapeIndex& index, Piece* piece) {
        if (piece->subshape().IsArray()) {
          memset(piece->untyped_data(), 0, piece->size_bytes_dense());
        }
      });
  return literal;
}

// static
Literal LiteralBase::CreateFromShapeWithUnknownLeafArrays(const Shape& shape) {
  Literal literal(shape, /*allocate_arrays=*/false, ArrayValueState::kUnknown);
  return literal;
}

// static
Literal LiteralBase::CreateFromShapeWithUndeterminedLeafArrays(
    const Shape& shape) {
  Literal literal(shape, /*allocate_arrays=*/false,
                  ArrayValueState::kUndetermined);
  return literal;
}

int32_t LiteralBase::GetDynamicSize(int64_t dim_index) const {
  return GetDynamicSize(dim_index, {});
}

int32_t LiteralBase::GetDynamicSize(int64_t dim_index,
                                    const ShapeIndex& shape_index) const {
  return piece(shape_index).GetDynamicSize(dim_index);
}

std::optional<int64_t> LiteralBase::GetFirstInteger() const {
  if (!primitive_util::IsIntegralType(shape().element_type())) {
    return std::nullopt;
  }
  return primitive_util::IntegralTypeSwitch<std::optional<int64_t>>(
      [&](auto primitive_type_constant) -> std::optional<int64_t> {
        using NativeT = primitive_util::NativeTypeOf<primitive_type_constant>;
        auto first_element = GetFirstElement<NativeT>();
        if constexpr (primitive_util::IsBigIntType(primitive_type_constant)) {
          // Check if value fits in int64_t (upper limbs must be zero)
          for (size_t i = 1; i < NativeT::kLimbNums; ++i) {
            if (first_element[i] != 0) {
              return std::nullopt;
            }
          }
          uint64_t lower = static_cast<uint64_t>(first_element);
          if (lower >
              static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            return std::nullopt;
          }
          return static_cast<int64_t>(lower);
        } else {
          if constexpr (std::is_same_v<NativeT, uint64_t>) {
            int64_t v = static_cast<int64_t>(first_element);
            if (v < 0) {
              return std::nullopt;
            }
          }
          return first_element;
        }
      },
      shape().element_type());
}

void LiteralBase::BuildPieceSubtree(const Shape& shape, Piece* piece) {
  CHECK(shape.IsTuple());
  for (int i = 0; i < ShapeUtil::TupleElementCount(shape); ++i) {
    const Shape& subshape = shape.tuple_shapes(i);

    Piece child_piece;
    child_piece.set_subshape(&subshape);

    if (subshape.IsTuple()) {
      BuildPieceSubtree(subshape, &child_piece);
    }

    piece->emplace_back(std::move(child_piece));
  }
}

template <typename NativeT>
absl::Status MutableLiteralBase::CopySliceFromInternal(
    const LiteralBase& src_literal, absl::Span<const int64_t> src_base,
    absl::Span<const int64_t> dest_base, absl::Span<const int64_t> copy_size) {
  auto linear_index = [](const Shape& shape,
                         absl::Span<const int64_t> multi_index) {
    return IndexUtil::MultidimensionalIndexToLinearIndex(shape, multi_index);
  };

  // `this->` is needed to workaround MSVC bug: #16882
  NativeT* dest_data = this->data<NativeT>().data();
  const NativeT* src_data = src_literal.data<NativeT>().data();
  if (src_literal.shape().rank() == 0 || shape().rank() == 0) {
    // If any of the two shapes are scalars, just assign the value once.
    TF_RET_CHECK(copy_size.empty());
    dest_data[linear_index(shape(), dest_base)] =
        src_data[linear_index(src_literal.shape(), src_base)];
  } else if (!ShapeUtil::IsZeroElementArray(shape()) &&
             !ShapeUtil::IsZeroElementArray(src_literal.shape()) &&
             absl::c_none_of(copy_size, [](auto d) { return d == 0; })) {
    // Perform copy if none of src, dest and copy_size has dimensions with zero
    // element, otherwise it's a no-op.
    TF_RET_CHECK(src_base.size() == dest_base.size());
    TF_RET_CHECK(src_base.size() == copy_size.size());

    // Scan the source from minor, stepping in copy size blocks, then within
    // the index enumeration functor, do a strided copy advancing source index
    // by one (walking through the minor dimension), and destination index by
    // proper stride size at the matching dimension.
    DimensionVector src_indexes(src_base.size(), 0);
    DimensionVector dest_indexes(dest_base.size(), 0);
    StrideConfig stride_config(src_literal.shape(), shape(), copy_size);

    auto copy_proc = [&](absl::Span<const int64_t> indexes) {
      // Map from multi-dimensional index, to source index.
      std::transform(indexes.begin(), indexes.end(), src_base.begin(),
                     src_indexes.begin(), std::plus<int64_t>());
      // Map from multi-dimensional index, to destination index.
      std::transform(indexes.begin(), indexes.end(), dest_base.begin(),
                     dest_indexes.begin(), std::plus<int64_t>());

      int64_t src_index = linear_index(src_literal.shape(), src_indexes);
      int64_t dest_index = linear_index(shape(), dest_indexes);

      StridedCopy(dest_data + dest_index, stride_config.dest_stride,
                  src_data + src_index, stride_config.source_stride,
                  stride_config.minor_loop_size);
      return true;
    };

    ShapeUtil::ForEachIndex(src_literal.shape(), stride_config.base,
                            stride_config.dimensions, stride_config.step,
                            copy_proc);
  }
  return absl::OkStatus();
}

void MutableLiteralBase::CopyElementFrom(const LiteralSlice& src_literal,
                                         absl::Span<const int64_t> src_index,
                                         absl::Span<const int64_t> dest_index) {
  DCHECK(LayoutUtil::IsDenseArray(shape()));
  DCHECK_EQ(shape().element_type(), src_literal.shape().element_type());
  const int64_t src_linear_index =
      IndexUtil::MultidimensionalIndexToLinearIndex(src_literal.shape(),
                                                    src_index);
  const int64_t dest_linear_index =
      IndexUtil::MultidimensionalIndexToLinearIndex(shape(), dest_index);
  const int64_t primitive_size =
      ShapeUtil::ByteSizeOfPrimitiveType(shape().element_type());

  char* dest_address =
      static_cast<char*>(untyped_data()) + dest_linear_index * primitive_size;
  const char* source_address =
      static_cast<const char*>(src_literal.untyped_data()) +
      src_linear_index * primitive_size;
  if (dest_address != source_address) {
    memcpy(dest_address, source_address, primitive_size);
  }
}

// static
absl::StatusOr<Literal> MutableLiteralBase::CreateFromProto(
    const LiteralProto& proto, bool prohibit_empty_literal) {
  if (!proto.has_shape()) {
    return absl::InvalidArgumentError("LiteralProto has no shape");
  }
  Shape shape(proto.shape());
  if (ShapeUtil::HasPrimitiveType(shape, OPAQUE_TYPE)) {
    return absl::InvalidArgumentError(
        "Literal shape cannot include OPAQUE_TYPE sub-shape");
  }
  if (!LayoutUtil::HasLayout(shape)) {
    return absl::InvalidArgumentError("LiteralProto has no layout");
  }
  if (LayoutUtil::IsSparseArray(shape)) {
    return absl::UnimplementedError("Sparse literals are not supported");
  }

  TF_RETURN_IF_ERROR(ShapeUtil::ValidateShapeWithOptionalLayout(shape));

  Literal literal(shape);

  TF_RETURN_IF_ERROR(literal.root_piece_.ForEachMutableSubpieceWithStatus(
      [&](const ShapeIndex& index, Piece* piece) -> absl::Status {
        const LiteralProto* proto_element = &proto;
        for (int64_t i : index) {
          CHECK(i < proto_element->tuple_literals_size());
          proto_element = &proto_element->tuple_literals(i);
        }

        if (piece->subshape().IsTuple()) {
          if (proto_element->tuple_literals_size() !=
              ShapeUtil::TupleElementCount(piece->subshape())) {
            return absl::InvalidArgumentError(absl::StrFormat(
                "Expected %d tuple elements in LiteralProto, has %d",
                ShapeUtil::TupleElementCount(piece->subshape()),
                proto_element->tuple_literals_size()));
          }
          return absl::OkStatus();
        }
        if (piece->subshape().element_type() == TOKEN) {
          return absl::OkStatus();
        }

        CHECK(piece->subshape().IsArray());

        // When prohibit_empty_literal is false (allowing literal with no
        // values), only copy from proto if the literal proto has values. This
        // mode is used for a learned cost model.
        if (prohibit_empty_literal || LiteralProtoHasValues(*proto_element)) {
          TF_RETURN_IF_ERROR(piece->CopyFromProto(*proto_element));
        }

        return absl::OkStatus();
      }));

  return std::move(literal);
}

std::vector<Literal> Literal::DecomposeTuple() {
  CHECK(shape().IsTuple());
  std::vector<Literal> elements;
  const auto tuple_element_count = ShapeUtil::TupleElementCount(shape());
  elements.reserve(tuple_element_count);
  for (int i = 0; i < tuple_element_count; ++i) {
    elements.push_back(Literal(ShapeUtil::GetSubshape(shape(), {i}),
                               /*allocate_arrays=*/false));
    Literal& element = elements.back();
    element.root_piece_.ForEachMutableSubpiece(
        [&](const ShapeIndex& index, Piece* dest_piece) {
          if (dest_piece->subshape().IsTuple()) {
            return;
          }
          ShapeIndex src_index = {i};
          for (int64_t j : index) {
            src_index.push_back(j);
          }
          Piece& src_piece = piece(src_index);

          // Move the respective buffer over to the element Literal.
          dest_piece->MoveDataFrom(src_piece);
        });
  }
  // Set this literal to be nil-shaped.
  *this = Literal();
  return elements;
}

namespace {

// Copies the elements in 'src' to 'dest'. The shape and layout of the data in
// the array slices are indicated by dest_shape and src_shape respectively.
template <typename NativeT>
void CopyElementsBetween(absl::Span<NativeT> dest,
                         absl::Span<const NativeT> src, const Shape& dest_shape,
                         const Shape& src_shape) {
  DCHECK(LayoutUtil::IsDenseArray(dest_shape));
  DCHECK(LayoutUtil::IsDenseArray(src_shape));
  DCHECK(ShapeUtil::Compatible(dest_shape, src_shape));
  if (ShapeUtil::IsZeroElementArray(dest_shape)) {
    return;
  }
  std::vector<int64_t> index(dest_shape.rank());
  do {
    dest[IndexUtil::MultidimensionalIndexToLinearIndex(dest_shape, index)] =
        src[IndexUtil::MultidimensionalIndexToLinearIndex(src_shape, index)];
  } while (IndexUtil::BumpIndices(dest_shape, absl::MakeSpan(index)));
}

}  // namespace

int32_t LiteralBase::Piece::GetDynamicSize(int64_t dim_index) const {
  CHECK(LayoutUtil::IsDenseArray(subshape()));
  if (!subshape_->is_dynamic_dimension(dim_index)) {
    // This is a static dimension, return size.
    return subshape_->dimensions(dim_index);
  }
  return dynamic_size_buffer()[dim_index];
}

void LiteralBase::Piece::SetDynamicSize(int64_t dim_index, int32_t size) {
  CHECK(LayoutUtil::IsDenseArray(subshape()));
  CHECK(subshape_->is_dynamic_dimension(dim_index));
  dynamic_size_buffer()[dim_index] = size;
}

void LiteralBase::Piece::AllocateBuffers() {
  const int64_t bytes = total_bytes_dense();
  if (bytes > kMaxInlinedBytes) {
    CHECK_EQ(buffer(), nullptr);
    rep_.emplace<DenseRep>();
    char* buffer =
        static_cast<char*>(tsl::port::AlignedMalloc(bytes, kMinimumAlignment));
    CHECK(buffer != nullptr) << "Failed to allocate buffer for Literal";
    set_buffer(buffer);
  } else {
    rep_.emplace<DenseInlinedRep>();
  }
}

void LiteralBase::Piece::DeallocateBuffers() {
  if (auto* array_rep = GetDenseRep()) {
    tsl::port::AlignedFree(array_rep->data);
    rep_.emplace<Uninitialized>();
  }
}

template <typename NativeT>
void LiteralBase::Piece::CopyElementsWithDynamicBound(
    const LiteralBase::Piece& src) {
  auto& dest_shape = subshape();
  auto& src_shape = src.subshape();

  // At least one shape has to be static as bound.
  CHECK(dest_shape.is_static() || src_shape.is_static());
  auto& bound_shape = dest_shape.is_static() ? src_shape : dest_shape;
  if (ShapeUtil::IsZeroElementArray(dest_shape)) {
    return;
  }
  if (dest_shape.rank() == 1) {
    // Fast path for rank 1 arrays.
    int64_t count = std::min(GetDynamicSize(0), src.GetDynamicSize(0));
    std::copy_n(src.data<NativeT>().begin(), count, data<NativeT>().begin());
    return;
  }
  std::vector<int64_t> index(dest_shape.rank());
  do {
    bool out_of_bound = false;
    for (int64_t i = 0; i < index.size(); ++i) {
      // Do not copy elements beyond dynamic bound.
      if (index[i] >= GetDynamicSize(i) || index[i] >= src.GetDynamicSize(i)) {
        out_of_bound = true;
      }
    }
    if (out_of_bound) {
      continue;
    }
    data<NativeT>()[IndexUtil::MultidimensionalIndexToLinearIndex(dest_shape,
                                                                  index)] =
        src.data<NativeT>()[IndexUtil::MultidimensionalIndexToLinearIndex(
            src_shape, index)];
  } while (IndexUtil::BumpIndices(bound_shape, absl::MakeSpan(index)));
}

absl::Status LiteralBase::Piece::CopyFrom(const LiteralBase::Piece& src,
                                          bool only_dynamic_bound) {
  CHECK(subshape_ != nullptr);
  CHECK(src.subshape_ != nullptr);
  CHECK(LayoutUtil::IsDenseArray(subshape()))
      << __func__ << " is only supported for dense arrays: " << subshape();
  CHECK(LayoutUtil::IsDenseArray(src.subshape()))
      << __func__ << " is only supported for dense arrays: " << src.subshape();
  if (!only_dynamic_bound) {
    CHECK(ShapeUtil::Compatible(subshape(), src.subshape()));
  }
  if (src.array_value_state_ == ArrayValueState::kUnknown ||
      src.array_value_state_ == ArrayValueState::kUndetermined) {
    if (array_value_state_ == ArrayValueState::kKnown) {
      DeallocateBuffers();
    }
    array_value_state_ = src.array_value_state_;
    return absl::OkStatus();
  } else {
    CHECK(src.array_value_state_ == ArrayValueState::kKnown);
    if (array_value_state_ == ArrayValueState::kUndetermined ||
        array_value_state_ == ArrayValueState::kUnknown) {
      AllocateBuffers();
    }
    array_value_state_ = src.array_value_state_;
  }

  if (ShapeUtil::Equal(subshape(), src.subshape())) {
    // If the layouts are equal it's faster just to memcpy.
    memcpy(buffer(), src.buffer(), src.size_bytes_dense());
  } else {
    std::vector<int64_t> origin(subshape().rank(), 0);
    primitive_util::ArrayTypeSwitch<void>(
        [&](auto primitive_type_constant) {
          using NativeT = primitive_util::NativeTypeOf<primitive_type_constant>;
          if (only_dynamic_bound) {
            CopyElementsWithDynamicBound<NativeT>(src);
          } else {
            CopyElementsBetween<NativeT>(this->data<NativeT>(),
                                         src.data<NativeT>(), subshape(),
                                         src.subshape());
          }
        },
        subshape().element_type());
  }
  DCHECK_EQ(dynamic_size_buffer_bytes(), src.dynamic_size_buffer_bytes());
  if (subshape().is_dynamic() && src.subshape().is_dynamic()) {
    memcpy(dynamic_size_buffer(), src.dynamic_size_buffer(),
           src.dynamic_size_buffer_bytes());
  }
  return absl::OkStatus();
}

namespace {

template <typename RepeatedFieldT, typename NativeT>
void CopyToRepeatedField(RepeatedFieldT* dest,
                         const absl::Span<const NativeT> src) {
  *dest = RepeatedFieldT(src.begin(), src.end());
}

}  // namespace

void LiteralBase::Piece::WriteToProto(LiteralProto* proto) const {
  *proto->mutable_shape() = subshape().ToProto();
  switch (subshape().element_type()) {
    case PRED:
      CopyToRepeatedField(proto->mutable_preds(), data<bool>());
      break;
    case U1:
      *proto->mutable_u1s() = std::string(
          reinterpret_cast<const char*>(data<u1>().data()), size_bytes_dense());
      break;
    case U2:
      *proto->mutable_u2s() = std::string(
          reinterpret_cast<const char*>(data<u2>().data()), size_bytes_dense());
      break;
    case U4:
      *proto->mutable_u4s() = std::string(
          reinterpret_cast<const char*>(data<u4>().data()), size_bytes_dense());
      break;
    case U8:
      proto->set_u8s(static_cast<const unsigned char*>(data<uint8_t>().data()),
                     element_count());
      break;
    case U16:
      *proto->mutable_u16s() =
          std::string(reinterpret_cast<const char*>(data<uint16_t>().data()),
                      size_bytes_dense());
      if (!kLittleEndian) {
        ConvertEndianShort(proto->mutable_u16s());
      }
      break;
    case U32:
      CopyToRepeatedField(proto->mutable_u32s(), data<uint32_t>());
      break;
    case U64:
      CopyToRepeatedField(proto->mutable_u64s(), data<uint64_t>());
      break;
    case U128:
      *proto->mutable_u128s() =
          std::string(reinterpret_cast<const char*>(data<u128>().data()),
                      size_bytes_dense());
      break;
    case U256:
      *proto->mutable_u256s() =
          std::string(reinterpret_cast<const char*>(data<u256>().data()),
                      size_bytes_dense());
      break;
    case S1:
      *proto->mutable_s1s() = std::string(
          reinterpret_cast<const char*>(data<s1>().data()), size_bytes_dense());
      break;
    case S2:
      *proto->mutable_s2s() = std::string(
          reinterpret_cast<const char*>(data<s2>().data()), size_bytes_dense());
      break;
    case S4:
      *proto->mutable_s4s() = std::string(
          reinterpret_cast<const char*>(data<s4>().data()), size_bytes_dense());
      break;
    case S8:
      proto->set_s8s(static_cast<const signed char*>(data<int8_t>().data()),
                     element_count());
      break;
    case S16:
      *proto->mutable_s16s() =
          std::string(reinterpret_cast<const char*>(data<int16_t>().data()),
                      size_bytes_dense());
      if (!kLittleEndian) {
        ConvertEndianShort(proto->mutable_s16s());
      }
      break;
    case S32:
      CopyToRepeatedField(proto->mutable_s32s(), data<int32_t>());
      break;
    case S64:
      CopyToRepeatedField(proto->mutable_s64s(), data<int64_t>());
      break;
    case S128:
      *proto->mutable_s128s() =
          std::string(reinterpret_cast<const char*>(data<s128>().data()),
                      size_bytes_dense());
      break;
    case S256:
      *proto->mutable_s256s() =
          std::string(reinterpret_cast<const char*>(data<s256>().data()),
                      size_bytes_dense());
      break;
    case TUPLE:
    case TOKEN:
      // Nothing to do but assign the shape which is done above.
      return;
#define ZK_DTYPES_CASE(cpp_type, unused, enum, member_name)                 \
  case enum:                                                                \
    *proto->mutable_##member_name##s() =                                    \
        std::string(reinterpret_cast<const char*>(data<cpp_type>().data()), \
                    size_bytes_dense());                                    \
    break;
      ZK_DTYPES_PUBLIC_TYPE_LIST(ZK_DTYPES_CASE)
#undef ZK_DTYPES_CASE
    default:
      // TODO(b/111551621): Support serializing more PrimitiveTypes.
      LOG(FATAL) << "Unhandled primitive type "
                 << PrimitiveType_Name(subshape().element_type());
  }
}

const void* LiteralBase::Piece::untyped_data() const {
  DCHECK(LayoutUtil::IsDenseArray(subshape()))
      << ShapeUtil::HumanString(subshape());
  return buffer();
}

void* LiteralBase::Piece::untyped_data() {
  DCHECK(LayoutUtil::IsDenseArray(subshape()))
      << ShapeUtil::HumanString(subshape());
  return buffer();
}

bool LiteralBase::Piece::IsDetermined() const {
  if (array_value_state_ == ArrayValueState::kUndetermined) {
    return false;
  }
  if (subshape().IsTuple()) {
    bool are_all_leaf_arrays_determined = true;
    ForEachSubpiece([&are_all_leaf_arrays_determined](const ShapeIndex& index,
                                                      const Piece& piece) {
      if (!piece.subshape().IsArray()) {
        return;
      }
      are_all_leaf_arrays_determined &= piece.IsDetermined();
    });
    return are_all_leaf_arrays_determined;
  }
  return true;
}

namespace {

template <typename RepeatedFieldT, typename NativeT>
absl::Status CopyFromRepeatedField(absl::Span<NativeT> dest,
                                   const RepeatedFieldT& src) {
  if (dest.size() != src.size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Expected %lu elements in LiteralProto repeated field, has %d",
        dest.size(), src.size()));
  }
  std::copy(src.begin(), src.end(), dest.begin());
  return absl::OkStatus();
}

}  // namespace

absl::Status LiteralBase::Piece::CopyFromProto(const LiteralProto& proto) {
  // These conditions should have been checked in
  // MutableLiteralBase::CreateFromProto.
  TF_RET_CHECK(proto.has_shape());
  Shape shape(proto.shape());
  TF_RET_CHECK(LayoutUtil::HasLayout(shape));
  TF_RET_CHECK(ShapeUtil::Equal(shape, subshape()));

  switch (subshape().element_type()) {
    case PRED:
      TF_RETURN_IF_ERROR(CopyFromRepeatedField(data<bool>(), proto.preds()));
      break;
    case S2: {
      const std::string& s(proto.s2s());
      TF_RET_CHECK(data<s2>().size() * sizeof(s2) == s.size());
      memcpy(untyped_data(), s.data(), s.size());
      break;
    }
    case S4: {
      const std::string& s(proto.s4s());
      TF_RET_CHECK(data<s4>().size() * sizeof(s4) == s.size());
      memcpy(untyped_data(), s.data(), s.size());
      break;
    }
    case S8: {
      auto s8_data = data<int8_t>();
      TF_RET_CHECK(proto.s8s().size() == s8_data.size());
      std::copy(proto.s8s().begin(), proto.s8s().end(), s8_data.begin());
      break;
    }
    case S16: {
      const std::string& s(proto.s16s());
      TF_RET_CHECK(data<int16_t>().size() * sizeof(int16_t) == s.size());
      memcpy(untyped_data(), s.data(), s.size());
      if (!kLittleEndian) {
        ConvertEndianShort(reinterpret_cast<char*>(untyped_data()), s.size());
      }
      break;
    }
    case S32:
      TF_RETURN_IF_ERROR(CopyFromRepeatedField(data<int32_t>(), proto.s32s()));
      break;
    case S64:
      TF_RETURN_IF_ERROR(CopyFromRepeatedField(data<int64_t>(), proto.s64s()));
      break;
    case S128: {
      const std::string& s(proto.s128s());
      TF_RET_CHECK(data<s128>().size() * sizeof(s128) == s.size());
      memcpy(untyped_data(), s.data(), s.size());
      break;
    }
    case S256: {
      const std::string& s(proto.s256s());
      TF_RET_CHECK(data<s256>().size() * sizeof(s256) == s.size());
      memcpy(untyped_data(), s.data(), s.size());
      break;
    }
    case U2: {
      const std::string& s(proto.u2s());
      TF_RET_CHECK(data<u2>().size() * sizeof(u2) == s.size());
      memcpy(untyped_data(), s.data(), s.size());
      break;
    }
    case U4: {
      const std::string& s(proto.u4s());
      TF_RET_CHECK(data<u4>().size() * sizeof(u4) == s.size());
      memcpy(untyped_data(), s.data(), s.size());
      break;
    }
    case U8: {
      auto u8_data = data<uint8_t>();
      TF_RET_CHECK(proto.u8s().size() == u8_data.size());
      std::copy(proto.u8s().begin(), proto.u8s().end(), u8_data.begin());
      break;
    }
    case U16: {
      const std::string& s(proto.u16s());
      TF_RET_CHECK(data<uint16_t>().size() * sizeof(uint16_t) == s.size());
      memcpy(untyped_data(), s.data(), s.size());
      if (!kLittleEndian) {
        ConvertEndianShort(reinterpret_cast<char*>(untyped_data()), s.size());
      }
      break;
    }
    case U32:
      TF_RETURN_IF_ERROR(CopyFromRepeatedField(data<uint32_t>(), proto.u32s()));
      break;
    case U64:
      TF_RETURN_IF_ERROR(CopyFromRepeatedField(data<uint64_t>(), proto.u64s()));
      break;
    case U128: {
      const std::string& s(proto.u128s());
      TF_RET_CHECK(data<u128>().size() * sizeof(u128) == s.size());
      memcpy(untyped_data(), s.data(), s.size());
      break;
    }
    case U256: {
      const std::string& s(proto.u256s());
      TF_RET_CHECK(data<u256>().size() * sizeof(u256) == s.size());
      memcpy(untyped_data(), s.data(), s.size());
      break;
    }
    case TUPLE:
      return absl::InvalidArgumentError(
          absl::StrFormat("Should not be called on tuple shapes: %s",
                          ShapeUtil::HumanString(subshape())));
#define ZK_DTYPES_CASE(cpp_type, unused, enum, member_name)               \
  case enum: {                                                            \
    const std::string& s(proto.member_name##s());                         \
    TF_RET_CHECK(data<cpp_type>().size() * sizeof(cpp_type) == s.size()); \
    memcpy(untyped_data(), s.data(), s.size());                           \
    break;                                                                \
  }
      ZK_DTYPES_PUBLIC_TYPE_LIST(ZK_DTYPES_CASE)
#undef ZK_DTYPES_CASE
    default:
      return absl::InvalidArgumentError(
          absl::StrFormat("Is called on unsupported shape: %s",
                          ShapeUtil::HumanString(subshape())));
  }
  return absl::OkStatus();
}

bool LiteralBase::Piece::IsKnown() const {
  if (array_value_state_ != ArrayValueState::kKnown) {
    return false;
  }
  if (subshape().IsTuple()) {
    bool are_all_leaf_arrays_known = true;
    ForEachSubpiece([&are_all_leaf_arrays_known](const ShapeIndex& index,
                                                 const Piece& piece) {
      if (!piece.subshape().IsArray()) {
        return;
      }
      are_all_leaf_arrays_known &= piece.IsKnown();
    });
    return are_all_leaf_arrays_known;
  }
  return true;
}

LiteralProto LiteralBase::ToProto() const {
  LiteralProto proto;
  root_piece().ForEachSubpiece(
      [&](const ShapeIndex& index, const Piece& piece) {
        LiteralProto* proto_piece = &proto;
        for (int64_t i : index) {
          while (proto_piece->tuple_literals_size() <= i) {
            proto_piece->add_tuple_literals();
          }
          proto_piece = proto_piece->mutable_tuple_literals(i);
        }
        piece.WriteToProto(proto_piece);
      });

  return proto;
}

const void* LiteralBase::untyped_data(const ShapeIndex& shape_index) const {
  return piece(shape_index).untyped_data();
}

void* MutableLiteralBase::untyped_data(const ShapeIndex& shape_index) {
  return piece(shape_index).untyped_data();
}

int64_t LiteralBase::size_bytes(const ShapeIndex& shape_index) const {
  return piece(shape_index).size_bytes_dense();
}

void MutableBorrowingLiteral::CopyPieceSubtree(const Shape& shape,
                                               const Piece* src_piece,
                                               Piece* dest_piece) {
  DCHECK(ShapeUtil::Equal(src_piece->subshape(), dest_piece->subshape()))
      << "src_piece has shape: "
      << ShapeUtil::HumanString(src_piece->subshape())
      << "dest_piece has shape: "
      << ShapeUtil::HumanString(dest_piece->subshape());
  dest_piece->set_array_value_state(src_piece->get_array_value_state());
  if (shape.IsTuple()) {
    for (int i = 0; i < ShapeUtil::TupleElementCount(shape); ++i) {
      const Shape& subshape = shape.tuple_shapes(i);

      Piece child_piece;
      child_piece.set_subshape(&subshape);

      CopyPieceSubtree(subshape, &src_piece->child(i), &child_piece);

      dest_piece->emplace_back(std::move(child_piece));
    }
  } else if (shape.IsArray()) {
    dest_piece->set_buffer(const_cast<char*>(src_piece->buffer()));
  }
}

MutableLiteralBase::~MutableLiteralBase() = default;

MutableBorrowingLiteral::MutableBorrowingLiteral(
    const MutableBorrowingLiteral& literal) {
  shape_ = literal.shape_.Clone();
  CHECK(LayoutUtil::HasLayout(*shape_));

  root_piece_ = new Piece();
  root_piece_->set_subshape(shape_.get());

  CopyPieceSubtree(*shape_, &literal.root_piece(), root_piece_);
}

MutableBorrowingLiteral& MutableBorrowingLiteral::operator=(
    const MutableBorrowingLiteral& literal) {
  shape_ = literal.shape_.Clone();
  CHECK(LayoutUtil::HasLayout(*shape_));

  root_piece_ = new Piece();
  root_piece_->set_subshape(shape_.get());

  CopyPieceSubtree(*shape_, &literal.root_piece(), root_piece_);

  return *this;
}

MutableBorrowingLiteral::MutableBorrowingLiteral(MutableLiteralBase* literal) {
  shape_ = literal->shape_.Clone();
  CHECK(LayoutUtil::HasLayout(*shape_));

  root_piece_ = new Piece();
  root_piece_->set_subshape(shape_.get());

  CopyPieceSubtree(*shape_, &literal->root_piece(), root_piece_);
}

MutableBorrowingLiteral::MutableBorrowingLiteral(
    MutableBorrowingLiteral literal, const ShapeIndex& view_root) {
  shape_ = std::make_unique<Shape>(literal.piece(view_root).subshape());
  CHECK(LayoutUtil::HasLayout(*shape_));

  root_piece_ = new Piece();
  root_piece_->set_subshape(shape_.get());

  CopyPieceSubtree(*shape_, &literal.piece(view_root), root_piece_);
}

MutableBorrowingLiteral::MutableBorrowingLiteral(const char* src_buf_ptr,
                                                 const Shape& shape) {
  shape_ = std::make_unique<Shape>(shape);
  CHECK(LayoutUtil::HasLayout(*shape_));
  CHECK(!shape_->IsTuple());

  root_piece_ = new Piece();
  root_piece_->set_subshape(shape_.get());
  root_piece_->set_buffer(const_cast<char*>(src_buf_ptr));
}

MutableBorrowingLiteral::MutableBorrowingLiteral(absl::Span<char*> src_buf_ptrs,
                                                 const Shape& shape) {
  shape_ = std::make_unique<Shape>(shape);
  if (!shape_->IsTuple()) {
    CHECK_EQ(src_buf_ptrs.size(), 1);
    root_piece_ = new Piece();
    root_piece_->set_subshape(shape_.get());
    root_piece_->set_buffer(const_cast<char*>(src_buf_ptrs[0]));
  } else {
    CHECK(!ShapeUtil::IsNestedTuple(*shape_));
    CHECK_EQ(src_buf_ptrs.size(), ShapeUtil::TupleElementCount(*shape_));
    root_piece_ = new Piece();
    root_piece_->set_subshape(shape_.get());

    for (int i = 0; i < src_buf_ptrs.size(); ++i) {
      Piece child_piece;
      const auto& src_shape = shape_->tuple_shapes(i);
      CHECK(src_shape.IsArray());
      child_piece.set_subshape(&src_shape);
      child_piece.set_buffer(src_buf_ptrs[i]);
      root_piece_->emplace_back(std::move(child_piece));
    }
  }
}

MutableBorrowingLiteral::MutableBorrowingLiteral(
    ShapeTree<char*> src_buf_ptrs) {
  shape_ = std::make_unique<Shape>(src_buf_ptrs.shape());

  root_piece_ = new Piece();
  root_piece_->set_subshape(shape_.get());
  BuildPieceSubtree(*shape_, root_piece_);

  root_piece_->ForEachMutableSubpiece(
      [&](const ShapeIndex& index, Piece* piece) {
        if (ShapeUtil::GetSubshape(*shape_, index).IsTuple()) {
          DCHECK_EQ(src_buf_ptrs.element(index), nullptr)
              << "Tuples should not have buffer pointers";
          return;
        }
        piece->set_buffer(const_cast<char*>(src_buf_ptrs.element(index)));
      });
}

MutableBorrowingLiteral::~MutableBorrowingLiteral() {
  if (root_piece_ != nullptr) {
    delete root_piece_;
  }
}

void MutableLiteralBase::SetDynamicSize(int64_t dim_index, int32_t size) {
  return SetDynamicSize(dim_index, {}, size);
}

void MutableLiteralBase::SetDynamicSize(int64_t dim_index,
                                        const ShapeIndex& shape_index,
                                        int32_t size) {
  Shape* subshape =
      ShapeUtil::GetMutableSubshape(mutable_shape_do_not_use(), shape_index);
  CHECK(LayoutUtil::IsDenseArray(*subshape))
      << __func__ << " is only supported for dense arrays: " << *subshape;
  CHECK_GE(subshape->dimensions(dim_index), size);
  subshape->set_dynamic_dimension(dim_index, true);
  CHECK_EQ(&piece(shape_index).subshape(), subshape);

  piece(shape_index).SetDynamicSize(dim_index, size);
}

absl::Status Literal::MoveFrom(Literal&& src_literal,
                               const ShapeIndex& dest_shape_index) {
  const Shape& dest_subshape =
      ShapeUtil::GetSubshape(shape(), dest_shape_index);
  if (!ShapeUtil::Equal(dest_subshape, src_literal.shape())) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Destination subshape not equal to source shape: %s vs %s",
        ShapeUtil::HumanString(dest_subshape),
        ShapeUtil::HumanString(src_literal.shape())));
  }

  src_literal.root_piece_.ForEachMutableSubpiece(
      [&](const ShapeIndex& src_index, Piece* src_piece) {
        if (!src_piece->subshape().IsArray()) {
          return;
        }

        ShapeIndex dest_index = dest_shape_index;
        for (int64_t i : src_index) {
          dest_index.push_back(i);
        }
        Piece& dest_piece = piece(dest_index);
        dest_piece.DeallocateBuffers();
        dest_piece.MoveDataFrom(*src_piece);
      });

  src_literal.shape_ = MaybeOwningShapePtr(&NilShape());
  src_literal.root_piece_ = Piece();
  src_literal.root_piece_.set_subshape(src_literal.shape_.get());

  return absl::OkStatus();
}

absl::Status MutableLiteralBase::CopySliceFrom(
    const LiteralSlice& src_literal, absl::Span<const int64_t> src_base,
    absl::Span<const int64_t> dest_base, absl::Span<const int64_t> copy_size) {
  TF_RET_CHECK(LayoutUtil::IsDenseArray(shape())) << shape();
  TF_RET_CHECK(LayoutUtil::IsDenseArray(src_literal.shape()))
      << src_literal.shape();
  TF_RET_CHECK(ShapeUtil::SameElementType(src_literal.shape(), shape()));
  TF_RET_CHECK(src_literal.shape().rank() == src_base.size());
  TF_RET_CHECK(shape().rank() == dest_base.size());

  return primitive_util::ArrayTypeSwitch<absl::Status>(
      [&](auto primitive_type_constant) -> absl::Status {
        using NativeT = primitive_util::NativeTypeOf<primitive_type_constant>;
        return CopySliceFromInternal<NativeT>(src_literal, src_base, dest_base,
                                              copy_size);
      },
      shape().element_type());
}

void MutableLiteralBase::PopulateR1(const tsl::core::Bitmap& values) {
  CHECK(shape().IsArray());
  CHECK_EQ(shape().rank(), 1);
  CHECK_EQ(element_count(), values.bits());
  CHECK_EQ(shape().element_type(), PRED);
  for (int64_t i = 0; i < static_cast<int64_t>(values.bits()); ++i) {
    Set({i}, values.get(i));
  }
}

void MutableLiteralBase::PopulateInplaceInternal(
    absl::FunctionRef<void(void*, absl::Span<const int64_t>, int)> populator,
    bool parallel) {
  const Shape& this_shape = shape();
  const int64_t rank = this_shape.rank();
  DCHECK(LayoutUtil::IsDenseArray(this_shape));
  char* const dest_base = static_cast<char*>(untyped_data());
  if (rank > 0) {
    StrideConfig stride_config(this_shape, this_shape, this_shape.dimensions());
    const int64_t primitive_size =
        ShapeUtil::ByteSizeOfPrimitiveType(shape().element_type());
    const int64_t num_elements = ShapeUtil::ElementsIn(shape());
    // If we are rank-1 and we are `parallel`, it is better to use a smaller
    // `step` than what `StrideConfig` does: stick the entire dimension in the
    // inner-most loop.
    if (parallel && this_shape.rank() == 1) {
      const int64_t thread_count =
          ShapeUtil::GetForEachIndexParallelThreadCount();
      // Let's just divide up the array into small amounts per thread.
      stride_config.dest_stride = stride_config.minor_loop_size =
          num_elements > 32 ? std::max<int64_t>(num_elements / thread_count, 1)
                            : num_elements;
      stride_config.step = {stride_config.minor_loop_size};
    }

    auto init_function = [&](absl::Span<const int64_t> indexes,
                             int thread_id) -> absl::StatusOr<bool> {
      const int64_t index =
          IndexUtil::MultidimensionalIndexToLinearIndex(shape(), indexes);
      DimensionVector minor_scan_indexes(rank, 0);
      std::copy(indexes.begin(), indexes.end(), minor_scan_indexes.begin());
      char* dest_ptr = dest_base + index * primitive_size;
      char* const dest_end =
          dest_base +
          // This handles the case where minor_loop_size does not evenly divide
          // the most minor dimension.
          std::min(index + stride_config.minor_loop_size, num_elements) *
              primitive_size;
      while (dest_ptr < dest_end) {
        populator(dest_ptr, minor_scan_indexes, thread_id);
        ++minor_scan_indexes[stride_config.minor_dimension];
        dest_ptr += primitive_size;
      }
      return true;
    };
    if (parallel) {
      ShapeUtil::ForEachIndexParallel(this_shape, stride_config.base,
                                      stride_config.dimensions,
                                      stride_config.step, init_function);
    } else {
      ShapeUtil::ForEachIndex(
          this_shape, stride_config.base, stride_config.dimensions,
          stride_config.step,
          [&init_function](
              absl::Span<const int64_t> indexes) -> absl::StatusOr<bool> {
            auto result_ignored = init_function(indexes, /*thread_id=*/-1);
            return true;
          });
    }
  } else {
    // For scalars.
    populator(dest_base, {}, /*thread_id=*/-1);
  }
}

absl::Status MutableLiteralBase::PopulateInplace(
    absl::FunctionRef<void(void*, absl::Span<const int64_t>)> populator) {
  TF_RET_CHECK(LayoutUtil::IsDenseArray(shape()))
      << __func__ << " is only supported for dense arrays: " << shape();
  PopulateInplaceInternal(
      [&](void* dest, absl::Span<const int64_t> indexes, int /*thread_id*/) {
        return populator(dest, indexes);
      },
      /*parallel=*/false);
  return absl::OkStatus();
}

absl::Status MutableLiteralBase::PopulateInplaceParallel(
    absl::FunctionRef<void(void*, absl::Span<const int64_t>, int)> populator) {
  TF_RET_CHECK(LayoutUtil::IsDenseArray(shape()))
      << __func__ << " is only supported for dense arrays: " << shape();
  PopulateInplaceInternal(populator,
                          /*parallel=*/element_count() > 32);
  return absl::OkStatus();
}

absl::Status MutableLiteralBase::CopyFrom(const LiteralSlice& src_literal,
                                          const ShapeIndex& dest_shape_index,
                                          const ShapeIndex& src_shape_index,
                                          bool only_dynamic_bound) {
  const Shape& dest_subshape =
      ShapeUtil::GetSubshape(shape(), dest_shape_index);
  const Shape& src_subshape =
      ShapeUtil::GetSubshape(src_literal.shape(), src_shape_index);
  if (only_dynamic_bound) {
    auto& bound_shape =
        dest_subshape.is_static() ? src_subshape : dest_subshape;
    auto& compact_shape =
        dest_subshape.is_static() ? dest_subshape : src_subshape;
    CHECK(ShapeUtil::DynamicShapeIsCompatible(compact_shape, bound_shape))
        << compact_shape.ToString() << " vs " << bound_shape.ToString();
  } else {
    if (!ShapeUtil::Compatible(dest_subshape, src_subshape)) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Destination subshape incompatible with source subshape: %s vs %s",
          ShapeUtil::HumanString(dest_subshape),
          ShapeUtil::HumanString(src_subshape)));
    }
  }
  return mutable_root_piece().ForEachMutableSubpieceWithStatus(
      [&](const ShapeIndex& index, Piece* piece) {
        if (!piece->subshape().IsArray()) {
          return absl::OkStatus();
        }

        // Determine if this index is in the part of this literal that we want
        // to copy over from src_literal.
        bool in_subtree_to_copy = true;
        for (int i = 0; i < dest_shape_index.size(); ++i) {
          if (index[i] != dest_shape_index[i]) {
            in_subtree_to_copy = false;
            break;
          }
        }
        if (!in_subtree_to_copy) {
          return absl::OkStatus();
        }
        // Construct the index of the corresponding piece in the source literal.
        ShapeIndex src_piece_index = src_shape_index;
        for (int64_t i = dest_shape_index.size(), end = index.size(); i < end;
             ++i) {
          src_piece_index.push_back(index[i]);
        }
        TF_RETURN_IF_ERROR(
            piece->CopyFrom(src_literal.piece(src_piece_index),
                            /*only_dynamic_bound=*/only_dynamic_bound));
        return absl::OkStatus();
      });
}

Literal LiteralBase::Relayout(const Layout& new_layout,
                              const ShapeIndex& shape_index) const {
  // Create new shape with 'new_layout' set at the given shape index.
  Shape new_shape = shape();
  Shape* subshape = ShapeUtil::GetMutableSubshape(&new_shape, shape_index);
  CHECK_OK(LayoutUtil::ValidateLayoutForShape(new_layout, *subshape));
  *subshape->mutable_layout() = new_layout;
  // LINT.IfChange
  // s4 literals are stored in uint8_t/int8_t, therefore element_size_in_bits
  // must be removed.
  if (subshape->layout().element_size_in_bits() == 4) {
    subshape->mutable_layout()->set_element_size_in_bits(0);
  }
  // LINT.ThenChange(//tensorflow/compiler/xla/types.h)
  Literal result(new_shape);
  CHECK_OK(result.CopyFrom(*this));
  return result;
}

Literal LiteralBase::Relayout(const Shape& shape_with_layout) const {
  CHECK(ShapeUtil::Compatible(shape_with_layout, shape()))
      << "Given shape_with_layout " << ShapeUtil::HumanString(shape_with_layout)
      << " not compatible with literal shape "
      << ShapeUtil::HumanString(shape());
  Literal result = CreateFromShape(shape_with_layout);
  ShapeUtil::ForEachSubshape(
      result.shape(),
      [this, &result](const Shape& subshape, const ShapeIndex& index) {
        if (subshape.IsArray()) {
          CHECK_OK(result.CopyFrom(*this,
                                   /*dest_shape_index=*/index,
                                   /*src_shape_index=*/index));
        }
      });
  return result;
}

Literal LiteralBase::ToStatic() const {
  // Materialize a static shape by fixing dynamic dims to runtime sizes and
  // copying only valid data.
  Shape new_shape = shape();
  ShapeUtil::ForEachMutableSubshape(
      &new_shape, [this](Shape* subshape, const ShapeIndex& index) {
        if (!subshape->IsArray()) {
          return;
        }
        for (int64_t i = 0; i < subshape->rank(); ++i) {
          // GetDynamicSize has a 32-bit return type and may truncate static
          // dimensions, so make sure to skip.
          if (!subshape->is_dynamic_dimension(i)) continue;
          subshape->set_dynamic_dimension(i, false);
          subshape->set_dimensions(i, GetDynamicSize(i, index));
        }
      });
  Literal result(new_shape);
  CHECK_OK(result.CopyFrom(*this, {}, {}, /*only_dynamic_bound=*/true));
  return result;
}

Literal LiteralBase::ToBoundedDynamic(const Shape& bounded_shape) const {
  CHECK(bounded_shape.is_dynamic());
  Literal result(bounded_shape);
  ShapeUtil::ForEachSubshape(
      shape(), [&](const Shape& subshape, const ShapeIndex& index) {
        if (!subshape.IsArray()) {
          return;
        }
        for (int64_t i = 0; i < subshape.rank(); ++i) {
          if (!bounded_shape.is_dynamic_dimension(i)) continue;
          result.SetDynamicSize(i, subshape.dimensions(i));
        }
      });
  CHECK_OK(result.CopyFrom(*this, {}, {}, /*only_dynamic_bound=*/true));

  return result;
}

BorrowingLiteral::BorrowingLiteral(const char* src_buf_ptr, const Shape& shape)
    : shape_(std::make_unique<Shape>(shape)) {
  CHECK(shape_->IsArray());
  CHECK(LayoutUtil::HasLayout(*shape_));

  root_piece_ = Piece();
  root_piece_.set_subshape(shape_.get());
  root_piece_.set_buffer(const_cast<char*>(src_buf_ptr));
}

BorrowingLiteral::BorrowingLiteral(absl::Span<const char* const> src_buf_ptrs,
                                   const Shape& shape)
    : LiteralBase(), shape_(std::make_unique<Shape>(shape)) {
  CHECK(shape_->IsTuple());
  CHECK(!ShapeUtil::IsNestedTuple(*shape_));
  CHECK_EQ(src_buf_ptrs.size(), ShapeUtil::TupleElementCount(*shape_));
  root_piece_ = Piece();
  root_piece_.set_subshape(shape_.get());
  BuildPieceSubtree(*shape_, &root_piece_);

  for (int i = 0, end = src_buf_ptrs.size(); i < end; ++i) {
    const auto& src_shape = shape_->tuple_shapes(i);
    CHECK(src_shape.IsArray());
    root_piece_.child(i).set_buffer(const_cast<char*>(src_buf_ptrs[i]));
  }
}

BorrowingLiteral::BorrowingLiteral(ShapeTree<const char*> src_buf_ptrs)
    : LiteralBase(), shape_(std::make_unique<Shape>(src_buf_ptrs.shape())) {
  root_piece_ = Piece();
  root_piece_.set_subshape(shape_.get());
  BuildPieceSubtree(*shape_, &root_piece_);

  root_piece_.ForEachMutableSubpiece(
      [&](const ShapeIndex& index, Piece* piece) {
        if (ShapeUtil::GetSubshape(*shape_, index).IsTuple()) {
          DCHECK_EQ(src_buf_ptrs.element(index), nullptr)
              << "Tuples should not have buffer pointers";
          return;
        }
        piece->set_buffer(const_cast<char*>(src_buf_ptrs.element(index)));
      });
}

}  // namespace zkx
