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

#ifndef ZKX_LITERAL_UTIL_H_
#define ZKX_LITERAL_UTIL_H_

#include <array>
#include <initializer_list>
#include <iterator>
#include <vector>

#include "absl/types/span.h"

#include "xla/tsl/lib/core/bitmap.h"
#include "zkx/array4d.h"
#include "zkx/literal.h"
#include "zkx/primitive_util.h"
#include "zkx/shape_util.h"

namespace zkx {

class LiteralUtil {
 public:
  // Returns a literal scalar representing the first element.
  static Literal GetFirstScalarLiteral(const LiteralSlice& literal);
  // Returns a literal scalar representing the element at `multi_index`.
  static Literal GetScalarLiteral(const LiteralBase& literal,
                                  absl::Span<const int64_t> multi_index);
  // Sets the value of the element at `multi_index` with a scalar literal.
  static void SetScalarLiteral(MutableLiteralBase& literal,
                               absl::Span<const int64_t> multi_index,
                               const LiteralBase& scalar);

  // Creates a new literal of a given rank. To minimize ambiguity (for users
  // and the compiler) these CreateR[0-2] methods should explicitly specify the
  // native type. For example:
  //
  //  CreateR1<float>({1.0, 42.0});
  //  CreateR2<uint32_t>({{1, 2}, {3, 4}});
  //
  // The variants not ending with WithLayout use the default ZKX layout for the
  // literal's linear representation in memory.
  template <typename NativeT>
  static Literal CreateR0(NativeT value);
  template <typename T>
  static Literal CreateR0(PrimitiveType primitive_type, T value);
  template <typename NativeT>
  static Literal CreateR1(absl::Span<const NativeT> values);
  static Literal CreateR1(const tsl::core::Bitmap& values);
  template <typename NativeT>
  static Literal CreateR2(
      std::initializer_list<std::initializer_list<NativeT>> values);
  template <typename NativeT>
  static Literal CreateR2WithLayout(
      std::initializer_list<std::initializer_list<NativeT>> values,
      const Layout& layout);
  template <typename NativeT>
  static Literal CreateR3(std::initializer_list<
                          std::initializer_list<std::initializer_list<NativeT>>>
                              values);
  template <typename NativeT>
  static Literal CreateR3WithLayout(
      std::initializer_list<
          std::initializer_list<std::initializer_list<NativeT>>>
          values,
      const Layout& layout);

  // Creates a scalar literal value zero of the given primitive type.
  static Literal Zero(PrimitiveType primitive_type);
  // Creates a scalar literal value one of the given primitive type.
  static Literal One(PrimitiveType primitive_type);
  // Creates a scalar literal value containing the minimum value of the given
  // primitive type.
  static Literal MinValue(PrimitiveType primitive_type);
  // Creates a scalar literal value containing the maximum value of the given
  // primitive type.
  static Literal MaxValue(PrimitiveType primitive_type);

  // Creates a literal of the given shape where each element is `value`.
  template <typename NativeT>
  static Literal CreateFullWithDescendingLayout(
      absl::Span<const int64_t> dimensions, NativeT value);
  template <typename NativeT>
  static Literal CreateFull(absl::Span<const int64_t> dimensions,
                            NativeT value);

  // Creates a new literal from an Array type. The variants not ending with
  // WithLayout use the default XLA layout for the literal's linear
  // representation in memory.
  template <typename NativeT>
  static Literal CreateFromArray(const Array<NativeT>& values);
  template <typename NativeT>
  static Literal CreateFromArrayWithLayout(const Array<NativeT>& values,
                                           const Layout& layout);
  template <typename NativeT>
  static Literal CreateR2FromArray2D(const Array2D<NativeT>& values);
  template <typename NativeT>
  static Literal CreateR2FromArray2DWithLayout(const Array2D<NativeT>& values,
                                               const Layout& layout);
  template <typename NativeT>
  static Literal CreateR3FromArray3D(const Array3D<NativeT>& values);
  template <typename NativeT>
  static Literal CreateR3FromArray3DWithLayout(const Array3D<NativeT>& values,
                                               const Layout& layout);
  template <typename NativeT>
  static Literal CreateR4(
      std::initializer_list<std::initializer_list<
          std::initializer_list<std::initializer_list<NativeT>>>>
          values);
  template <typename NativeT>
  static Literal CreateR4WithLayout(
      std::initializer_list<std::initializer_list<
          std::initializer_list<std::initializer_list<NativeT>>>>
          values,
      const Layout& layout);
  template <typename NativeT>
  static Literal CreateR4FromArray4D(const Array4D<NativeT>& values);
  template <typename NativeT>
  static Literal CreateR4FromArray4DWithLayout(const Array4D<NativeT>& values,
                                               const Layout& layout);

  // Returns a tuple literal composed of given literals. Data is copied from the
  // given elements into the returned literal.
  static Literal MakeTuple(absl::Span<const Literal* const> elements);

  static Literal MakeTupleFromSlices(absl::Span<const LiteralSlice> elements);

  // As above, but intended to be invoked with move semantics; i.e.
  //
  //  std::vector<Literal> elements = ...;
  //  auto result = LiteralUtil::MakeTupleOwned(std::move(elements));
  //
  // This would have been declared as an overload, but there is ambiguity
  // in invocation between the above signature and this one.
  static Literal MakeTupleOwned(std::vector<Literal> elements);

  // This overload lets you pass a list of Literals to MakeTupleOwned:
  //
  //   LiteralUtil::MakeTupleOwned(LiteralUtil::CreateR1(...), ...).
  //
  // Simply relying on the MakeTupleOwned(std::vector<Literal>)
  // overload doesn't work because std::initializer_list's elements are always
  // const.
  //
  // The arguments to this function must all be Literal.
  template <typename... Ts>
  static Literal MakeTupleOwned(Ts... elements) {
    std::array<Literal, sizeof...(Ts)> arr{std::move(elements)...};
    std::vector<Literal> v;
    v.insert(v.begin(), std::make_move_iterator(arr.begin()),
             std::make_move_iterator(arr.end()));
    return MakeTupleOwned(std::move(v));
  }

  // Create a constant token literal. Token types have no value.
  static Literal CreateToken();

  // Creates a new Literal object with its values having the primitive_type
  // type, and with dimensions defined by the dimensions parameter.
  // The content of the literal values is the default value of the primitive
  // type of literal itself (0 for numeric types, and false for predicates).
  static Literal CreateFromDimensions(PrimitiveType primitive_type,
                                      absl::Span<const int64_t> dimensions);

  // Returns a multi-dimensional index as a string. For example: '{7, 8}' will
  // be returned for a 2-dimensional index with dimension 0 index equal to 7,
  // dimension 1 equal to 8.
  static std::string MultiIndexAsString(absl::Span<const int64_t> multi_index);

  // Converts the given literal to a scalar int64_t, if possible.
  //
  // Fails if the literal is not an integral type or if the value it contains
  // cannot be represented as an int64_t.
  static std::optional<int64_t> LiteralAsScalarInt64(const Literal& l);
};

// static
template <typename NativeT>
Literal LiteralUtil::CreateR0(NativeT value) {
  Literal literal(ShapeUtil::MakeShape(
      primitive_util::NativeToPrimitiveType<NativeT>(), {}));
  literal.Set({}, value);
  return literal;
}

// static
template <typename T>
Literal LiteralUtil::CreateR0(PrimitiveType primitive_type, T value) {
  return primitive_util::ArrayTypeSwitch<Literal>(
      [&value](auto type) {
        using NativeT = primitive_util::NativeTypeOf<type>;
        return CreateR0(static_cast<NativeT>(value));
      },
      primitive_type);
}

// static
template <typename NativeT>
Literal LiteralUtil::CreateR1(absl::Span<const NativeT> values) {
  Literal literal(
      ShapeUtil::MakeShape(primitive_util::NativeToPrimitiveType<NativeT>(),
                           {static_cast<int64_t>(values.size())}));
  literal.PopulateR1(values);
  return literal;
}

// static
template <typename NativeT>
Literal LiteralUtil::CreateR2WithLayout(
    std::initializer_list<std::initializer_list<NativeT>> values,
    const Layout& layout) {
  Literal literal(ShapeUtil::MakeShapeWithDenseLayout(
      primitive_util::NativeToPrimitiveType<NativeT>(),
      {static_cast<int64_t>(values.size()),
       static_cast<int64_t>(values.begin()->size())},
      layout.minor_to_major()));
  literal.PopulateR2(values);
  return literal;
}

// static
template <typename NativeT>
Literal LiteralUtil::CreateR2(
    std::initializer_list<std::initializer_list<NativeT>> values) {
  return CreateR2WithLayout(values, LayoutUtil::GetDefaultLayoutForR2());
}

// static
template <typename NativeT>
Literal LiteralUtil::CreateR3WithLayout(
    std::initializer_list<std::initializer_list<std::initializer_list<NativeT>>>
        values,
    const Layout& layout) {
  const int64_t d0 = values.size();
  const int64_t d1 = values.begin()->size();
  const int64_t d2 = values.begin()->begin()->size();
  Array3D<NativeT> tmp(d0, d1, d2);
  int64_t i0 = 0;
  for (auto d1_values : values) {
    int64_t i1 = 0;
    for (auto d2_values : d1_values) {
      int64_t i2 = 0;
      for (auto value : d2_values) {
        tmp(i0, i1, i2) = value;
        ++i2;
      }
      ++i1;
    }
    ++i0;
  }
  return CreateR3FromArray3DWithLayout(tmp, layout);
}

// static
template <typename NativeT>
Literal LiteralUtil::CreateR3(
    std::initializer_list<std::initializer_list<std::initializer_list<NativeT>>>
        values) {
  return CreateR3WithLayout(values, LayoutUtil::GetDefaultLayoutForR3());
}

// static
template <typename NativeT>
Literal LiteralUtil::CreateFromArrayWithLayout(const Array<NativeT>& values,
                                               const Layout& layout) {
  Literal literal(ShapeUtil::MakeShapeWithDenseLayout(
      primitive_util::NativeToPrimitiveType<NativeT>(), values.dimensions(),
      layout.minor_to_major()));
  literal.PopulateFromArray(values);
  return literal;
}

// static
template <typename NativeT>
Literal LiteralUtil::CreateFromArray(const Array<NativeT>& values) {
  return CreateFromArrayWithLayout(
      values, LayoutUtil::GetDefaultLayoutForRank(values.num_dimensions()));
}

// static
template <typename NativeT>
Literal LiteralUtil::CreateR2FromArray2DWithLayout(
    const Array2D<NativeT>& values, const Layout& layout) {
  return CreateFromArrayWithLayout(values, layout);
}

// static
template <typename NativeT>
Literal LiteralUtil::CreateR2FromArray2D(const Array2D<NativeT>& values) {
  return CreateFromArray(values);
}

// static
template <typename NativeT>
Literal LiteralUtil::CreateR3FromArray3DWithLayout(
    const Array3D<NativeT>& values, const Layout& layout) {
  return CreateFromArrayWithLayout(values, layout);
}

// static
template <typename NativeT>
Literal LiteralUtil::CreateR3FromArray3D(const Array3D<NativeT>& values) {
  return CreateFromArray(values);
}

// static
template <typename NativeT>
Literal LiteralUtil::CreateR4WithLayout(
    std::initializer_list<std::initializer_list<
        std::initializer_list<std::initializer_list<NativeT>>>>
        values,
    const Layout& layout) {
  const int64_t d0 = values.size();
  const int64_t d1 = values.begin()->size();
  const int64_t d2 = values.begin()->begin()->size();
  const int64_t d3 = values.begin()->begin()->begin()->size();
  Array4D<NativeT> tmp(d0, d1, d2, d3);
  int64_t i0 = 0;
  for (auto d1_values : values) {
    int64_t i1 = 0;
    for (auto d2_values : d1_values) {
      int64_t i2 = 0;
      for (auto d3_values : d2_values) {
        int64_t i3 = 0;
        for (auto value : d3_values) {
          tmp(i0, i1, i2, i3) = value;
          ++i3;
        }
        ++i2;
      }
      ++i1;
    }
    ++i0;
  }
  return CreateR4FromArray4DWithLayout(tmp, layout);
}

// static
template <typename NativeT>
Literal LiteralUtil::CreateR4(
    std::initializer_list<std::initializer_list<
        std::initializer_list<std::initializer_list<NativeT>>>>
        values) {
  return CreateR4WithLayout(values, LayoutUtil::GetDefaultLayoutForR4());
}

// static
template <typename NativeT>
Literal LiteralUtil::CreateR4FromArray4DWithLayout(
    const Array4D<NativeT>& values, const Layout& layout) {
  return CreateFromArrayWithLayout(values, layout);
}

// static
template <typename NativeT>
Literal LiteralUtil::CreateR4FromArray4D(const Array4D<NativeT>& values) {
  return CreateFromArray(values);
}

// static
template <typename NativeT>
Literal LiteralUtil::CreateFullWithDescendingLayout(
    absl::Span<const int64_t> dimensions, NativeT value) {
  Literal literal(ShapeUtil::MakeShapeWithDescendingLayout(
      primitive_util::NativeToPrimitiveType<NativeT>(), dimensions));
  literal.PopulateWithValue(value);
  return literal;
}

// static
template <typename NativeT>
Literal LiteralUtil::CreateFull(absl::Span<const int64_t> dimensions,
                                NativeT value) {
  Literal literal(ShapeUtil::MakeShape(
      primitive_util::NativeToPrimitiveType<NativeT>(), dimensions));
  literal.PopulateWithValue(value);
  return literal;
}

// Generates fake data in a literal of the given shape, or returns an error
// status if the element type is currently unhandled for fake data
// generation. See below for documentation of pseudo_random.
absl::StatusOr<Literal> MakeFakeLiteral(const Shape& shape,
                                        bool pseudo_random = true);

// Similar to MakeFakeLiteral above but takes a random number generator engine
// to enable reusing the engine across randomly generated literals. `limit` is a
// optional pair that contains the min and the max values to be sample for
// integers (integer format only). `is_sorted` sorts the sample data for
// integers (integer format only). `no_duplicates` indicates that there should
// be no duplicate values in each generated array. This is uniqueness is
// best-effort only. Some types (half and bfloat16) are not supported and
// uniqueness cannot be guaranteed if the number of elements exceeds the number
// of different values supported by the type. (floating point format only)
// `max_bits_of_precision` sets the data to have the given number of bits or
// less (integer or floating point formats only).
absl::StatusOr<Literal> MakeFakeLiteral(
    const Shape& shape, std::minstd_rand0* engine,
    std::optional<std::pair<int64_t, int64_t>> limit, bool is_sorted,
    bool no_duplicates, std::optional<int64_t> max_bits_of_precision);

}  // namespace zkx

#endif  // ZKX_LITERAL_UTIL_H_
