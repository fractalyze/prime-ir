/* Copyright 2022 The OpenXLA Authors.
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

#ifndef ZKX_PYTHON_IFRT_DTYPE_H_
#define ZKX_PYTHON_IFRT_DTYPE_H_

#include <optional>
#include <ostream>
#include <string>

#include "absl/status/statusor.h"

#include "zk_dtypes/include/all_types.h"
#include "zkx/primitive_util.h"
#include "zkx/python/ifrt/dtype.pb.h"
#include "zkx/python/ifrt/serdes_default_version_accessor.h"
#include "zkx/python/ifrt/serdes_version.h"

namespace zkx::ifrt {

// Data type of an element.
//
// Based on `zkx::PrimitiveType`. Differences:
//
// * Match the Google C++ style guide for enumerator naming.
// * Rename PRIMITIVE_TYPE_INVALID to kInvalid.
// * Remove TUPLE, OPAQUE_TYPE.
// * Add kString.
class DType {
 public:
  // LINT.IfChange
  enum Kind {
    // Invalid data type.
    kInvalid = 0,

    // Predicates are two-state booleans.
    kPred = 1,

    // Signed integral values of fixed width.
    kS2 = 3,
    kS4 = 4,
    kS8 = 5,
    kS16 = 6,
    kS32 = 7,
    kS64 = 8,
    kS128 = 59,
    kS256 = 60,

    // Unsigned integral values of fixed width.
    kU2 = 10,
    kU4 = 11,
    kU8 = 12,
    kU16 = 13,
    kU32 = 14,
    kU64 = 15,
    kU128 = 57,
    kU256 = 58,

    // Opaque objects.
    kOpaque = 17,

    // A token type threaded between side-effecting operations. Shapes of this
    // dtype will have empty dimensions.
    kToken = 18,

#define ADD_ENUM(unused, dtype_enum, enum, unused2) \
  k##dtype_enum = static_cast<PrimitiveType>(enum),
    ZK_DTYPES_PUBLIC_TYPE_LIST(ADD_ENUM)
#undef ADD_ENUM

    // Variable-length string represented as raw bytes, as in `bytes` in Python,
    // i.e., no encoding enforcement. String is not support in ZKX. DType.Kind
    // needs to match zkx.PrimitiveType enum, so choose a large enum to avoid
    // collision.
    kString = 99,
  };
  // LINT.ThenChange(dtype.proto:DTypeProtoKind)

  explicit DType(Kind kind) : kind_(kind) {}
  DType(const DType&) = default;
  DType(DType&&) = default;
  DType& operator=(const DType&) = default;
  DType& operator=(DType&&) = default;

  Kind kind() const { return kind_; }

  bool operator==(const DType& other) const { return kind_ == other.kind_; }
  bool operator!=(const DType& other) const { return kind_ != other.kind_; }

  template <typename H>
  friend H AbslHashValue(H h, const DType& value) {
    return H::combine(std::move(h), value.kind());
  }

  // Returns the byte size of a single element of this DType. Returns
  // std::nullopt if not aligned to a byte boundary or there is no fixed size
  // (such as kString).
  std::optional<int> byte_size() const;

  // Returns the bit size of a single element of this DType. Returns
  // std::nullopt if there is no fixed size.
  std::optional<int> bit_size() const;

  // Constructs `DType` from `DTypeProto`.
  static absl::StatusOr<DType> FromProto(const DTypeProto& proto);

  // Returns a `DTypeProto` representation.
  DTypeProto ToProto(
      SerDesVersion version = SerDesDefaultVersionAccessor::Get()) const;

  // TODO(hyeontaek): Remove this method in favor of AbslStringify.
  std::string DebugString() const;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const DType& dtype) {
    sink.Append(dtype.DebugString());
  }

 private:
  Kind kind_;
};

std::ostream& operator<<(std::ostream& os, const DType& dtype);

}  // namespace zkx::ifrt

#endif  // ZKX_PYTHON_IFRT_DTYPE_H_
