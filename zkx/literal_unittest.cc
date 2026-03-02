/* Copyright 2026 The ZKX Authors.

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

#include <cstdint>
#include <vector>

#include "gtest/gtest.h"

#include "xla/tsl/platform/statusor.h"
#include "zk_dtypes/include/all_types.h"
#include "zkx/literal_util.h"
#include "zkx/primitive_util.h"
#include "zkx/tests/literal_test_util.h"

namespace zkx {
namespace {

/// Helper: returns the narrowest unsigned integer PrimitiveType that is at
/// least as wide as the given field type.
template <typename F>
constexpr PrimitiveType IntTypeForField() {
  constexpr auto kFieldType = primitive_util::NativeToPrimitiveType<F>();
  constexpr int kBytes = primitive_util::ByteWidth(kFieldType);
  if constexpr (kBytes <= 4)
    return U32;
  else if constexpr (kBytes <= 8)
    return U64;
  else if constexpr (kBytes <= 16)
    return U128;
  else
    return U256;
}

// ---------------------------------------------------------------------------
// Prime field → integer conversion
// ---------------------------------------------------------------------------

template <typename T>
class PrimeFieldToIntegerTest : public testing::Test {};

// Use PUBLIC list to avoid internal-only types that lack NativeToPrimitiveType.
// Trailing type absorbs the macro's trailing comma.
using PrimeFieldTypes = testing::Types<
#define PRIME_FIELD_TYPE(ActualType, ...) ActualType,
    ZK_DTYPES_PUBLIC_PRIME_FIELD_TYPE_LIST(PRIME_FIELD_TYPE)
#undef PRIME_FIELD_TYPE
        zk_dtypes::BabybearMont>;

TYPED_TEST_SUITE(PrimeFieldToIntegerTest, PrimeFieldTypes);

TYPED_TEST(PrimeFieldToIntegerTest, RoundTripR0) {
  using F = TypeParam;
  constexpr auto kFieldType = primitive_util::NativeToPrimitiveType<F>();
  constexpr PrimitiveType kIntType = IntTypeForField<F>();

  F field_val = F::FromUnchecked(42);
  Literal field_lit = LiteralUtil::CreateR0<F>(field_val);

  TF_ASSERT_OK_AND_ASSIGN(Literal int_lit, field_lit.Convert(kIntType));

  // The integer literal should hold the raw stored value (42).
  if constexpr (kIntType == U32) {
    EXPECT_EQ(int_lit.Get<uint32_t>({}), uint32_t{42});
  } else if constexpr (kIntType == U64) {
    EXPECT_EQ(int_lit.Get<uint64_t>({}), uint64_t{42});
  } else if constexpr (kIntType == U128) {
    EXPECT_EQ(static_cast<uint64_t>(int_lit.Get<u128>({})), uint64_t{42});
  } else if constexpr (kIntType == U256) {
    EXPECT_EQ(static_cast<uint64_t>(int_lit.Get<u256>({})), uint64_t{42});
  }

  // Round-trip: integer → field should give back the same element.
  TF_ASSERT_OK_AND_ASSIGN(Literal round_trip, int_lit.Convert(kFieldType));
  EXPECT_TRUE(LiteralTestUtil::Equal(field_lit, round_trip));
}

TYPED_TEST(PrimeFieldToIntegerTest, RoundTripR1) {
  using F = TypeParam;
  constexpr auto kFieldType = primitive_util::NativeToPrimitiveType<F>();
  constexpr PrimitiveType kIntType = IntTypeForField<F>();

  std::vector<F> vals = {F::FromUnchecked(0), F::FromUnchecked(1),
                         F::FromUnchecked(100)};
  Literal field_lit = LiteralUtil::CreateR1<F>(vals);

  TF_ASSERT_OK_AND_ASSIGN(Literal int_lit, field_lit.Convert(kIntType));
  TF_ASSERT_OK_AND_ASSIGN(Literal round_trip, int_lit.Convert(kFieldType));
  EXPECT_TRUE(LiteralTestUtil::Equal(field_lit, round_trip));
}

TYPED_TEST(PrimeFieldToIntegerTest, NarrowIntegerReturnsError) {
  using F = TypeParam;

  // All prime fields are >= 4 bytes, so U8 should always fail.
  Literal field_lit = LiteralUtil::CreateR0<F>(F::FromUnchecked(1));
  auto result = field_lit.Convert(U8);
  EXPECT_FALSE(result.ok());
}

// ---------------------------------------------------------------------------
// Extension field → integer conversion (one-way only)
// ---------------------------------------------------------------------------
// Note: ext_field → integer extracts values()[0].value() (a base-field scalar),
// but the width check requires the integer type to be >= the extension field
// width. The inverse (integer → ext_field) requires the integer to be <=
// base-field width. These constraints are incompatible, so round-trip is not
// possible through Convert(). We only test one-way conversion here.

template <typename T>
class ExtFieldToIntegerTest : public testing::Test {};

using ExtFieldTypes = testing::Types<
#define EXT_FIELD_TYPE(ActualType, ...) ActualType,
    ZK_DTYPES_PUBLIC_EXT_FIELD_TYPE_LIST(EXT_FIELD_TYPE)
#undef EXT_FIELD_TYPE
        zk_dtypes::BabybearX4Mont>;

TYPED_TEST_SUITE(ExtFieldToIntegerTest, ExtFieldTypes);

TYPED_TEST(ExtFieldToIntegerTest, ConvertToInteger) {
  using F = TypeParam;
  using BasePF = typename F::BasePrimeField;
  // The integer type must be >= the full extension field width for the
  // ByteWidth check to pass.
  constexpr PrimitiveType kIntType = IntTypeForField<F>();

  F field_val = F(BasePF::FromUnchecked(77));
  Literal field_lit = LiteralUtil::CreateR0<F>(field_val);

  TF_ASSERT_OK_AND_ASSIGN(Literal int_lit, field_lit.Convert(kIntType));

  // Verify the extracted value matches the base-field raw value.
  if constexpr (kIntType == U32) {
    EXPECT_EQ(int_lit.Get<uint32_t>({}), uint32_t{77});
  } else if constexpr (kIntType == U64) {
    EXPECT_EQ(int_lit.Get<uint64_t>({}), uint64_t{77});
  } else if constexpr (kIntType == U128) {
    EXPECT_EQ(static_cast<uint64_t>(int_lit.Get<u128>({})), uint64_t{77});
  } else if constexpr (kIntType == U256) {
    EXPECT_EQ(static_cast<uint64_t>(int_lit.Get<u256>({})), uint64_t{77});
  }
}

}  // namespace
}  // namespace zkx
