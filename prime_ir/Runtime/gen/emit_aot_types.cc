// Copyright 2026 The PrimeIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

// Emits AOT runtime type list using zk_dtypes X-macros.
// Output: one line per type, consumed by gen_aot_runtime.py.
//   curve <lower_snake> <rank>
//   ext_field <lower_snake>
//
// Extension fields are filtered by the AOT eligibility criteria:
//   (degree >= 4) OR (degree >= 2 AND prime storage bits > 64)

#include <cstddef>
#include <cstdio>

#include "zk_dtypes/include/all_types.h"

#define EMIT_R1_CURVE(ActualType, UpperCamelCase, UpperSnake, LowerSnake)      \
  printf("curve %s 1\n", #LowerSnake);
#define EMIT_R2_CURVE(ActualType, UpperCamelCase, UpperSnake, LowerSnake)      \
  printf("curve %s 2\n", #LowerSnake);

// Only emit extension fields that qualify for AOT:
//   degree >= 4 (always expensive) OR (degree >= 2 AND large prime)
#define EMIT_EXT_FIELD(ActualType, UpperCamelCase, UpperSnake, LowerSnake)     \
  {                                                                            \
    constexpr size_t deg = ActualType::ExtensionDegree();                      \
    constexpr size_t bits =                                                    \
        ActualType::Config::BaseField::Config::kStorageBits;                   \
    if (deg >= 4 || (deg >= 2 && bits > 64))                                   \
      printf("ext_field %s\n", #LowerSnake);                                   \
  }

int main() {
  ZK_DTYPES_PUBLIC_R1_AFFINE_POINT_TYPE_LIST(EMIT_R1_CURVE)
  ZK_DTYPES_PUBLIC_R2_AFFINE_POINT_TYPE_LIST(EMIT_R2_CURVE)
  ZK_DTYPES_ALL_EXT_FIELD_TYPE_LIST(EMIT_EXT_FIELD)
  return 0;
}
