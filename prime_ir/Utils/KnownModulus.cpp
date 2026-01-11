/* Copyright 2025 The PrimeIR Authors.

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

#include "prime_ir/Utils/KnownModulus.h"

#include <mutex> // NOLINT(build/c++11)

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/bit.h"
#include "prime_ir/Utils/ZkDtypes.h"
#include "zk_dtypes/include/all_types.h"

namespace mlir::prime_ir {

namespace {

template <typename T>
void registerKnownModulusAliases(DenseMap<APInt, std::string> &map,
                                 std::string_view name) {
  if (name.size() >= 4 && name.substr(name.size() - 4) == "_std") {
    return;
  }

  APInt modulus;
  modulus = convertToAPInt(T::Config::kModulus, T::Config::kModulusBits);
  map[modulus] = name;
  if constexpr (!isPowerOfTwo(T::Config::kModulusBits)) {
    map[modulus.zext(llvm::bit_ceil(T::Config::kModulusBits))] = name;
  }
}

} // namespace

std::optional<std::string> getKnownModulusAlias(const APInt &modulus) {
  // NOTE(chokobole): We intentionally leak this object. See
  // https://google.github.io/styleguide/cppguide.html#Static_and_Global_Variables
  static auto &knownModulusAliases = *new DenseMap<APInt, std::string>();
  static std::once_flag onceFlag;
  std::call_once(onceFlag, []() {
#define REGISTER_KNOWN_MODULUS_ALIAS(ActualType, unused, unused2,              \
                                     LowerSnakeCaseName)                       \
  registerKnownModulusAliases<ActualType>(knownModulusAliases,                 \
                                          #LowerSnakeCaseName);
    ZK_DTYPES_ALL_PRIME_FIELD_TYPE_LIST(REGISTER_KNOWN_MODULUS_ALIAS)
#undef REGISTER_KNOWN_MODULUS_ALIAS
  });
  auto it = knownModulusAliases.find(modulus);
  if (it != knownModulusAliases.end()) {
    return it->second;
  }
  return std::nullopt;
}

} // namespace mlir::prime_ir
