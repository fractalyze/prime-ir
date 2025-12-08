/* Copyright 2025 The ZKIR Authors.

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

#include "zkir/Utils/KnownModulus.h"

#include <mutex>

#include "llvm/ADT/DenseMap.h"

namespace mlir::zkir {

namespace {

std::tuple<APInt, APInt> getBnScalarFieldModuluses(const APInt &x) {
  // See section 2 "The proposed method for k = 12" in
  // https://eprint.iacr.org/2005/133.pdf
  APInt x2 = x * x;
  APInt x3 = x2 * x;
  APInt x4 = x3 * x;
  APInt scalarFieldModulus = 36 * x4 + 36 * x3 + 18 * x2 + 6 * x + 1;
  APInt baseFieldModulus = scalarFieldModulus + 6 * x2;
  return {scalarFieldModulus, baseFieldModulus};
}

void registerBn254Modulus(
    llvm::DenseMap<APInt, std::string> &knownModulusAliases) {
  APInt x = APInt(254, 4965661367192848881);
  auto [scalarFieldModulus, baseFieldModulus] = getBnScalarFieldModuluses(x);
  knownModulusAliases[scalarFieldModulus] = "bn254_sf";
  knownModulusAliases[baseFieldModulus] = "bn254_bf";
}

} // namespace

std::optional<std::string> getKnownModulusAlias(const APInt &modulus) {
  // NOTE(chokobole): We intentionally leak this object. See
  // https://google.github.io/styleguide/cppguide.html#Static_and_Global_Variables
  // clang-format off
  static auto &knownModulusAliases = *new llvm::DenseMap<APInt, std::string> {
    // Babybear: 2³¹ - 2²⁷ + 1
    {APInt(31, 2013265921), "babybear"},
    // Koalabear: 2³¹ - 2²⁴ + 1
    {APInt(31, 2130706433), "koalabear"},
    // Mersenne31: 2³¹ - 1
    {APInt(31, 2147483647), "mersenne31"},
    // Goldilocks: 2⁶⁴ - 2³² + 1
    {APInt(64, UINT64_C(18446744069414584321)), "goldilocks"},
  };
  // clang-format on
  static std::once_flag onceFlag;
  std::call_once(onceFlag, []() { registerBn254Modulus(knownModulusAliases); });
  auto it = knownModulusAliases.find(modulus);
  if (it != knownModulusAliases.end()) {
    return it->second;
  }
  return std::nullopt;
}

} // namespace mlir::zkir
