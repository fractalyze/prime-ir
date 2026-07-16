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

#ifndef PRIME_IR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_PRIMEFIELDCODEGEN_H_
#define PRIME_IR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_PRIMEFIELDCODEGEN_H_

#include <optional>
#include <utility>

#include "llvm/ADT/APInt.h"
#include "mlir/IR/Value.h"

namespace zk_dtypes {

template <typename>
class QuadraticExtensionFieldOperation;
template <typename>
class CubicExtensionFieldOperation;
template <typename>
class QuarticExtensionFieldOperation;
template <typename>
class KaratsubaOperation;
template <typename>
class ToomCookOperation;
template <typename>
class ExtensionFieldOperation;

} // namespace zk_dtypes

namespace mlir::prime_ir::field {

class FieldCodeGen;
template <size_t N, typename BaseFieldT, bool GeneralModulus>
class ExtensionFieldCodeGen;

// NOTE(chokobole): This class is not used directly. It is used to generate
// MLIR operations that implement prime field arithmetic. User should use
// FieldCodeGen instead.
class PrimeFieldCodeGen {
public:
  PrimeFieldCodeGen() = default;
  explicit PrimeFieldCodeGen(Value value) : value(value) {}
  ~PrimeFieldCodeGen() = default;

  operator Value() const { return value; }

  // Whether `modulus` is a Solinas prime p = 2^a - 2^b + 1, for which
  // Inverse() emits an addition chain instead of a generic pow / safegcd.
  static bool hasInverseChain(const APInt &modulus);

  PrimeFieldCodeGen operator+(const PrimeFieldCodeGen &other) const;
  PrimeFieldCodeGen &operator+=(const PrimeFieldCodeGen &other);
  PrimeFieldCodeGen operator-(const PrimeFieldCodeGen &other) const;
  PrimeFieldCodeGen &operator-=(const PrimeFieldCodeGen &other);
  PrimeFieldCodeGen operator*(const PrimeFieldCodeGen &other) const;
  PrimeFieldCodeGen &operator*=(const PrimeFieldCodeGen &other);
  PrimeFieldCodeGen operator-() const;

private:
  friend class FieldCodeGen;
  template <size_t, typename, bool>
  friend class ExtensionFieldCodeGen;
  template <typename>
  friend class zk_dtypes::QuadraticExtensionFieldOperation;
  template <typename>
  friend class zk_dtypes::CubicExtensionFieldOperation;
  template <typename>
  friend class zk_dtypes::QuarticExtensionFieldOperation;
  template <typename>
  friend class zk_dtypes::KaratsubaOperation;
  template <typename>
  friend class zk_dtypes::ToomCookOperation;
  template <typename>
  friend class zk_dtypes::ExtensionFieldOperation;

  PrimeFieldCodeGen Double() const;
  PrimeFieldCodeGen Square() const;
  PrimeFieldCodeGen Inverse() const;

  // If `modulus` is a Solinas prime p = 2^a - 2^b + 1, returns (a, b).
  static std::optional<std::pair<unsigned, unsigned>>
  detectSolinasForm(const APInt &modulus);

  // x -> x^(p-2) for p = 2^a - 2^b + 1, as a branch-free addition chain.
  PrimeFieldCodeGen solinasInverseChain(unsigned a, unsigned b) const;

  PrimeFieldCodeGen CreateConst(int64_t constant) const;
  PrimeFieldCodeGen CreateRationalConst(int64_t num, int64_t denom) const;

  Value value;
};

} // namespace mlir::prime_ir::field

// NOLINTNEXTLINE(whitespace/line_length)
#endif // PRIME_IR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_PRIMEFIELDCODEGEN_H_
