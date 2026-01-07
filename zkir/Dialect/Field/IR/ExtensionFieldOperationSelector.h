// Copyright 2026 The ZKIR Authors.
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

#ifndef ZKIR_DIALECT_FIELD_IR_EXTENSIONFIELDOPERATIONSELECTOR_H_
#define ZKIR_DIALECT_FIELD_IR_EXTENSIONFIELDOPERATIONSELECTOR_H_

#include <cstddef>

#include "zk_dtypes/include/field/cubic_extension_field_operation.h"
#include "zk_dtypes/include/field/quadratic_extension_field_operation.h"
#include "zk_dtypes/include/field/quartic_extension_field_operation.h"

namespace mlir::zkir::field {

// Selects the appropriate zk_dtypes extension field operation based on degree.
template <size_t N>
struct ExtensionFieldOperationSelector;

template <>
struct ExtensionFieldOperationSelector<2> {
  template <typename Derived>
  using Type = zk_dtypes::QuadraticExtensionFieldOperation<Derived>;
};

template <>
struct ExtensionFieldOperationSelector<3> {
  template <typename Derived>
  using Type = zk_dtypes::CubicExtensionFieldOperation<Derived>;
};

template <>
struct ExtensionFieldOperationSelector<4> {
  template <typename Derived>
  using Type = zk_dtypes::QuarticExtensionFieldOperation<Derived>;
};

} // namespace mlir::zkir::field

#endif // ZKIR_DIALECT_FIELD_IR_EXTENSIONFIELDOPERATIONSELECTOR_H_
