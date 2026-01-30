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

#ifndef ZKX_CORE_COLLECTIVES_REDUCTION_UTIL_H_
#define ZKX_CORE_COLLECTIVES_REDUCTION_UTIL_H_

#include "absl/status/statusor.h"

#include "zk_dtypes/include/comparable_traits.h"
#include "zk_dtypes/include/field/field.h"
#include "zk_dtypes/include/group/group.h"
#include "zkx/service/collective_ops_utils.h"

namespace zkx {

namespace internal {

template <typename T, typename Selector, typename Result>
Result SelectReductionImpl(ReductionKind kind, Selector selector) {
  switch (kind) {
    case ReductionKind::kSum:
      if constexpr (zk_dtypes::IsAdditiveGroup<T> || std::is_integral_v<T>)
        return selector.sum();
      else
        return Result(absl::InvalidArgumentError(
            "Sum reduction is only supported for additive groups or integral "
            "types"));
    case ReductionKind::kProduct:
      if constexpr (zk_dtypes::IsMultiplicativeGroup<T> ||
                    std::is_integral_v<T>)
        return selector.product();
      else
        return Result(absl::InvalidArgumentError(
            "Product reduction is only supported for multiplicative groups or "
            "integral types"));
    case ReductionKind::kMin:
      if constexpr (zk_dtypes::IsComparable<T> || std::is_integral_v<T>)
        return selector.min();
      else
        return Result(absl::InvalidArgumentError(
            "Min reduction is only supported for comparable types or integral "
            "types"));
    case ReductionKind::kMax:
      if constexpr (zk_dtypes::IsComparable<T> || std::is_integral_v<T>)
        return selector.max();
      else
        return Result(absl::InvalidArgumentError(
            "Max reduction is only supported for comparable types or integral "
            "types"));
    default:
      break;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unsupported reduction: ", static_cast<int>(kind)));
}

}  // namespace internal

template <typename T, typename RetType, typename Selector>
absl::StatusOr<RetType> SelectReduction(ReductionKind kind, Selector selector) {
  return internal::SelectReductionImpl<T, Selector, absl::StatusOr<RetType>>(
      kind, selector);
}

template <typename T, typename Selector>
absl::Status SelectReduction(ReductionKind kind, Selector selector) {
  return internal::SelectReductionImpl<T, Selector, absl::Status>(kind,
                                                                  selector);
}

}  // namespace zkx

#endif  // ZKX_CORE_COLLECTIVES_REDUCTION_UTIL_H_
