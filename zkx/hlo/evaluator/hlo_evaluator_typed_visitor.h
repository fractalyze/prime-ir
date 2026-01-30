/* Copyright 2018 The OpenXLA Authors.
Copyright 2026 The ZKX Authors.

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

#ifndef ZKX_HLO_EVALUATOR_HLO_EVALUATOR_TYPED_VISITOR_H_
#define ZKX_HLO_EVALUATOR_HLO_EVALUATOR_TYPED_VISITOR_H_

#include <stdint.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <type_traits>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/log/check.h"
#include "absl/numeric/bits.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"

#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "zk_dtypes/include/comparable_traits.h"
#include "zk_dtypes/include/field/field.h"
#include "zk_dtypes/include/geometry/point_declarations.h"
#include "zk_dtypes/include/group/group.h"
#include "zkx/hlo/evaluator/hlo_evaluator.h"
#include "zkx/hlo/ir/hlo_casting_utils.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/ir/hlo_instructions.h"
#include "zkx/index_util.h"
#include "zkx/literal.h"
#include "zkx/service/shape_inference.h"
#include "zkx/shape_util.h"
#include "zkx/status_macros.h"
#include "zkx/types.h"
#include "zkx/util.h"

namespace zkx {
namespace detail {

template <typename T>
using unsigned_promoted_type_t =
    std::make_unsigned_t<decltype(std::declval<T>() + std::declval<T>())>;

}  // namespace detail

// ToArithmeticSafeType(T t):
//  - converts `t` to an unsigned integer at least as wide as `int` if T is an
//    integer, and
//  - otherwise returns `t` unchanged.
//
// It's UB in C++ to under/overflow a signed integer, so we wrap all arithmetic
// in this type to force 2's complement behavior.
template <typename T>
auto ToArithmeticSafeType(T t) {
  if constexpr (std::is_integral_v<T>) {
    return static_cast<detail::unsigned_promoted_type_t<T>>(t);
  } else {
    return std::move(t);
  }
}

// Templated DfsHloVisitor for use by HloEvaluator.
//
// Typically ReturnT here indicates the resulting literal type of each evaluated
// Handle* method of a TypedVisitor.  There are however a few exceptions to this
// rule, notably:
// - HandleCompare and HandleIsFinite: where the resulting literal type is
//   always boolean.
// - HandleImag and HandleReal: where the resulting literal type is always float
//   and the operand is always complex, or real in the case of HandleReal.
// These operations are handled outside of the parent HloEvaluator handlers
// instead of from within TypedVisitor.
//
// Type params:
//   - ReturnT: The type of input and output of each operation.
//   - ElementwiseT: The type in which internal computation are done.
//
// This is logically a private part of HloEvaluator.  It lives in this header
// file rather than in hlo_evaluator.cc because we use extern templates and a
// bunch of independent cc files to speed up compiling the many instantiations
// of this class.
//
// NOTE: Prefer putting new implementation to HloEvaluator rather than
// HloEvaluatorTypedVisitor whenever possible, because this class is templated
// for all primitive types and is an order of magnitude larger in code size as
// well as compile time. Only put op handling that involves compute using native
// C++ types here, such as elementwise ops with compute, convolution, dot, etc.
template <typename ReturnT, typename ElementwiseT = ReturnT>
class HloEvaluatorTypedVisitor : public ConstDfsHloVisitorWithDefault {
 private:
  ABSL_ATTRIBUTE_NOINLINE absl::Status UnsupportedTypeError(
      const HloInstruction* instruction) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Unsupported type for %s: %s", HloOpcodeString(instruction->opcode()),
        PrimitiveType_Name(instruction->shape().element_type())));
  }

 public:
  explicit HloEvaluatorTypedVisitor(HloEvaluator* p) : parent_(p) {}

  // The following higher-order functions convert a function with ElementwiseT
  // to a function with ReturnT.
  std::function<ReturnT(ReturnT)> ConvertUnaryFunction(
      const std::function<ElementwiseT(ElementwiseT)>& unary_op) {
    return [&unary_op](ReturnT arg) {
      return static_cast<ReturnT>(unary_op(static_cast<ElementwiseT>(arg)));
    };
  }
  std::function<ReturnT(ReturnT, ReturnT)> ConvertBinaryFunction(
      const std::function<ElementwiseT(ElementwiseT, ElementwiseT)>&
          binary_op) {
    return [&binary_op](ReturnT arg1, ReturnT arg2) {
      return static_cast<ReturnT>(binary_op(static_cast<ElementwiseT>(arg1),
                                            static_cast<ElementwiseT>(arg2)));
    };
  }
  std::function<ReturnT(ReturnT, ReturnT, ReturnT)> ConvertTernaryFunction(
      const std::function<ElementwiseT(ElementwiseT, ElementwiseT,
                                       ElementwiseT)>& ternary_op) {
    return [&ternary_op](ReturnT arg1, ReturnT arg2, ReturnT arg3) {
      return static_cast<ReturnT>(ternary_op(static_cast<ElementwiseT>(arg1),
                                             static_cast<ElementwiseT>(arg2),
                                             static_cast<ElementwiseT>(arg3)));
    };
  }

  absl::Status DefaultAction(const HloInstruction* hlo_instruction) override {
    return absl::UnimplementedError(
        absl::StrFormat("unhandled HLO ops for HloEvaluator: %s.",
                        HloOpcodeString(hlo_instruction->opcode())));
  }

  template <typename NativeT,
            typename std::enable_if_t<std::is_unsigned_v<NativeT>>* = nullptr>
  absl::Status HandleAbs(const HloInstruction* abs) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[abs],
                        ElementWiseUnaryOp(abs, [](NativeT elem_operand) {
                          return elem_operand;
                        }));
    return absl::OkStatus();
  }

  template <typename NativeT,
            typename std::enable_if_t<std::is_signed_v<NativeT>>* = nullptr>
  absl::Status HandleAbs(const HloInstruction* abs) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[abs],
                        ElementWiseUnaryOp(abs, [](NativeT elem_operand) {
                          return std::abs(elem_operand);
                        }));
    return absl::OkStatus();
  }

  template <typename NativeT,
            typename std::enable_if_t<!std::is_integral_v<NativeT>>* = nullptr>
  absl::Status HandleAbs(const HloInstruction* abs) {
    return UnsupportedTypeError(abs);
  }

  absl::Status HandleAbs(const HloInstruction* abs) override {
    return HandleAbs<ElementwiseT>(abs);
  }

  absl::Status HandleNot(const HloInstruction* not_) override {
    if constexpr (std::is_arithmetic_v<ElementwiseT>) {
      TF_ASSIGN_OR_RETURN(
          parent_->evaluated_[not_],
          ElementWiseUnaryOp(not_, [](ElementwiseT elem_operand) {
            if constexpr (std::is_same_v<ElementwiseT, bool>) {
              return !elem_operand;
            } else {
              static_assert(std::is_integral_v<ElementwiseT>);
              return ~elem_operand;
            }
          }));
      return absl::OkStatus();
    }
    return UnsupportedTypeError(not_);
  }

  // Negation that is safe and consistent across types.
  // Key idea:
  // - For *signed integers*, plain `-x` is UB when x == INT_MIN.
  //   C++ cannot represent the positive counterpart, so the operation overflows
  //   → UB.
  // - For *unsigned integers*, arithmetic is modulo 2^N, so `-x` is always
  // well-defined.
  //
  // Therefore we split the implementation with SFINAE:
  //   (A) signed integer → negate via an unsigned cast to avoid UB
  //   (B) otherwise (unsigned integer or floating point) → plain `-x` is fine
  template <typename NativeT,
            typename std::enable_if_t<std::is_signed_v<NativeT>>* = nullptr>
  absl::Status HandleNegate(const HloInstruction* negate) {
    using type = std::make_unsigned_t<NativeT>;
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[negate],
        ElementWiseUnaryOp(negate, [](ElementwiseT elem_operand) {
          return NativeT(-type(elem_operand));
        }));
    return absl::OkStatus();
  }

  template <typename NativeT,
            typename std::enable_if_t<!std::is_signed_v<NativeT>>* = nullptr>
  absl::Status HandleNegate(const HloInstruction* negate) {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[negate],
        ElementWiseUnaryOp(
            negate, [](ElementwiseT elem_operand) { return -elem_operand; }));
    return absl::OkStatus();
  }

  absl::Status HandleNegate(const HloInstruction* negate) override {
    return HandleNegate<ReturnT>(negate);
  }

  absl::Status HandleSign(const HloInstruction* sign) override {
    if constexpr (std::is_integral_v<ElementwiseT>) {
      TF_ASSIGN_OR_RETURN(
          parent_->evaluated_[sign],
          ElementWiseUnaryOp(sign, [](ElementwiseT elem_operand) {
            return (ElementwiseT(0) < elem_operand) -
                   (elem_operand < ElementwiseT(0));
          }));
      return absl::OkStatus();
    }
    return UnsupportedTypeError(sign);
  }

  absl::Status HandleMultiply(const HloInstruction* multiply) override {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[multiply],
        ElementWiseBinaryOp(
            multiply, [](ElementwiseT lhs_elem, ElementwiseT rhs_elem) {
              if constexpr (zk_dtypes::IsMultiplicativeGroup<ElementwiseT> ||
                            std::is_integral_v<ElementwiseT>) {
                return ElementwiseT(ToArithmeticSafeType(lhs_elem) *
                                    ToArithmeticSafeType(rhs_elem));
              } else {
                // TODO(chokobole): Handle scalar–matrix/vec multiplication.
                // If multiplication currently assumes identical element types,
                // we may need a dedicated opcode for scalar multiplication.
                ABSL_UNREACHABLE();
                return ElementwiseT(0);
              }
            }));
    return absl::OkStatus();
  }

  absl::Status HandleSubtract(const HloInstruction* subtract) override {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[subtract],
        ElementWiseBinaryOp(
            subtract, [](ElementwiseT lhs_elem, ElementwiseT rhs_elem) {
              if constexpr (zk_dtypes::IsAffinePoint<ElementwiseT>) {
                return (ToArithmeticSafeType(lhs_elem) -
                        ToArithmeticSafeType(rhs_elem))
                    .ToAffine();
              } else {
                return ElementwiseT(ToArithmeticSafeType(lhs_elem) -
                                    ToArithmeticSafeType(rhs_elem));
              }
            }));
    return absl::OkStatus();
  }

  absl::Status HandleAdd(const HloInstruction* add) override {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[add],
        ElementWiseBinaryOp(
            add, [](ElementwiseT lhs_elem, ElementwiseT rhs_elem) {
              if constexpr (zk_dtypes::IsAffinePoint<ElementwiseT>) {
                return (ToArithmeticSafeType(lhs_elem) +
                        ToArithmeticSafeType(rhs_elem))
                    .ToAffine();
              } else {
                return ElementwiseT(ToArithmeticSafeType(lhs_elem) +
                                    ToArithmeticSafeType(rhs_elem));
              }
            }));
    return absl::OkStatus();
  }

  absl::Status HandleDivide(const HloInstruction* divide) override {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[divide],
        ElementWiseBinaryOp(
            divide,
            [](ElementwiseT lhs_elem, ElementwiseT rhs_elem) -> ElementwiseT {
              if constexpr (zk_dtypes::IsMultiplicativeGroup<ElementwiseT>) {
                return lhs_elem / rhs_elem;
              } else if constexpr (std::is_integral_v<ElementwiseT>) {
                if constexpr (std::is_unsigned_v<ElementwiseT>) {
                  if (rhs_elem == 0) {
                    return std::numeric_limits<ElementwiseT>::max();
                  }
                } else {
                  if (rhs_elem == 0) {
                    return static_cast<ElementwiseT>(-1);
                  }
                  if (rhs_elem == -1 &&
                      lhs_elem == std::numeric_limits<ElementwiseT>::min()) {
                    return lhs_elem;
                  }
                }
                return lhs_elem / rhs_elem;
              } else {
                ABSL_UNREACHABLE();
                return ElementwiseT(0);
              }
            }));
    return absl::OkStatus();
  }

  absl::Status HandleMaximum(const HloInstruction* maximum) override {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[maximum],
        ElementWiseBinaryOp(maximum, [](ElementwiseT lhs, ElementwiseT rhs) {
          if constexpr (zk_dtypes::IsComparable<ElementwiseT> ||
                        std::is_integral_v<ElementwiseT>) {
            return std::max(lhs, rhs);
          } else {
            ABSL_UNREACHABLE();
            return ElementwiseT(0);
          }
        }));
    return absl::OkStatus();
  }

  absl::Status HandleMinimum(const HloInstruction* minimum) override {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[minimum],
        ElementWiseBinaryOp(minimum, [](ElementwiseT lhs, ElementwiseT rhs) {
          if constexpr (zk_dtypes::IsComparable<ElementwiseT> ||
                        std::is_integral_v<ElementwiseT>) {
            return std::min(lhs, rhs);
          } else {
            ABSL_UNREACHABLE();
            return ElementwiseT(0);
          }
        }));
    return absl::OkStatus();
  }

  absl::Status HandlePower(const HloInstruction* power) override {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[power],
        ElementWiseBinaryOp(power, [](ElementwiseT lhs_el,
                                      ElementwiseT rhs_el) {
          if constexpr (zk_dtypes::IsExtensionField<ElementwiseT>) {
            // TODO(chokobole): Handle extension field power.
            ABSL_UNREACHABLE();
            return ElementwiseT(0);
          } else if constexpr (zk_dtypes::IsMultiplicativeGroup<ElementwiseT>) {
            return lhs_el.Pow(rhs_el);
          } else if (lhs_el == ElementwiseT(1) || rhs_el == ElementwiseT(0)) {
            // Case 0: 1^x = 1 and x^0 = 1, regardless of X, see
            // Branch Cuts for Complex Elementary Functions or Much Ado About
            // Nothing's Sign Bit, W. Kahan, Section 10.
            return static_cast<ElementwiseT>(1);
          } else if constexpr (std::is_same_v<ElementwiseT, bool>) {
            return lhs_el || !rhs_el;
          } else if constexpr (std::is_integral_v<ElementwiseT>) {
            if constexpr (std::is_signed_v<ElementwiseT>) {
              if (rhs_el < static_cast<ElementwiseT>(0)) {
                return static_cast<ElementwiseT>(
                    lhs_el == static_cast<ElementwiseT>(1) ? 1 : 0);
              }
            }
            return static_cast<ElementwiseT>(
                IPow<std::make_unsigned_t<ElementwiseT>>(lhs_el, rhs_el));
          } else {
            ABSL_UNREACHABLE();
            return ElementwiseT(0);
          }
        }));
    return absl::OkStatus();
  }

  absl::Status HandleRemainder(const HloInstruction* remainder) override {
    if constexpr (std::is_integral_v<ElementwiseT>) {
      TF_ASSIGN_OR_RETURN(
          parent_->evaluated_[remainder],
          ElementWiseBinaryOp(
              remainder,
              [](ElementwiseT lhs_el, ElementwiseT rhs_el) -> ElementwiseT {
                if (rhs_el == 0) {
                  return lhs_el;
                }
                if constexpr (std::is_signed_v<ElementwiseT>) {
                  if (rhs_el == -1 &&
                      lhs_el == std::numeric_limits<ElementwiseT>::min()) {
                    return 0;
                  }
                }
                return lhs_el % rhs_el;
              }));
      return absl::OkStatus();
    }
    return UnsupportedTypeError(remainder);
  }

  absl::Status HandleAnd(const HloInstruction* and_inst) override {
    if constexpr (std::is_integral_v<ElementwiseT>) {
      TF_ASSIGN_OR_RETURN(
          parent_->evaluated_[and_inst],
          ElementWiseBinaryOp(and_inst,
                              [](ElementwiseT lhs_el, ElementwiseT rhs_el) {
                                return lhs_el & rhs_el;
                              }));
      return absl::OkStatus();
    }
    return UnsupportedTypeError(and_inst);
  }

  absl::Status HandleOr(const HloInstruction* or_inst) override {
    if constexpr (std::is_integral_v<ElementwiseT>) {
      TF_ASSIGN_OR_RETURN(parent_->evaluated_[or_inst],
                          ElementWiseBinaryOp(or_inst, [](ElementwiseT lhs_el,
                                                          ElementwiseT rhs_el) {
                            return lhs_el | rhs_el;
                          }));
      return absl::OkStatus();
    }
    return UnsupportedTypeError(or_inst);
  }

  absl::Status HandleXor(const HloInstruction* xor_inst) override {
    if constexpr (std::is_integral_v<ElementwiseT>) {
      TF_ASSIGN_OR_RETURN(
          parent_->evaluated_[xor_inst],
          ElementWiseBinaryOp(xor_inst,
                              [](ElementwiseT lhs_el, ElementwiseT rhs_el) {
                                return lhs_el ^ rhs_el;
                              }));
      return absl::OkStatus();
    }
    return UnsupportedTypeError(xor_inst);
  }

  absl::Status HandleShiftLeft(const HloInstruction* shl) override {
    if constexpr (std::is_integral_v<ElementwiseT> &&
                  !std::is_same_v<ElementwiseT, bool>) {
      TF_ASSIGN_OR_RETURN(parent_->evaluated_[shl],
                          ElementWiseBinaryOp(shl, [](ElementwiseT lhs_elem,
                                                      ElementwiseT rhs_elem) {
                            return IsShiftOutOfBounds<ReturnT>(rhs_elem)
                                       ? 0
                                       : (lhs_elem << rhs_elem);
                          }));
      return absl::OkStatus();
    }
    return UnsupportedTypeError(shl);
  }

  absl::Status HandleShiftRightArithmetic(const HloInstruction* shr) override {
    if constexpr (std::is_integral_v<ElementwiseT> &&
                  !std::is_same_v<ElementwiseT, bool>) {
      using SignedT = make_specialized_signed_t<ReturnT>;
      TF_ASSIGN_OR_RETURN(
          parent_->evaluated_[shr],
          ElementWiseBinaryOp(
              shr, [](ElementwiseT lhs_elem, ElementwiseT rhs_elem) {
                SignedT lhs_signed = static_cast<SignedT>(lhs_elem);
                if (IsShiftOutOfBounds<ReturnT>(rhs_elem)) {
                  return lhs_signed < 0 ? static_cast<ElementwiseT>(-1) : 0;
                } else {
                  return static_cast<ElementwiseT>(lhs_signed >> rhs_elem);
                }
              }));
      return absl::OkStatus();
    }
    return UnsupportedTypeError(shr);
  }

  absl::Status HandleShiftRightLogical(const HloInstruction* shr) override {
    if constexpr (std::is_integral_v<ElementwiseT> &&
                  !std::is_same_v<ElementwiseT, bool>) {
      using UnsignedT = make_specialized_unsigned_t<ReturnT>;
      TF_ASSIGN_OR_RETURN(parent_->evaluated_[shr],
                          ElementWiseBinaryOp(shr, [](ElementwiseT lhs_elem,
                                                      ElementwiseT rhs_elem) {
                            // If shift amount is greater than the number of
                            // bits, then return 0.
                            if (IsShiftOutOfBounds<ReturnT>(rhs_elem)) {
                              return static_cast<ElementwiseT>(0);
                            }
                            return static_cast<ElementwiseT>(
                                static_cast<UnsignedT>(lhs_elem) >> rhs_elem);
                          }));
      return absl::OkStatus();
    }
    return UnsupportedTypeError(shr);
  }

  absl::Status HandleClamp(const HloInstruction* clamp) override {
    if constexpr (std::is_integral_v<ElementwiseT> ||
                  zk_dtypes::IsComparable<ElementwiseT>) {
      auto clamp_op = [](ElementwiseT low, ElementwiseT value,
                         ElementwiseT high) {
        return std::min(high, std::max(value, low));
      };
      TF_ASSIGN_OR_RETURN(
          parent_->evaluated_[clamp],
          ElementwiseTernaryOp(clamp,
                               std::move(ConvertTernaryFunction(clamp_op))));
      return absl::OkStatus();
    }
    return UnsupportedTypeError(clamp);
  }

  absl::Status HandleSelect(const HloInstruction* select) override {
    CHECK(!ShapeUtil::IsScalar(select->operand(0)->shape()));
    CHECK(select->shape().IsArray());
    std::function<ReturnT(bool, ReturnT, ReturnT)> select_op =
        [](bool pred, ReturnT on_true, ReturnT on_false) {
          if (pred) {
            return on_true;
          }
          return on_false;
        };
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[select],
                        ElementwiseTernaryOp(select, std::move(select_op)));
    return absl::OkStatus();
  }

  absl::Status HandleDot(const HloInstruction* dot) override {
    return HandleDotSlowPath(dot);
  }

  absl::Status HandleDotSlowPathWithLiterals(const HloInstruction* dot,
                                             const Literal& lhs_literal,
                                             const Literal& rhs_literal) {
    const auto& dnums = dot->dot_dimension_numbers();

    const auto lhs_rank = lhs_literal.shape().rank();
    const auto rhs_rank = rhs_literal.shape().rank();

    CHECK(ShapeUtil::SameElementType(lhs_literal.shape(), rhs_literal.shape()));
    CHECK(ShapeUtil::SameElementType(lhs_literal.shape(), dot->shape()));

    CHECK_EQ(dnums.lhs_batch_dimensions_size(),
             dnums.rhs_batch_dimensions_size());

    DimensionVector lhs_non_contracting_dims =
        GetNonContractingDims(lhs_rank, dnums.lhs_contracting_dimensions(),
                              dnums.lhs_batch_dimensions());
    DimensionVector rhs_non_contracting_dims =
        GetNonContractingDims(rhs_rank, dnums.rhs_contracting_dimensions(),
                              dnums.rhs_batch_dimensions());

    DimensionVector contracting_dim_sizes;
    contracting_dim_sizes.reserve(dnums.lhs_contracting_dimensions_size());
    DimensionVector lhs_contracting_dims;
    DimensionVector rhs_contracting_dims;
    for (int64_t i = 0; i < dnums.lhs_contracting_dimensions_size(); ++i) {
      const int64_t lhs_dnum = dnums.lhs_contracting_dimensions(i);
      const int64_t rhs_dnum = dnums.rhs_contracting_dimensions(i);
      lhs_contracting_dims.push_back(lhs_dnum);
      rhs_contracting_dims.push_back(rhs_dnum);
      const int64_t dim_size = lhs_literal.shape().dimensions(lhs_dnum);
      contracting_dim_sizes.push_back(dim_size);
    }
    const int64_t total_contraction_size = Product(contracting_dim_sizes);
    Literal result(dot->shape());
    TF_RETURN_IF_ERROR(result.PopulateParallel<ReturnT>(
        [&](absl::Span<const int64_t> result_index, int /*thread_id*/) {
          // Locations in LHS and RHS that we read from.
          DimensionVector lhs_index(lhs_rank);
          DimensionVector rhs_index(rhs_rank);

          // First come the batch dimensions.
          int64_t idx = 0;
          for (int64_t i = 0; i < dnums.lhs_batch_dimensions_size(); i++) {
            lhs_index[dnums.lhs_batch_dimensions(i)] = result_index[idx];
            rhs_index[dnums.rhs_batch_dimensions(i)] = result_index[idx];
            idx++;
          }

          // Next we have non-contracting dimensions, if any.
          for (int64_t i = 0; i < lhs_non_contracting_dims.size(); i++) {
            lhs_index[lhs_non_contracting_dims[i]] = result_index[idx++];
          }
          for (int64_t i = 0; i < rhs_non_contracting_dims.size(); i++) {
            rhs_index[rhs_non_contracting_dims[i]] = result_index[idx++];
          }

          // Accumulate resulting product along the contracting dimensions.
          ElementwiseT result_val = static_cast<ElementwiseT>(0);
          for (int64_t k = 0; k < total_contraction_size; k++) {
            if constexpr (zk_dtypes::IsField<ElementwiseT> ||
                          std::is_integral_v<ElementwiseT>) {
              const auto lhs = static_cast<ElementwiseT>(
                  lhs_literal.Get<ReturnT>(lhs_index));
              const auto rhs = static_cast<ElementwiseT>(
                  rhs_literal.Get<ReturnT>(rhs_index));
              result_val +=
                  ToArithmeticSafeType(lhs) * ToArithmeticSafeType(rhs);
            } else {
              ABSL_UNREACHABLE();
              return ElementwiseT(0);
            }

            if (parent_->trace_mac_handler_ != nullptr) {
              const int64_t result_linear_index =
                  IndexUtil::MultidimensionalIndexToLinearIndex(dot->shape(),
                                                                result_index);
              const int64_t lhs_linear_index =
                  IndexUtil::MultidimensionalIndexToLinearIndex(
                      lhs_literal.shape(), lhs_index);
              const int64_t rhs_linear_index =
                  IndexUtil::MultidimensionalIndexToLinearIndex(
                      rhs_literal.shape(), rhs_index);

              parent_->trace_mac_handler_(result_linear_index, lhs_linear_index,
                                          rhs_linear_index);
            }

            // If there are no contracting dimensions, do not try to count down
            // from -1 to 0; that's an infinite loop.
            if (!contracting_dim_sizes.empty()) {
              for (int64_t i = contracting_dim_sizes.size() - 1; i >= 0; --i) {
                lhs_index[lhs_contracting_dims[i]]++;
                rhs_index[rhs_contracting_dims[i]]++;
                if (lhs_index[lhs_contracting_dims[i]] !=
                    contracting_dim_sizes[i]) {
                  break;
                }
                lhs_index[lhs_contracting_dims[i]] = 0;
                rhs_index[rhs_contracting_dims[i]] = 0;
              }
            }
          }

          return static_cast<ReturnT>(result_val);
        }));

    parent_->evaluated_[dot] = std::move(result);
    return absl::OkStatus();
  }

  absl::Status HandleDotSlowPath(const HloInstruction* dot) {
    auto lhs = dot->operand(0);
    auto rhs = dot->operand(1);
    CHECK(dot->shape().IsArray());
    CHECK(lhs->shape().IsArray());
    CHECK(rhs->shape().IsArray());
    const bool lhs_same =
        ShapeUtil::SameElementType(lhs->shape(), dot->shape());
    const bool rhs_same =
        ShapeUtil::SameElementType(rhs->shape(), dot->shape());
    const Literal& lhs_literal = parent_->GetEvaluatedLiteralFor(lhs);
    const Literal& rhs_literal = parent_->GetEvaluatedLiteralFor(rhs);
    if (lhs_same && rhs_same) {
      return HandleDotSlowPathWithLiterals(dot, lhs_literal, rhs_literal);
    }
    if (lhs_same) {
      return HandleDotSlowPathWithLiterals(
          dot, lhs_literal,
          rhs_literal.Convert(dot->shape().element_type()).value());
    }
    if (rhs_same) {
      return HandleDotSlowPathWithLiterals(
          dot, lhs_literal.Convert(dot->shape().element_type()).value(),
          rhs_literal);
    }
    return HandleDotSlowPathWithLiterals(
        dot, lhs_literal.Convert(dot->shape().element_type()).value(),
        rhs_literal.Convert(dot->shape().element_type()).value());
  }

  absl::Status HandlePad(const HloInstruction* pad) override {
    CHECK(pad->operand(0)->shape().IsArray());
    // Padding value must be scalar.
    CHECK(ShapeUtil::IsScalar(pad->operand(1)->shape()));
    CHECK_EQ(pad->operand(0)->shape().rank(),
             pad->padding_config().dimensions_size());

    TF_ASSIGN_OR_RETURN(auto inferred_return_shape,
                        ShapeInference::InferPadShape(
                            /*operand_shape=*/pad->operand(0)->shape(),
                            /*padding_value_shape=*/pad->operand(1)->shape(),
                            /*padding_config=*/pad->padding_config()));
    // Try to convert the element type if the inferred type is not compatible.
    bool convert_element_type =
        pad->shape().element_type() != inferred_return_shape.element_type();
    if (convert_element_type) {
      inferred_return_shape.set_element_type(pad->shape().element_type());
    }
    CHECK(ShapeUtil::Compatible(pad->shape(), inferred_return_shape))
        << "return shape is set to: " << ShapeUtil::HumanString(pad->shape())
        << " but is inferred to be: "
        << ShapeUtil::HumanString(inferred_return_shape);
    ReturnT scalar;
    if (convert_element_type) {
      TF_ASSIGN_OR_RETURN(auto literal,
                          parent_->GetEvaluatedLiteralFor(pad->operand(1))
                              .Convert(inferred_return_shape.element_type()));
      scalar = literal.Get<ReturnT>({});
    } else {
      scalar =
          parent_->GetEvaluatedLiteralFor(pad->operand(1)).Get<ReturnT>({});
    }

    // Create new HLO of padded shape with padding value.
    Literal result(pad->shape());
    TF_RETURN_IF_ERROR(result.PopulateParallel<ReturnT>(
        [&scalar](absl::Span<const int64_t> multi_index, int) {
          return scalar;
        }));

    const Literal& evaluated_operand =
        parent_->GetEvaluatedLiteralFor(pad->operand(0));

    std::vector<int64_t> target_index(result.shape().rank(), 0);

    // Loop through each element of the operand, assign them to the
    // corresponding index of the resulting padded literal.
    const PaddingConfig& pad_config = pad->padding_config();

    auto func = [&](absl::Span<const int64_t> input_index) {
      for (auto i = 0; i < input_index.size(); ++i) {
        // Interior padding occurs logically before edge padding, so in the case
        // of negative edge padding elements are removed from the
        // interior-padded operand.
        // TODO(chokobole): Do we need this? Dependency: interior_padding
        // target_index[i] =
        //     pad_config.dimensions(i).edge_padding_low() +
        //     input_index[i] * (pad_config.dimensions(i).interior_padding() +
        //     1);
        target_index[i] =
            pad_config.dimensions(i).edge_padding_low() + input_index[i];

        // Account for negative low and high padding: skip assignment if the
        // any target index is out of range.
        if (!(target_index[i] >= 0 &&
              target_index[i] < pad->shape().dimensions(i))) {
          return true;
        }
      }
      result.Set<ReturnT>(target_index,
                          evaluated_operand.Get<ReturnT>(input_index));
      return true;
    };

    std::vector<int64_t> zero_base(evaluated_operand.shape().dimensions_size(),
                                   0);
    std::vector<int64_t> step(evaluated_operand.shape().dimensions_size(), 1);

    ShapeUtil::ForEachIndexNoStatus(evaluated_operand.shape(), zero_base,
                                    evaluated_operand.shape().dimensions(),
                                    step, func);

    parent_->evaluated_[pad] = std::move(result);
    return absl::OkStatus();
  }

  absl::Status HandleClz(const HloInstruction* clz) override {
    // Enable CLZ only for integer types.
    if constexpr (std::is_integral_v<ElementwiseT> &&
                  !std::is_same_v<ElementwiseT, bool>) {
      TF_ASSIGN_OR_RETURN(
          parent_->evaluated_[clz],
          ElementWiseUnaryOp(clz, [](ElementwiseT elem_operand) {
            using UnsignedT = make_specialized_unsigned_t<ElementwiseT>;
            if (elem_operand == 0) {
              return static_cast<ElementwiseT>(
                  std::numeric_limits<UnsignedT>::digits);
            }
            return static_cast<ElementwiseT>(
                absl::countl_zero(static_cast<UnsignedT>(elem_operand)));
          }));
      return absl::OkStatus();
    }
    return UnsupportedTypeError(clz);
  }

  absl::Status HandlePopulationCount(const HloInstruction* popcnt) override {
    if constexpr (std::is_integral_v<ElementwiseT> &&
                  !std::is_same_v<ElementwiseT, bool>) {
      TF_ASSIGN_OR_RETURN(
          parent_->evaluated_[popcnt],
          ElementWiseUnaryOp(popcnt, [](ElementwiseT elem_operand) {
            using UnsignedT = make_specialized_unsigned_t<ElementwiseT>;
            return static_cast<ElementwiseT>(
                absl::popcount(static_cast<UnsignedT>(elem_operand)));
          }));
      return absl::OkStatus();
    }
    return UnsupportedTypeError(popcnt);
  }

  absl::Status HandleIota(const HloInstruction* instruction) override {
    auto* iota = Cast<HloIotaInstruction>(instruction);
    if constexpr (std::is_integral_v<ElementwiseT> ||
                  zk_dtypes::IsAdditiveGroup<ElementwiseT> ||
                  zk_dtypes::IsMultiplicativeGroup<ElementwiseT>) {
      Literal result(iota->shape());
      ShapeUtil::ForEachIndexNoStatus(
          iota->shape(), [&](absl::Span<const int64_t> idx) {
            result.Set(idx, static_cast<ReturnT>(idx[iota->iota_dimension()]));
            return true;
          });
      parent_->evaluated_[iota] = std::move(result);
      return absl::OkStatus();
    }
    return UnsupportedTypeError(iota);
  }

 private:
  absl::StatusOr<Literal> ElementWiseUnaryOp(
      const HloInstruction* instruction,
      const std::function<ElementwiseT(ElementwiseT)>& unary_op) {
    const Literal& operand_literal =
        parent_->GetEvaluatedLiteralFor(instruction->operand(0));
    TF_ASSIGN_OR_RETURN(
        auto result_literal,
        (HloEvaluator::ElementWiseUnaryOpImpl<ReturnT, ReturnT>(
            instruction, ConvertUnaryFunction(unary_op), operand_literal)));

    return std::move(result_literal);
  }

  absl::StatusOr<Literal> ElementWiseBinaryOp(
      const HloInstruction* instruction,
      const std::function<ElementwiseT(ElementwiseT, ElementwiseT)>&
          binary_op) {
    const auto& shape = instruction->shape();
    const auto* lhs = instruction->operand(0);
    const auto* rhs = instruction->operand(1);
    TF_RET_CHECK(ShapeUtil::SameDimensions(shape, rhs->shape()));
    TF_RET_CHECK(ShapeUtil::SameDimensions(lhs->shape(), rhs->shape()));

    const Literal& lhs_literal = parent_->GetEvaluatedLiteralFor(lhs);
    const Literal& rhs_literal = parent_->GetEvaluatedLiteralFor(rhs);

    Literal result(shape);

    TF_RETURN_IF_ERROR(result.PopulateParallel<ReturnT>(
        [&](absl::Span<const int64_t> multi_index, int) {
          return ConvertBinaryFunction(binary_op)(
              lhs_literal.Get<ReturnT>(multi_index),
              rhs_literal.Get<ReturnT>(multi_index));
        }));
    return std::move(result);
  }

  template <typename LhsType, typename RhsType, typename EhsType>
  absl::StatusOr<Literal> ElementwiseTernaryOp(
      const HloInstruction* instruction,
      const std::function<ReturnT(LhsType, RhsType, EhsType)>& ternary_op) {
    const auto& shape = instruction->shape();
    const auto* lhs = instruction->operand(0);
    const auto* rhs = instruction->operand(1);
    const auto* ehs = instruction->operand(2);
    TF_RET_CHECK(ShapeUtil::SameDimensions(shape, lhs->shape()));
    TF_RET_CHECK(ShapeUtil::SameDimensions(lhs->shape(), rhs->shape()));
    TF_RET_CHECK(ShapeUtil::SameDimensions(rhs->shape(), ehs->shape()));

    const Literal& lhs_literal = parent_->GetEvaluatedLiteralFor(lhs);
    const Literal& rhs_literal = parent_->GetEvaluatedLiteralFor(rhs);
    const Literal& ehs_literal = parent_->GetEvaluatedLiteralFor(ehs);

    Literal result(shape);

    TF_RETURN_IF_ERROR(result.PopulateParallel<ReturnT>(
        [&](absl::Span<const int64_t> multi_index, int) {
          return ternary_op(lhs_literal.Get<LhsType>(multi_index),
                            rhs_literal.Get<RhsType>(multi_index),
                            ehs_literal.Get<EhsType>(multi_index));
        }));

    return std::move(result);
  }

  template <typename NativeT>
  static bool IsShiftOutOfBounds(ElementwiseT rhs) {
    using UnsignedT = make_specialized_unsigned_t<NativeT>;
    UnsignedT lhs_bits_unsigned =
        static_cast<UnsignedT>(std::numeric_limits<UnsignedT>::digits);
    UnsignedT rhs_unsigned = static_cast<UnsignedT>(rhs);
    return rhs_unsigned >= lhs_bits_unsigned;
  }

  HloEvaluator* parent_;  // not owned
};

// These extern templates prevent users of this class from implicitly
// instantiating it.  We explicitly instantiate this class in the various
// hlo_evaluator_typed_visitor*.cc files.
extern template class HloEvaluatorTypedVisitor<bool>;
extern template class HloEvaluatorTypedVisitor<u1, uint64_t>;
extern template class HloEvaluatorTypedVisitor<u2, uint64_t>;
extern template class HloEvaluatorTypedVisitor<u4, uint64_t>;
extern template class HloEvaluatorTypedVisitor<uint8_t, uint64_t>;
extern template class HloEvaluatorTypedVisitor<uint16_t, uint64_t>;
extern template class HloEvaluatorTypedVisitor<uint32_t, uint64_t>;
extern template class HloEvaluatorTypedVisitor<uint64_t>;
extern template class HloEvaluatorTypedVisitor<s1, int64_t>;
extern template class HloEvaluatorTypedVisitor<s2, int64_t>;
extern template class HloEvaluatorTypedVisitor<s4, int64_t>;
extern template class HloEvaluatorTypedVisitor<int8_t, int64_t>;
extern template class HloEvaluatorTypedVisitor<int16_t, int64_t>;
extern template class HloEvaluatorTypedVisitor<int32_t, int64_t>;
extern template class HloEvaluatorTypedVisitor<int64_t>;

}  // namespace zkx

#endif  // ZKX_HLO_EVALUATOR_HLO_EVALUATOR_TYPED_VISITOR_H_
