/* Copyright 2025 The ZKX Authors.

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

#include "zkx/mlir/codegen_utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/TypeUtilities.h"

#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveOps.h"
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveTypes.h"
#include "prime_ir/Dialect/Field/IR/FieldOps.h"
#include "zkx/mlir/mlir_utils.h"

namespace zkx::mlir_utils {

using namespace mlir;

namespace {

template <typename U, typename S>
Value DivideOrRemainderIntegerHelper(ImplicitLocOpBuilder& b, Value lhs,
                                     Value rhs, bool is_signed,
                                     Value returned_on_zero,
                                     Value returned_on_signed_overflow) {
  Type type = lhs.getType();
  auto element_type = cast<IntegerType>(getElementTypeOrSelf(type));
  Value zero = b.create<arith::ConstantOp>(b.getZeroAttr(type));
  auto make_constant = [&](const APInt& i) {
    return mlir_utils::GetConstantOrSplat(b, type,
                                          b.getIntegerAttr(element_type, i));
  };
  Value one = make_constant(APInt(element_type.getWidth(), 1));
  Value rhs_is_zero =
      b.create<arith::CmpIOp>(arith::CmpIPredicate::eq, rhs, zero);

  // For unsigned just set the divisor to 1 when it would be 0.
  if (!is_signed) {
    Value safe_rhs = b.create<arith::SelectOp>(rhs_is_zero, one, rhs);
    Value safe_div = b.create<U>(lhs, safe_rhs);
    return b.create<arith::SelectOp>(rhs_is_zero, returned_on_zero, safe_div);
  }

  // For signed also check for INT_SMIN / -1.
  Value smin = make_constant(APInt::getSignedMinValue(element_type.getWidth()));
  Value lhs_is_smin =
      b.create<arith::CmpIOp>(arith::CmpIPredicate::eq, lhs, smin);
  Value minus_one = make_constant(APInt::getAllOnes(element_type.getWidth()));
  Value rhs_is_minus_one =
      b.create<arith::CmpIOp>(arith::CmpIPredicate::eq, rhs, minus_one);
  Value has_int_min_overflow =
      b.create<arith::AndIOp>(lhs_is_smin, rhs_is_minus_one);
  Value rhs_is_unsafe =
      b.create<arith::OrIOp>(rhs_is_zero, has_int_min_overflow);
  Value safe_rhs = b.create<arith::SelectOp>(rhs_is_unsafe, one, rhs);
  Value safe_div = b.create<S>(lhs, safe_rhs);
  Value safe_smin = b.create<arith::SelectOp>(
      has_int_min_overflow, returned_on_signed_overflow, safe_div);
  return b.create<arith::SelectOp>(rhs_is_zero, returned_on_zero, safe_smin);
}

// Construct operations to select the saturated value if the shift amount is
// greater than the bitwidth of the type.
Value SelectShiftedOrSaturated(ImplicitLocOpBuilder& b, Value rhs,
                               Value shifted, Value saturated, Type type) {
  Type etype =
      isa<ShapedType>(type) ? cast<ShapedType>(type).getElementType() : type;
  auto bit_width_int = cast<IntegerType>(etype).getWidth();
  Value bit_width = mlir_utils::GetConstantOrSplat(
      b, type, b.getIntegerAttr(etype, bit_width_int));
  Value cmp =
      b.create<arith::CmpIOp>(arith::CmpIPredicate::ugt, bit_width, rhs);
  return b.create<arith::SelectOp>(cmp, shifted, saturated);
}

}  // namespace

Value ConvertInteger(ImplicitLocOpBuilder& b, ArrayRef<Type> result_types,
                     Type source_type, Type target_type, ValueRange args,
                     bool is_signed, ArrayRef<NamedAttribute> attributes) {
  if (target_type.isInteger(/*width=*/1)) {
    // When casting to bool, we need to compare whether the value is equal to
    // zero.
    Value zero =
        b.create<arith::ConstantOp>(b.getZeroAttr(args.front().getType()));
    return b.create<arith::CmpIOp>(arith::CmpIPredicate::ne, args.front(),
                                   zero);
  }
  auto src = cast<IntegerType>(getElementTypeOrSelf(source_type));
  auto res = cast<IntegerType>(getElementTypeOrSelf(target_type));
  if (src.getWidth() > res.getWidth()) {
    return b.create<arith::TruncIOp>(result_types, args, attributes);
  }
  if (src.getWidth() < res.getWidth()) {
    // Special case boolean values, so they get casted to `1` instead of `-1`.
    if (is_signed) {
      return b.create<arith::ExtSIOp>(result_types, args, attributes);
    } else {
      return b.create<arith::ExtUIOp>(result_types, args, attributes);
    }
  }
  // No conversion is needed for the same width integers
  return args.front();
}

Value DivideInteger(ImplicitLocOpBuilder& b, Value lhs, Value rhs,
                    bool is_signed) {
  // Integer division overflow behavior:
  //
  // X / 0 == -1
  // INT_SMIN /s -1 = INT_SMIN
  Type type = lhs.getType();
  Type element_type = getElementTypeOrSelf(type);
  auto integer_element_type = cast<IntegerType>(element_type);
  auto make_constant = [&](const APInt& i) {
    return mlir_utils::GetConstantOrSplat(
        b, type, b.getIntegerAttr(integer_element_type, i));
  };
  Value minus_one =
      make_constant(APInt::getAllOnes(integer_element_type.getWidth()));
  Value smin =
      make_constant(APInt::getSignedMinValue(integer_element_type.getWidth()));
  return DivideOrRemainderIntegerHelper<arith::DivUIOp, arith::DivSIOp>(
      b, lhs, rhs, is_signed,
      /*returned_on_zero=*/minus_one,
      /*returned_on_signed_overflow=*/smin);
}

Value PowerInteger(ImplicitLocOpBuilder& b, Value lhs, Value rhs,
                   bool is_signed) {
  // Exponentiation by squaring:
  // https://en.wikipedia.org/wiki/Exponentiation_by_squaring;
  Value neg_one =
      b.create<arith::ConstantOp>(b.getIntegerAttr(lhs.getType(), -1));
  Value zero = b.create<arith::ConstantOp>(b.getIntegerAttr(lhs.getType(), 0));
  Value one = b.create<arith::ConstantOp>(b.getIntegerAttr(lhs.getType(), 1));
  Value two = b.create<arith::ConstantOp>(b.getIntegerAttr(lhs.getType(), 2));
  Value step = b.create<arith::ConstantIndexOp>(1);
  Value lower_bound = b.create<arith::ConstantIndexOp>(0);
  // The number of iterations is chosen to be sufficient for the bit-width of
  // the exponent. For a 32-bit exponent, 32 iterations are needed.
  Value upper_bound = b.create<arith::ConstantIndexOp>(
      cast<IntegerType>(rhs.getType()).getWidth());
  Value original_base = lhs;
  Value original_exponent = rhs;

  Value accum =
      b.create<scf::ForOp>(
           lower_bound, upper_bound, step,
           SmallVector<Value>({one, original_base, original_exponent}),
           [&](OpBuilder& b, Location loc, Value /*v*/, ValueRange iters) {
             ImplicitLocOpBuilder nested_b(loc, b);
             Value accum = iters[0];
             Value base = iters[1];
             Value exponent = iters[2];

             Value condition = nested_b.create<arith::CmpIOp>(
                 arith::CmpIPredicate::eq,
                 nested_b.create<arith::AndIOp>(exponent, one), one);
             Value multiplied = nested_b.create<arith::MulIOp>(accum, base);
             accum =
                 nested_b.create<arith::SelectOp>(condition, multiplied, accum);
             base = nested_b.create<arith::MulIOp>(base, base);
             exponent = nested_b.create<arith::ShRUIOp>(exponent, one);
             nested_b.create<scf::YieldOp>(
                 SmallVector<Value>({accum, base, exponent}));
           })
          .getResult(0);

  if (!is_signed) {
    return accum;
  }

  Value rhs_is_even = b.create<arith::CmpIOp>(
      arith::CmpIPredicate::eq, b.create<arith::RemSIOp>(rhs, two), zero);
  Value rhs_is_negative =
      b.create<arith::CmpIOp>(arith::CmpIPredicate::slt, rhs, zero);
  Value lhs_is_one =
      b.create<arith::CmpIOp>(arith::CmpIPredicate::eq, lhs, one);
  Value lhs_is_neg_one =
      b.create<arith::CmpIOp>(arith::CmpIPredicate::eq, lhs, neg_one);

  // The accum is correct when the rhs is non-negative. When rhs is
  // negative, we return 0 for integer, with the exception of lhs values of 1
  // and -1 which have integer results for negative exponents. Specifically, the
  // calculation is the following:
  //
  // - Return accum if the rhs is not negative.
  // - Return 1 or -1 depending on the parity of rhs when the lhs is -1.
  // - Return 1 if lhs is 1.
  // - Else return 0.
  Value if_lhs_is_one = b.create<arith::SelectOp>(lhs_is_one, one, zero);
  Value if_lhs_is_neg_one = b.create<arith::SelectOp>(
      lhs_is_neg_one, b.create<arith::SelectOp>(rhs_is_even, one, neg_one),
      if_lhs_is_one);
  return b.create<arith::SelectOp>(rhs_is_negative, if_lhs_is_neg_one, accum);
}

Value RemainderInteger(ImplicitLocOpBuilder& b, Value lhs, Value rhs,
                       bool is_signed) {
  // Integer remainder overflow behavior:
  //
  // X % 0 == X
  // INT_SMIN %s -1 = 0
  Type type = lhs.getType();
  Value zero = b.create<arith::ConstantOp>(b.getZeroAttr(type));
  return DivideOrRemainderIntegerHelper<arith::RemUIOp, arith::RemSIOp>(
      b, lhs, rhs, is_signed,
      /*returned_on_zero=*/lhs,
      /*returned_on_signed_overflow=*/zero);
}

Value SignInteger(ImplicitLocOpBuilder& b, Value value) {
  // sign(x) = x == 0 ? 0 : ((x >>s (width - 1)) | 1)
  IntegerType integer_type =
      cast<IntegerType>(getElementTypeOrSelf(value.getType()));
  Value zero = mlir_utils::GetConstantOrSplat(b, value.getType(),
                                              b.getZeroAttr(integer_type));
  Value bit_width_minus_one = mlir_utils::GetConstantOrSplat(
      b, value.getType(),
      b.getIntegerAttr(integer_type, integer_type.getWidth() - 1));
  Value one = mlir_utils::GetConstantOrSplat(b, value.getType(),
                                             b.getOneAttr(integer_type));
  Value cmp = b.create<arith::CmpIOp>(arith::CmpIPredicate::eq, value, zero);
  Value shr = b.create<arith::ShRSIOp>(value, bit_width_minus_one);
  Value or_op = b.create<arith::OrIOp>(shr, one);
  return b.create<arith::SelectOp>(cmp, zero, or_op);
}

Value ShiftLeftInteger(ImplicitLocOpBuilder& b, Value lhs, Value rhs) {
  Type type = lhs.getType();

  // "Saturate" if the shift is greater than the bitwidth of the type
  Value zero = b.create<arith::ConstantOp>(b.getZeroAttr(type));
  Value shifted = b.create<arith::ShLIOp>(lhs, rhs);
  return SelectShiftedOrSaturated(b, rhs, shifted, zero, type);
}

Value ShiftRightArithmeticInteger(ImplicitLocOpBuilder& b, Value lhs,
                                  Value rhs) {
  Type type = lhs.getType();
  Type etype =
      isa<ShapedType>(type) ? cast<ShapedType>(type).getElementType() : type;
  auto bit_width_int = cast<IntegerType>(etype).getWidth();

  // "Saturate" if the shift is greater than the bitwidth of the type
  Value max_shift = mlir_utils::GetConstantOrSplat(
      b, type, b.getIntegerAttr(etype, bit_width_int - 1));
  Value saturated_shifted = b.create<arith::ShRSIOp>(lhs, max_shift);
  Value shifted = b.create<arith::ShRSIOp>(lhs, rhs);
  return SelectShiftedOrSaturated(b, rhs, shifted, saturated_shifted, type);
}

Value ShiftRightLogicalInteger(ImplicitLocOpBuilder& b, Value lhs, Value rhs) {
  Type type = lhs.getType();

  // "Saturate" if the shift is greater than the bitwidth of the type
  Value zero = b.create<arith::ConstantOp>(b.getZeroAttr(type));
  Value shifted = b.create<arith::ShRUIOp>(lhs, rhs);
  return SelectShiftedOrSaturated(b, rhs, shifted, zero, type);
}

Value ConvertField(ImplicitLocOpBuilder& b, ArrayRef<Type> result_types,
                   Type source_type, Type target_type, ValueRange args,
                   ArrayRef<NamedAttribute> attributes) {
  auto source_element_type = getElementTypeOrSelf(source_type);
  auto target_element_type = getElementTypeOrSelf(target_type);

  if (auto integer_type = dyn_cast<IntegerType>(source_element_type)) {
    if (auto pf_element_type =
            dyn_cast<prime_ir::field::PrimeFieldType>(target_element_type)) {
      unsigned src_width = integer_type.getWidth();
      unsigned dst_width = pf_element_type.getStorageType().getWidth();
      Value current_val = args[0];

      if (dst_width < src_width) {
        b.emitError("Cannot convert integer to prime field with smaller width");
        return nullptr;
      }

      if (dst_width > src_width) {
        Type target_storage_type = pf_element_type.getStorageType();

        if (auto shaped_type = dyn_cast<ShapedType>(current_val.getType())) {
          target_storage_type = shaped_type.clone(target_storage_type);
        }
        current_val =
            b.create<arith::ExtUIOp>(target_storage_type, current_val);
      }

      Type result_type = result_types.front();
      if (pf_element_type.isMontgomery()) {
        Type std_source_type =
            prime_ir::field::getStandardFormType(result_type);
        auto bitcast =
            b.create<prime_ir::field::BitcastOp>(std_source_type, current_val);
        return b.create<prime_ir::field::ToMontOp>(result_type, bitcast);
      }

      return b.create<prime_ir::field::BitcastOp>(result_type, current_val);
    }
  }

  if (prime_ir::field::isMontgomery(source_type)) {
    if (prime_ir::field::isMontgomery(target_type)) {
      return args[0];
    } else {
      return b.create<prime_ir::field::FromMontOp>(result_types, args,
                                                   attributes);
    }
  } else {
    if (prime_ir::field::isMontgomery(target_type)) {
      return b.create<prime_ir::field::ToMontOp>(result_types, args,
                                                 attributes);
    } else {
      return args[0];
    }
  }
}

mlir::Value ConvertEcPoint(mlir::ImplicitLocOpBuilder& b,
                           mlir::ArrayRef<mlir::Type> result_types,
                           mlir::Type source_type, mlir::Type target_type,
                           mlir::ValueRange args,
                           mlir::ArrayRef<mlir::NamedAttribute> attributes) {
  auto source_element_type = getElementTypeOrSelf(source_type);
  auto target_element_type = getElementTypeOrSelf(target_type);
  if (source_element_type == target_element_type) {
    return args[0];
  }

  auto is_ec_point_type = [](mlir::Type t) {
    return isa<prime_ir::elliptic_curve::AffineType>(t) ||
           isa<prime_ir::elliptic_curve::JacobianType>(t) ||
           isa<prime_ir::elliptic_curve::XYZZType>(t);
  };

  if (is_ec_point_type(source_element_type) &&
      is_ec_point_type(target_element_type)) {
    return b.create<prime_ir::elliptic_curve::ConvertPointTypeOp>(
        result_types, args, attributes);
  }

  return args[0];
}

}  // namespace zkx::mlir_utils
