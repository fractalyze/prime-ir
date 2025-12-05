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

#include "zkir/Dialect/ModArith/IR/ModArithOps.h"

#include "llvm/ADT/SmallVectorExtras.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "zkir/Dialect/ModArith/IR/ModArithDialect.h"
#include "zkir/Dialect/ModArith/IR/ModArithTypes.h"
#include "zkir/Utils/APIntUtils.h"

// IWYU pragma: begin_keep
// Headers needed for ModArithCanonicalization.cpp.inc
#include "mlir/IR/Matchers.h"
// IWYU pragma: end_keep

namespace mlir::zkir::mod_arith {

Type getStandardFormType(Type type) {
  auto modArithType = cast<ModArithType>(getElementTypeOrSelf(type));
  if (modArithType.isMontgomery()) {
    auto standardType =
        ModArithType::get(type.getContext(), modArithType.getModulus());
    if (auto memrefType = dyn_cast<MemRefType>(type)) {
      return MemRefType::get(memrefType.getShape(), standardType,
                             memrefType.getLayout(),
                             memrefType.getMemorySpace());
    } else if (auto shapedType = dyn_cast<ShapedType>(type)) {
      return shapedType.cloneWith(shapedType.getShape(), standardType);
    } else {
      return standardType;
    }
  } else {
    return type;
  }
}

Type getMontgomeryFormType(Type type) {
  auto modArithType = cast<ModArithType>(getElementTypeOrSelf(type));
  if (!modArithType.isMontgomery()) {
    auto montType =
        ModArithType::get(type.getContext(), modArithType.getModulus(), true);
    if (auto memrefType = dyn_cast<MemRefType>(type)) {
      return MemRefType::get(memrefType.getShape(), montType,
                             memrefType.getLayout(),
                             memrefType.getMemorySpace());
    } else if (auto shapedType = dyn_cast<ShapedType>(type)) {
      return shapedType.cloneWith(shapedType.getShape(), montType);
    } else {
      return montType;
    }
  } else {
    return type;
  }
}

bool BitcastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  Type inputType = getElementTypeOrSelf(inputs.front());
  Type outputType = getElementTypeOrSelf(outputs.front());

  return getIntOrModArithBitWidth(inputType) ==
         getIntOrModArithBitWidth(outputType);
}

LogicalResult MontReduceOp::verify() {
  IntegerType integerType =
      cast<IntegerType>(getElementTypeOrSelf(getLow().getType()));
  ModArithType modArithType = getResultModArithType(*this);
  unsigned intWidth = integerType.getWidth();
  unsigned modWidth = modArithType.getStorageBitWidth();
  if (intWidth != modWidth)
    return emitOpError() << "Expected operand width to be " << modWidth
                         << ", but got " << intWidth << " instead.";
  return success();
}

LogicalResult ToMontOp::verify() {
  ModArithType resultType = getResultModArithType(*this);
  if (!resultType.isMontgomery())
    return emitOpError() << "ToMontOp result should be a Montgomery type, "
                         << "but got " << resultType << ".";
  return success();
}

LogicalResult FromMontOp::verify() {
  ModArithType resultType = getResultModArithType(*this);
  if (resultType.isMontgomery())
    return emitOpError() << "FromMontOp result should be a standard type, "
                         << "but got " << resultType << ".";
  return success();
}

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
  return adaptor.getValue();
}

ConstantOp ConstantOp::materialize(OpBuilder &builder, Attribute value,
                                   Type type, Location loc) {
  if (!isa<ModArithType>(getElementTypeOrSelf(type))) {
    return nullptr;
  }

  if (auto intAttr = dyn_cast<IntegerAttr>(value)) {
    return builder.create<ConstantOp>(loc, type, intAttr);
  } else if (auto denseElementsAttr =
                 dyn_cast<DenseModArithElementsAttr>(value)) {
    return builder.create<ConstantOp>(loc, type, denseElementsAttr);
  }
  return nullptr;
}

Operation *ModArithDialect::materializeConstant(OpBuilder &builder,
                                                Attribute value, Type type,
                                                Location loc) {
  if (auto boolAttr = dyn_cast<BoolAttr>(value)) {
    return builder.create<arith::ConstantOp>(loc, boolAttr);
  } else if (auto denseElementsAttr = dyn_cast<DenseIntElementsAttr>(value)) {
    return builder.create<arith::ConstantOp>(loc, denseElementsAttr);
  }
  return ConstantOp::materialize(builder, value, type, loc);
}

OpFoldResult BitcastOp::fold(FoldAdaptor adaptor) {
  if (isa_and_present<IntegerAttr>(adaptor.getInput())) {
    return adaptor.getInput();
  } else if (auto denseElementsAttr = dyn_cast_if_present<DenseIntElementsAttr>(
                 adaptor.getInput())) {
    // TODO(chokobole): Can we remove this clone?
    return ZkirDenseElementsAttr::get(
        denseElementsAttr.getType().clone(getResultModArithType(*this)),
        SmallVector<APInt>(denseElementsAttr.getValues<APInt>().begin(),
                           denseElementsAttr.getValues<APInt>().end()));
  } else if (auto denseElementsAttr =
                 dyn_cast_if_present<DenseModArithElementsAttr>(
                     adaptor.getInput())) {
    return denseElementsAttr.bitcast(
        cast<ModArithType>(denseElementsAttr.getElementType())
            .getStorageType());
  }
  return {};
}

OpFoldResult ToMontOp::fold(FoldAdaptor adaptor) {
  auto modArithType = cast<ModArithType>(getElementTypeOrSelf(getType()));
  MontgomeryAttr montAttr = modArithType.getMontgomeryAttr();
  APInt modulus = modArithType.getModulus().getValue();

  auto toMontConversion = [montAttr, modulus](APInt value) {
    return mulMod(value, montAttr.getR().getValue(), modulus);
  };

  if (auto intAttr = dyn_cast_if_present<IntegerAttr>(adaptor.getInput())) {
    return IntegerAttr::get(intAttr.getType(),
                            toMontConversion(intAttr.getValue()));
  } else if (auto denseElementsAttr =
                 dyn_cast_if_present<DenseModArithElementsAttr>(
                     adaptor.getInput())) {
    return ZkirDenseElementsAttr::get(
        denseElementsAttr.getType(),
        llvm::map_to_vector(denseElementsAttr.getValues<APInt>(),
                            toMontConversion));
  }
  return {};
}

OpFoldResult FromMontOp::fold(FoldAdaptor adaptor) {
  auto modArithType = cast<ModArithType>(getElementTypeOrSelf(getType()));
  MontgomeryAttr montAttr = modArithType.getMontgomeryAttr();
  APInt modulus = modArithType.getModulus().getValue();

  auto fromMontConversion = [montAttr, modulus](APInt value) {
    return mulMod(value, montAttr.getRInv().getValue(), modulus);
  };

  if (auto intAttr = dyn_cast_if_present<IntegerAttr>(adaptor.getInput())) {
    return IntegerAttr::get(intAttr.getType(),
                            fromMontConversion(intAttr.getValue()));
  } else if (auto denseElementsAttr =
                 dyn_cast_if_present<DenseModArithElementsAttr>(
                     adaptor.getInput())) {
    return ZkirDenseElementsAttr::get(
        denseElementsAttr.getType(),
        llvm::map_to_vector(denseElementsAttr.getValues<APInt>(),
                            fromMontConversion));
  }
  return {};
}

OpFoldResult CmpOp::fold(FoldAdaptor adaptor) {
  auto compare = [](APInt lhs, APInt rhs, arith::CmpIPredicate predicate) {
    switch (predicate) {
    case arith::CmpIPredicate::eq:
      return lhs.eq(rhs);
    case arith::CmpIPredicate::ne:
      return lhs.ne(rhs);
    case arith::CmpIPredicate::slt:
      return lhs.slt(rhs);
    case arith::CmpIPredicate::sle:
      return lhs.sle(rhs);
    case arith::CmpIPredicate::sgt:
      return lhs.sgt(rhs);
    case arith::CmpIPredicate::sge:
      return lhs.sge(rhs);
    case arith::CmpIPredicate::ult:
      return lhs.ult(rhs);
    case arith::CmpIPredicate::ule:
      return lhs.ule(rhs);
    case arith::CmpIPredicate::ugt:
      return lhs.ugt(rhs);
    case arith::CmpIPredicate::uge:
      return lhs.uge(rhs);
    }
  };

  auto predicate = adaptor.getPredicate();
  if (auto lhs = dyn_cast_if_present<IntegerAttr>(adaptor.getLhs())) {
    if (auto rhs = dyn_cast_if_present<IntegerAttr>(adaptor.getRhs())) {
      return BoolAttr::get(getType().getContext(),
                           compare(lhs.getValue(), rhs.getValue(), predicate));
    }
  } else if (auto lhs = dyn_cast_if_present<DenseModArithElementsAttr>(
                 adaptor.getLhs())) {
    if (auto rhs =
            dyn_cast_if_present<DenseModArithElementsAttr>(adaptor.getRhs())) {
      return DenseIntElementsAttr::get(
          lhs.getType().clone(IntegerType::get(lhs.getType().getContext(), 1)),
          llvm::map_to_vector(
              llvm::zip(lhs.getValues<APInt>(), rhs.getValues<APInt>()),
              [compare, predicate](const auto &values) {
                const auto &[lhs, rhs] = values;
                return compare(lhs, rhs, predicate);
              }));
    }
  }
  return {};
}

OpFoldResult NegateOp::fold(FoldAdaptor adaptor) {
  auto modArithType = cast<ModArithType>(getElementTypeOrSelf(getType()));
  APInt modulus = modArithType.getModulus().getValue();

  auto negateMod = [modulus](APInt value) {
    return value.isZero() ? value : modulus - value;
  };

  if (auto intAttr = dyn_cast_if_present<IntegerAttr>(adaptor.getInput())) {
    return IntegerAttr::get(intAttr.getType(), negateMod(intAttr.getValue()));
  } else if (auto denseElementsAttr =
                 dyn_cast_if_present<DenseModArithElementsAttr>(
                     adaptor.getInput())) {
    return ZkirDenseElementsAttr::get(
        denseElementsAttr.getType(),
        llvm::map_to_vector(denseElementsAttr.getValues<APInt>(), negateMod));
  }
  return {};
}

OpFoldResult DoubleOp::fold(FoldAdaptor adaptor) {
  auto modArithType = cast<ModArithType>(getElementTypeOrSelf(getType()));
  APInt modulus = modArithType.getModulus().getValue();

  auto doubleMod = [modulus](APInt value) {
    assert(modulus.getBitWidth() > modulus.getActiveBits());
    APInt resultValue = value.shl(1);
    if (resultValue.uge(modulus)) {
      resultValue -= modulus;
    }
    return resultValue;
  };

  if (auto intAttr = dyn_cast_if_present<IntegerAttr>(adaptor.getInput())) {
    return IntegerAttr::get(intAttr.getType(), doubleMod(intAttr.getValue()));
  } else if (auto denseElementsAttr =
                 dyn_cast_if_present<DenseModArithElementsAttr>(
                     adaptor.getInput())) {
    return ZkirDenseElementsAttr::get(
        denseElementsAttr.getType(),
        llvm::map_to_vector(denseElementsAttr.getValues<APInt>(), doubleMod));
  }
  return {};
}

OpFoldResult SquareOp::fold(FoldAdaptor adaptor) {
  auto modArithType = cast<ModArithType>(getElementTypeOrSelf(getType()));
  APInt modulus = modArithType.getModulus().getValue();

  auto squareMod = [modArithType, modulus](APInt value) {
    auto square = mulMod(value, value, modulus);
    if (modArithType.isMontgomery()) {
      MontgomeryAttr montAttr = modArithType.getMontgomeryAttr();
      square = mulMod(square, montAttr.getRInv().getValue(), modulus);
    }
    return square;
  };

  if (auto intAttr = dyn_cast_if_present<IntegerAttr>(adaptor.getInput())) {
    return IntegerAttr::get(intAttr.getType(), squareMod(intAttr.getValue()));
  } else if (auto denseElementsAttr =
                 dyn_cast_if_present<DenseModArithElementsAttr>(
                     adaptor.getInput())) {
    return ZkirDenseElementsAttr::get(
        denseElementsAttr.getType(),
        llvm::map_to_vector(denseElementsAttr.getValues<APInt>(), squareMod));
  }
  return {};
}

OpFoldResult MontSquareOp::fold(FoldAdaptor adaptor) {
  auto modArithType = cast<ModArithType>(getElementTypeOrSelf(getType()));
  APInt modulus = modArithType.getModulus().getValue();

  auto montSquareMod = [modArithType, modulus](APInt value) {
    auto square = mulMod(value, value, modulus);
    MontgomeryAttr montAttr = modArithType.getMontgomeryAttr();
    square = mulMod(square, montAttr.getRInv().getValue(), modulus);
    return square;
  };

  if (auto intAttr = dyn_cast_if_present<IntegerAttr>(adaptor.getInput())) {
    return IntegerAttr::get(intAttr.getType(),
                            montSquareMod(intAttr.getValue()));
  } else if (auto denseElementsAttr =
                 dyn_cast_if_present<DenseModArithElementsAttr>(
                     adaptor.getInput())) {
    return ZkirDenseElementsAttr::get(
        denseElementsAttr.getType(),
        llvm::map_to_vector(denseElementsAttr.getValues<APInt>(),
                            montSquareMod));
  }
  return {};
}

OpFoldResult InverseOp::fold(FoldAdaptor adaptor) {
  auto modArithType = cast<ModArithType>(getElementTypeOrSelf(getType()));
  APInt modulus = modArithType.getModulus().getValue();

  auto inverseMod = [modArithType, modulus](APInt value) {
    auto inverse = multiplicativeInverse(value, modulus);
    if (modArithType.isMontgomery()) {
      MontgomeryAttr montAttr = modArithType.getMontgomeryAttr();
      inverse = mulMod(inverse, montAttr.getRSquared().getValue(), modulus);
    }
    return inverse;
  };

  if (auto intAttr = dyn_cast_if_present<IntegerAttr>(adaptor.getInput())) {
    return IntegerAttr::get(intAttr.getType(), inverseMod(intAttr.getValue()));
  } else if (auto denseElementsAttr =
                 dyn_cast_if_present<DenseModArithElementsAttr>(
                     adaptor.getInput())) {
    return ZkirDenseElementsAttr::get(
        denseElementsAttr.getType(),
        llvm::map_to_vector(denseElementsAttr.getValues<APInt>(), inverseMod));
  }
  return {};
}

OpFoldResult MontInverseOp::fold(FoldAdaptor adaptor) {
  auto modArithType = cast<ModArithType>(getElementTypeOrSelf(getType()));
  APInt modulus = modArithType.getModulus().getValue();

  auto montInverseMod = [modArithType, modulus](APInt value) {
    auto inverse = multiplicativeInverse(value, modulus);
    MontgomeryAttr montAttr = modArithType.getMontgomeryAttr();
    inverse = mulMod(inverse, montAttr.getRSquared().getValue(), modulus);
    return inverse;
  };

  if (auto intAttr = dyn_cast_if_present<IntegerAttr>(adaptor.getInput())) {
    return IntegerAttr::get(intAttr.getType(),
                            montInverseMod(intAttr.getValue()));
  } else if (auto denseElementsAttr =
                 dyn_cast_if_present<DenseModArithElementsAttr>(
                     adaptor.getInput())) {
    return ZkirDenseElementsAttr::get(
        denseElementsAttr.getType(),
        llvm::map_to_vector(denseElementsAttr.getValues<APInt>(),
                            montInverseMod));
  }
  return {};
}

OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  auto modArithType = cast<ModArithType>(getElementTypeOrSelf(getType()));
  APInt modulus = modArithType.getModulus().getValue();
  auto addMod = [modulus](const APInt &a, const APInt &b) -> APInt {
    APInt sum = a + b;
    if (sum.uge(modulus)) {
      sum -= modulus;
    }
    return sum;
  };

  if (auto rhsIntAttr = dyn_cast_if_present<IntegerAttr>(adaptor.getRhs())) {
    APInt rhsValue = rhsIntAttr.getValue();
    if (auto lhsIntAttr = dyn_cast_if_present<IntegerAttr>(adaptor.getLhs())) {
      APInt lhsValue = lhsIntAttr.getValue();
      APInt resultValue = addMod(lhsValue, rhsValue);
      return IntegerAttr::get(lhsIntAttr.getType(), resultValue);
    } else if (rhsValue.isZero()) {
      // x + 0 -> x
      return getLhs();
    }
  } else if (auto rhsDenseElementsAttr =
                 dyn_cast_if_present<DenseModArithElementsAttr>(
                     adaptor.getRhs())) {
    auto rhsValues = rhsDenseElementsAttr.getValues<APInt>();
    if (auto lhsDenseElementsAttr =
            dyn_cast_if_present<DenseModArithElementsAttr>(adaptor.getLhs())) {
      auto lhsValues = lhsDenseElementsAttr.getValues<APInt>();
      return ZkirDenseElementsAttr::get(
          rhsDenseElementsAttr.getType(),
          llvm::map_to_vector(llvm::zip(lhsValues, rhsValues),
                              [addMod](const auto &values) {
                                const auto &[lhs, rhs] = values;
                                return addMod(lhs, rhs);
                              }));
    } else {
      // NOLINTNEXTLINE(whitespace/newline)
      if (llvm::all_of(rhsValues, [](APInt value) { return value.isZero(); })) {
        // x + 0 -> x
        return getLhs();
      }
    }
  }
  return {};
}

OpFoldResult SubOp::fold(FoldAdaptor adaptor) {
  auto modArithType = cast<ModArithType>(getElementTypeOrSelf(getType()));
  APInt modulus = modArithType.getModulus().getValue();
  auto subMod = [modulus](const APInt &a, const APInt &b) -> APInt {
    auto diff = a + modulus - b;
    if (diff.uge(modulus)) {
      diff -= modulus;
    }
    return diff;
  };

  if (auto rhsIntAttr = dyn_cast_if_present<IntegerAttr>(adaptor.getRhs())) {
    APInt rhsValue = rhsIntAttr.getValue();
    if (auto lhsIntAttr = dyn_cast_if_present<IntegerAttr>(adaptor.getLhs())) {
      APInt lhsValue = lhsIntAttr.getValue();
      APInt resultValue = subMod(lhsValue, rhsValue);
      return IntegerAttr::get(lhsIntAttr.getType(), resultValue);
    } else if (rhsValue.isZero()) {
      // x - 0 -> x
      return getLhs();
    }
  } else if (auto rhsDenseElementsAttr =
                 dyn_cast_if_present<DenseModArithElementsAttr>(
                     adaptor.getRhs())) {
    auto rhsValues = rhsDenseElementsAttr.getValues<APInt>();
    if (auto lhsDenseElementsAttr =
            dyn_cast_if_present<DenseModArithElementsAttr>(adaptor.getLhs())) {
      auto lhsValues = lhsDenseElementsAttr.getValues<APInt>();
      return ZkirDenseElementsAttr::get(
          rhsDenseElementsAttr.getType(),
          llvm::map_to_vector(llvm::zip(lhsValues, rhsValues),
                              [subMod](const auto &values) {
                                const auto &[lhs, rhs] = values;
                                return subMod(lhs, rhs);
                              }));
    } else {
      // NOLINTNEXTLINE(whitespace/newline)
      if (llvm::all_of(rhsValues, [](APInt value) { return value.isZero(); })) {
        // x - 0 -> x
        return getLhs();
      }
    }
  }
  return {};
}

OpFoldResult MulOp::fold(FoldAdaptor adaptor) {
  auto modArithType = cast<ModArithType>(getElementTypeOrSelf(getType()));
  MontgomeryAttr montAttr = modArithType.getMontgomeryAttr();
  APInt modulus = modArithType.getModulus().getValue();

  if (auto rhsIntAttr = dyn_cast_if_present<IntegerAttr>(adaptor.getRhs())) {
    if (auto lhsIntAttr = dyn_cast_if_present<IntegerAttr>(adaptor.getLhs())) {
      APInt lhsValue = lhsIntAttr.getValue();
      APInt rhsValue = rhsIntAttr.getValue();
      if (modArithType.isMontgomery()) {
        rhsValue = mulMod(rhsValue, montAttr.getRInv().getValue(), modulus);
      }
      APInt resultValue = mulMod(lhsValue, rhsValue, modulus);
      return IntegerAttr::get(lhsIntAttr.getType(), resultValue);
    } else if (rhsIntAttr.getValue().isZero()) {
      // x * 0 -> 0
      return getRhs();
    } else {
      if (modArithType.isMontgomery()) {
        if (rhsIntAttr.getValue() == montAttr.getR().getValue()) {
          // x * 1 -> x
          return getLhs();
        }
      } else {
        if (rhsIntAttr.getValue().isOne()) {
          // x * 1 -> x
          return getLhs();
        }
      }
    }
  } else if (auto rhsDenseElementsAttr =
                 dyn_cast_if_present<DenseModArithElementsAttr>(
                     adaptor.getRhs())) {
    auto rhsValues = rhsDenseElementsAttr.getValues<APInt>();
    if (auto lhsDenseElementsAttr =
            dyn_cast_if_present<DenseModArithElementsAttr>(adaptor.getLhs())) {
      auto lhsValues = lhsDenseElementsAttr.getValues<APInt>();
      return ZkirDenseElementsAttr::get(
          rhsDenseElementsAttr.getType(),
          llvm::map_to_vector(
              llvm::zip(lhsValues, rhsValues),
              [modulus, montAttr, modArithType](const auto &values) {
                auto [lhs, rhs] = values;
                if (modArithType.isMontgomery()) {
                  rhs = mulMod(rhs, montAttr.getRInv().getValue(), modulus);
                }
                return mulMod(lhs, rhs, modulus);
              }));
    } else {
      // NOLINTNEXTLINE(whitespace/newline)
      if (llvm::all_of(rhsValues, [](APInt value) { return value.isZero(); })) {
        // x * 0 -> 0
        return getRhs();
      } else {
        if (modArithType.isMontgomery()) {
          if (llvm::all_of(rhsValues, [montAttr](APInt value) {
                return value == montAttr.getR().getValue();
              })) {
            // x * 1 -> x
            return getLhs();
          }
        } else {
          if (llvm::all_of(rhsValues,
                           [](APInt value) { return value.isOne(); })) {
            // x * 1 -> x
            return getLhs();
          }
        }
      }
    }
  }
  return {};
}

OpFoldResult MontMulOp::fold(FoldAdaptor adaptor) {
  auto modArithType = cast<ModArithType>(getElementTypeOrSelf(getType()));
  MontgomeryAttr montAttr = modArithType.getMontgomeryAttr();
  APInt modulus = modArithType.getModulus().getValue();

  if (auto rhsIntAttr = dyn_cast_if_present<IntegerAttr>(adaptor.getRhs())) {
    APInt rhsValue =
        mulMod(rhsIntAttr.getValue(), montAttr.getRInv().getValue(), modulus);
    if (auto lhsIntAttr = dyn_cast_if_present<IntegerAttr>(adaptor.getLhs())) {
      APInt lhsValue = lhsIntAttr.getValue();
      APInt resultValue = mulMod(lhsValue, rhsValue, modulus);
      return IntegerAttr::get(lhsIntAttr.getType(), resultValue);
    } else if (rhsValue.isZero()) {
      // x * 0 -> 0
      return getRhs();
    } else if (rhsValue.isOne()) {
      // x * 1 -> x
      return getLhs();
    }
  } else if (auto rhsDenseElementsAttr =
                 dyn_cast_if_present<DenseModArithElementsAttr>(
                     adaptor.getRhs())) {
    auto rhsValues = rhsDenseElementsAttr.getValues<APInt>();
    if (auto lhsDenseElementsAttr =
            dyn_cast_if_present<DenseModArithElementsAttr>(adaptor.getLhs())) {
      auto lhsValues = lhsDenseElementsAttr.getValues<APInt>();
      return ZkirDenseElementsAttr::get(
          rhsDenseElementsAttr.getType(),
          llvm::map_to_vector(
              llvm::zip(lhsValues, rhsValues),
              [modulus, montAttr, modArithType](const auto &values) {
                auto [lhs, rhs] = values;
                if (modArithType.isMontgomery()) {
                  rhs = mulMod(rhs, montAttr.getRInv().getValue(), modulus);
                }
                return mulMod(lhs, rhs, modulus);
              }));
    } else {
      // NOLINTNEXTLINE(whitespace/newline)
      if (llvm::all_of(rhsValues, [](APInt value) { return value.isZero(); })) {
        // x * 0 -> 0
        return getRhs();
      } else if (llvm::all_of(rhsValues, [montAttr](APInt value) {
                   return value == montAttr.getR().getValue();
                 })) {
        // x * 1 -> x
        return getLhs();
      }
    }
  }
  return {};
}

ParseResult ConstantOp::parse(OpAsmParser &parser, OperationState &result) {
  APInt parsedInt;
  Type parsedType;
  ZkirDenseElementsAttr valueAttr;

  if (parser.parseOptionalInteger(parsedInt).has_value()) {
    if (failed(parser.parseColonType(parsedType)))
      return failure();

    if (parsedInt.isNegative()) {
      parser.emitError(parser.getCurrentLocation(),
                       "negative value is not allowed");
      return failure();
    }

    auto modArithType = dyn_cast<ModArithType>(parsedType);
    if (!modArithType)
      return failure();

    APInt modulus = modArithType.getModulus().getValue();
    if (modulus.isNegative() || modulus.isZero()) {
      parser.emitError(parser.getCurrentLocation(), "modulus must be positive");
      return failure();
    }

    unsigned outputBitWidth = modArithType.getStorageBitWidth();
    if (parsedInt.getActiveBits() > outputBitWidth) {
      parser.emitError(parser.getCurrentLocation(),
                       "constant value is too large for the underlying type");
      return failure();
    }

    // zero-extend or truncate to the correct bitwidth
    parsedInt = parsedInt.zextOrTrunc(outputBitWidth).urem(modulus);
    result.addAttribute(
        "value", IntegerAttr::get(modArithType.getStorageType(), parsedInt));
    result.addTypes(parsedType);
    return success();
  }

  if (failed(parser.parseAttribute(valueAttr))) {
    parser.emitError(parser.getCurrentLocation(),
                     "expected value to be a scalar or dense elements attr");
    return failure();
  }
  result.addAttribute("value", valueAttr);
  result.addTypes(valueAttr.getType());
  return success();
}

void ConstantOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printAttributeWithoutType(getValue());
  p << " : ";
  p.printType(getType());
}

namespace {
bool isNegativeOf(Attribute attr, Value val, uint32_t offset) {
  IntegerAttr intAttr = dyn_cast_if_present<IntegerAttr>(attr);
  if (auto denseIntAttr = dyn_cast_if_present<SplatElementsAttr>(attr)) {
    intAttr = denseIntAttr.getSplatValue<IntegerAttr>();
  }
  if (intAttr) {
    auto modArithType = cast<ModArithType>(getElementTypeOrSelf(val.getType()));
    APInt modulus = modArithType.getModulus().getValue();
    if (modArithType.isMontgomery()) {
      MontgomeryAttr montAttr = modArithType.getMontgomeryAttr();
      APInt montReduced =
          mulMod(intAttr.getValue(), montAttr.getRInv().getValue(), modulus);
      return montReduced == modulus - offset;
    } else {
      auto intAttr = cast<IntegerAttr>(attr);
      return intAttr.getValue() == modulus - offset;
    }
  }
  return false;
}

bool isEqualTo(Attribute attr, uint32_t offset) {
  IntegerAttr intAttr = dyn_cast_if_present<IntegerAttr>(attr);
  if (auto denseIntAttr = dyn_cast_if_present<SplatElementsAttr>(attr)) {
    intAttr = denseIntAttr.getSplatValue<IntegerAttr>();
  }
  if (intAttr) {
    return (intAttr.getValue() - offset).isZero();
  }
  return false;
}
} // namespace

namespace {
#include "zkir/Dialect/ModArith/IR/ModArithCanonicalization.cpp.inc"
}

void AddOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                        MLIRContext *context) {
  patterns.add<AddConstantTwice>(context);
  patterns.add<AddConstantToSubLhs>(context);
  patterns.add<AddConstantToSubRhs>(context);
  patterns.add<AddSelfIsDouble>(context);
  patterns.add<AddBothNegated>(context);
  patterns.add<AddAfterSub>(context);
  patterns.add<AddAfterNegLhs>(context);
  patterns.add<AddAfterNegRhs>(context);
  patterns.add<FactorMulAdd>(context);
}

void SubOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                        MLIRContext *context) {
  patterns.add<SubConstantFromAdd>(context);
  patterns.add<SubConstantTwiceLhs>(context);
  patterns.add<SubConstantTwiceRhs>(context);
  patterns.add<SubAddFromConstant>(context);
  patterns.add<SubSubFromConstantLhs>(context);
  patterns.add<SubSubFromConstantRhs>(context);
  patterns.add<SubSelfIsZero>(context);
  patterns.add<SubLhsAfterAdd>(context);
  patterns.add<SubRhsAfterAdd>(context);
  patterns.add<SubLhsAfterSub>(context);
  patterns.add<SubAfterNegLhs>(context);
  patterns.add<SubAfterNegRhs>(context);
  patterns.add<SubBothNegated>(context);
  patterns.add<SubAfterSquareBoth>(context);
  patterns.add<FactorMulSub>(context);
}

void MulOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                        MLIRContext *context) {
  patterns.add<MulByTwoIsDouble>(context);
  patterns.add<MulSelfIsSquare>(context);
  patterns.add<MulNegativeOneRhs>(context);
  patterns.add<MulNegativeTwoRhs>(context);
  patterns.add<MulNegativeThreeRhs>(context);
  patterns.add<MulNegativeFourRhs>(context);
  patterns.add<MulConstantTwice>(context);
  patterns.add<MulOfMulByConstant>(context);
  patterns.add<MulAddDistributeConstant>(context);
  patterns.add<MulSubDistributeConstantRhs>(context);
  patterns.add<MulSubDistributeConstantLhs>(context);
}

} // namespace mlir::zkir::mod_arith
