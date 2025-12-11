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

#include "zkir/Dialect/Field/IR/FieldOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "zkir/Dialect/ModArith/IR/ModArithTypes.h"

// IWYU pragma: begin_keep
// Headers needed for FieldCanonicalization.cpp.inc
#include "mlir/IR/Matchers.h"
#include "zkir/Dialect/TensorExt/IR/TensorExtOps.h"
// IWYU pragma: end_keep
namespace mlir::zkir::field {

Type getStandardFormType(Type type) {
  Type standardType = getElementTypeOrSelf(type);
  if (auto pfType = dyn_cast<PrimeFieldType>(standardType)) {
    if (pfType.isMontgomery()) {
      standardType =
          PrimeFieldType::get(type.getContext(), pfType.getModulus());
    }
  } else if (auto extField =
                 dyn_cast<ExtensionFieldTypeInterface>(standardType)) {
    if (extField.isMontgomery()) {
      auto baseField = cast<PrimeFieldType>(extField.getBaseFieldType());
      auto pfType =
          PrimeFieldType::get(type.getContext(), baseField.getModulus());
      auto nonResidue = cast<IntegerAttr>(extField.getNonResidue());
      standardType = extField.cloneWith(
          pfType,
          mod_arith::getAttrAsStandardForm(baseField.getModulus(), nonResidue));
    }
  }
  if (auto memrefType = dyn_cast<MemRefType>(type)) {
    return MemRefType::get(memrefType.getShape(), standardType,
                           memrefType.getLayout(), memrefType.getMemorySpace());
  } else if (auto shapedType = dyn_cast<ShapedType>(type)) {
    return shapedType.cloneWith(shapedType.getShape(), standardType);
  } else {
    return standardType;
  }
}

Type getMontgomeryFormType(Type type) {
  Type montType = getElementTypeOrSelf(type);
  if (auto pfType = dyn_cast<PrimeFieldType>(montType)) {
    if (!pfType.isMontgomery()) {
      montType =
          PrimeFieldType::get(type.getContext(), pfType.getModulus(), true);
    }
  } else if (auto extField = dyn_cast<ExtensionFieldTypeInterface>(montType)) {
    if (!extField.isMontgomery()) {
      auto baseField = cast<PrimeFieldType>(extField.getBaseFieldType());
      auto pfType =
          PrimeFieldType::get(type.getContext(), baseField.getModulus(), true);
      auto nonResidue = cast<IntegerAttr>(extField.getNonResidue());
      montType =
          extField.cloneWith(pfType, mod_arith::getAttrAsMontgomeryForm(
                                         baseField.getModulus(), nonResidue));
    }
  }
  if (auto memrefType = dyn_cast<MemRefType>(type)) {
    return MemRefType::get(memrefType.getShape(), montType,
                           memrefType.getLayout(), memrefType.getMemorySpace());
  } else if (auto shapedType = dyn_cast<ShapedType>(type)) {
    return shapedType.cloneWith(shapedType.getShape(), montType);
  } else {
    return montType;
  }
}

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
  return adaptor.getValue();
}

ParseResult ConstantOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<APInt> parsedInt;
  Type parsedType;

  if (failed(parser.parseCommaSeparatedList(
          [&]() { return parser.parseInteger(parsedInt.emplace_back()); })) ||
      failed(parser.parseColonType(parsedType)))
    return failure();

  if (auto pfType = dyn_cast<PrimeFieldType>(parsedType)) {
    if (parsedInt.size() != 1) {
      parser.emitError(parser.getCurrentLocation(),
                       "prime field constant must have exactly one value");
      return failure();
    }

    auto modulus = pfType.getModulus().getValue();

    // TODO(batzor): Check if the modulus is a prime number.
    if (modulus.isNegative() || modulus.isZero()) {
      parser.emitError(parser.getCurrentLocation(), "modulus must be positive");
      return failure();
    }

    auto outputBitWidth = pfType.getStorageBitWidth();
    if (parsedInt[0].getActiveBits() > outputBitWidth)
      return parser.emitError(
          parser.getCurrentLocation(),
          "constant value is too large for the underlying type");

    // zero-extend or truncate to the correct bitwidth
    parsedInt[0] = parsedInt[0].zextOrTrunc(outputBitWidth);
    result.addAttribute(
        "value", IntegerAttr::get(pfType.getStorageType(), parsedInt[0]));
    result.addTypes(pfType);
    return success();
  } else if (auto extField =
                 dyn_cast<ExtensionFieldTypeInterface>(parsedType)) {
    unsigned numCoeffs = extField.getDegreeOverBase();
    if (parsedInt.size() != numCoeffs) {
      parser.emitError(parser.getCurrentLocation(),
                       "extension field constant must have exactly " +
                           std::to_string(numCoeffs) + " values");
      return failure();
    }

    auto baseField = cast<PrimeFieldType>(extField.getBaseFieldType());
    auto modulus = baseField.getModulus().getValue();

    // TODO(batzor): Check if the modulus is a prime number.
    if (modulus.isNegative() || modulus.isZero()) {
      parser.emitError(parser.getCurrentLocation(), "modulus must be positive");
      return failure();
    }

    auto outputBitWidth = baseField.getStorageBitWidth();
    for (const auto &value : parsedInt) {
      if (value.getActiveBits() > outputBitWidth)
        return parser.emitError(
            parser.getCurrentLocation(),
            "constant value is too large for the underlying type");
    }

    // zero-extend or truncate to the correct bitwidth
    for (unsigned i = 0; i < numCoeffs; ++i) {
      parsedInt[i] = parsedInt[i].zextOrTrunc(outputBitWidth);
    }
    result.addAttribute("value", extField.createConstantAttr(parsedInt));
    result.addTypes(parsedType);
    return success();
  }
  parser.emitError(parser.getCurrentLocation(),
                   "invalid constant type: expected prime or extension field");
  return failure();
}

void ConstantOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printAttributeWithoutType(getValue());
  p << " : ";
  p.printType(getType());
}

template <typename OpType>
static LogicalResult disallowShapedTypeOfExtField(OpType op) {
  // FIXME(batzor): In the prime field case, we rely on elementwise trait but in
  // the quadratic extension case, `linalg.generic` introduced by the
  // elementwise pass will be ill-formed due to the 1:N conversion.
  auto resultType = op.getType();
  if (isa<ShapedType>(resultType)) {
    auto elementType = cast<ShapedType>(resultType).getElementType();
    if (isa<QuadraticExtFieldType>(elementType)) {
      return op->emitOpError("shaped type is not supported for quadratic "
                             "extension field type");
    }
  }
  return success();
}

LogicalResult CmpOp::verify() {
  auto operandType = getElementTypeOrSelf(getLhs());
  if (isa<QuadraticExtFieldType>(operandType)) {
    arith::CmpIPredicate predicate = getPredicate();
    if (predicate == arith::CmpIPredicate::eq ||
        predicate == arith::CmpIPredicate::ne) {
      return success();
    } else {
      return emitOpError() << "only 'eq' and 'ne' comparisons are supported "
                              "for quadratic extension field type";
    }
  }
  return success();
}
LogicalResult FromMontOp::verify() {
  bool isMont = isMontgomery(getType());
  if (isMont) {
    return emitOpError()
           << "FromMontOp result should be a standard type, but got "
           << getElementTypeOrSelf(getType()) << ".";
  }
  return success();
}
LogicalResult ToMontOp::verify() {
  bool isMont = isMontgomery(getType());
  if (!isMont) {
    return emitOpError()
           << "ToMontOp result should be a Montgomery type, but got "
           << getElementTypeOrSelf(getType()) << ".";
  }
  return success();
}

bool BitcastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  Type inputType = getElementTypeOrSelf(inputs.front());
  Type outputType = getElementTypeOrSelf(outputs.front());

  return getIntOrPrimeFieldBitWidth(inputType) ==
         getIntOrPrimeFieldBitWidth(outputType);
}

LogicalResult ExtToCoeffsOp::verify() {
  Type inputType = getInput().getType();
  if (auto extField = dyn_cast<ExtensionFieldTypeInterface>(inputType)) {
    unsigned expected = extField.getDegreeOverBase();
    if (getOutput().size() == expected) {
      return success();
    }
    return emitOpError() << "expected " << expected
                         << " output types for extension field input, but got "
                         << getOutput().size();
  }
  return emitOpError() << "input type must be an extension field; got "
                       << inputType;
}

LogicalResult ExtFromCoeffsOp::verify() {
  Type outputType = getType();
  if (auto extField = dyn_cast<ExtensionFieldTypeInterface>(outputType)) {
    unsigned expected = extField.getDegreeOverBase();
    if (getInput().size() == expected) {
      return success();
    }
    return emitOpError() << "expected " << expected
                         << " input types for extension field output, but got "
                         << getInput().size();
  }
  return emitOpError() << "output type must be an extension field; got "
                       << outputType;
}

namespace {
#include "zkir/Dialect/Field/IR/FieldCanonicalization.cpp.inc"
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
  patterns.add<SubLhsAfterAdd>(context);
  patterns.add<SubRhsAfterAdd>(context);
  patterns.add<SubLhsAfterSub>(context);
  patterns.add<SubAfterNegLhs>(context);
  patterns.add<SubAfterNegRhs>(context);
  patterns.add<SubBothNegated>(context);
  patterns.add<SubAfterSquareBoth>(context);
  patterns.add<SubAfterSumSquare>(context);
  patterns.add<FactorMulSub>(context);
}

void MulOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                        MLIRContext *context) {
  patterns.add<MulSelfIsSquare>(context);
  patterns.add<MulConstantTwice>(context);
  patterns.add<MulOfMulByConstant>(context);
  patterns.add<BitReverseMulBitReverse>(context);
}

} // namespace mlir::zkir::field
