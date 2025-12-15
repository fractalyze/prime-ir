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

#include "zkir/Utils/AssemblyFormatUtils.h"

namespace mlir::zkir {

void printModulus(AsmPrinter &printer, const APInt &modulus,
                  const Type &storageType, bool isMontgomery) {
  printer << "<";
  modulus.print(printer.getStream(), false);
  printer << " : " << storageType;
  if (isMontgomery) {
    printer << ", true";
  }
  printer << ">";
}

ParseResult validateModularInteger(OpAsmParser &parser, const APInt &modulus,
                                   APInt &parsedInt) {
  if (parsedInt.getActiveBits() > modulus.getBitWidth()) {
    return parser.emitError(parser.getCurrentLocation(),
                            "value is too large for the underlying type");
  }
  parsedInt = parsedInt.zextOrTrunc(modulus.getBitWidth());
  if (parsedInt.uge(modulus)) {
    return parser.emitError(parser.getCurrentLocation(),
                            "value is not in the field defined by modulus");
  }
  return success();
}

ParseResult parseModularInteger(OpAsmParser &parser, APInt &parsedInt,
                                Type &parsedType,
                                GetModulusCallback getModulusCallback) {
  auto parseResult = parseOptionalModularInteger(parser, parsedInt, parsedType,
                                                 getModulusCallback);
  if (parseResult.has_value()) {
    return parseResult.value();
  }
  return failure();
}

OptionalParseResult
parseOptionalModularInteger(OpAsmParser &parser, APInt &parsedInt,
                            Type &parsedType,
                            GetModulusCallback getModulusCallback) {
  if (!parser.parseOptionalInteger(parsedInt).has_value()) {
    return std::nullopt;
  }

  if (failed(parser.parseColonType(parsedType))) {
    return failure();
  }

  APInt modulus;
  if (failed(getModulusCallback(modulus))) {
    return failure();
  }

  return validateModularInteger(parser, modulus, parsedInt);
}

ParseResult parseModularIntegerList(OpAsmParser &parser,
                                    SmallVector<APInt> &parsedInts,
                                    Type &parsedType,
                                    GetModulusCallback getModulusCallback) {
  if (failed(parser.parseKeyword("dense")) || failed(parser.parseLess())) {
    return failure();
  }

  SmallVector<int64_t> parsedShape;
  auto parseTensor = [&](auto &&parseTensor, SmallVector<int64_t> &curShape,
                         int level = 0) -> ParseResult {
    int64_t dimCount = 0;
    auto checkpoint = parser.getCurrentLocation();
    do {
      APInt val;
      if (parser.parseOptionalInteger(val).has_value()) {
        parsedInts.push_back(std::move(val));
        ++dimCount;
      } else if (failed(parser.parseLSquare()) ||
                 parseTensor(parseTensor, curShape, level + 1)) {
        return failure();
      } else {
        ++dimCount;
      }
    } while (succeeded(parser.parseOptionalComma()));

    if (static_cast<int64_t>(curShape.size()) <= level)
      curShape.resize(level + 1);
    if (curShape[level] == 0)
      curShape[level] = dimCount;
    else if (curShape[level] != dimCount)
      return parser.emitError(checkpoint, "non-uniform tensor at dimension ")
             << level;
    return parser.parseRSquare();
  };

  bool isSplat =
      parser.parseOptionalInteger(parsedInts.emplace_back()).has_value();
  if (!isSplat) {
    parsedInts.pop_back();
    if (failed(parser.parseLSquare()) ||
        failed(parseTensor(parseTensor, parsedShape))) {
      return failure();
    }
  }

  if (failed(parser.parseGreater()) ||
      failed(parser.parseColonType(parsedType))) {
    return failure();
  }

  auto shapedType = dyn_cast<ShapedType>(parsedType);
  if (!shapedType) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected shaped type for parsed type");
  }
  if (shapedType.hasStaticShape()) {
    if (!isSplat) {
      ArrayRef<int64_t> expectedShape = shapedType.getShape();
      if (expectedShape.size() != parsedShape.size() ||
          !std::equal(expectedShape.begin(), expectedShape.end(),
                      parsedShape.begin())) {
        return parser.emitError(parser.getCurrentLocation(),
                                "tensor constant shape [")
               << llvm::make_range(parsedShape.begin(), parsedShape.end())
               << "] does not match type shape " << shapedType;
      }
    }
  } else {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected static shape for parsed type");
  }

  APInt modulus;
  if (failed(getModulusCallback(modulus))) {
    return failure();
  }

  for (APInt &parsedInt : parsedInts) {
    if (failed(validateModularInteger(parser, modulus, parsedInt))) {
      return failure();
    }
  }
  return success();
}

ParseResult parseModularOrExtendedModularInteger(
    OpAsmParser &parser, SmallVector<APInt> &parsedInts, Type &parsedType,
    GetModulusCallback getModulusCallback) {
  auto parseResult = parseOptionalModularOrExtendedModularInteger(
      parser, parsedInts, parsedType, getModulusCallback);
  if (parseResult.has_value()) {
    return parseResult.value();
  }
  return failure();
}

OptionalParseResult parseOptionalModularOrExtendedModularInteger(
    OpAsmParser &parser, SmallVector<APInt> &parsedInts, Type &parsedType,
    GetModulusCallback getModulusCallback) {
  assert(parsedInts.empty() && "parsedInts must be empty");
  if (failed(parser.parseOptionalLSquare())) {
    APInt val;
    auto res = parseOptionalModularInteger(parser, val, parsedType,
                                           getModulusCallback);
    if (res.has_value() && succeeded(*res)) {
      parsedInts.push_back(val);
    }
    return res;
  }

  if (failed(parser.parseCommaSeparatedList(
          [&]() { return parser.parseInteger(parsedInts.emplace_back()); })) ||
      failed(parser.parseRSquare())) {
    parsedInts.clear();
    return failure();
  }

  if (failed(parser.parseColonType(parsedType))) {
    parsedInts.clear();
    return failure();
  }

  APInt modulus;
  if (failed(getModulusCallback(modulus))) {
    parsedInts.clear();
    return failure();
  }

  for (APInt &parsedInt : parsedInts) {
    if (failed(validateModularInteger(parser, modulus, parsedInt))) {
      parsedInts.clear();
      return failure();
    }
  }
  return success();
}

} // namespace mlir::zkir
