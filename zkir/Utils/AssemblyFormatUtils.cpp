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

namespace {

ParseResult doParseModularInteger(OpAsmParser &parser, APInt &parsedInt,
                                  Type &parsedType,
                                  GetModulusCallback getModulusCallback) {
  if (failed(parser.parseColonType(parsedType))) {
    return failure();
  }

  APInt modulus;
  if (failed(getModulusCallback(modulus))) {
    return failure();
  }

  return validateModularInteger(parser, modulus, parsedInt);
}

} // namespace

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
  if (!parser.parseOptionalInteger(parsedInt).has_value()) {
    return failure();
  }
  return doParseModularInteger(parser, parsedInt, parsedType,
                               getModulusCallback);
}

OptionalParseResult
parseOptionalModularInteger(OpAsmParser &parser, APInt &parsedInt,
                            Type &parsedType,
                            GetModulusCallback getModulusCallback) {
  if (!parser.parseOptionalInteger(parsedInt).has_value()) {
    return std::nullopt;
  }
  return doParseModularInteger(parser, parsedInt, parsedType,
                               getModulusCallback);
}

} // namespace mlir::zkir
