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

#ifndef ZKIR_UTILS_ASSEMBLYFORMATUTILS_H_
#define ZKIR_UTILS_ASSEMBLYFORMATUTILS_H_

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/bit.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::zkir {

template <typename T>
Type parseModulus(AsmParser &parser) {
  if (failed(parser.parseLess())) {
    return nullptr;
  }
  APInt modulus;
  if (failed(parser.parseInteger(modulus))) {
    return nullptr;
  }
  // NOTE(chokobole): **DO NOT** check `!modulus.isPositive()` here.
  // The 'modulus' is treated as an **unsigned integer** .
  // Interpreting the most significant bit (MSB) as a sign bit (as
  // `isPositive()` would implicitly do) could incorrectly classify a large
  // positive unsigned modulus as a negative number.
  if (modulus.isZero()) {
    parser.emitError(parser.getCurrentLocation(), "modulus cannot be zero");
    return nullptr;
  }
  if (modulus.isPowerOf2()) {
    // TODO(chokobole): This is because the present PrimeField logic cannot
    // correctly handle the storage bit width requirements for Binary Fields.
    //
    // The intended logic for Binary Fields is different: the storage bit width
    // should typically be (modulus_bit_width - 1), whereas PrimeField currently
    // assumes the storage width equals the modulus bit width.
    parser.emitError(parser.getCurrentLocation(),
                     "modulus must not be a power of 2");
    return nullptr;
  }
  Type storageType;
  if (succeeded(parser.parseOptionalColon())) {
    if (failed(parser.parseType(storageType))) {
      return nullptr;
    }

    if (auto integerType = dyn_cast<IntegerType>(storageType)) {
      if (integerType.getWidth() < modulus.getActiveBits()) {
        parser.emitError(
            parser.getCurrentLocation(),
            "storage type must be at least as wide as the modulus");
        return nullptr;
      }
      modulus = modulus.zextOrTrunc(integerType.getWidth());
    } else {
      parser.emitError(parser.getCurrentLocation(),
                       "expected integer type for storage type");
      return nullptr;
    }
  } else {
    auto storageBitWidth = llvm::bit_ceil(modulus.getActiveBits());
    modulus = modulus.zextOrTrunc(storageBitWidth);
    storageType = IntegerType::get(parser.getContext(), storageBitWidth);
  }
  IntegerAttr modulusAttr = IntegerAttr::get(storageType, modulus);

  bool isMontgomery = false;
  if (succeeded(parser.parseOptionalComma())) {
    if (succeeded(parser.parseKeyword("true"))) {
      isMontgomery = true;
    } else if (succeeded(parser.parseKeyword("false"))) {
      isMontgomery = false;
    } else {
      parser.emitError(parser.getCurrentLocation(),
                       "expected 'true' or 'false' for isMontgomery");
      return nullptr;
    }
  }
  if (failed(parser.parseGreater())) {
    return nullptr;
  }
  return T::get(parser.getContext(), modulusAttr, isMontgomery);
}

void printModulus(AsmPrinter &printer, const APInt &modulus,
                  const Type &storageType, bool isMontgomery);

using GetModulusCallback = llvm::function_ref<ParseResult(APInt &)>;

// Attempts to validate a parsed integer value against the constraints of a
// **modular integer** defined by the 'modulus'.
//
// 1. **Size Check:** Ensures the bit width of the parsed value ('parsedInt')
//    does not exceed the bit width of the modulus/underlying storage type.
// 2. **Size Adjustment:** Adjusts the parsed integer's size (zero-extend or
//    truncate) to match the exact bit width of the modulus.
// 3. **Range Check:** Verifies that the adjusted value is strictly less than
//    the modulus (i.e., 0 <= value < modulus). Fails if the value is outside
//    this valid modular range.
ParseResult validateModularInteger(OpAsmParser &parser, const APInt &modulus,
                                   APInt &parsedInt);

ParseResult parseModularInteger(OpAsmParser &parser, APInt &parsedInt,
                                Type &parsedType,
                                GetModulusCallback getModulusCallback);
OptionalParseResult
parseOptionalModularInteger(OpAsmParser &parser, APInt &parsedInt,
                            Type &parsedType,
                            GetModulusCallback getModulusCallback);

ParseResult parseModularIntegerList(OpAsmParser &parser,
                                    SmallVector<APInt> &parsedInts,
                                    Type &parsedType,
                                    GetModulusCallback getModulusCallback);

ParseResult parseModularOrExtendedModularInteger(
    OpAsmParser &parser, SmallVector<APInt> &parsedInts, Type &parsedType,
    GetModulusCallback getModulusCallback);

OptionalParseResult parseOptionalModularOrExtendedModularInteger(
    OpAsmParser &parser, SmallVector<APInt> &parsedInts, Type &parsedType,
    GetModulusCallback getModulusCallback);

} // namespace mlir::zkir

#endif // ZKIR_UTILS_ASSEMBLYFORMATUTILS_H_
