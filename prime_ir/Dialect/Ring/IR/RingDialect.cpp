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

#include "prime_ir/Dialect/Ring/IR/RingDialect.h"

#include <cstdint>
#include <optional>

#include "llvm/ADT/STLExtras.h" // IWYU pragma: keep (interleaveComma)
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h" // IWYU pragma: keep
#include "mlir/IR/Builders.h"    // IWYU pragma: keep
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h" // IWYU pragma: keep (AsmParser, FieldParser)
#include "prime_ir/Dialect/Ring/IR/RingOps.h"
#include "prime_ir/Dialect/Ring/IR/RingTypes.h"

// Generated definitions
#include "prime_ir/Dialect/Ring/IR/RingDialect.cpp.inc"
#include "prime_ir/Dialect/Ring/IR/RingEnums.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "prime_ir/Dialect/Ring/IR/RingTypes.cpp.inc"

#define GET_OP_CLASSES
#include "prime_ir/Dialect/Ring/IR/RingOps.cpp.inc"

namespace mlir::prime_ir::ring {

LogicalResult RqType::verify(function_ref<InFlightDiagnostic()> emitError,
                             DenseI64ArrayAttr moduli, IntegerAttr ringDegree,
                             IntegerType storageType, Domain domain) {
  if (moduli.empty()) {
    return emitError() << "ring.rq must have at least one modulus";
  }
  if (!storageType) {
    return emitError() << "ring.rq residue storage type must be provided";
  }
  llvm::SmallSet<int64_t, 8> seen;
  for (int64_t q : moduli.asArrayRef()) {
    if (q <= 1) {
      return emitError() << "ring.rq modulus must be > 1, got " << q;
    }
    if (!seen.insert(q).second) {
      return emitError() << "ring.rq moduli must be distinct (coprime RNS "
                            "basis), got repeated "
                         << q;
    }
    // Residues live in [0, q), so the word has to hold q - 1 unsigned.
    if (APInt(64, q - 1).getActiveBits() > storageType.getWidth()) {
      return emitError() << "ring.rq modulus " << q << " does not fit in i"
                         << storageType.getWidth() << " storage";
    }
  }
  if (!ringDegree) {
    return emitError() << "ring.rq degree N must be provided";
  }
  int64_t n = ringDegree.getValue().getSExtValue();
  if (n <= 0 || (n & (n - 1)) != 0) {
    return emitError() << "ring.rq degree N must be a positive power of two, "
                          "got "
                       << n;
  }
  // The evaluation basis is the image of the CRT map, which exists only when
  // X^N+1 splits into linear factors over every Z_q_i — i.e. when each q_i
  // admits a 2N-th root of unity. A coefficient-basis value needs no such root.
  if (domain == Domain::Eval) {
    for (int64_t q : moduli.asArrayRef()) {
      if ((q - 1) % (2 * n) != 0) {
        return emitError()
               << "ring.rq eval basis needs X^N+1 to split over each modulus, "
                  "but 2N = "
               << (2 * n) << " does not divide " << q << " - 1";
      }
    }
  }
  return success();
}

// Format: !ring.rq<[q0, q1, ...], N : iW> with an optional storage type and an
// optional `coeff|eval` basis after it (absent means i64 and coeff). The two
// tails are told apart by whether the token parses as a type.
Type RqType::parse(AsmParser &parser) {
  llvm::SmallVector<int64_t> moduli;
  if (parser.parseLess() ||
      parser.parseCommaSeparatedList(AsmParser::Delimiter::Square,
                                     [&]() {
                                       int64_t v;
                                       if (parser.parseInteger(v)) {
                                         return failure();
                                       }
                                       moduli.push_back(v);
                                       return success();
                                     }) ||
      parser.parseComma()) {
    return {};
  }
  IntegerAttr ringDegree;
  if (parser.parseAttribute(ringDegree)) {
    return {};
  }
  auto storageType = IntegerType::get(parser.getContext(), 64);
  Domain domain = Domain::Coeff;
  bool haveStorage = false;
  while (succeeded(parser.parseOptionalComma())) {
    Type parsedType;
    OptionalParseResult typeResult = parser.parseOptionalType(parsedType);
    if (typeResult.has_value()) {
      if (failed(*typeResult)) {
        return {};
      }
      auto intType = llvm::dyn_cast<IntegerType>(parsedType);
      if (!intType || haveStorage) {
        parser.emitError(parser.getNameLoc())
            << "expected a single integer residue storage type, got "
            << parsedType;
        return {};
      }
      storageType = intType;
      haveStorage = true;
      continue;
    }
    llvm::StringRef keyword;
    if (parser.parseKeyword(&keyword)) {
      return {};
    }
    std::optional<Domain> parsed = symbolizeDomain(keyword);
    if (!parsed) {
      parser.emitError(parser.getNameLoc())
          << "expected 'coeff' or 'eval' basis, got '" << keyword << "'";
      return {};
    }
    domain = *parsed;
    break;
  }
  if (parser.parseGreater()) {
    return {};
  }
  return getChecked([&] { return parser.emitError(parser.getNameLoc()); },
                    parser.getContext(),
                    DenseI64ArrayAttr::get(parser.getContext(), moduli),
                    ringDegree, storageType, domain);
}

void RqType::print(AsmPrinter &printer) const {
  printer << "<[";
  llvm::interleaveComma(getModuli().asArrayRef(), printer.getStream());
  printer << "], " << getRingDegree();
  if (getStorageWidth() != 64) {
    printer << ", " << getStorageType();
  }
  if (getDomain() != Domain::Coeff) {
    printer << ", " << stringifyDomain(getDomain());
  }
  printer << ">";
}

void RingDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "prime_ir/Dialect/Ring/IR/RingTypes.cpp.inc" // NOLINT(build/include)
      >();
  addOperations<
#define GET_OP_LIST
#include "prime_ir/Dialect/Ring/IR/RingOps.cpp.inc" // NOLINT(build/include)
      >();
}

} // namespace mlir::prime_ir::ring
