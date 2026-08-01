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

#include "prime_ir/Dialect/Field/IR/FieldTypes.h"

#include <algorithm>
#include <cstring>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/bit.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "prime_ir/Dialect/Field/IR/FieldOperation.h"
#include "prime_ir/Dialect/Field/IR/FieldOps.h"
#include "prime_ir/Dialect/Field/IR/TowerFieldConfig.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithAttributes.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithOps.h"
#include "prime_ir/IR/DenseElementBytes.h"
#include "prime_ir/Utils/AssemblyFormatUtils.h"

namespace mlir::prime_ir::field {

#include "prime_ir/Dialect/Field/IR/FieldTypesInterfaces.cpp.inc"

bool isMontgomery(Type type) {
  Type elementType = getElementTypeOrSelf(type);
  if (auto fieldType = dyn_cast<FieldTypeInterface>(elementType)) {
    return fieldType.isMontgomery();
  }
  return false;
}

unsigned getIntOrPrimeFieldBitWidth(Type type) {
  assert((llvm::isa<PrimeFieldType, IntegerType>(type)));
  if (auto pfType = dyn_cast<PrimeFieldType>(type)) {
    return pfType.getTypeSizeInBits();
  }
  return cast<IntegerType>(type).getWidth();
}

mod_arith::ModArithType convertPrimeFieldType(PrimeFieldType type) {
  IntegerAttr modulus = type.getModulus();
  bool isMontgomery = type.isMontgomery();
  return mod_arith::ModArithType::get(type.getContext(), modulus, isMontgomery);
}

ParseResult parseColonFieldType(AsmParser &parser, Type &type) {
  if (failed(parser.parseColonType(type)))
    return failure();

  if (isa<PrimeFieldType, BinaryFieldType, ExtensionFieldType>(type)) {
    return success();
  }
  return parser.emitError(
      parser.getCurrentLocation(),
      "expected prime field, binary field, or extension field type");
}

ParseResult validateAttribute(AsmParser &parser, SMLoc loc, Type type,
                              Attribute attr, std::string_view attrName) {
  // Callers feed the validated attribute straight into field arithmetic, which
  // compares APInts against the modulus and requires equal bit widths — a
  // narrower or wider literal aborts rather than diagnosing. So the storage
  // type has to be checked here, not just the attribute kind.
  if (auto pfType = dyn_cast<PrimeFieldType>(type)) {
    auto intAttr = dyn_cast<IntegerAttr>(attr);
    if (!intAttr) {
      return parser.emitError(loc, "expected integer attribute for " +
                                       std::string(attrName));
    }
    if (intAttr.getType() != pfType.getStorageType()) {
      return parser.emitError(loc)
             << "expected " << pfType.getStorageType() << " attribute for "
             << attrName << ", but got " << intAttr.getType();
    }
    return success();
  }

  auto denseAttr = dyn_cast<DenseIntElementsAttr>(attr);
  if (!denseAttr) {
    return parser.emitError(loc, "expected dense int elements attribute for " +
                                     std::string(attrName));
  }
  if (auto efType = dyn_cast<ExtensionFieldType>(type)) {
    Type storageType = efType.getBasePrimeField().getStorageType();
    SmallVector<int64_t> expectedShape = efType.getAttrShape();
    ShapedType attrType = denseAttr.getType();
    if (attrType.getElementType() != storageType ||
        attrType.getShape() != ArrayRef<int64_t>(expectedShape)) {
      return parser.emitError(loc)
             << "expected " << RankedTensorType::get(expectedShape, storageType)
             << " attribute for " << attrName << ", but got " << attrType;
    }
  }
  return success();
}

Attribute maybeToMontgomery(Type type, Attribute attr) {
  IntegerAttr modulus;
  bool isMont = false;

  if (auto pfType = dyn_cast<field::PrimeFieldType>(type)) {
    modulus = pfType.getModulus();
    isMont = pfType.isMontgomery();
  } else if (auto efType = dyn_cast<field::ExtensionFieldType>(type)) {
    modulus = efType.getBasePrimeField().getModulus();
    isMont = efType.isMontgomery();
  }

  if (!isMont)
    return attr;

  if (auto intAttr = dyn_cast<IntegerAttr>(attr))
    return mod_arith::getAttrAsMontgomeryForm(modulus, intAttr);
  return mod_arith::getAttrAsMontgomeryForm(modulus,
                                            cast<DenseElementsAttr>(attr));
}

Value createFieldConstant(Type fieldType, ImplicitLocOpBuilder &builder,
                          uint64_t value) {
  auto constantLike = cast<ConstantLikeInterface>(fieldType);
  TypedAttr attr = constantLike.createConstantAttr(static_cast<int64_t>(value));
  return ConstantOp::create(builder, fieldType, attr)->getResult(0);
}

Attribute maybeToStandard(Type type, Attribute attr) {
  IntegerAttr modulus;
  bool isMont = false;

  if (auto pfType = dyn_cast<field::PrimeFieldType>(type)) {
    modulus = pfType.getModulus();
    isMont = pfType.isMontgomery();
  } else if (auto efType = dyn_cast<field::ExtensionFieldType>(type)) {
    modulus = efType.getBasePrimeField().getModulus();
    isMont = efType.isMontgomery();
  }

  if (!isMont)
    return attr;

  if (auto intAttr = dyn_cast<IntegerAttr>(attr))
    return mod_arith::getAttrAsStandardForm(modulus, intAttr);
  return mod_arith::getAttrAsStandardForm(modulus,
                                          cast<DenseElementsAttr>(attr));
}

//===----------------------------------------------------------------------===//
// PrimeFieldType
//===----------------------------------------------------------------------===//

// static
Type PrimeFieldType::parse(AsmParser &parser) {
  return parseModulus<PrimeFieldType>(parser);
}

void PrimeFieldType::print(AsmPrinter &printer) const {
  printModulus(printer, getModulus().getValue(), getStorageType(),
               isMontgomery());
}

llvm::TypeSize PrimeFieldType::getTypeSizeInBits(
    DataLayout const &, llvm::ArrayRef<DataLayoutEntryInterface>) const {
  return llvm::TypeSize::getFixed(getTypeSizeInBits());
}

uint64_t PrimeFieldType::getABIAlignment(
    DataLayout const &dataLayout,
    llvm::ArrayRef<DataLayoutEntryInterface>) const {
  return dataLayout.getTypeABIAlignment(getStorageType());
}

bool PrimeFieldType::isMontgomery() const { return getIsMontgomery(); }

SmallVector<int64_t> PrimeFieldType::getAttrShape() const { return {}; }

unsigned PrimeFieldType::getDegreeOverPrime() const { return 1; }

TypedAttr PrimeFieldType::createConstantAttr(int64_t c) const {
  PrimeFieldOperation pfOp(c, *this);
  return pfOp.getIntegerAttr();
}

TypedAttr
PrimeFieldType::createConstantAttrFromValues(ArrayRef<APInt> values) const {
  assert(values.size() == 1);
  PrimeFieldOperation pfOp(values[0], *this);
  return pfOp.getIntegerAttr();
}

ShapedType PrimeFieldType::overrideShapedType(ShapedType type) const {
  return type.clone(getStorageType());
}

//===----------------------------------------------------------------------===//
// BinaryFieldType
//===----------------------------------------------------------------------===//

// GF(2)[y] helpers for the flat-modulus irreducibility check. Operands are
// < 2^n with n <= 64, so products fit APInt(2n) and reduced values fit
// uint64_t.
static llvm::APInt polyClmul(uint64_t a, uint64_t b, unsigned n) {
  llvm::APInt r(2 * n, 0);
  llvm::APInt wide(2 * n, a);
  for (unsigned i = 0; i < n; ++i) {
    if (b >> i & 1)
      r ^= wide.shl(i);
  }
  return r;
}

static uint64_t polyReduce(llvm::APInt p, uint64_t fLow, unsigned n) {
  llvm::APInt low(p.getBitWidth(), fLow);
  for (int i = static_cast<int>(p.getActiveBits()) - 1;
       i >= static_cast<int>(n); i = static_cast<int>(p.getActiveBits()) - 1) {
    p.clearBit(i);
    p ^= low.shl(i - n);
  }
  return p.extractBitsAsZExtValue(std::min(64u, p.getBitWidth()), 0);
}

// Rabin for power-of-two degree n: f = y^n + fLow is irreducible over GF(2)
// iff y^(2^n) == y (mod f) and gcd(y^(2^(n/2)) + y, f) == 1 (2 is the only
// prime dividing n).
static bool isIrreduciblePoly(uint64_t fLow, unsigned n) {
  if (n == 1)
    return fLow == 1; // y + 1
  auto modmul = [&](uint64_t a, uint64_t b) {
    return polyReduce(polyClmul(a, b, n), fLow, n);
  };
  auto yPow2Exp = [&](unsigned e) {
    uint64_t r = 2;
    for (unsigned i = 0; i < e; ++i)
      r = modmul(r, r);
    return r;
  };
  if (yPow2Exp(n) != 2)
    return false;
  // Euclid on GF(2)[y]: gcd(y^(2^(n/2)) + y, f).
  llvm::APInt a(n + 1, yPow2Exp(n / 2) ^ 2);
  llvm::APInt b(n + 1, fLow);
  b.setBit(n);
  while (!a.isZero()) {
    while (!a.isZero() && a.getActiveBits() >= b.getActiveBits())
      a ^= b.shl(a.getActiveBits() - b.getActiveBits());
    std::swap(a, b);
    if (b.getActiveBits() <= 1)
      break;
  }
  return b.getActiveBits() == 1;
}

// static
LogicalResult
BinaryFieldType::verify(function_ref<InFlightDiagnostic()> emitError,
                        unsigned towerLevel, bool isFlat, uint64_t flatModLow) {
  if (towerLevel > kMaxTowerLevel) {
    return emitError() << "binary field tower level must be between 0 and "
                       << kMaxTowerLevel << ", got " << towerLevel;
  }
  if (!isFlat) {
    if (flatModLow != 0) {
      return emitError() << "the tower basis carries no flat modulus";
    }
    return success();
  }
  if (towerLevel == 0) {
    return emitError() << "GF(2) has a single basis; a flat basis exists "
                          "only at tower levels 1-"
                       << kMaxTowerLevel;
  }
  const unsigned n = 1u << towerLevel;
  // Level 7 admits only the canonical GHASH modulus: a wider low part needs
  // a different reduction algorithm than the two-fold clmad schedule.
  if (towerLevel == 7 && flatModLow != kCanonicalFlatModLow[7]) {
    return emitError() << "level-7 flat modulus must be the canonical GHASH "
                          "polynomial (low part 0x87)";
  }
  if ((flatModLow & 1) == 0) {
    return emitError() << "flat modulus must have a constant term (bit 0), "
                          "got low part 0x"
                       << llvm::utohexstr(flatModLow);
  }
  const unsigned deg =
      flatModLow == 0 ? 0 : 64 - llvm::countl_zero(flatModLow) - 1;
  if (2 * deg > n) {
    return emitError() << "flat modulus low part must satisfy 2*deg <= " << n
                       << " so the two-fold reduction converges; 0x"
                       << llvm::utohexstr(flatModLow) << " has degree " << deg;
  }
  if (towerLevel <= 6 && !isIrreduciblePoly(flatModLow, n)) {
    return emitError() << "flat modulus y^" << n << " + 0x"
                       << llvm::utohexstr(flatModLow)
                       << " is reducible over GF(2)";
  }
  return success();
}

// static
Type BinaryFieldType::parse(AsmParser &parser) {
  if (failed(parser.parseLess())) {
    return nullptr;
  }

  unsigned towerLevel;
  if (failed(parser.parseInteger(towerLevel))) {
    return nullptr;
  }

  if (towerLevel > kMaxTowerLevel) {
    parser.emitError(parser.getCurrentLocation(),
                     "binary field tower level must be between 0 and ")
        << kMaxTowerLevel;
    return nullptr;
  }

  // Optional basis selector: `flat` = the canonical modulus of the level,
  // `ghash`/`aes` = canonical sugar at levels 7/3, `poly<f>` = an explicit
  // modulus given as the full polynomial bitmask (bit i = coeff of y^i). A
  // canonical modulus spelled via poly<> uniques to the same type and prints
  // back under its canonical name.
  bool isFlat = false;
  uint64_t flatModLow = 0;
  if (succeeded(parser.parseOptionalComma())) {
    StringRef basis;
    llvm::SMLoc basisLoc = parser.getCurrentLocation();
    if (failed(parser.parseKeyword(&basis))) {
      return nullptr;
    }
    if (basis == "ghash") {
      if (towerLevel != kMaxTowerLevel) {
        parser.emitError(basisLoc,
                         "the ghash basis is GF(2¹²⁸), only valid at tower "
                         "level ")
            << kMaxTowerLevel;
        return nullptr;
      }
      flatModLow = kCanonicalFlatModLow[towerLevel];
    } else if (basis == "aes") {
      if (towerLevel != kAesTowerLevel) {
        parser.emitError(basisLoc,
                         "the aes basis is GF(2⁸), only valid at tower level ")
            << kAesTowerLevel;
        return nullptr;
      }
      flatModLow = kCanonicalFlatModLow[towerLevel];
    } else if (basis == "flat") {
      if (towerLevel == 0) {
        parser.emitError(basisLoc, "GF(2) has a single basis");
        return nullptr;
      }
      flatModLow = kCanonicalFlatModLow[towerLevel];
    } else if (basis == "poly") {
      llvm::APInt modulus;
      if (failed(parser.parseLess())) {
        return nullptr;
      }
      OptionalParseResult parsedInt = parser.parseOptionalInteger(modulus);
      if (!parsedInt.has_value() || failed(*parsedInt) ||
          failed(parser.parseGreater())) {
        parser.emitError(basisLoc, "expected poly<modulus-bitmask>");
        return nullptr;
      }
      const unsigned n = 1u << towerLevel;
      if (modulus.getActiveBits() != n + 1) {
        parser.emitError(basisLoc, "flat modulus must have degree exactly ")
            << n << " (leading bit " << n << " set, none above)";
        return nullptr;
      }
      flatModLow = modulus.extractBitsAsZExtValue(
                       std::min(64u, modulus.getBitWidth()), 0) &
                   (n >= 64 ? ~uint64_t{0} : (uint64_t{1} << n) - 1);
    } else {
      parser.emitError(basisLoc,
                       "expected 'ghash', 'aes', 'flat', or 'poly<...>', "
                       "got '")
          << basis << "'";
      return nullptr;
    }
    isFlat = true;
  }

  if (failed(parser.parseGreater())) {
    return nullptr;
  }

  return BinaryFieldType::getChecked(
      [&] {
        return parser.emitError(parser.getNameLoc(), "invalid binary field: ");
      },
      parser.getContext(), towerLevel, isFlat, flatModLow);
}

void BinaryFieldType::print(AsmPrinter &printer) const {
  printer << "<" << getTowerLevel();
  if (isGhash()) {
    printer << ", ghash";
  } else if (isAes()) {
    printer << ", aes";
  } else if (isCanonicalFlat()) {
    printer << ", flat";
  } else if (getIsFlat()) {
    const unsigned n = getBitWidth();
    llvm::APInt modulus(n + 1, getFlatModLow());
    modulus.setBit(n);
    llvm::SmallString<40> hex;
    modulus.toString(hex, /*Radix=*/16, /*Signed=*/false);
    printer << ", poly<0x" << hex << ">";
  }
  printer << ">";
}

// FieldTypeInterface. A binary field is a leaf in prime-ir's field-type
// algebra: its GF(2^(2^level)) tower is internal to BinaryFieldToArith, not a
// prime-field coefficient decomposition. So, like a prime field, attr-shape is
// empty and degree over the prime field is 1 (keeping the invariant
// getDegreeOverPrime == product(getAttrShape)). Declaring the interface is what
// lets passes gated on it (e.g. stablehlo's ConvertFieldMul) treat a binary
// multiply as a field op instead of skipping it.
bool BinaryFieldType::isMontgomery() const { return false; }

SmallVector<int64_t> BinaryFieldType::getAttrShape() const { return {}; }

unsigned BinaryFieldType::getDegreeOverPrime() const { return 1; }

llvm::TypeSize BinaryFieldType::getTypeSizeInBits(
    DataLayout const &, llvm::ArrayRef<DataLayoutEntryInterface>) const {
  return llvm::TypeSize::getFixed(getTypeSizeInBits());
}

uint64_t BinaryFieldType::getABIAlignment(
    DataLayout const &dataLayout,
    llvm::ArrayRef<DataLayoutEntryInterface>) const {
  return dataLayout.getTypeABIAlignment(getStorageType());
}

TypedAttr BinaryFieldType::createConstantAttr(int64_t c) const {
  APInt value(getBitWidth(), static_cast<uint64_t>(c));
  // Mask to valid range
  value = value.zextOrTrunc(getBitWidth());
  return IntegerAttr::get(getStorageType(), value);
}

TypedAttr
BinaryFieldType::createConstantAttrFromValues(ArrayRef<APInt> values) const {
  assert(values.size() == 1);
  APInt value = values[0].zextOrTrunc(getBitWidth());
  return IntegerAttr::get(getStorageType(), value);
}

ShapedType BinaryFieldType::overrideShapedType(ShapedType type) const {
  return type.clone(getStorageType());
}

//===----------------------------------------------------------------------===//
// ExtensionFieldType utilities
//===----------------------------------------------------------------------===//

namespace ext_field_utils {
namespace {

// Create constant attribute using tower-aware dispatch.
// Uses getFlatDenseIntElementsAttr() which works for both tower and non-tower.
TypedAttr createConstantAttrImpl(ArrayRef<APInt> coeffs,
                                 ExtensionFieldType efType) {
  assert(coeffs.size() == efType.getDegreeOverPrime());
  SmallVector<APInt> coeffsVec(coeffs.begin(), coeffs.end());
  auto sig = getTowerSignature(efType);
#define CREATE_CONSTANT_ATTR(unused_sig, TypeName)                             \
  auto efOp = TypeName::fromUnchecked(coeffsVec, efType);                      \
  return efOp.getFlatDenseIntElementsAttr();
  DISPATCH_TOWER_BY_SIGNATURE(sig, CREATE_CONSTANT_ATTR,
                              ExtensionFieldOperation, Op)
#undef CREATE_CONSTANT_ATTR
}

template <unsigned kDegreeOverBase>
Value buildStructFromCoeffs(ImplicitLocOpBuilder &builder, Type structType,
                            llvm::ArrayRef<Value> coeffs) {
  return prime_ir::SimpleStructBuilder<kDegreeOverBase>::initialized(
      builder, builder.getLoc(), structType, coeffs);
}

template <unsigned kDegreeOverBase>
llvm::SmallVector<Value> extractCoeffsFromStruct(ImplicitLocOpBuilder &builder,
                                                 Value structValue) {
  prime_ir::SimpleStructBuilder<kDegreeOverBase> extFieldStruct(structValue);
  return extFieldStruct.getValues(builder, builder.getLoc());
}

// Compile-time dispatch helpers for ExtensionFieldType
template <unsigned... Degrees>
Value dispatchBuildStructFromCoeffs(
    unsigned degree, ImplicitLocOpBuilder &builder, Type structType,
    ArrayRef<Value> coeffs, std::integer_sequence<unsigned, Degrees...>) {
  Value result;
  (void)((Degrees == degree ? (result = buildStructFromCoeffs<Degrees>(
                                   builder, structType, coeffs),
                               true)
                            : false) ||
         ...);
  assert(result && "unsupported extension field degree");
  return result;
}

template <unsigned... Degrees>
SmallVector<Value>
dispatchExtractCoeffsFromStruct(unsigned degree, ImplicitLocOpBuilder &builder,
                                Value structValue,
                                std::integer_sequence<unsigned, Degrees...>) {
  SmallVector<Value> result;
  (void)((Degrees == degree ? (result = extractCoeffsFromStruct<Degrees>(
                                   builder, structValue),
                               true)
                            : false) ||
         ...);
  assert(!result.empty() && "unsupported extension field degree");
  return result;
}

// Generate sequence from kMinExtDegree to kMaxExtDegree (2, 3, 4)
template <size_t Start, size_t... Is>
constexpr auto makeExtDegreeSequence(std::index_sequence<Is...>) {
  return std::integer_sequence<unsigned,
                               static_cast<unsigned>(Start + Is)...>{};
}

constexpr auto kExtDegreeSequence = makeExtDegreeSequence<kMinExtDegree>(
    std::make_index_sequence<kNumExtDegrees>{});

} // namespace
} // namespace ext_field_utils

//===----------------------------------------------------------------------===//
// ExtensionFieldType
//===----------------------------------------------------------------------===//

LogicalResult
ExtensionFieldType::verify(function_ref<InFlightDiagnostic()> emitError,
                           unsigned degree, Type baseField,
                           Attribute nonResidue) {
  if (degree < 2 || degree > kMaxExtDegree) {
    return emitError() << "extension field degree must be between 2 and "
                       << kMaxExtDegree << ", got " << degree;
  }

  // Base field must be either a prime field or an extension field
  if (!isa<PrimeFieldType, ExtensionFieldType>(baseField)) {
    return emitError() << "base field must be a prime field or extension field";
  }

  // For tower extensions, validate non-residue using extension field
  // arithmetic. The non-residue must satisfy: nr^((q - 1) / n) ≢ 1 in the base
  // field, where q = p^(base degree over prime) is the order of the base
  // extension field.
  if (isa<ExtensionFieldType>(baseField)) {
    auto baseEfType = cast<ExtensionFieldType>(baseField);
    PrimeFieldType pfType = baseEfType.getBasePrimeField();
    APInt p = pfType.getModulus().getValue();
    unsigned baseDegreeOverPrime = baseEfType.getDegreeOverPrime();

    // Compute q = p^baseDegreeOverPrime (order of base extension field)
    // Use extended precision to avoid overflow
    unsigned resultBitWidth = p.getBitWidth() * baseDegreeOverPrime + 1;
    APInt q(resultBitWidth, 1);
    APInt pExt = p.zext(resultBitWidth);
    for (unsigned i = 0; i < baseDegreeOverPrime; ++i) {
      q *= pExt;
    }

    // exp = (q - 1) / degree
    APInt exp = (q - 1).udiv(APInt(resultBitWidth, degree));

    // Dispatch to the correct tower type and check nr^exp != 1
    auto sig = getTowerSignature(baseEfType);

    // Handle scalar non-residue: embed as [nr, 0, 0, ...] in base extension
    // field
    if (auto intAttr = dyn_cast<IntegerAttr>(nonResidue)) {
#define CHECK_SCALAR_NON_RESIDUE(unused_sig, TypeName)                         \
  auto nrOp = TypeName::fromUnchecked(intAttr.getValue(), baseEfType);         \
  if (nrOp.power(exp).isOne()) {                                               \
    return emitError() << "nonResidue must satisfy nonResidue^((q - 1) / "     \
                       << degree << ") != 1 in base extension field";          \
  }
      DISPATCH_TOWER_BY_SIGNATURE(sig, CHECK_SCALAR_NON_RESIDUE,
                                  ExtensionFieldOperation, Op)
#undef CHECK_SCALAR_NON_RESIDUE
      return success();
    }

    // Handle full extension field element non-residue
    auto denseAttr = cast<DenseIntElementsAttr>(nonResidue);

#define CHECK_NON_RESIDUE(unused_sig, TypeName)                                \
  auto nrOp = TypeName::fromUnchecked(denseAttr, baseEfType);                  \
  if (nrOp.power(exp).isOne()) {                                               \
    return emitError() << "nonResidue must satisfy nonResidue^((q - 1) / "     \
                       << degree << ") != 1 in base extension field";          \
  }
    DISPATCH_TOWER_BY_SIGNATURE(sig, CHECK_NON_RESIDUE, ExtensionFieldOperation,
                                Op)
#undef CHECK_NON_RESIDUE

    return success();
  }

  // A DenseIntElementsAttr on a prime base encodes a general monic modulus
  // uᴺ ≡ Σⱼ mⱼ·uʲ (e.g. pil2-stark's x³ - x - 1 as [1, 1, 0]) rather than a
  // binomial non-residue. The binomial power test below does not apply, and
  // irreducibility of a general modulus has no comparably cheap check — it is
  // the registrant's responsibility (the zk_dtypes dtype this mirrors is
  // golden-tested against its reference implementation).
  if (auto denseAttr = dyn_cast<DenseIntElementsAttr>(nonResidue)) {
    if (denseAttr.getNumElements() != static_cast<int64_t>(degree)) {
      return emitError() << "modulus low-coefficient count ("
                         << denseAttr.getNumElements()
                         << ") must equal the extension degree (" << degree
                         << ")";
    }
    return success();
  }

  // For direct extensions over prime fields, validate that nonResidue is
  // actually a non-residue: nonResidue^((p - 1) / n) ≢ 1 (mod p)
  auto pfType = cast<PrimeFieldType>(baseField);
  auto nrOp =
      PrimeFieldOperation::fromUnchecked(cast<IntegerAttr>(nonResidue), pfType);
  APInt p = pfType.getModulus().getValue();
  APInt exp = (p - 1).udiv(APInt(p.getBitWidth(), degree));
  if (nrOp.power(exp).isOne()) {
    return emitError() << "nonResidue must satisfy nonResidue^((p - 1) / "
                       << degree << ") != 1 (mod p)";
  }

  return success();
}

// static
Type ExtensionFieldType::parse(AsmParser &parser) {
  if (failed(parser.parseLess())) {
    return nullptr;
  }

  // Parse "Nx" format (e.g., "2x", "3x", "4x")
  unsigned degree;
  if (failed(parser.parseInteger(degree))) {
    return nullptr;
  }

  // Validate degree early
  if (degree < 2 || degree > kMaxExtDegree) {
    parser.emitError(parser.getCurrentLocation(),
                     "extension field degree must be between 2 and ")
        << kMaxExtDegree;
    return nullptr;
  }

  if (failed(parser.parseKeyword("x"))) {
    return nullptr;
  }

  // Parse base field type (can be prime or extension field)
  Type baseFieldType;
  if (failed(parser.parseType(baseFieldType))) {
    return nullptr;
  }
  if (!isa<PrimeFieldType, ExtensionFieldType>(baseFieldType)) {
    parser.emitError(parser.getCurrentLocation(),
                     "base field must be a prime field or extension field");
    return nullptr;
  }

  // Parse non-residue: a scalar IntegerAttr (binomial uᴺ - ξ), or on a prime
  // base a DenseIntElementsAttr of the general monic modulus's low
  // coefficients (uᴺ ≡ Σⱼ mⱼ·uʲ).
  if (failed(parser.parseComma())) {
    return nullptr;
  }
  Attribute nonResidueAttr;
  if (failed(parser.parseAttribute(nonResidueAttr))) {
    return nullptr;
  }
  if (auto denseAttr = dyn_cast<DenseIntElementsAttr>(nonResidueAttr)) {
    if (auto pfType = dyn_cast<PrimeFieldType>(baseFieldType);
        pfType && pfType.getIsMontgomery()) {
      nonResidueAttr = mod_arith::getAttrAsMontgomeryForm(
          pfType.getModulus(), cast<DenseIntElementsAttr>(nonResidueAttr));
    }
    if (failed(parser.parseGreater())) {
      return nullptr;
    }
    return ExtensionFieldType::get(parser.getContext(), degree, baseFieldType,
                                   nonResidueAttr);
  }
  IntegerAttr nonResidue = dyn_cast<IntegerAttr>(nonResidueAttr);
  if (!nonResidue) {
    parser.emitError(parser.getCurrentLocation(),
                     "non-residue must be an integer or dense-int attribute");
    return nullptr;
  }

  // Convert non-residue to Montgomery form if base field is in Montgomery form
  if (auto pfType = dyn_cast<PrimeFieldType>(baseFieldType)) {
    if (pfType.getIsMontgomery()) {
      nonResidue =
          mod_arith::getAttrAsMontgomeryForm(pfType.getModulus(), nonResidue);
    }
  } else if (auto efType = dyn_cast<ExtensionFieldType>(baseFieldType)) {
    if (efType.isMontgomery()) {
      // For tower extensions, convert the scalar non-residue to Montgomery form
      // using the underlying prime field's modulus. The scalar will later be
      // embedded as [value, 0, 0, ...] in the base extension field.
      nonResidue = mod_arith::getAttrAsMontgomeryForm(
          efType.getBasePrimeField().getModulus(), nonResidue);
    }
  }

  if (failed(parser.parseGreater())) {
    return nullptr;
  }
  return ExtensionFieldType::get(parser.getContext(), degree, baseFieldType,
                                 nonResidue);
}

void ExtensionFieldType::print(AsmPrinter &printer) const {
  Type baseField = getBaseField();
  Attribute nonResidue = getNonResidue();

  if (auto pfType = dyn_cast<PrimeFieldType>(baseField)) {
    if (pfType.getIsMontgomery()) {
      // Scalar = binomial non-residue; dense = general monic modulus low
      // coefficients (prime base only).
      if (auto intAttr = dyn_cast<IntegerAttr>(nonResidue)) {
        nonResidue =
            mod_arith::getAttrAsStandardForm(pfType.getModulus(), intAttr);
      } else {
        nonResidue = mod_arith::getAttrAsStandardForm(
            pfType.getModulus(), cast<DenseIntElementsAttr>(nonResidue));
      }
    }
  } else {
    auto efType = cast<ExtensionFieldType>(baseField);
    if (efType.isMontgomery()) {
      auto modulus = efType.getBasePrimeField().getModulus();
      if (auto intAttr = dyn_cast<IntegerAttr>(nonResidue)) {
        nonResidue = mod_arith::getAttrAsStandardForm(modulus, intAttr);
      } else {
        nonResidue = mod_arith::getAttrAsStandardForm(
            modulus, cast<DenseIntElementsAttr>(nonResidue));
      }
    }
  }
  printer << "<" << getDegree() << "x" << baseField << ", " << nonResidue
          << ">";
}

llvm::TypeSize ExtensionFieldType::getTypeSizeInBits(
    DataLayout const &, llvm::ArrayRef<DataLayoutEntryInterface>) const {
  return llvm::TypeSize::getFixed(getTypeSizeInBits());
}

uint64_t ExtensionFieldType::getABIAlignment(
    DataLayout const &dataLayout,
    llvm::ArrayRef<DataLayoutEntryInterface> params) const {
  Type baseField = getBaseField();
  if (auto pfType = dyn_cast<PrimeFieldType>(baseField)) {
    return dataLayout.getTypeABIAlignment(pfType.getStorageType());
  }
  // For tower: use alignment of the underlying prime field
  return dataLayout.getTypeABIAlignment(getBasePrimeField().getStorageType());
}

bool ExtensionFieldType::isMontgomery() const {
  return getBasePrimeField().getIsMontgomery();
}

TypedAttr ExtensionFieldType::createConstantAttr(int64_t c) const {
  PrimeFieldType pfType = getBasePrimeField();
  PrimeFieldOperation pfOp(c, pfType); // Handles Montgomery conversion
  unsigned degreeOverPrime = getDegreeOverPrime();
  unsigned bitWidth = pfType.getTypeSizeInBits();

  SmallVector<APInt> coeffs(degreeOverPrime, APInt::getZero(bitWidth));
  coeffs[0] = static_cast<APInt>(pfOp);
  return createConstantAttrFromValues(coeffs);
}

TypedAttr
ExtensionFieldType::createConstantAttrFromValues(ArrayRef<APInt> coeffs) const {
  return ext_field_utils::createConstantAttrImpl(coeffs, *this);
}

ShapedType ExtensionFieldType::overrideShapedType(ShapedType type) const {
  return type;
}

size_t ExtensionFieldType::getDenseElementBitSize() const {
  return getTypeSizeInBits();
}

// An EF element serializes as its `degreeOverPrime` prime-field coefficients,
// each a storage-int's worth of little-endian bytes. The (de)serialization is
// symmetric with the `tensor<degree x iStorage>` cover the AsmPrinter walks.
Attribute
ExtensionFieldType::convertToAttribute(::llvm::ArrayRef<char> rawData) const {
  PrimeFieldType pfType = getBasePrimeField();
  unsigned primeBits = pfType.getTypeSizeInBits();
  // Sub-byte prime storage isn't supported — would need bit-packing per coeff.
  if (primeBits % 8 != 0)
    return Attribute{};
  unsigned primeBytes = primeBits / 8;
  unsigned degree = getDegreeOverPrime();
  if (rawData.size() != degree * primeBytes)
    return Attribute{};

  auto tensorTy = RankedTensorType::get({static_cast<int64_t>(degree)},
                                        pfType.getStorageType());
  return DenseElementsAttr::getFromRawBuffer(tensorTy, rawData);
}

::llvm::LogicalResult ExtensionFieldType::convertFromAttribute(
    Attribute attr, ::llvm::SmallVectorImpl<char> &result) const {
  PrimeFieldType pfType = getBasePrimeField();
  unsigned primeBits = pfType.getTypeSizeInBits();
  if (primeBits % 8 != 0)
    return failure();
  unsigned primeBytes = primeBits / 8;
  unsigned degree = getDegreeOverPrime();

  auto denseAttr = dyn_cast<DenseElementsAttr>(attr);
  if (!denseAttr)
    return failure();
  if (denseAttr.getNumElements() != static_cast<int64_t>(degree))
    return failure();

  ArrayRef<char> raw = denseAttr.getRawData();
  size_t want = static_cast<size_t>(degree) * primeBytes;
  if (denseAttr.isSplat()) {
    if (raw.size() != primeBytes)
      return failure();
    result.reserve(result.size() + want);
    for (unsigned i = 0; i < degree; ++i)
      result.append(raw.begin(), raw.end());
  } else {
    if (raw.size() != want)
      return failure();
    result.append(raw.begin(), raw.end());
  }
  return success();
}

unsigned ExtensionFieldType::getDegreeOverPrime() const {
  auto baseField = cast<FieldTypeInterface>(getBaseField());
  return getDegree() * baseField.getDegreeOverPrime();
}

PrimeFieldType ExtensionFieldType::getBasePrimeField() const {
  Type baseField = getBaseField();
  if (auto pfType = dyn_cast<PrimeFieldType>(baseField)) {
    return pfType;
  }
  // Recursively find the prime field at the base of the tower
  return cast<ExtensionFieldType>(baseField).getBasePrimeField();
}

bool ExtensionFieldType::isTower() const {
  return isa<ExtensionFieldType>(getBaseField());
}

unsigned ExtensionFieldType::getTowerDepth() const {
  Type baseField = getBaseField();
  if (isa<PrimeFieldType>(baseField)) {
    return 1;
  }
  return 1 + cast<ExtensionFieldType>(baseField).getTowerDepth();
}

SmallVector<int64_t> ExtensionFieldType::getAttrShape() const {
  SmallVector<int64_t> shape;
  shape.push_back(static_cast<int64_t>(getDegree()));
  auto baseShape = cast<FieldTypeInterface>(getBaseField()).getAttrShape();
  shape.append(baseShape.begin(), baseShape.end());
  return shape;
}

Type ExtensionFieldType::cloneWith(Type baseField, Attribute element) const {
  return ExtensionFieldType::get(getContext(), getDegree(), baseField, element);
}

Value ExtensionFieldType::createNonResidueValue(
    ImplicitLocOpBuilder &builder) const {
  Type baseFieldType = getBaseField();
  Attribute nonResidueAttr = getNonResidue();

  // Prime field base: return mod_arith constant directly.
  // The non-residue attribute is already in the correct form (standard or
  // Montgomery), so we use it directly without conversion.
  if (auto pfType = dyn_cast<PrimeFieldType>(baseFieldType)) {
    auto intAttr = cast<IntegerAttr>(nonResidueAttr);
    return mod_arith::ConstantOp::create(builder, convertPrimeFieldType(pfType),
                                         intAttr);
  }

  // Extension field base: use ConstantLikeInterface for proper embedding
  auto baseEfType = cast<ExtensionFieldType>(baseFieldType);
  auto constantLike = cast<ConstantLikeInterface>(baseEfType);

  // If integer attr, embed as [value, 0, 0, ...] in the extension field
  if (auto intAttr = dyn_cast<IntegerAttr>(nonResidueAttr)) {
    nonResidueAttr = constantLike.createConstantAttr(intAttr.getInt());
  }

  return ConstantOp::materialize(builder, nonResidueAttr, baseEfType,
                                 builder.getLoc());
}

bool ExtensionFieldType::hasGeneralModulus() const {
  return isa<PrimeFieldType>(getBaseField()) &&
         isa<DenseIntElementsAttr>(getNonResidue());
}

SmallVector<Value> ExtensionFieldType::createModulusLowCoeffValues(
    ImplicitLocOpBuilder &builder) const {
  // General-monic-modulus counterpart of createNonResidueValue: one mod_arith
  // constant per low coefficient of uᴺ ≡ Σⱼ mⱼ·uʲ. Prime base only — a tower
  // over a general-modulus field is future work.
  assert(hasGeneralModulus());
  auto pfType = cast<PrimeFieldType>(getBaseField());
  auto denseAttr = cast<DenseIntElementsAttr>(getNonResidue());
  SmallVector<Value> coeffs;
  for (const APInt &c : denseAttr.getValues<APInt>()) {
    coeffs.push_back(mod_arith::ConstantOp::create(
        builder, convertPrimeFieldType(pfType),
        IntegerAttr::get(pfType.getStorageType(), c)));
  }
  return coeffs;
}

Value ExtensionFieldType::buildStructFromCoeffs(
    ImplicitLocOpBuilder &builder, Type structType,
    llvm::ArrayRef<Value> coeffs) const {
  return ext_field_utils::dispatchBuildStructFromCoeffs(
      getDegree(), builder, structType, coeffs,
      ext_field_utils::kExtDegreeSequence);
}

llvm::SmallVector<Value>
ExtensionFieldType::extractCoeffsFromStruct(ImplicitLocOpBuilder &builder,
                                            Value structValue) const {
  return ext_field_utils::dispatchExtractCoeffsFromStruct(
      getDegree(), builder, structValue, ext_field_utils::kExtDegreeSequence);
}

//===----------------------------------------------------------------------===//
// ExtensionFieldType field-level coefficient methods
//===----------------------------------------------------------------------===//

Operation::result_range ExtensionFieldType::toCoeffs(ImplicitLocOpBuilder &b,
                                                     Value val) const {
  SmallVector<Type> resultTypes(getDegree(), getBaseField());
  return ExtToCoeffsOp::create(b, resultTypes, val).getResults();
}

Value ExtensionFieldType::fromCoeffs(ImplicitLocOpBuilder &b,
                                     ValueRange coeffs) const {
  return ExtFromCoeffsOp::create(b, *this, coeffs);
}

Value ExtensionFieldType::fromPrimeCoeffs(ImplicitLocOpBuilder &b,
                                          ArrayRef<Value> primeCoeffs) const {
  Type baseField = getBaseField();
  unsigned degree = getDegree();
  if (isa<PrimeFieldType>(baseField)) {
    return fromCoeffs(b, primeCoeffs);
  }
  auto baseEf = cast<ExtensionFieldType>(baseField);
  unsigned baseDeg = baseEf.getDegreeOverPrime();
  SmallVector<Value> baseCoeffs;
  for (unsigned i = 0; i < degree; ++i) {
    baseCoeffs.push_back(
        baseEf.fromPrimeCoeffs(b, primeCoeffs.slice(i * baseDeg, baseDeg)));
  }
  return fromCoeffs(b, baseCoeffs);
}

//===----------------------------------------------------------------------===//
// Standalone convenience wrappers
//===----------------------------------------------------------------------===//

Operation::result_range toCoeffs(ImplicitLocOpBuilder &b, Value val) {
  return cast<ExtensionFieldType>(val.getType()).toCoeffs(b, val);
}

Value fromCoeffs(ImplicitLocOpBuilder &b, Type type, ValueRange coeffs) {
  return cast<ExtensionFieldType>(type).fromCoeffs(b, coeffs);
}

Value fromPrimeCoeffs(ImplicitLocOpBuilder &b, ExtensionFieldType efType,
                      ArrayRef<Value> primeCoeffs) {
  return efType.fromPrimeCoeffs(b, primeCoeffs);
}

Value createFieldConstant(PrimeFieldType pfType, ImplicitLocOpBuilder &builder,
                          const APInt &value) {
  auto attr = IntegerAttr::get(pfType.getStorageType(), value);
  return ConstantOp::create(builder, pfType, attr);
}

} // namespace mlir::prime_ir::field
