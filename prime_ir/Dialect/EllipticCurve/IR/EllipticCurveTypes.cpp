/* Copyright 2026 The PrimeIR Authors.

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

#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveTypes.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveOps.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"

namespace mlir::prime_ir::elliptic_curve {

#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveTypesInterfaces.cpp.inc"

#define DEFINE_ELLIPTIC_CURVE_TYPE_INTERFACE_METHODS(TYPE, N)                  \
  unsigned TYPE##Type::getNumCoords() const { return N; }                      \
  Type TYPE##Type::getBaseFieldType() const {                                  \
    return getCurve().getBaseField();                                          \
  }                                                                            \
  Attribute TYPE##Type::getCurveAttr() const { return getCurve(); }            \
  PointKind TYPE##Type::getPointKind() const { return PointKind::k##TYPE; }

DEFINE_ELLIPTIC_CURVE_TYPE_INTERFACE_METHODS(Affine, 2);
DEFINE_ELLIPTIC_CURVE_TYPE_INTERFACE_METHODS(Jacobian, 3);
DEFINE_ELLIPTIC_CURVE_TYPE_INTERFACE_METHODS(XYZZ, 4);
DEFINE_ELLIPTIC_CURVE_TYPE_INTERFACE_METHODS(EdAffine, 2);
DEFINE_ELLIPTIC_CURVE_TYPE_INTERFACE_METHODS(EdExtended, 4);

#undef DEFINE_ELLIPTIC_CURVE_TYPE_INTERFACE_METHODS

//===----------------------------------------------------------------------===//
// ConstantLikeInterface implementations
//===----------------------------------------------------------------------===//

TypedAttr AffineType::createConstantAttr(int64_t c) const {
  // TODO(chokobole): Implement this.
  return {};
}

TypedAttr
AffineType::createConstantAttrFromValues(ArrayRef<APInt> values) const {
  // TODO(chokobole): Implement this.
  return {};
}

ShapedType AffineType::overrideShapedType(ShapedType type) const {
  return type;
}

TypedAttr JacobianType::createConstantAttr(int64_t c) const {
  // TODO(chokobole): Implement this.
  return {};
}

TypedAttr
JacobianType::createConstantAttrFromValues(ArrayRef<APInt> values) const {
  // TODO(chokobole): Implement this.
  return {};
}

ShapedType JacobianType::overrideShapedType(ShapedType type) const {
  return type;
}

TypedAttr XYZZType::createConstantAttr(int64_t c) const {
  // TODO(chokobole): Implement this.
  return {};
}

TypedAttr XYZZType::createConstantAttrFromValues(ArrayRef<APInt> values) const {
  // TODO(chokobole): Implement this.
  return {};
}

ShapedType XYZZType::overrideShapedType(ShapedType type) const { return type; }

TypedAttr EdAffineType::createConstantAttr(int64_t c) const { return {}; }

TypedAttr
EdAffineType::createConstantAttrFromValues(ArrayRef<APInt> values) const {
  return {};
}

ShapedType EdAffineType::overrideShapedType(ShapedType type) const {
  return type;
}

TypedAttr EdExtendedType::createConstantAttr(int64_t c) const {
  // TODO(chokobole): Implement this.
  return {};
}

TypedAttr
EdExtendedType::createConstantAttrFromValues(ArrayRef<APInt> values) const {
  // TODO(chokobole): Implement this.
  return {};
}

ShapedType EdExtendedType::overrideShapedType(ShapedType type) const {
  return type;
}

//===----------------------------------------------------------------------===//
// DenseElementTypeInterface implementations
//
// EC points lay out as `numCoords × baseFieldStorageBits` in dense storage.
// Without this interface, MLIR's `getDenseElementBitWidth` hits
// `llvm_unreachable("unsupported element type")` when any pass touches a
// DenseElementsAttr typed `tensor<...x!ec.point>` (e.g. via getValues, byte
// width queries, canonicalize). In release builds the unreachable becomes
// UB and manifests as heap corruption — observed as a "free(): invalid
// pointer" SIGSEGV at process teardown after a GPU compile of a fused
// scalar EC constant + broadcast.
//===----------------------------------------------------------------------===//

namespace {
size_t pointDenseElementBitSize(unsigned numCoords, Type baseField) {
  if (auto pf = dyn_cast<field::PrimeFieldType>(baseField))
    return numCoords * pf.getStorageBitWidth();
  if (auto ef = dyn_cast<field::ExtensionFieldType>(baseField))
    return numCoords * ef.getStorageBitWidth();
  if (auto bf = dyn_cast<field::BinaryFieldType>(baseField))
    return numCoords * bf.getStorageBitWidth();
  llvm_unreachable("Unsupported EC base field type");
}
}  // namespace

namespace {
// Encode `rawData` (numCoords * baseFieldStorageBits bits, little-endian) as
// a DenseIntElementsAttr typed `tensor<numCoords x i{baseStorageBits}>`. The
// IR pipeline already encodes EC constants this way at JAX level (each
// element is a per-coord storage int), so the round-trip via the textual
// AsmPrinter is symmetric.
Attribute pointConvertToAttribute(unsigned numCoords, Type baseField,
                                  ::llvm::ArrayRef<char> rawData) {
  unsigned baseBits = 0;
  if (auto pf = dyn_cast<field::PrimeFieldType>(baseField))
    baseBits = pf.getStorageBitWidth();
  else if (auto ef = dyn_cast<field::ExtensionFieldType>(baseField))
    baseBits = ef.getStorageBitWidth();
  else if (auto bf = dyn_cast<field::BinaryFieldType>(baseField))
    baseBits = bf.getStorageBitWidth();
  if (baseBits == 0) return {};
  unsigned baseBytes = (baseBits + 7) / 8;
  if (rawData.size() != numCoords * baseBytes) return {};

  ::llvm::SmallVector<APInt, 4> coordVals;
  coordVals.reserve(numCoords);
  for (unsigned c = 0; c < numCoords; ++c) {
    unsigned numWords = (baseBits + 63) / 64;
    ::llvm::SmallVector<uint64_t, 8> words(numWords, 0);
    std::memcpy(words.data(), rawData.data() + c * baseBytes, baseBytes);
    coordVals.emplace_back(baseBits, words);
  }
  auto intTy = IntegerType::get(baseField.getContext(), baseBits);
  auto tensorTy = RankedTensorType::get({static_cast<int64_t>(numCoords)},
                                         intTy);
  return DenseIntElementsAttr::get(tensorTy, coordVals);
}

::llvm::LogicalResult pointConvertFromAttribute(
    unsigned numCoords, Type baseField, Attribute attr,
    ::llvm::SmallVectorImpl<char>& result) {
  unsigned baseBits = 0;
  if (auto pf = dyn_cast<field::PrimeFieldType>(baseField))
    baseBits = pf.getStorageBitWidth();
  else if (auto ef = dyn_cast<field::ExtensionFieldType>(baseField))
    baseBits = ef.getStorageBitWidth();
  else if (auto bf = dyn_cast<field::BinaryFieldType>(baseField))
    baseBits = bf.getStorageBitWidth();
  if (baseBits == 0) return ::llvm::failure();
  unsigned baseBytes = (baseBits + 7) / 8;

  auto denseAttr = dyn_cast<DenseIntElementsAttr>(attr);
  if (!denseAttr) return ::llvm::failure();
  if (denseAttr.getNumElements() != static_cast<int64_t>(numCoords))
    return ::llvm::failure();

  size_t prevSize = result.size();
  result.resize(prevSize + numCoords * baseBytes);
  unsigned c = 0;
  for (const APInt& v : denseAttr.getValues<APInt>()) {
    if (c >= numCoords) break;
    std::memcpy(result.data() + prevSize + c * baseBytes,
                v.getRawData(), baseBytes);
    ++c;
  }
  return ::llvm::success();
}
}  // namespace

#define DEFINE_EC_DENSE_ELEMENT_TYPE_INTERFACE(TYPE, N)                        \
  size_t TYPE##Type::getDenseElementBitSize() const {                          \
    return pointDenseElementBitSize(N, getBaseFieldType());                    \
  }                                                                            \
  Attribute TYPE##Type::convertToAttribute(                                    \
      ::llvm::ArrayRef<char> rawData) const {                                  \
    return pointConvertToAttribute(N, getBaseFieldType(), rawData);            \
  }                                                                            \
  ::llvm::LogicalResult TYPE##Type::convertFromAttribute(                      \
      Attribute attr, ::llvm::SmallVectorImpl<char>& result) const {           \
    return pointConvertFromAttribute(N, getBaseFieldType(), attr, result);     \
  }

DEFINE_EC_DENSE_ELEMENT_TYPE_INTERFACE(Affine, 2)
DEFINE_EC_DENSE_ELEMENT_TYPE_INTERFACE(Jacobian, 3)
DEFINE_EC_DENSE_ELEMENT_TYPE_INTERFACE(XYZZ, 4)

#undef DEFINE_EC_DENSE_ELEMENT_TYPE_INTERFACE

} // namespace mlir::prime_ir::elliptic_curve
