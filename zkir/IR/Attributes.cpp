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

#include "zkir/IR/Attributes.h"

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "zkir/Dialect/ModArith/IR/ModArithAttributes.h"
#include "zkir/Dialect/ModArith/IR/ModArithTypes.h"

namespace mlir::zkir {

//===----------------------------------------------------------------------===//
// ZkirDenseElementsAttr Utilities
//===----------------------------------------------------------------------===//

namespace {

size_t getDenseElementBitWidth(Type eltType) {
  if (auto modArithType = dyn_cast<mod_arith::ModArithType>(eltType)) {
    return cast<IntegerType>(modArithType.getModulus().getType()).getWidth();
  }
  llvm_unreachable("expected mod_arith::ModArithType");
  return 0;
}

// Get the bitwidth of a dense element type within the buffer.
// ZkirDenseElementsAttr requires bitwidths greater than 1 to be aligned by 8.
size_t getDenseElementStorageWidth(size_t origWidth) {
  return origWidth == 1 ? origWidth : llvm::alignTo<8>(origWidth);
}

size_t getDenseElementStorageWidth(Type elementType) {
  return getDenseElementStorageWidth(getDenseElementBitWidth(elementType));
}

template <typename Values>
bool hasSameNumElementsOrSplat(ShapedType type, const Values &values) {
  return (values.size() == 1) ||
         (type.getNumElements() == static_cast<int64_t>(values.size()));
}

} // namespace

//===----------------------------------------------------------------------===//
// DenseElementsAttr Iterators
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// AttributeElementIterator
//===----------------------------------------------------------------------===//

ZkirDenseElementsAttr::AttributeElementIterator::AttributeElementIterator(
    ZkirDenseElementsAttr attr, size_t index)
    : llvm::indexed_accessor_iterator<AttributeElementIterator, const void *,
                                      Attribute, Attribute, Attribute>(
          attr.getAsOpaquePointer(), index) {}

Attribute ZkirDenseElementsAttr::AttributeElementIterator::operator*() const {
  auto owner = cast<ZkirDenseElementsAttr>(getFromOpaquePointer(base));
  Type eltTy = owner.getElementType();
  if (auto modArithType = dyn_cast<mod_arith::ModArithType>(eltTy))
    return IntegerAttr::get(modArithType.getStorageType(),
                            *IntElementIterator(owner, index));
  llvm_unreachable("expected mod_arith::ModArithType");
  return nullptr;
}

//===----------------------------------------------------------------------===//
// IntElementIterator
//===----------------------------------------------------------------------===//

ZkirDenseElementsAttr::IntElementIterator::IntElementIterator(
    ZkirDenseElementsAttr attr, size_t dataIndex)
    : DenseElementIndexedIteratorImpl<IntElementIterator, APInt, APInt, APInt>(
          attr.getRawData().data(), attr.isSplat(), dataIndex),
      bitWidth(getDenseElementBitWidth(attr.getElementType())) {}

APInt ZkirDenseElementsAttr::IntElementIterator::operator*() const {
  return readBits(getData(),
                  getDataIndex() * getDenseElementStorageWidth(bitWidth),
                  bitWidth);
}

//===----------------------------------------------------------------------===//
// ZkirDenseElementsAttr methods
//===----------------------------------------------------------------------===//

// static
bool ZkirDenseElementsAttr::classof(Attribute attr) {
  return llvm::isa<mod_arith::DenseModArithElementsAttr>(attr);
}

// static
bool ZkirDenseElementsAttr::isValidType(ShapedType type) {
  return isa<mod_arith::ModArithType>(type.getElementType());
}

// static
ZkirDenseElementsAttr ZkirDenseElementsAttr::get(ShapedType type,
                                                 ArrayRef<Attribute> values) {
  assert(hasSameNumElementsOrSplat(type, values));

  Type eltType = type.getElementType();

  if (auto modArithType = dyn_cast<mod_arith::ModArithType>(eltType)) {
    // Otherwise, get the raw storage width to use for the allocation.
    size_t bitWidth = getDenseElementBitWidth(eltType);
    size_t storageBitWidth = getDenseElementStorageWidth(bitWidth);

    // Compress the attribute values into a character buffer.
    SmallVector<char, 8> data(
        llvm::divideCeil(storageBitWidth * values.size(), CHAR_BIT));
    APInt intVal;
    for (unsigned i = 0, e = values.size(); i < e; ++i) {
      auto intAttr = cast<IntegerAttr>(values[i]);
      assert(intAttr.getType() ==
                 cast<mod_arith::ModArithType>(eltType).getStorageType() &&
             "expected integer attribute type to equal element type");
      intVal = intAttr.getValue();

      assert(intVal.getBitWidth() == bitWidth &&
             "expected value to have same bitwidth as element type");
      writeBits(data.data(), i * storageBitWidth, intVal);
    }

    // Handle the special encoding of splat of bool.
    if (values.size() == 1 && eltType.isInteger(1))
      data[0] = data[0] ? -1 : 0;

    return mod_arith::DenseModArithElementsAttr::getRaw(type, data);
  }
  return nullptr;
}

// static
ZkirDenseElementsAttr ZkirDenseElementsAttr::get(ShapedType type,
                                                 ArrayRef<APInt> values) {
  if (auto modArithType =
          dyn_cast<mod_arith::ModArithType>(type.getElementType())) {
    assert(hasSameNumElementsOrSplat(type, values));
    size_t storageBitWidth = getDenseElementStorageWidth(modArithType);
    return mod_arith::DenseModArithElementsAttr::getRaw(type, storageBitWidth,
                                                        values);
  }
  llvm_unreachable("expected mod_arith::ModArithType");
  return nullptr;
}

// static
ZkirDenseElementsAttr
ZkirDenseElementsAttr::getFromRawBuffer(ShapedType type,
                                        ArrayRef<char> rawBuffer) {
  if (auto modArithType =
          dyn_cast<mod_arith::ModArithType>(type.getElementType())) {
    return mod_arith::DenseModArithElementsAttr::getRaw(type, rawBuffer);
  }
  llvm_unreachable("expected mod_arith::ModArithType");
  return nullptr;
}

// static
ZkirDenseElementsAttr
ZkirDenseElementsAttr::getAttr(detail::TensorLiteralParser &parser, SMLoc loc,
                               ShapedType type) {
  Type eltType = type.getElementType();

  // Check to see if we parse the literal from a hex string.
  if (parser.hexStorage &&
      (eltType.isIntOrIndexOrFloat() || isa<ComplexType>(eltType)))
    return getHexAttr(parser, loc, type);

  // Check that the parsed storage size has the same number of elements to the
  // type, or is a known splat.
  if (!parser.shape.empty() && parser.getShape() != type.getShape()) {
    parser.p.emitError(loc)
        << "inferred shape of elements literal ([" << parser.getShape()
        << "]) does not match type ([" << type.getShape() << "])";
    return nullptr;
  }

  // Handle the case where no elements were parsed.
  if (!parser.hexStorage && parser.storage.empty() && type.getNumElements()) {
    parser.p.emitError(loc) << "parsed zero elements, but type (" << type
                            << ") expected at least 1";
    return nullptr;
  }

  if (auto modArithType = dyn_cast<mod_arith::ModArithType>(eltType)) {
    std::vector<APInt> intValues;
    if (failed(parser.getIntAttrElements(loc, modArithType.getStorageType(),
                                         intValues)))
      return nullptr;
    return ZkirDenseElementsAttr::get(type, intValues);
  }
  llvm_unreachable("expected mod_arith::ModArithType");
  return nullptr;
}

namespace {

ParseResult parseElementAttrHexValues(detail::Parser &parser, Token tok,
                                      std::string &result) {
  if (std::optional<std::string> value = tok.getHexStringValue()) {
    result = std::move(*value);
    return success();
  }
  return parser.emitError(
      tok.getLoc(), "expected string containing hex digits starting with `0x`");
}

} // namespace

// static
ZkirDenseElementsAttr
ZkirDenseElementsAttr::getHexAttr(detail::TensorLiteralParser &parser,
                                  SMLoc loc, ShapedType type) {
  std::string data;
  if (parseElementAttrHexValues(parser.p, *parser.hexStorage, data))
    return nullptr;

  ArrayRef<char> rawData(data);
  bool detectedSplat = false;
  if (!isValidRawBuffer(type, rawData, detectedSplat)) {
    parser.p.emitError(loc)
        << "elements hex data size is invalid for provided type: " << type;
    return nullptr;
  }

  if (llvm::endianness::native == llvm::endianness::big) {
    // Convert endianness in big-endian(BE) machines. `rawData` is
    // little-endian(LE) because HEX in raw data of dense element attribute
    // is always LE format. It is converted into BE here to be used in BE
    // machines.
    SmallVector<char, 64> outDataVec(rawData.size());
    MutableArrayRef<char> convRawData(outDataVec);
    DenseIntOrFPElementsAttr::convertEndianOfArrayRefForBEmachine(
        rawData, convRawData, type);
    return ZkirDenseElementsAttr::getFromRawBuffer(type, convRawData);
  }

  return ZkirDenseElementsAttr::getFromRawBuffer(type, rawData);
}

bool ZkirDenseElementsAttr::isValidRawBuffer(ShapedType type,
                                             ArrayRef<char> rawBuffer,
                                             bool &detectedSplat) {
  size_t storageWidth = getDenseElementStorageWidth(type.getElementType());
  size_t rawBufferWidth = rawBuffer.size() * CHAR_BIT;
  int64_t numElements = type.getNumElements();

  // The initializer is always a splat if the result type has a single element.
  detectedSplat = numElements == 1;

  // Storage width of 1 is special as it is packed by the bit.
  if (storageWidth == 1) {
    // Check for a splat, or a buffer equal to the number of elements which
    // consists of either all 0's or all 1's.
    if (rawBuffer.size() == 1) {
      auto rawByte = static_cast<uint8_t>(rawBuffer[0]);
      if (rawByte == 0 || rawByte == 0xff) {
        detectedSplat = true;
        return true;
      }
    }

    // This is a valid non-splat buffer if it has the right size.
    return rawBufferWidth == llvm::alignTo<8>(numElements);
  }

  // All other types are 8-bit aligned, so we can just check the buffer width
  // to know if only a single initializer element was passed in.
  if (rawBufferWidth == storageWidth) {
    detectedSplat = true;
    return true;
  }

  // The raw buffer is valid if it has the right size.
  return rawBufferWidth == storageWidth * numElements;
}

ArrayRef<char> ZkirDenseElementsAttr::getRawData() const {
  Type eltType = getType().getElementType();
  if (auto modArithType = dyn_cast<mod_arith::ModArithType>(eltType)) {
    return static_cast<mod_arith::detail::DenseModArithElementsAttrStorage *>(
               impl)
        ->data;
  }
  llvm_unreachable("expected mod_arith::ModArithType");
  return {};
}

DenseElementsAttr ZkirDenseElementsAttr::bitcast(Type newEltType) const {
  ShapedType curType = getType();
  Type curEltType = curType.getElementType();

  if (auto modArithType = dyn_cast<mod_arith::ModArithType>(curEltType)) {
    assert(modArithType.getModulus().getType() == newEltType);
    return DenseIntElementsAttr::getRaw(curType.clone(newEltType),
                                        getRawData());
  }
  llvm_unreachable("expected mod_arith::ModArithType");
  return nullptr;
}

bool ZkirDenseElementsAttr::isSplat() const {
  return static_cast<ZkirDenseElementsAttributeStorage *>(impl)->isSplat;
}

ShapedType ZkirDenseElementsAttr::getType() const {
  return static_cast<const ZkirDenseElementsAttributeStorage *>(impl)->type;
}

Type ZkirDenseElementsAttr::getElementType() const {
  return getType().getElementType();
}

int64_t ZkirDenseElementsAttr::getNumElements() const {
  return getType().getNumElements();
}

} // namespace mlir::zkir
