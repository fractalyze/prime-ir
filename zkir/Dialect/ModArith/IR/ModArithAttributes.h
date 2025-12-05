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

#ifndef ZKIR_DIALECT_MODARITH_IR_MODARITHATTRIBUTES_H_
#define ZKIR_DIALECT_MODARITH_IR_MODARITHATTRIBUTES_H_

#include <string.h>

#include <tuple>
#include <utility>

#include "llvm/ADT/Hashing.h"
#include "mlir/IR/AttributeSupport.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "zkir/IR/Attributes.h" // IWYU pragma: keep

namespace mlir::zkir::mod_arith::detail {

struct MontgomeryAttrStorage : public AttributeStorage {
  using KeyTy = IntegerAttr;

  MontgomeryAttrStorage(IntegerAttr modulus, IntegerAttr nPrime,
                        IntegerAttr nInv, IntegerAttr r, IntegerAttr rInv,
                        IntegerAttr bInv, IntegerAttr rSquared,
                        SmallVector<IntegerAttr> invTwoPowers)
      : modulus(std::move(modulus)), nPrime(std::move(nPrime)),
        nInv(std::move(nInv)), r(std::move(r)), rInv(std::move(rInv)),
        bInv(std::move(bInv)), rSquared(std::move(rSquared)),
        invTwoPowers(std::move(invTwoPowers)) {}

  KeyTy getAsKey() const { return KeyTy(modulus); }

  bool operator==(const KeyTy &key) const { return key == KeyTy(modulus); }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key);
  }

  static MontgomeryAttrStorage *construct(AttributeStorageAllocator &allocator,
                                          KeyTy &&key);

  IntegerAttr modulus;
  IntegerAttr nPrime;
  IntegerAttr nInv;
  IntegerAttr r;
  IntegerAttr rInv;
  IntegerAttr bInv;
  IntegerAttr rSquared;
  SmallVector<IntegerAttr> invTwoPowers;
};

struct BYAttrStorage : public AttributeStorage {
  using KeyTy = IntegerAttr;

  BYAttrStorage(IntegerAttr modulus, IntegerAttr divsteps, IntegerAttr mInv,
                IntegerAttr newBitWidth)
      : modulus(std::move(modulus)), divsteps(std::move(divsteps)),
        mInv(std::move(mInv)), newBitWidth(std::move(newBitWidth)) {}

  KeyTy getAsKey() const { return KeyTy(modulus); }

  bool operator==(const KeyTy &key) const { return key == KeyTy(modulus); }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key);
  }

  static BYAttrStorage *construct(AttributeStorageAllocator &allocator,
                                  KeyTy &&key);

  IntegerAttr modulus;
  IntegerAttr divsteps;
  IntegerAttr mInv;
  IntegerAttr newBitWidth;
};

// This is taken and modified from
// https://github.com/llvm/llvm-project/blob/5ed852f/mlir/lib/IR/AttributeDetail.h#L55-L201
struct DenseModArithElementsAttrStorage
    : public ZkirDenseElementsAttributeStorage {
  DenseModArithElementsAttrStorage(ShapedType ty, ArrayRef<char> data,
                                   bool isSplat = false)
      : ZkirDenseElementsAttributeStorage(ty, isSplat), data(data) {}

  struct KeyTy {
    KeyTy(ShapedType type, ArrayRef<char> data, llvm::hash_code hashCode,
          bool isSplat = false)
        : type(type), data(data), hashCode(hashCode), isSplat(isSplat) {}

    // The type of the dense elements.
    ShapedType type;

    // The raw buffer for the data storage.
    ArrayRef<char> data;

    // The computed hash code for the storage data.
    llvm::hash_code hashCode;

    // A boolean that indicates if this data is a splat or not.
    bool isSplat;
  };

  // Compare this storage instance with the provided key.
  bool operator==(const KeyTy &key) const {
    return key.type == type && key.data == data;
  }

  // Construct a key from a shaped type, raw data buffer, and a flag that
  // signals if the data is already known to be a splat. Callers to this
  // function are expected to tag preknown splat values when possible, e.g. one
  // element shapes.
  static KeyTy getKey(ShapedType ty, ArrayRef<char> data, bool isKnownSplat) {
    // Handle an empty storage instance.
    if (data.empty())
      return KeyTy(ty, data, 0);

    // If the data is already known to be a splat, the key hash value is
    // directly the data buffer.
    bool isBoolData = ty.getElementType().isInteger(1);
    if (isKnownSplat) {
      if (isBoolData)
        return getKeyForSplatBoolData(ty, data[0] != 0);
      return KeyTy(ty, data, llvm::hash_value(data), isKnownSplat);
    }

    // Otherwise, we need to check if the data corresponds to a splat or not.

    // Handle the simple case of only one element.
    size_t numElements = ty.getNumElements();
    std::ignore = numElements;
    assert(numElements != 1 && "splat of 1 element should already be detected");

    // Handle boolean values directly as they are packed to 1-bit.
    if (isBoolData)
      return getKeyForBoolData(ty, data, numElements);

    size_t elementWidth = getDenseElementBitWidth(ty.getElementType());
    // Non 1-bit dense elements are padded to 8-bits.
    size_t storageSize = llvm::divideCeil(elementWidth, CHAR_BIT);
    assert(((data.size() / storageSize) == numElements) &&
           "data does not hold expected number of elements");

    // Create the initial hash value with just the first element.
    auto firstElt = data.take_front(storageSize);
    auto hashVal = llvm::hash_value(firstElt);

    // Check to see if this storage represents a splat. If it doesn't then
    // combine the hash for the data starting with the first non splat element.
    for (size_t i = storageSize, e = data.size(); i != e; i += storageSize)
      if (memcmp(data.data(), &data[i], storageSize))
        return KeyTy(ty, data, llvm::hash_combine(hashVal, data.drop_front(i)));

    // Otherwise, this is a splat so just return the hash of the first element.
    return KeyTy(ty, firstElt, hashVal, /*isSplat=*/true);
  }

  // Construct a key with a set of boolean data.
  static KeyTy getKeyForBoolData(ShapedType ty, ArrayRef<char> data,
                                 size_t numElements) {
    ArrayRef<char> splatData = data;
    bool splatValue = splatData.front() & 1;

    // Check the simple case where the data matches the known splat value.
    if (splatData == ArrayRef<char>(splatValue ? kSplatTrue : kSplatFalse))
      return getKeyForSplatBoolData(ty, splatValue);

    // Handle the case where the potential splat value is 1 and the number of
    // elements is non 8-bit aligned.
    size_t numOddElements = numElements % CHAR_BIT;
    if (splatValue && numOddElements != 0) {
      // Check that all bits are set in the last value.
      char lastElt = splatData.back();
      if (lastElt != llvm::maskTrailingOnes<unsigned char>(numOddElements))
        return KeyTy(ty, data, llvm::hash_value(data));

      // If this is the only element, the data is known to be a splat.
      if (splatData.size() == 1)
        return getKeyForSplatBoolData(ty, splatValue);
      splatData = splatData.drop_back();
    }

    // Check that the data buffer corresponds to a splat of the proper mask.
    char mask = splatValue ? ~0 : 0;
    return llvm::all_of(splatData, [mask](char c) { return c == mask; })
               ? getKeyForSplatBoolData(ty, splatValue)
               : KeyTy(ty, data, llvm::hash_value(data));
  }

  // Return a key to use for a boolean splat of the given value.
  static KeyTy getKeyForSplatBoolData(ShapedType type, bool splatValue) {
    const char &splatData = splatValue ? kSplatTrue : kSplatFalse;
    return KeyTy(type, splatData, llvm::hash_value(splatData),
                 /*isSplat=*/true);
  }

  // Hash the key for the storage.
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key.type, key.hashCode);
  }

  // Construct a new storage instance.
  static DenseModArithElementsAttrStorage *
  construct(AttributeStorageAllocator &allocator, KeyTy key) {
    // If the data buffer is non-empty, we copy it into the allocator with a
    // 64-bit alignment.
    ArrayRef<char> copy, data = key.data;
    if (!data.empty()) {
      char *rawData = reinterpret_cast<char *>(
          allocator.allocate(data.size(), alignof(uint64_t)));
      memcpy(rawData, data.data(), data.size());
      copy = ArrayRef<char>(rawData, data.size());
    }

    return new (allocator.allocate<DenseModArithElementsAttrStorage>())
        DenseModArithElementsAttrStorage(key.type, copy, key.isSplat);
  }

  ArrayRef<char> data;

  // The values used to denote a boolean splat value.
  // This is not using constexpr declaration due to compilation failure
  // encountered with MSVC where it would inline these values, which makes it
  // unsafe to refer by reference in KeyTy.
  static const char kSplatTrue;
  static const char kSplatFalse;

private:
  static size_t getDenseElementBitWidth(Type ty);
};

} // namespace mlir::zkir::mod_arith::detail

#define GET_ATTRDEF_CLASSES
#include "zkir/Dialect/ModArith/IR/ModArithAttributes.h.inc"

#endif // ZKIR_DIALECT_MODARITH_IR_MODARITHATTRIBUTES_H_
