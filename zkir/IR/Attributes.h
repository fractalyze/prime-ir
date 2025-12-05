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

#ifndef ZKIR_IR_ATTRIBUTES_H_
#define ZKIR_IR_ATTRIBUTES_H_

#include <cstdint>

#include "mlir/IR/AttributeSupport.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Types.h"
#include "mlir/lib/AsmParser/TensorLiteralParser.h"

namespace mlir::zkir {
namespace mod_arith {

class DenseModArithElementsAttr;

} // namespace mod_arith

struct ZkirDenseElementsAttributeStorage : public AttributeStorage {
public:
  ZkirDenseElementsAttributeStorage(ShapedType type, bool isSplat)
      : type(type), isSplat(isSplat) {}

  ShapedType type;
  bool isSplat;
};

// This is taken and modified from
// https://github.com/llvm/llvm-project/blob/5ed852f/mlir/include/mlir/IR/BuiltinAttributes.h#L80-L683
// An attribute that represents a reference to a dense vector or tensor
// object.
class ZkirDenseElementsAttr : public Attribute {
public:
  using Attribute::Attribute;

  // Allow implicit conversion to ElementsAttr.
  operator ElementsAttr() const { return cast_if_present<ElementsAttr>(*this); }
  // Allow implicit conversion to TypedAttr.
  operator TypedAttr() const { return ElementsAttr(*this); }

  // Method for support type inquiry through isa, cast and dyn_cast.
  static bool classof(Attribute attr);

  static bool isValidType(ShapedType type);

  // Constructs a dense elements attribute from an array of element values.
  // Each element attribute value is expected to be an element of 'type'.
  // 'type' must be a vector or tensor with static shape.
  static ZkirDenseElementsAttr get(ShapedType type, ArrayRef<Attribute> values);

  // Constructs a dense integer elements attribute from an array of APInt
  // values. Each APInt value is expected to have the same bitwidth as the
  // element type of 'type'. 'type' must be a vector or tensor with static
  // shape.
  static ZkirDenseElementsAttr get(ShapedType type, ArrayRef<APInt> values);

  // Construct a dense elements attribute from a raw buffer representing the
  // data for this attribute. Users are encouraged to use one of the
  // constructors above, which provide more safeties. However, this
  // constructor is useful for tools which may want to interop and can
  // follow the precise definition.
  //
  // The format of the raw buffer is a densely packed array of values that
  // can be bitcast to the storage format of the element type specified.
  static ZkirDenseElementsAttr getFromRawBuffer(ShapedType type,
                                                ArrayRef<char> rawBuffer);

  static ZkirDenseElementsAttr getAttr(detail::TensorLiteralParser &p,
                                       SMLoc loc, ShapedType type);

  // Returns true if the given buffer is a valid raw buffer for the given
  // type. `detectedSplat` is set if the buffer is valid and represents a
  // splat buffer.
  //
  // User code should be prepared for additional, conformant patterns to be
  // identified as splats in the future.
  static bool isValidRawBuffer(ShapedType type, ArrayRef<char> rawBuffer,
                               bool &detectedSplat);

  //===--------------------------------------------------------------------===//
  // Iterators
  //===--------------------------------------------------------------------===//

  // The iterator range over the given iterator type T.
  template <typename IteratorT>
  using iterator_range_impl = detail::ElementsAttrRange<IteratorT>;

  // The iterator for the given element type T.
  template <typename T, typename AttrT = ZkirDenseElementsAttr>
  using iterator = decltype(std::declval<AttrT>().template value_begin<T>());
  // The iterator range over the given element T.
  template <typename T, typename AttrT = ZkirDenseElementsAttr>
  using iterator_range =
      decltype(std::declval<AttrT>().template getValues<T>());

  // A utility iterator that allows walking over the internal Attribute values
  // of a ZkirDenseElementsAttr.
  class AttributeElementIterator
      : public llvm::indexed_accessor_iterator<AttributeElementIterator,
                                               const void *, Attribute,
                                               Attribute, Attribute> {
  public:
    // Accesses the Attribute value at this iterator position.
    Attribute operator*() const;

  private:
    friend ZkirDenseElementsAttr;
    friend mod_arith::DenseModArithElementsAttr;

    // Constructs a new iterator.
    AttributeElementIterator(ZkirDenseElementsAttr attr, size_t index);
  };

  // Iterator for walking raw element values of the specified type 'T', which
  // may be any c++ data type matching the stored representation: int32_t,
  // float, etc.
  template <typename T>
  class ElementIterator
      : public detail::DenseElementIndexedIteratorImpl<ElementIterator<T>,
                                                       const T> {
  public:
    // Accesses the raw value at this iterator position.
    const T &operator*() const {
      return reinterpret_cast<const T *>(this->getData())[this->getDataIndex()];
    }

  private:
    friend ZkirDenseElementsAttr;
    friend mod_arith::DenseModArithElementsAttr;

    // Constructs a new iterator.
    ElementIterator(const char *data, bool isSplat, size_t dataIndex)
        : detail::DenseElementIndexedIteratorImpl<ElementIterator<T>, const T>(
              data, isSplat, dataIndex) {}
  };

  // A utility iterator that allows walking over the internal raw APInt values.
  class IntElementIterator
      : public detail::DenseElementIndexedIteratorImpl<IntElementIterator,
                                                       APInt, APInt, APInt> {
  public:
    // Accesses the raw APInt value at this iterator position.
    APInt operator*() const;

  private:
    friend ZkirDenseElementsAttr;
    friend mod_arith::DenseModArithElementsAttr;

    // Constructs a new iterator.
    IntElementIterator(ZkirDenseElementsAttr attr, size_t dataIndex);

    // The bitwidth of the element type.
    size_t bitWidth;
  };

  //===--------------------------------------------------------------------===//
  // Value Querying
  //===--------------------------------------------------------------------===//

  // Returns true if this attribute corresponds to a splat, i.e. if all
  // element values are the same.
  bool isSplat() const;

  // Return the raw storage data held by this attribute. Users should
  // generally not use this directly, as the internal storage format is not
  // always in the form the user might expect.
  ArrayRef<char> getRawData() const;

  // Return the type of this ElementsAttr, guaranteed to be a vector or tensor
  // with static shape.
  ShapedType getType() const;

  // Return the element type of this DenseElementsAttr.
  Type getElementType() const;

  // Returns the number of elements held by this attribute.
  int64_t getNumElements() const;

  // Returns the number of elements held by this attribute.
  int64_t size() const { return getNumElements(); }

  // Returns if the number of elements held by this attribute is 0.
  bool empty() const { return size() == 0; }

  //===--------------------------------------------------------------------===//
  // Mutation Utilities
  //===--------------------------------------------------------------------===//

  // Return a new DenseElementsAttr that has the same data as the current
  // attribute, but has bitcast elements to 'newElType'. The new type must have
  // the same bitwidth as the current element type.
  DenseElementsAttr bitcast(Type newElType) const;

protected:
  // Iterators to various elements that require out-of-line definition. These
  // are hidden from the user to encourage consistent use of the
  // getValues/value_begin/value_end API.
  IntElementIterator raw_int_begin() const {
    return IntElementIterator(*this, 0);
  }
  IntElementIterator raw_int_end() const {
    return IntElementIterator(*this, getNumElements());
  }

  static ZkirDenseElementsAttr getHexAttr(detail::TensorLiteralParser &parser,
                                          SMLoc loc, ShapedType type);
};

} // namespace mlir::zkir

#endif // ZKIR_IR_ATTRIBUTES_H_
