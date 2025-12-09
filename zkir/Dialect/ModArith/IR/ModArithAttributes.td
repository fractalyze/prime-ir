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

#ifndef ZKIR_DIALECT_MODARITH_IR_MODARITHATTRS_TD_
#define ZKIR_DIALECT_MODARITH_IR_MODARITHATTRS_TD_

include "mlir/IR/BuiltinAttributeInterfaces.td"
include "zkir/Dialect/ModArith/IR/ModArithDialect.td"

class ModArith_Attr<string name, string attrMnemonic, list<Trait> traits = [],
                    string baseCppClass = "::mlir::Attribute">
  : AttrDef<ModArith_Dialect, name, traits, baseCppClass> {
  let mnemonic = attrMnemonic;
}

def ModArith_MontgomeryAttr : ModArith_Attr<"Montgomery", "montgomery"> {
  let summary = "An attribute with precomputed Montgomery parameters";
  let description = [{
    This attribute contains the precomputed Montgomery parameters for a given
    modulus. It is used for efficient multiplication when the modulus is fixed
    at compile time.
  }];
  let parameters = (ins "IntegerAttr":$modulus);
  let extraClassDeclaration = [{
    IntegerAttr getNPrime() const;
    IntegerAttr getNInv() const;
    IntegerAttr getR() const;
    IntegerAttr getRInv() const;
    IntegerAttr getRSquared() const;
    IntegerAttr getBInv() const;
    const SmallVector<IntegerAttr> &getInvTwoPowers() const;
  }];
  let assemblyFormat = "`<`$modulus`>`";
  let genStorageClass = 0;
}

def ModArith_BYAttr : ModArith_Attr<"BY", "by"> {
  let summary = "An attribute with precomputed Bernstein-Yang inverse parameters";
  let description = [{
    This attribute contains the precomputed Bernstein-Yang inverse parameters
    for a given modulus. It is used for efficient inverse when the modulus is
    fixed at compile time.
  }];
  let parameters = (ins "IntegerAttr":$modulus);
  let extraClassDeclaration = [{
    IntegerAttr getDivsteps() const;
    IntegerAttr getMInv() const;
    IntegerAttr getNewBitWidth() const;
  }];
  let assemblyFormat = "`<`$modulus`>`";
  let genStorageClass = 0;
}

def ModArith_DenseModArithElementsAttr : ModArith_Attr<
    "DenseModArithElements", "", [ElementsAttrInterface],
    "ZkirDenseElementsAttr"
  > {
  let summary = "An attribute containing a dense multi-dimensional array of "
                "modular arithmetic values";
  let description = [{
    Syntax:

    ```
    tensor-literal ::= integer-literal | [] | [tensor-literal (, tensor-literal)* ]
    dense-mod-arith-elements-attribute ::= `dense` `<` tensor-literal `>` `:`
                                            ( tensor-type | vector-type )
    ```

    A dense modular arithmetic elements attribute is an elements attribute containing
    a densely packed vector or tensor of modular arithmetic values. The
    element type of this attribute is required to be a `ModArithType`.

    Examples:

    ```
    // A splat tensor of integer values.
    dense<10> : tensor<2x!mod_arith.int<65537 : i32>>
    // A tensor of 2 modular arithmetic elements.
    dense<[10, 11]> : tensor<2x!mod_arith.int<65537 : i32>>
    ```
  }];
  let parameters = (ins AttributeSelfTypeParameter<"", "ShapedType">:$type,
                        "ArrayRef<char>":$rawData);
  let extraClassDeclaration = [{
    using ZkirDenseElementsAttr::empty;
    using ZkirDenseElementsAttr::getNumElements;
    using ZkirDenseElementsAttr::getElementType;
    using ZkirDenseElementsAttr::isSplat;
    using ZkirDenseElementsAttr::size;

    // The set of data types that can be iterated by this attribute.
    using ContiguousIterableTypesT = std::tuple<
      // Integer types.
      uint8_t, uint16_t, uint32_t, uint64_t,
      int8_t, int16_t, int32_t, int64_t,
      short, unsigned short, int, unsigned, long, unsigned long
    >;
    using NonContiguousIterableTypesT = std::tuple<
      Attribute,
      // Integer types.
      APInt
    >;

    // Provide a `try_value_begin_impl` to enable iteration within
    // ElementsAttr.
    template <typename T>
    auto try_value_begin_impl(OverloadToken<T>) const {
      return try_value_begin<T>();
    }

    //===--------------------------------------------------------------------===//
    // DenseElementsAttr methods
    //===--------------------------------------------------------------------===//

    // Return the splat value for this attribute. This asserts that the attribute
    // corresponds to a splat.
    template <typename T>
    std::enable_if_t<!std::is_base_of<Attribute, T>::value ||
                        std::is_same<Attribute, T>::value,
                    T>
    getSplatValue() const {
      assert(isSplat() && "expected the attribute to be a splat");
      return *value_begin<T>();
    }

    // Return the splat value for derived attribute element types.
    template <typename T>
    std::enable_if_t<std::is_base_of<Attribute, T>::value &&
                        !std::is_same<Attribute, T>::value,
                    T>
    getSplatValue() const {
      return llvm::cast<T>(getSplatValue<Attribute>());
    }

    // Try to get an iterator of the given type to the start of the held element
    // values. Return failure if the type cannot be iterated.
    template <typename T>
    auto try_value_begin() const {
      auto range = tryGetValues<T>();
      using iterator = decltype(range->begin());
      return failed(range) ? FailureOr<iterator>(failure()) : range->begin();
    }

    // Try to get an iterator of the given type to the end of the held element
    // values. Return failure if the type cannot be iterated.
    template <typename T>
    auto try_value_end() const {
      auto range = tryGetValues<T>();
      using iterator = decltype(range->begin());
      return failed(range) ? FailureOr<iterator>(failure()) : range->end();
    }

    // Return the held element values as a range of the given type.
    template <typename T>
    auto getValues() const {
      auto range = tryGetValues<T>();
      assert(succeeded(range) && "element type cannot be iterated");
      return std::move(*range);
    }

    // Get an iterator of the given type to the start of the held element values.
    template <typename T>
    auto value_begin() const {
      return getValues<T>().begin();
    }

    // Get an iterator of the given type to the end of the held element values.
    template <typename T>
    auto value_end() const {
      return getValues<T>().end();
    }

    // Try to get the held element values as a range of integer or floating-point
    // values.
    template <typename T>
    using IntValueTemplateCheckT =
        std::enable_if_t<(!std::is_same<T, bool>::value &&
                          std::numeric_limits<T>::is_integer)>;
    template <typename T, typename = IntValueTemplateCheckT<T>>
    FailureOr<iterator_range_impl<ElementIterator<T>>> tryGetValues() const {
      if (!isValidInt(sizeof(T), std::numeric_limits<T>::is_signed))
        return failure();
      const char *rawData = getRawData().data();
      bool splat = isSplat();
      return iterator_range_impl<ElementIterator<T>>(
          getType(), ElementIterator<T>(rawData, splat, 0),
          ElementIterator<T>(rawData, splat, getNumElements()));
    }

    // Try to get the held element values as a range of Attributes.
    template <typename T>
    using AttributeValueTemplateCheckT =
        std::enable_if_t<std::is_same<T, Attribute>::value>;
    template <typename T, typename = AttributeValueTemplateCheckT<T>>
    FailureOr<iterator_range_impl<AttributeElementIterator>>
    tryGetValues() const {
      return iterator_range_impl<AttributeElementIterator>(
          getType(), AttributeElementIterator(*this, 0),
          AttributeElementIterator(*this, getNumElements()));
    }

    // Try to get the held element values a range of T, where T is a derived
    // attribute type.
    template <typename T>
    using DerivedAttrValueTemplateCheckT =
        std::enable_if_t<std::is_base_of<Attribute, T>::value &&
                        !std::is_same<Attribute, T>::value>;
    template <typename T>
    struct DerivedAttributeElementIterator
        : public llvm::mapped_iterator_base<DerivedAttributeElementIterator<T>,
                                            AttributeElementIterator, T> {
      using llvm::mapped_iterator_base<DerivedAttributeElementIterator<T>,
                                      AttributeElementIterator,
                                      T>::mapped_iterator_base;

      /// Map the element to the iterator result type.
      T mapElement(Attribute attr) const { return llvm::cast<T>(attr); }
    };
    template <typename T, typename = DerivedAttrValueTemplateCheckT<T>>
    FailureOr<iterator_range_impl<DerivedAttributeElementIterator<T>>>
    tryGetValues() const {
      using DerivedIterT = DerivedAttributeElementIterator<T>;
      return iterator_range_impl<DerivedIterT>(
          getType(), DerivedIterT(value_begin<Attribute>()),
          DerivedIterT(value_end<Attribute>()));
    }

    // Try to get the held element values as a range of APInts.
    template <typename T>
    using APIntValueTemplateCheckT =
        std::enable_if_t<std::is_same<T, APInt>::value>;
    template <typename T, typename = APIntValueTemplateCheckT<T>>
    FailureOr<iterator_range_impl<IntElementIterator>> tryGetValues() const {
      return iterator_range_impl<IntElementIterator>(getType(), raw_int_begin(),
                                                    raw_int_end());
    }

    // Iterator access to the float element values.
    IntElementIterator begin() const { return tryGetValues<APInt>()->begin(); }
    IntElementIterator end() const { return tryGetValues<APInt>()->end(); }

  protected:
    friend ZkirDenseElementsAttr;

    // Constructs a dense elements attribute from an array of raw APInt values.
    // Each APInt value is expected to have the same bitwidth as the element
    // type of 'type'. 'type' must be a vector or tensor with static shape.
    //
    // If the `values` array only has a single element, then this constructs
    // splat of that value.
    static ZkirDenseElementsAttr getRaw(ShapedType type, size_t storageWidth,
                                        ArrayRef<APInt> values);

    // Get or create a new dense elements attribute instance with the given raw
    // data buffer. 'type' must be a vector or tensor with static shape.
    //
    // If the `values` array only has a single element, then this constructs
    // splat of that value.
    static ZkirDenseElementsAttr getRaw(ShapedType type, ArrayRef<char> data);

    // Check the information for a C++ data type, check if this type is valid for
    // the current attribute. This method is used to verify specific type
    // invariants that the templatized 'getValues' method cannot.
    bool isValidInt(int64_t dataEltSize, bool isSigned) const;

  public:
  }];
  let hasCustomAssemblyFormat = 1;
  let genAccessors = 0;
  let genStorageClass = 0;
  let skipDefaultBuilders = 1;
}

#endif  // ZKIR_DIALECT_MODARITH_IR_MODARITHATTRS_TD_
