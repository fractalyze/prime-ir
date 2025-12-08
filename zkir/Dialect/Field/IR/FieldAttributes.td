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

#ifndef ZKIR_DIALECT_FIELD_IR_FIELDATTRS_TD_
#define ZKIR_DIALECT_FIELD_IR_FIELDATTRS_TD_

include "zkir/Dialect/Field/IR/FieldDialect.td"
include "zkir/Dialect/Field/IR/FieldTypes.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/OpBase.td"

class Field_Attr<string name, string attrMnemonic, list<Trait> traits = []>
  : AttrDef<Field_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

// Extension field attribute interface.
// Provides a unified interface for extension field attributes (Fp2, Fp3, etc.)
def ExtensionFieldAttrInterface : AttrInterface<"ExtensionFieldAttrInterface"> {
  let cppNamespace = "mlir::zkir::field";

  let description = [{
    Interface for extension field attributes that allows generic handling
    of different extension degrees without explicit type checking.
  }];

  let methods = [
    InterfaceMethod<"Get the coefficient at index",
      "mlir::Attribute", "getCoeff", (ins "unsigned":$index)>,
  ];
}

def Field_PrimeFieldAttr : Field_Attr<"PrimeField", "pf.elem", [TypedAttrInterface]> {
  let summary = "An attribute representing a prime field element";
  let parameters = (ins "PrimeFieldType":$type, "IntegerAttr":$value);
  let assemblyFormat = "`<` $value `>` `:` $type";
  let builders = [
    AttrBuilderWithInferredContext<(ins "PrimeFieldType":$type,
                                        "const llvm::APInt":$value), [{
      return $_get(
        type.getContext(),
        type,
        IntegerAttr::get(type.getStorageType(), value));
    }]>,
    AttrBuilderWithInferredContext<(ins "PrimeFieldType":$type,
                                        "const uint64_t":$value), [{
      return $_get(
        type.getContext(),
        type,
        IntegerAttr::get(type.getStorageType(), value));
    }]>
  ];
  let extraClassDeclaration = [{
    using ValueType = Attribute;
  }];
  let genVerifyDecl = 1;
}

def Field_QuadraticExtFieldAttr : Field_Attr<"QuadraticExtField", "f2.elem", [TypedAttrInterface, ExtensionFieldAttrInterface]> {
  let summary = "An attribute representing a quadratic extension field element";
  let parameters = (ins "QuadraticExtFieldType":$type, Field_PrimeFieldAttr:$c0, Field_PrimeFieldAttr:$c1);
  let assemblyFormat = "`<` $c0 `,` $c1 `>` `:` $type";
  let extraClassDeclaration = [{
    using ValueType = Attribute;

    // ExtensionFieldAttrInterface method
    mlir::Attribute getCoeff(unsigned index) const {
      assert(index < 2 && "Index out of bounds");
      return index == 0 ? getC0() : getC1();
    }
  }];
}

def Field_CubicExtFieldAttr : Field_Attr<"CubicExtField", "f3.elem", [TypedAttrInterface, ExtensionFieldAttrInterface]> {
  let summary = "An attribute representing a cubic extension field element";
  let parameters = (ins "CubicExtFieldType":$type, Field_PrimeFieldAttr:$c0, Field_PrimeFieldAttr:$c1, Field_PrimeFieldAttr:$c2);
  let assemblyFormat = "`<` $c0 `,` $c1 `,` $c2 `>` `:` $type";
  let extraClassDeclaration = [{
    using ValueType = Attribute;

    // ExtensionFieldAttrInterface method
    mlir::Attribute getCoeff(unsigned index) const {
      assert(index < 3 && "Index out of bounds");
      switch (index) {
        case 0: return getC0();
        case 1: return getC1();
        default: return getC2();
      }
    }
  }];
}

def Field_QuarticExtFieldAttr : Field_Attr<"QuarticExtField", "f4.elem", [TypedAttrInterface, ExtensionFieldAttrInterface]> {
  let summary = "An attribute representing a quartic extension field element";
  let parameters = (ins "QuarticExtFieldType":$type, Field_PrimeFieldAttr:$c0, Field_PrimeFieldAttr:$c1, Field_PrimeFieldAttr:$c2, Field_PrimeFieldAttr:$c3);
  let assemblyFormat = "`<` $c0 `,` $c1 `,` $c2 `,` $c3 `>` `:` $type";
  let extraClassDeclaration = [{
    using ValueType = Attribute;

    // ExtensionFieldAttrInterface method
    mlir::Attribute getCoeff(unsigned index) const {
      assert(index < 4 && "Index out of bounds");
      switch (index) {
        case 0: return getC0();
        case 1: return getC1();
        case 2: return getC2();
        default: return getC3();
      }
    }
  }];
}

def Field_RootOfUnityAttr : Field_Attr<"RootOfUnity", "root_of_unity", [TypedAttrInterface]> {
  let summary = "An attribute representing a root of unity in a prime field";
  let parameters = (ins Field_PrimeFieldAttr:$root, "IntegerAttr":$degree);
  let assemblyFormat = "`<` $root `,` $degree `>`";
  let extraClassDeclaration = [{
    PrimeFieldType getType() const {
      return getRoot().getType();
    }
  }];
  let genVerifyDecl = 1;
}

def Field_AnyFieldAttr : AnyAttrOf<[Field_PrimeFieldAttr, Field_QuadraticExtFieldAttr, Field_CubicExtFieldAttr, Field_QuarticExtFieldAttr]> {
  string cppType = "mlir::TypedAttr";
}

#endif  // ZKIR_DIALECT_FIELD_IR_FIELDATTRS_TD_
