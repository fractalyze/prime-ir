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

#ifndef ZKIR_DIALECT_FIELD_IR_FIELDTYPES_TD_
#define ZKIR_DIALECT_FIELD_IR_FIELDTYPES_TD_

include "zkir/Dialect/Field/IR/FieldDialect.td"

include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/AttrTypeBase.td"

// Base type definition for Field types.
class Field_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Field_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

// Field type interface.
// Provides a unified interface for all field types (Fp, Fp2, Fp3, etc.)
def FieldTypeInterface : TypeInterface<"FieldTypeInterface"> {
  let description = [{
    Interface for all field types that provides common operations.
  }];

  let methods = [
    InterfaceMethod<"Check if the field is in Montgomery form",
      "bool", "isMontgomery">,
    InterfaceMethod<"Create zero constant",
      "mlir::Value", "createZeroConstant",
      (ins "mlir::ImplicitLocOpBuilder &":$builder)>,
    InterfaceMethod<"Create one constant",
      "mlir::Value", "createOneConstant",
      (ins "mlir::ImplicitLocOpBuilder &":$builder)>,
  ];
}

// Extension field type interface.
// Provides a unified interface for extension field types (Fp2, Fp3, etc.)
def ExtensionFieldTypeInterface : TypeInterface<"ExtensionFieldTypeInterface", [
  DeclareTypeInterfaceMethods<FieldTypeInterface>
  ]> {
  let description = [{
    Interface for extension field types that allows generic handling
    of different extension degrees without explicit type checking.
  }];

  let methods = [
    InterfaceMethod<"Get extension degree over prime field",
      "unsigned", "getDegreeOverPrime">,
    InterfaceMethod<"Get extension degree over base field",
      "unsigned", "getDegreeOverBase">,
    InterfaceMethod<"Get the base field type",
      "mlir::Type", "getBaseFieldType">,
    InterfaceMethod<"Get the non-residue element",
      "mlir::Attribute", "getNonResidue">,
    InterfaceMethod<"Create a new type with different base field type and attribute",
      "mlir::Type", "cloneWith",
      (ins "mlir::Type":$baseField, "mlir::Attribute":$element)>,
    InterfaceMethod<"Create constant attribute from coefficient values",
      "mlir::TypedAttr", "createConstantAttr",
      (ins "llvm::ArrayRef<llvm::APInt>":$coeffs)>,
    InterfaceMethod<"Build LLVM struct from coefficients",
      "mlir::Value", "buildStructFromCoeffs",
      (ins "mlir::ImplicitLocOpBuilder &":$builder,
           "mlir::Type":$structType,
           "llvm::ArrayRef<mlir::Value>":$coeffs)>,
    InterfaceMethod<"Extract coefficients from LLVM struct",
      "llvm::SmallVector<mlir::Value>", "extractCoeffsFromStruct",
      (ins "mlir::ImplicitLocOpBuilder &":$builder,
           "mlir::Value":$structValue)>,
  ];
}

// Prime field type definition.
// This type represents an element in a prime field parameterized by a prime modulus.
def Field_PrimeFieldType : Field_Type<"PrimeField", "pf", [
  MemRefElementTypeInterface,
  VectorElementTypeInterface,
  DeclareTypeInterfaceMethods<FieldTypeInterface>,
  DeclareTypeInterfaceMethods<DataLayoutTypeInterface>
  ]> {
  let summary = "Prime field element type";
  let description = [{
    `field.pf<p>` represents an element in a prime field where arithmetic operations
    are performed modulo a prime modulus `p`.

    Example:
    ```
    !PF1 = !field.pf<7>
    ```
  }];
  let parameters = (ins
   "IntegerAttr": $modulus,
   DefaultValuedParameter<"bool", "false">: $isMontgomery
  );
  let builders = [
    TypeBuilder<(ins "IntegerAttr":$modulus), [{
      return $_get(context, modulus, false);
    }]>,
  ];
  let extraClassDeclaration = [{
    IntegerType getStorageType() const {
      // TODO(batzor): decouple the storage type from the modulus type
      return cast<IntegerType>(getModulus().getType());
    }
    unsigned getStorageBitWidth() const {
      return getStorageType().getWidth();
    }
  }];
  let hasCustomAssemblyFormat = 1;
}

// Quadratic extension field type definition.
// This type represents an element in a quadratic extension field parameterized by the underlying field.
// Currently, it is only defined for prime fields.
def Field_QuadraticExtFieldType : Field_Type<"QuadraticExtField", "f2", [
  MemRefElementTypeInterface,
  VectorElementTypeInterface,
  DeclareTypeInterfaceMethods<ExtensionFieldTypeInterface>,
  DeclareTypeInterfaceMethods<DataLayoutTypeInterface>
  ]> {
  let summary = "Quadratic extension field element type";
  let description = [{
    Represents an element in a quadratic extension field defined by the given
    prime field `baseField` and the non-residue `xi`.

    An element is represented as a₀ + a₁v where v² = ξ.

    Example:
    ```
    !PF = !field.pf<7:i32>
    #xi = #field.pf.elem<6:i32> : !PF
    !QF = !field.f2<!PF, #xi>
    ```
  }];
  let parameters = (ins
   Field_PrimeFieldType: $baseField,
   "PrimeFieldAttr": $nonResidue
  );
  let assemblyFormat = "`<` $baseField `,` $nonResidue `>`";
}

// Cubic extension field type definition.
// This type represents an element in a cubic extension field parameterized by the underlying field.
// Currently, it is only defined for prime fields.
def Field_CubicExtFieldType : Field_Type<"CubicExtField", "f3", [
  MemRefElementTypeInterface,
  VectorElementTypeInterface,
  DeclareTypeInterfaceMethods<ExtensionFieldTypeInterface>,
  DeclareTypeInterfaceMethods<DataLayoutTypeInterface>
  ]> {
  let summary = "Cubic extension field element type";
  let description = [{
    Represents an element in a cubic extension field defined by the given
    prime field `baseField` and the non-residue `xi`.

    An element is represented as a₀ + a₁v + a₂v² where v³ = ξ.

    Example:
    ```
    !PF = !field.pf<7:i32>
    #xi = #field.pf.elem<6:i32> : !PF
    !CF = !field.f3<!PF, #xi>
    ```
  }];
  let parameters = (ins
   Field_PrimeFieldType: $baseField,
   "PrimeFieldAttr": $nonResidue
  );
  let assemblyFormat = "`<` $baseField `,` $nonResidue `>`";
}

// Quartic extension field type definition.
// This type represents an element in a quartic extension field parameterized by the underlying field.
// Currently, it is only defined for prime fields.
def Field_QuarticExtFieldType : Field_Type<"QuarticExtField", "f4", [
  MemRefElementTypeInterface,
  VectorElementTypeInterface,
  DeclareTypeInterfaceMethods<ExtensionFieldTypeInterface>,
  DeclareTypeInterfaceMethods<DataLayoutTypeInterface>
  ]> {
  let summary = "Quartic extension field element type";
  let description = [{
    Represents an element in a quartic extension field defined by the given
    prime field `baseField` and the non-residue `xi`.

    An element is represented as a₀ + a₁v + a₂v² + a₃v³ where v⁴ = ξ.

    Example:
    ```
    !PF = !field.pf<7:i32>
    #xi = #field.pf.elem<6:i32> : !PF
    !QF = !field.f4<!PF, #xi>
    ```
  }];
  let parameters = (ins
   Field_PrimeFieldType: $baseField,
   "PrimeFieldAttr": $nonResidue
  );
  let assemblyFormat = "`<` $baseField `,` $nonResidue `>`";
}

def Field_AnyFieldType : AnyTypeOf<[Field_PrimeFieldType, Field_QuadraticExtFieldType, Field_CubicExtFieldType, Field_QuarticExtFieldType], "any-field-type">;
def Field_AnyExtFieldType : AnyTypeOf<[Field_QuadraticExtFieldType, Field_CubicExtFieldType, Field_QuarticExtFieldType], "any-ext-field-type">;

def PrimeFieldLike: TypeOrValueSemanticsContainer<Field_PrimeFieldType, "prime-field-like">;
def QuadraticExtFieldLike: TypeOrValueSemanticsContainer<Field_QuadraticExtFieldType, "quadratic-ext-field-like">;
def CubicExtFieldLike: TypeOrValueSemanticsContainer<Field_CubicExtFieldType, "cubic-ext-field-like">;
def QuarticExtFieldLike: TypeOrValueSemanticsContainer<Field_QuarticExtFieldType, "quartic-ext-field-like">;
def FieldLike : TypeConstraint<Or<[PrimeFieldLike.predicate, QuadraticExtFieldLike.predicate, CubicExtFieldLike.predicate, QuarticExtFieldLike.predicate]>, "field-like">;

#endif  // ZKIR_DIALECT_FIELD_IR_FIELDTYPES_TD_
