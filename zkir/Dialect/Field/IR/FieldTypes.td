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

// Extension field type interface.
// Provides a unified interface for extension field types (Fp2, Fp3, etc.)
def ExtensionFieldTypeInterface : TypeInterface<"ExtensionFieldTypeInterface"> {
  let description = [{
    Interface for extension field types that allows generic handling
    of different extension degrees without explicit type checking.
  }];

  let methods = [
    InterfaceMethod<"Get the number of coefficients",
      "unsigned", "getNumCoefficients">,
    InterfaceMethod<"Build LLVM struct from coefficients",
      "::mlir::Value", "buildStructFromCoeffs",
      (ins "::mlir::ImplicitLocOpBuilder &":$builder,
           "::mlir::Type":$structType,
           "::llvm::ArrayRef<::mlir::Value>":$coeffs)>,
    InterfaceMethod<"Extract coefficients from LLVM struct",
      "::llvm::SmallVector<::mlir::Value>", "extractCoeffsFromStruct",
      (ins "::mlir::ImplicitLocOpBuilder &":$builder,
           "::mlir::Value":$structValue)>,
  ];
}

// Prime field type definition.
// This type represents an element in a prime field parameterized by a prime modulus.
def Field_PrimeFieldType : Field_Type<"PrimeField", "pf", [
  MemRefElementTypeInterface,
  VectorElementTypeInterface,
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
  let assemblyFormat = "`<` params `>`";
  let builders = [
    TypeBuilder<(ins "IntegerAttr":$modulus), [{
      return $_get(context, modulus, false);
    }]>,
  ];
  let extraClassDeclaration = [{
    bool isMontgomery() const {
      return getIsMontgomery();
    }
    IntegerType getStorageType() const {
      // TODO(batzor): decouple the storage type from the modulus type
      return cast<IntegerType>(getModulus().getType());
    }
    unsigned getStorageBitWidth() const {
      return getStorageType().getWidth();
    }
  }];
}

// Quadratic extension field type definition.
// This type represents an element in a quadratic extension field parameterized by the underlying field.
// Currently, it is only defined for prime fields.
def Field_QuadraticExtFieldType : Field_Type<"QuadraticExtField", "f2", [
  MemRefElementTypeInterface,
  VectorElementTypeInterface,
  ExtensionFieldTypeInterface,
  DeclareTypeInterfaceMethods<DataLayoutTypeInterface>
  ]> {
  let summary = "Quadratic extension field element type";
  let description = [{
    Represents an element in a quadratic extension field defined by the given
    prime field `baseField` and the irreducible binomial `beta`.

    Example:
    ```
    !PF = !field.pf<7:i32>
    #beta = #field.pf.elem<6:i32> : !PF
    !QF = !field.f2<!PF, #beta>
    ```
  }];
  let parameters = (ins
   Field_PrimeFieldType: $baseField,
   "PrimeFieldAttr": $beta
  );
  let assemblyFormat = "`<` $baseField `,` $beta `>`";
  let extraClassDeclaration = [{
    // Check if the field is in Montgomery form.
    bool isMontgomery() const {
      return getBaseField().getIsMontgomery();
    }

    // ExtensionFieldTypeInterface methods
    unsigned getNumCoefficients() {
      return 2;
    }

    ::mlir::Value buildStructFromCoeffs(::mlir::ImplicitLocOpBuilder &builder,
                                        ::mlir::Type structType,
                                        ::llvm::ArrayRef<::mlir::Value> coeffs) {
      return ::mlir::zkir::SimpleStructBuilder<2>::initialized(
          builder, builder.getLoc(), structType, coeffs);
    }

    ::llvm::SmallVector<::mlir::Value> extractCoeffsFromStruct(
        ::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value structValue) {
      ::mlir::zkir::SimpleStructBuilder<2> extFieldStruct(structValue);
      return extFieldStruct.getValues(builder, builder.getLoc());
    }
  }];
}

// Cubic extension field type definition.
// This type represents an element in a cubic extension field parameterized by the underlying field.
// Currently, it is only defined for prime fields.
def Field_CubicExtFieldType : Field_Type<"CubicExtField", "f3", [
  MemRefElementTypeInterface,
  VectorElementTypeInterface,
  ExtensionFieldTypeInterface,
  DeclareTypeInterfaceMethods<DataLayoutTypeInterface>
  ]> {
  let summary = "Cubic extension field element type";
  let description = [{
    Represents an element in a cubic extension field defined by the given
    prime field `baseField` and the irreducible binomial `xi`.

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
   "PrimeFieldAttr": $xi
  );
  let assemblyFormat = "`<` $baseField `,` $xi `>`";
  let extraClassDeclaration = [{
    // Check if the field is in Montgomery form.
    bool isMontgomery() {
      return getBaseField().getIsMontgomery();
    }

    // ExtensionFieldTypeInterface methods
    unsigned getNumCoefficients() {
      return 3;
    }

    ::mlir::Value buildStructFromCoeffs(::mlir::ImplicitLocOpBuilder &builder,
                                        ::mlir::Type structType,
                                        ::llvm::ArrayRef<::mlir::Value> coeffs) {
      return ::mlir::zkir::SimpleStructBuilder<3>::initialized(
          builder, builder.getLoc(), structType, coeffs);
    }

    ::llvm::SmallVector<::mlir::Value> extractCoeffsFromStruct(
        ::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value structValue) {
      ::mlir::zkir::SimpleStructBuilder<3> extFieldStruct(structValue);
      return extFieldStruct.getValues(builder, builder.getLoc());
    }
  }];
}

def Field_AnyFieldType : AnyTypeOf<[Field_PrimeFieldType, Field_QuadraticExtFieldType, Field_CubicExtFieldType], "any-field-type">;

def PrimeFieldLike: TypeOrValueSemanticsContainer<Field_PrimeFieldType, "prime-field-like">;
def QuadraticExtFieldLike: TypeOrValueSemanticsContainer<Field_QuadraticExtFieldType, "quadratic-ext-field-like">;
def CubicExtFieldLike: TypeOrValueSemanticsContainer<Field_CubicExtFieldType, "cubic-ext-field-like">;
def FieldLike : TypeConstraint<Or<[PrimeFieldLike.predicate, QuadraticExtFieldLike.predicate, CubicExtFieldLike.predicate]>, "field-like">;

#endif  // ZKIR_DIALECT_FIELD_IR_FIELDTYPES_TD_
