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

#ifndef PRIME_IR_DIALECT_FIELD_IR_FIELDTYPES_TD_
#define PRIME_IR_DIALECT_FIELD_IR_FIELDTYPES_TD_

include "prime_ir/Dialect/Field/IR/FieldDialect.td"
include "prime_ir/Dialect/Field/IR/FieldAttributes.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/ConstantLikeInterface.td"

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
  ];
}

// Prime field type definition.
// This type represents an element in a prime field parameterized by a prime modulus.
def Field_PrimeFieldType : Field_Type<"PrimeField", "pf", [
  MemRefElementTypeInterface,
  VectorElementTypeInterface,
  DeclareTypeInterfaceMethods<FieldTypeInterface>,
  DeclareTypeInterfaceMethods<DataLayoutTypeInterface>,
  DeclareTypeInterfaceMethods<ConstantLikeInterface>
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
      APInt mod = getModulus().getValue();
      if (mod.isPowerOf2()) {
        // Binary field: 2ⁿ needs n bits for storage (not n + 1)
        unsigned storageBits = mod.logBase2();
        return IntegerType::get(getContext(), storageBits);
      }
      // Prime field: use modulus type as-is
      return cast<IntegerType>(getModulus().getType());
    }
    unsigned getStorageBitWidth() const {
      return getStorageType().getWidth();
    }
  }];
  let hasCustomAssemblyFormat = 1;
}

// Unified extension field type definition.
// This type represents an element in an extension field of degree N (2, 3, or 4).
// Supports tower extensions where base field can be either a prime field or
// another extension field.
def Field_ExtensionFieldType : Field_Type<"ExtensionField", "ef", [
  MemRefElementTypeInterface,
  VectorElementTypeInterface,
  DeclareTypeInterfaceMethods<FieldTypeInterface>,
  DeclareTypeInterfaceMethods<DataLayoutTypeInterface>,
  DeclareTypeInterfaceMethods<ConstantLikeInterface>
  ]> {
  let summary = "Extension field element type";
  let description = [{
    Represents an element in an extension field of degree N (2, 3, or 4)
    defined by the given base field and the non-residue `nonResidue`.
    The base field can be either a prime field or another extension field,
    allowing for tower constructions.

    An element is represented as a₀ + a₁v + ... + aₙ₋₁vⁿ⁻¹ where vⁿ = nonResidue.

    Example:
    ```
    !PF = !field.pf<7:i32>
    !EF2 = !field.ef<2x!PF, 6:i32>      // quadratic extension over prime
    !EF3 = !field.ef<3x!PF, 2:i32>      // cubic extension over prime
    !EF4 = !field.ef<4x!PF, 2:i32>      // quartic extension over prime
    !EF6 = !field.ef<3x!EF2, 2:i32>     // tower: Fp6 = (Fp2)^3
    !EF12 = !field.ef<2x!EF6, 2:i32>    // tower: Fp12 = (Fp6)^2
    ```
  }];
  let parameters = (ins
    "unsigned": $degree,
    "Type": $baseField,  // Can be PrimeFieldType or ExtensionFieldType
    "IntegerAttr": $nonResidue
  );
  let extraClassDeclaration = [{
    // Returns the total degree over the underlying prime field.
    // For Fp2 over Fp: returns 2
    // For Fp6 = (Fp2)^3: returns 6
    // For Fp12 = (Fp6)^2 = ((Fp2)^3)^2: returns 12
    unsigned getDegreeOverPrime() const;

    // Returns the underlying prime field at the base of the tower.
    PrimeFieldType getBasePrimeField() const;

    // Returns true if this is a tower extension (base field is an extension).
    bool isTower() const;

    // Returns the tower depth (1 for direct extension over prime).
    unsigned getTowerDepth() const;

    Type cloneWith(Type baseField, Attribute element) const;
    Value buildStructFromCoeffs(ImplicitLocOpBuilder &builder,
                                Type structType,
                                ArrayRef<Value> coeffs) const;
    SmallVector<Value> extractCoeffsFromStruct(
        ImplicitLocOpBuilder &builder, Value structValue) const;
  }];
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

def Field_AnyFieldType : AnyTypeOf<[Field_PrimeFieldType, Field_ExtensionFieldType], "any-field-type">;

def PrimeFieldLike: TypeOrValueSemanticsContainer<Field_PrimeFieldType, "prime-field-like">;
def ExtensionFieldLike: TypeOrValueSemanticsContainer<Field_ExtensionFieldType, "extension-field-like">;
def FieldLike : TypeConstraint<Or<[PrimeFieldLike.predicate, ExtensionFieldLike.predicate]>, "field-like">;

#endif  // PRIME_IR_DIALECT_FIELD_IR_FIELDTYPES_TD_
