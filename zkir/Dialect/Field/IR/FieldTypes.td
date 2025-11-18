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
  }];
}

def Field_AnyFieldType : AnyTypeOf<[Field_PrimeFieldType, Field_QuadraticExtFieldType], "any-field-type">;

def PrimeFieldLike: TypeOrValueSemanticsContainer<Field_PrimeFieldType, "prime-field-like">;
def QuadraticExtFieldLike: TypeOrValueSemanticsContainer<Field_QuadraticExtFieldType, "quadratic-ext-field-like">;
def FieldLike : TypeConstraint<Or<[PrimeFieldLike.predicate, QuadraticExtFieldLike.predicate]>, "field-like">;

#endif  // ZKIR_DIALECT_FIELD_IR_FIELDTYPES_TD_
