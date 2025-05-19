#ifndef ZKIR_DIALECT_FIELD_IR_FIELDTYPES_TD_
#define ZKIR_DIALECT_FIELD_IR_FIELDTYPES_TD_

include "zkir/Dialect/Field/IR/FieldDialect.td"

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
def Field_PrimeFieldType : Field_Type<"PrimeField", "pf", [MemRefElementTypeInterface]> {
  let summary = "Prime field element type";
  let description = [{
    `field.pf<p>` represents an element in a prime field where arithmetic operations
    are performed modulo a prime modulus `p`.

    Example:
    ```
    !PF1 = !field.pf<7>
    ```
  }];
  let parameters = (ins "IntegerAttr": $modulus);
  let assemblyFormat = "`<` $modulus `>`";
}

// Quadratic extension field type definition.
// This type represents an element in a quadratic extension field parameterized by the underlying field.
// Currently, it is only defined for prime fields.
def Field_QuadraticExtFieldType : Field_Type<"QuadraticExtField", "f2", [
  MemRefElementTypeInterface]> {
  let summary = "Quadratic extension field element type";
  let description = [{
    Represents an element in a quadratic extension field defined by the given
    prime field `base_field` and the irreducible binomial `beta`.

    Example:
    ```
    !PF = !field.pf<7:i32>
    #beta = #field.pf_elem<6:i32> : !PF
    !QF = !field.f2<!PF, #beta>
    ```
  }];
  let parameters = (ins
   Field_PrimeFieldType: $base_field,
   "PrimeFieldAttr": $beta
  );
  let assemblyFormat = "`<` $base_field `,` $beta `>`";
}

def PrimeFieldLike: TypeOrValueSemanticsContainer<Field_PrimeFieldType, "prime-field-like">;
def QuadraticExtFieldLike: TypeOrValueSemanticsContainer<Field_QuadraticExtFieldType, "quadratic-ext-field-like">;
def FieldLike : TypeConstraint<Or<[PrimeFieldLike.predicate, QuadraticExtFieldLike.predicate]>, "field-like">;

#endif  // ZKIR_DIALECT_FIELD_IR_FIELDTYPES_TD_
