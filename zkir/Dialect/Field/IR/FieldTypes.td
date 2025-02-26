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

    Examples:
    ```
    !PF1 = !field.pf<7>
    ```
  }];
  let parameters = (ins
    "IntegerAttr": $modulus
  );
  let assemblyFormat = "`<` $modulus `>`";
}

def PrimeFieldLike: TypeOrValueSemanticsContainer<Field_PrimeFieldType, "prime-field-like">;

#endif  // ZKIR_DIALECT_FIELD_IR_FIELDTYPES_TD_
