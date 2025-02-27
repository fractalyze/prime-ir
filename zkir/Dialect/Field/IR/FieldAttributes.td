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

def Field_PrimeFieldAttr : Field_Attr<"PrimeField", "pf", [TypedAttrInterface]> {
  let summary = "Attribute representing the prime field element";
  let parameters = (ins "PrimeFieldType":$type, "IntegerAttr":$value);
  let assemblyFormat = "$value `:` $type";
  let builders = [
    AttrBuilderWithInferredContext<(ins "PrimeFieldType":$type,
                                        "const llvm::APInt":$value), [{
      return $_get(
        type.getContext(),
        type,
        IntegerAttr::get(type.getModulus().getType(), value));
    }]>
  ];
  let extraClassDeclaration = [{
    using ValueType = Attribute;
  }];
}

#endif  // ZKIR_DIALECT_FIELD_IR_FIELDATTRS_TD_
