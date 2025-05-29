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
        IntegerAttr::get(type.getModulus().getType(), value));
    }]>,
    AttrBuilderWithInferredContext<(ins "PrimeFieldType":$type,
                                        "const uint64_t":$value), [{
      return $_get(
        type.getContext(),
        type,
        IntegerAttr::get(type.getModulus().getType(), value));
    }]>
  ];
  let extraClassDeclaration = [{
    using ValueType = Attribute;
  }];
  let genVerifyDecl = 1;
}

def Field_QuadraticExtFieldAttr : Field_Attr<"QuadraticExtField", "f2.elem", [TypedAttrInterface]> {
  let summary = "An attribute representing a quadratic extension field element";
  let parameters = (ins "QuadraticExtFieldType":$type, Field_PrimeFieldAttr:$low, Field_PrimeFieldAttr:$high);
  let assemblyFormat = "`<` $low `,` $high `>` `:` $type";
  let extraClassDeclaration = [{
    using ValueType = Attribute;
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

def Field_AnyFieldAttr : AnyAttrOf<[Field_PrimeFieldAttr, Field_QuadraticExtFieldAttr]> {
  string cppType = "::mlir::TypedAttr";
}

#endif  // ZKIR_DIALECT_FIELD_IR_FIELDATTRS_TD_
