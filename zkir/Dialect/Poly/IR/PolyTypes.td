#ifndef ZKIR_DIALECT_POLY_IR_POLYTYPES_TD_
#define ZKIR_DIALECT_POLY_IR_POLYTYPES_TD_

include "zkir/Dialect/Field/IR/FieldTypes.td"
include "zkir/Dialect/Poly/IR/PolyDialect.td"

class Poly_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Poly_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Poly_PolyType : Poly_Type<"Poly", "polynomial", [MemRefElementTypeInterface]> {
  let summary = "A type representing a polynomial ring.";
  let description = [{
    A type for polynomials over a polynomial ring or quotient ring.
  }];
  let parameters = (ins Field_PrimeFieldType:$baseField, "IntegerAttr":$maxDegree);
  let assemblyFormat = "`<` $baseField `,` $maxDegree `>`";
}

def PolyLike: TypeOrValueSemanticsContainer<Poly_PolyType, "poly-like">;

#endif  // ZKIR_DIALECT_POLY_IR_POLYTYPES_TD_
