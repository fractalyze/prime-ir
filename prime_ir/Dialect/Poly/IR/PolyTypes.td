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

#ifndef PRIME_IR_DIALECT_POLY_IR_POLYTYPES_TD_
#define PRIME_IR_DIALECT_POLY_IR_POLYTYPES_TD_

include "prime_ir/Dialect/Field/IR/FieldTypes.td"
include "prime_ir/Dialect/Poly/IR/PolyDialect.td"

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

#endif  // PRIME_IR_DIALECT_POLY_IR_POLYTYPES_TD_
