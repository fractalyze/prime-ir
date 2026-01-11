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

#ifndef PRIME_IR_DIALECT_FIELD_IR_FIELDATTRS_TD_
#define PRIME_IR_DIALECT_FIELD_IR_FIELDATTRS_TD_

include "prime_ir/Dialect/Field/IR/FieldDialect.td"
include "prime_ir/Dialect/Field/IR/FieldTypes.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/OpBase.td"

class Field_Attr<string name, string attrMnemonic, list<Trait> traits = []>
  : AttrDef<Field_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def Field_RootOfUnityAttr : Field_Attr<"RootOfUnity", "root_of_unity", [TypedAttrInterface]> {
  let summary = "An attribute representing a root of unity in a prime field";
  let parameters = (ins "PrimeFieldType":$type, "IntegerAttr":$root, "IntegerAttr":$degree);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

// TODO(chokobole): Refactor this with a proper attribute type. Currently it's used in
// `EllipticCurveAttributes.td`, so it can't be used.
def Field_AnyFieldAttr : AnyAttrOf<[APIntAttr, AnyIntElementsAttr]> {
  string cppType = "mlir::TypedAttr";
}

#endif  // PRIME_IR_DIALECT_FIELD_IR_FIELDATTRS_TD_
