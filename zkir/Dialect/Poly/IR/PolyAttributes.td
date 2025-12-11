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

#ifndef ZKIR_DIALECT_POLY_IR_POLYATTRIBUTES_TD_
#define ZKIR_DIALECT_POLY_IR_POLYATTRIBUTES_TD_

include "zkir/Dialect/Poly/IR/PolyDialect.td"
include "zkir/Dialect/Poly/IR/PolyTypes.td"

class Poly_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Poly_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def Poly_PrimitiveRootAttr: Poly_Attr<"PrimitiveRoot", "primitive_root"> {
  let summary = "an attribute containing a primitive root of unity and its twiddles";
  let description = [{
    A primitive root attribute stores a primitive root of unity specified by the `RootOfUnityAttr`
    attribute and also some other necessary precomputed values for (I)NTT operations.

    Example:
    ```
    !PF1 = !field.pf<7:i32>
    #root_of_unity = #field.root_of_unity<root=2:i32, degree=3> : !PF1
    #root = #poly.primitive_root<#root_of_unity>
    ```
  }];
  let extraClassDeclaration = [{
    IntegerAttr getDegree() const;
    IntegerAttr getRoot() const;
    IntegerAttr getInvRoot() const;
    IntegerAttr getInvDegree() const;
    DenseElementsAttr getRoots() const;
    DenseElementsAttr getInvRoots() const;
  }];
  let parameters = (ins Field_RootOfUnityAttr:$rootOfUnity, OptionalParameter<"mod_arith::MontgomeryAttr">:$montgomery);
  let assemblyFormat = "`<` params `>`";
  let genStorageClass = 0;
}

#endif  // ZKIR_DIALECT_POLY_IR_POLYATTRIBUTES_TD_
