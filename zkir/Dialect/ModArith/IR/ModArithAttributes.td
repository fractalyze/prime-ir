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

#ifndef ZKIR_DIALECT_MODARITH_IR_MODARITHATTRS_TD_
#define ZKIR_DIALECT_MODARITH_IR_MODARITHATTRS_TD_

include "mlir/IR/BuiltinAttributeInterfaces.td"
include "zkir/Dialect/ModArith/IR/ModArithDialect.td"

class ModArith_Attr<string name, string attrMnemonic, list<Trait> traits = []>
  : AttrDef<ModArith_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def ModArith_MontgomeryAttr : ModArith_Attr<"Montgomery", "montgomery"> {
  let summary = "An attribute with precomputed Montgomery parameters";
  let description = [{
    This attribute contains the precomputed Montgomery parameters for a given
    modulus. It is used for efficient multiplication when the modulus is fixed
    at compile time.
  }];
  let parameters = (ins "IntegerAttr":$modulus);
  let extraClassDeclaration = [{
    IntegerAttr getNPrime() const;
    IntegerAttr getNInv() const;
    IntegerAttr getR() const;
    IntegerAttr getRInv() const;
    IntegerAttr getRSquared() const;
    IntegerAttr getBInv() const;
    const SmallVector<IntegerAttr> &getInvTwoPowers() const;
  }];
  let assemblyFormat = "`<`$modulus`>`";
  let genStorageClass = 0;
}

def ModArith_BYAttr : ModArith_Attr<"BY", "by"> {
  let summary = "An attribute with precomputed Bernstein-Yang inverse parameters";
  let description = [{
    This attribute contains the precomputed Bernstein-Yang inverse parameters
    for a given modulus. It is used for efficient inverse when the modulus is
    fixed at compile time.
  }];
  let parameters = (ins "IntegerAttr":$modulus);
  let extraClassDeclaration = [{
    IntegerAttr getDivsteps() const;
    IntegerAttr getMInv() const;
    IntegerAttr getNewBitWidth() const;
  }];
  let assemblyFormat = "`<`$modulus`>`";
  let genStorageClass = 0;
}

#endif  // ZKIR_DIALECT_MODARITH_IR_MODARITHATTRS_TD_
