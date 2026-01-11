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

#ifndef PRIME_IR_DIALECT_MODARITH_IR_MODARITHTYPES_TD_
#define PRIME_IR_DIALECT_MODARITH_IR_MODARITHTYPES_TD_

include "prime_ir/Dialect/ModArith/IR/ModArithDialect.td"

include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/AttrTypeBase.td"

class ModArith_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<ModArith_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def ModArith_ModArithType : ModArith_Type<"ModArith", "int", [
  MemRefElementTypeInterface,
  VectorElementTypeInterface,
  DeclareTypeInterfaceMethods<DataLayoutTypeInterface>
  ]> {
  let summary = "Integer type with modular arithmetic";
  let description = [{
    `mod_arith.int<p>` represents an element of the ring of integers modulo $p$.
    The `modulus` attribute is the ring modulus, and `mod_arith` operations lower to
    `arith` operations that produce results in the range `[0, modulus)`, often called
    the _canonical representative_.

    `modulus` is specified with an integer type suffix, for example,
    `mod_arith.int<65537 : i32>`. This corresponds to the storage type for the
    modulus, and is the smallest power of two integer type that can represent the modulus.

    Passes may allow intermediate values that do not always produce a
    canonical representative in `[0, modulus)`. For example, if the machine storage
    type is `i64`, but the `modulus` fits within an `i32`, a lowering could
    allow intermediate arithmetic values to grow to as large as an `i64` before
    reducing them. However, all passes must ensure that values used outside
    the local scope (e.g., function return values or arguments to calls to linked
    functions) are appropriately reduced to the canonical representative.
    `modulus` is the modulus the arithmetic working with.

    Example:
    ```
    !Zp1 = !mod_arith.int<7> // implicitly being i4
    !Zp2 = !mod_arith.int<65537 : i32>
    !Zp3 = !mod_arith.int<536903681 : i64>
    ```
  }];
  let parameters = (ins
   "IntegerAttr":$modulus,
    DefaultValuedParameter<"bool", "false">:$isMontgomery
  );
  let builders = [
    TypeBuilder<(ins "IntegerAttr":$modulus), [{
      return $_get(context, modulus, false);
    }]>,
  ];
  let extraClassDeclaration = [{
    bool isMontgomery() const {
      return getIsMontgomery();
    }
    MontgomeryAttr getMontgomeryAttr() const {
      return MontgomeryAttr::get(getContext(), getModulus());
    }
    BYAttr getBYAttr() const {
      return BYAttr::get(getContext(), getModulus());
    }
    IntegerType getStorageType() const {
      APInt mod = getModulus().getValue();
      if (mod.isPowerOf2()) {
        // Binary field: 2ⁿ needs n bits for storage (not n + 1)
        unsigned storageBits = mod.logBase2();
        return IntegerType::get(getContext(), storageBits);
      }
      // Prime field: use modulus type as-is
      return cast<IntegerType>(getModulus().getType());
    }
    unsigned getStorageBitWidth() const {
      return getStorageType().getWidth();
    }
  }];
  let hasCustomAssemblyFormat = 1;
}

def ModArithLike: TypeOrValueSemanticsContainer<ModArith_ModArithType, "mod_arith-like">;

#endif  // PRIME_IR_DIALECT_MODARITH_IR_MODARITHTYPES_TD_
