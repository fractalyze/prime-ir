#ifndef ZKIR_DIALECT_MODARITH_IR_MODARITHATTRS_TD_
#define ZKIR_DIALECT_MODARITH_IR_MODARITHATTRS_TD_

include "mlir/IR/BuiltinAttributeInterfaces.td"
include "zkir/Dialect/ModArith/IR/ModArithDialect.td"

class ModArith_Attr<string name, string attrMnemonic, list<Trait> traits = []>
  : AttrDef<ModArith_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def ModArith_MontgomeryAttr : ModArith_Attr<"Montgomery", "montgomery"> {
  let summary = "Attribute with precomputed Montgomery parameters";
  let description = [{
    This attribute contains the precomputed Montgomery parameters for a given
    modulus. It is used to optimize Montgomery multiplication and
    other operations in the `mod_arith` dialect.
  }];
  let parameters = (ins "ModArithType":$modType);
  let extraClassDeclaration = [{
    IntegerAttr getNPrime() const;
    IntegerAttr getR() const;
    IntegerAttr getRInv() const;
    IntegerAttr getRSquared() const;
  }];
  let assemblyFormat = "`<`$modType`>`";
  let genStorageClass = 0;
}

#endif  // ZKIR_DIALECT_MODARITH_IR_MODARITHATTRS_TD_
