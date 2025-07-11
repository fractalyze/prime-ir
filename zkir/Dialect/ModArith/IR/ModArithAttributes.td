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
    IntegerAttr getR() const;
    IntegerAttr getRInv() const;
    IntegerAttr getRSquared() const;
    IntegerAttr getBInv() const;
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
