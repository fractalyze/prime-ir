#ifndef ZKIR_DIALECT_TENSOREXT_IR_TENSOREXTATTRIBUTES_TD_
#define ZKIR_DIALECT_TENSOREXT_IR_TENSOREXTATTRIBUTES_TD_

include "zkir/Dialect/TensorExt/IR/TensorExtDialect.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

class TensorExt_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<TensorExt_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def TensorExt_BitReverseIndicesAttr : TensorExt_Attr<"BitReverseIndices", "bit_reverse_indices"> {
  let summary = "The attribute representing the indices for bit-reversal";
  let description = [{
    A bit-reverse indices attribute stores the indices for bit-reversal.
  }];
  let parameters = (ins "IntegerAttr":$bitWidth);
  let assemblyFormat = "`<` $bitWidth `>`";
  let genStorageClass = 0;
  let extraClassDeclaration = [{
    DenseIntElementsAttr getIndices() const;
    Type getIndicesType() const;
  }];
  let builders = [
    AttrBuilderWithInferredContext<(ins "IntegerAttr":$bitWidth), [{
      return $_get(bitWidth.getContext(), bitWidth);
    }]>
  ];
}

#endif  // ZKIR_DIALECT_TENSOREXT_IR_TENSOREXTATTRIBUTES_TD_
