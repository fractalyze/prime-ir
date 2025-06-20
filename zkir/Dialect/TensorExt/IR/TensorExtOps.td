#ifndef ZKIR_DIALECT_TENSOREXT_IR_TENSOREXTOPS_TD_
#define ZKIR_DIALECT_TENSOREXT_IR_TENSOREXTOPS_TD_

include "zkir/Dialect/TensorExt/IR/TensorExtDialect.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"


class TensorExt_Op<string mnemonic, list<Trait> traits = []> :
        Op<TensorExt_Dialect, mnemonic, traits> {
  let cppNamespace = "::mlir::zkir::tensor_ext";
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def TensorExt_BitReverseOp : TensorExt_Op<"bit_reverse", [Pure, Involution, DestinationStyleOpInterface, SameOperandsAndResultType]> {
  let summary = "Bit-reverse the given order.";
  let description = [{
    Transforms a tensor to its bit-reversed order.

    Example:
    ```
    %1 = tensor_ext.bit_reverse %0 : tensor<16xi32>
    ```
  }];

  let arguments = (ins AnyRankedTensor:$input);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "operands attr-dict `:` type($output)";
  let extraClassDeclaration = [{
    MutableOperandRange getDpsInitsMutable() { return getInputMutable(); }
  }];
  let hasFolder = 1;
}

#endif  // ZKIR_DIALECT_TENSOREXT_IR_TENSOREXTOPS_TD_
