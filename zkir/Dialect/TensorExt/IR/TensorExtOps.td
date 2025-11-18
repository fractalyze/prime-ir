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

def TensorExt_BitReverseOp : TensorExt_Op<"bit_reverse", [Pure, DestinationStyleOpInterface, SameOperandsAndResultType]> {
  let summary = "Bit-reverse the given order.";
  let description = [{
    Transforms a tensor to its bit-reversed order.

    Example:
    ```
    %1 = tensor_ext.bit_reverse %0 : tensor<16xi32>
    ```
  }];

  let arguments = (ins AnyRankedTensor:$source, AnyRankedTensor:$dest);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "$source `into` $dest attr-dict `:` type($output)";
  let extraClassDeclaration = [{
    MutableOperandRange getDpsInitsMutable() { return getDestMutable(); }
  }];
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

#endif  // ZKIR_DIALECT_TENSOREXT_IR_TENSOREXTOPS_TD_
