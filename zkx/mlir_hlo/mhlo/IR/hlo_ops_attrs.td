/* Copyright 2021 The OpenXLA Authors.
Copyright 2025 The ZKX Authors.

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

#ifndef ZKX_MLIR_HLO_MHLO_IR_HLO_OPS_ATTRS_TD
#define ZKX_MLIR_HLO_MHLO_IR_HLO_OPS_ATTRS_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/Interfaces.td"
include "mlir/IR/TensorEncoding.td"

def MHLO_Dims : ArrayRefParameter<"int64_t", "Dimension"> {
  let parser = "parseDimSizes($_parser)";
  let printer = "printDimSizes($_printer, $_self)";
}

def MHLO_DotDimensionNumbers : AttrDef<MHLO_Dialect, "DotDimensionNumbers"> {
  let mnemonic = "dot";
  let summary = "Attribute that models the dimension information for dot.";
  let parameters = (ins
      MHLO_Dims:$lhsBatchingDimensions,
      MHLO_Dims:$rhsBatchingDimensions,
      MHLO_Dims:$lhsContractingDimensions,
      MHLO_Dims:$rhsContractingDimensions
  );
  let hasCustomAssemblyFormat = 1;
}

def MHLO_ScatterDimensionNumbers : AttrDef<MHLO_Dialect, "ScatterDimensionNumbers"> {
  let mnemonic = "scatter";
  let summary = "Attribute that models the dimension information for scatter";
  let parameters = (ins
      MHLO_Dims:$updateWindowDims,
      MHLO_Dims:$insertedWindowDims,
      MHLO_Dims:$inputBatchingDims,
      MHLO_Dims:$scatterIndicesBatchingDims,
      MHLO_Dims:$scatterDimsToOperandDims,
      "int64_t":$indexVectorDim
  );
  let hasCustomAssemblyFormat = 1;
}

def MHLO_GatherDimensionNumbers : AttrDef<MHLO_Dialect, "GatherDimensionNumbers"> {
  let mnemonic = "gather";
  let summary = "Attribute that models the dimension information for gather";
  let parameters = (ins
      MHLO_Dims:$offsetDims,
      MHLO_Dims:$collapsedSliceDims,
      MHLO_Dims:$operandBatchingDims,
      MHLO_Dims:$startIndicesBatchingDims,
      MHLO_Dims:$startIndexMap,
      "int64_t":$indexVectorDim
  );
  let hasCustomAssemblyFormat = 1;
}

// Note: This is an experimental attribute and shouldn't be relied upon for
// production.
def MHLO_TypeExtensions : AttrDef<MHLO_Dialect, "TypeExtensions", [
    DeclareAttrInterfaceMethods<VerifiableTensorEncoding>,
    DeclareAttrInterfaceMethods<HLO_BoundedAttrInterface>]> {
  let mnemonic = "type_extensions";

  // TODO(b/238903065): Move sparsity related info here from the standalone
  // attribute. That will allow composition of bounds and sparsity info.
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$bounds
  );

  let summary = "Attribute that extends tensor type with MHLO type properties.";

  let description = [{
    This attribute is used to extend MLIR tensor type with MHLO tensor specific
    properties. These properties aren't modeled in the MLIR type. This
    attribute is set in the `encoding` field of the tensor type.

    See `HLO_BoundedAttrInterface` for documentation for `bounds`.
  }];

  let assemblyFormat = "`<` `bounds` `=` ` ` custom<DimSizes>($bounds) `>`";
}

#endif // ZKX_MLIR_HLO_MHLO_IR_HLO_OPS_ATTRS_TD
