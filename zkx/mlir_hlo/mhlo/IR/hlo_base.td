/* Copyright 2019 The OpenXLA Authors.
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

#ifndef ZKX_MLIR_HLO_MHLO_IR_HLO_BASE_TD
#define ZKX_MLIR_HLO_MHLO_IR_HLO_BASE_TD

include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/Constraints.td"
include "stablehlo/dialect/Base.td"

//===----------------------------------------------------------------------===//
// MHLO type definitions.
//===----------------------------------------------------------------------===//

defvar MHLO_Pred = HLO_Pred;

defvar MHLO_SInt = HLO_SInt;
defvar MHLO_UInt = HLO_UInt;
defvar MHLO_Int = HLO_Int;
defvar MHLO_Field = HLO_Field;

// The broadcasting dimensions correspond to a tuple that describes how a
// smaller rank shape is broadcast into a larger rank shape. For example,
// given a 2x3x4 cuboid and a 3x4 matrix, a broadcasting tuple (1,2) means
// matching the matrix to dimensions 1 and 2 of the cuboid.
defvar MHLO_BroadcastDimAttr = I64ElementsAttr;

// Token type.
defvar MHLO_Token = HLO_Token;

// Any integer tensor types
defvar MHLO_IntTensor = HLO_IntTensor;

// Any integer tensor type with rank 0 (i.e. representing a single integer).
defvar MHLO_ScalarIntTensor = HLO_ScalarIntTensor;

defvar MHLO_IntOrFieldTensor = HLO_IntOrFieldTensor;

defvar MHLO_FieldTensor = RankedTensorOf<[HLO_Field]>;

defvar MHLO_PredTensor = HLO_PredTensor;

defvar MHLO_Tensor = HLO_Tensor;

defvar MHLO_AnyTensor = HLO_AnyTensor;

defvar MHLO_Tuple = HLO_Tuple;

defvar MHLO_TensorOrToken = HLO_TensorOrToken;

defvar MHLO_TensorOrTokenOrTuple = AnyTypeOf<[MHLO_Tensor, MHLO_Token, MHLO_Tuple]>;

defvar MHLO_DimensionValue = HLO_DimensionValue;

// Dynamic representation of a shape vector as a tensor.
defvar MHLO_DimensionTensor = HLO_DimensionTensor;

//===----------------------------------------------------------------------===//
// MHLO combined type definitions.
//===----------------------------------------------------------------------===//

// Any integer or predicate tensor types
defvar MHLO_PredOrIntTensor = HLO_PredOrIntTensor;

//===----------------------------------------------------------------------===//
// MHLO static shape type definitions.
//===----------------------------------------------------------------------===//

// In general, static shaped tensor constraints should be avoided unless
// it is for a legacy op which is only correct with static shapes.
defvar MHLO_StaticShapeTensor = HLO_StaticShapeTensor;

defvar MHLO_StaticShapeOrBoundedDimTensor = HLO_StaticShapeOrBoundedDimTensor;

defvar MHLO_StaticShapeTensorOrToken = HLO_StaticShapeTensorOrToken;

defvar MHLO_StaticShapeIntTensor = HLO_StaticShapeIntTensor;

defvar MHLO_StaticShapeIntOrFieldTensor = HLO_StaticShapeIntOrFieldTensor;

defvar MHLO_Static1DIntTensor = HLO_Static1DIntTensor;

#endif // ZKX_MLIR_HLO_MHLO_IR_HLO_BASE_TD
