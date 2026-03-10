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

// This is the operation definition file for MHLO ops.

#ifndef ZKX_MLIR_HLO_MHLO_IR_HLO_OPS_TD
#define ZKX_MLIR_HLO_MHLO_IR_HLO_OPS_TD

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "zkx/mlir_hlo/mhlo/IR/hlo_utils.td"
include "zkx/mlir_hlo/mhlo/IR/hlo_ops_common.td"

class MHLO_Op<string mnemonic, list<Trait> traits> :
    Op<MHLO_Dialect, mnemonic, traits> {
  // Whether this operation has a custom conversion to HLO or not.
  bit hasCustomHLOConverter = 0b0;

  let extraClassDeclaration = [{
    // Relax the strict default implementation with one that allows
    // for StableHLO-specific differences.
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      return mlir::hlo::isCompatibleForHloTypeInference(l, r);
    }
  }];
}

class MHLO_ShapedInterfaceOp<string mnemonic, list<Trait> traits> :
    MHLO_Op<mnemonic, traits # [DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
    ["reifyReturnTypeShapes"]>]> {
  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      return mlir::hlo::isCompatibleForHloTypeInference(l, r);
    }
  }];
}

//===----------------------------------------------------------------------===//
// MHLO nullary op definitions.
//===----------------------------------------------------------------------===//

def MHLO_ConstantOp : MHLO_Op<"constant",
    [ConstantLike, Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Constant operation";
  let description = [{
    Produces an `output` tensor from a constant `value`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#constant

    Example:
    ```mlir
    %output = mhlo.constant dense<[[0, 1], [2, 3]]> : tensor<2x2xi32>
    ```
  }];
  let arguments = (ins
    ElementsAttr:$value
  );

  let results = (outs
    MHLO_StaticShapeTensor:$output
  );

  let builders = [
    OpBuilder<(ins "Attribute":$value)>];

  let hasCustomAssemblyFormat = 1;

  // Constant has special conversion logic to HLO.
  let hasCustomHLOConverter = 1;

  let hasFolder = 1;

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
  }];
}

def MHLO_IotaOp : MHLO_Op<"iota", [Pure]> {
  let summary = "Iota operation";
  let description = [{
    Fills an `output` tensor with values in increasing order starting from zero
    along the `iota_dimension` dimension.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#iota

    Example:
    ```mlir
    %output = mhlo.iota dim = 0 : tensor<4x5xi32>
    ```
  }];
  let arguments = (ins
    ConfinedAttr<I64Attr, [IntNonNegative]>:$iota_dimension
  );

  let results = (outs MHLO_StaticShapeIntOrFieldTensor:$output);

  // TODO(b/130357376): Iota has special conversion logic to HLO.
  let hasCustomHLOConverter = 1;
  let hasCanonicalizer = 1;
  let hasFolder = 1;
  let hasVerifier = 1;
}

def MHLO_DynamicIotaOp: MHLO_ShapedInterfaceOp<"dynamic_iota", [Pure]> {
  let summary = "DynamicIota operation";
  let description = [{
    This operation is functionally identical to
    [iota](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#iota)
    op, but the result shape is specified dynamically via `output_shape`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#dynamic_iota

    Example:
    ```mlir
    %0 = mhlo.dynamic_iota %arg0, dim = 0 : (tensor<1xindex>) -> tensor<4xi32>
    ```
  }];

  let arguments = (ins
    MHLO_DimensionTensor:$output_shape,
    ConfinedAttr<I64Attr, [IntNonNegative]>:$iota_dimension
  );
  let results = (outs MHLO_Tensor:$result);

  let hasCanonicalizer = 1;
  // Cannot be exported to legacy formats.
  let hasCustomHLOConverter = 1;
}

def MHLO_CreateTokenOp : MHLO_Op<"create_token", [Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "CreateToken operation";
  let description = [{
    This operation is on its way out of StableHLO, so it is not included in
    the specification: https://github.com/openxla/stablehlo/issues/3.

    Informally, this operation does the same thing as AfterAllOp with 0 inputs:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#after_all

    Example:
    ```mlir
    %output = mhlo.create_token : !mhlo.token
    ```
  }];

  let results = (outs MHLO_Token:$output);

  let assemblyFormat = "attr-dict `:` type(results)";
}

//===----------------------------------------------------------------------===//
// MHLO unary elementwise op definitions.
//===----------------------------------------------------------------------===//
// See https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions

class MHLO_UnaryElementwiseOp<string mnemonic, list<Trait> traits,
    Type OperandType, Type ResultType = OperandType> : MHLO_Op<mnemonic, traits # [Elementwise,
    InferShapedTypeOpInterface, SameOperandsAndResultShape]> {
  let arguments = (ins OperandType:$operand);
  let results = (outs ResultType:$result);
  let extraClassDeclaration = [{
    LogicalResult reifyReturnTypeShapes(
        OpBuilder& builder, ValueRange operands,
        SmallVectorImpl<Value>& reifiedReturnShapes) {
      return ::mlir::hlo::deriveShapeFromOperand(&builder, getOperation(),
                                                operands.front(),
                                                &reifiedReturnShapes);
    }
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      return mlir::hlo::isCompatibleForHloTypeInference(l, r);
    }
  }];

  let assemblyFormat = [{
    $operand attr-dict
      `:` custom<SameOperandsAndResultType>(type($operand), type($result))
  }];
}

def MHLO_AbsOp: MHLO_UnaryElementwiseOp<"abs",
    [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>],
     RankedTensorOf<[MHLO_SInt]>,
     RankedTensorOf<[MHLO_SInt]>> {
  let summary = "Abs operation";
  let description = [{
    Performs element-wise abs operation on `operand` tensor and produces a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#abs

    Example:
    ```mlir
    %result = mhlo.abs %operand : tensor<3xi32>
    ```
  }];
  let hasFolder = 1;
}

def MHLO_ClzOp: MHLO_UnaryElementwiseOp<"count_leading_zeros",
    [Pure, HLO_CompatibleOperandsAndResultType], MHLO_IntTensor> {
  let summary = "Clz operation";
  let description = [{
    Performs element-wise count of the number of leading zero bits in the
    `operand` tensor and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#count_leading_zeros

    Example:
    ```mlir
    %result = mhlo.count_leading_zeros %operand : tensor<2x2xi8>
    ```
  }];
}

def MHLO_ConvertOp : MHLO_UnaryElementwiseOp<"convert",
    [Pure, SameOperandsAndResultShape], MHLO_Tensor> {
  let summary = "Convert operation";
  let description = [{
    Performs an element-wise conversion from one element type to another on
    `operand` tensor and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#convert

    Example:
    ```mlir
    %result = mhlo.convert %operand : (tensor<3xi32>) -> tensor<3xu32>
    ```
  }];
  let builders = [
    OpBuilder<(ins "Value":$operand, "Type":$result_element_ty)>];

  let hasFolder = 1;

  let hasCanonicalizer = 1;

  let hasCustomHLOConverter = 1;
}

def MHLO_NegOp: MHLO_UnaryElementwiseOp<"negate",
    [Pure, HLO_CompatibleOperandsAndResultType], MHLO_IntOrFieldTensor> {
  let summary = "Neg operation";
  let description = [{
    Performs element-wise negation of `operand` tensor and produces a `result`
    tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#negate

    Example:
    ```mlir
    %result = mhlo.negate %operand : tensor<2x3xi32>
    ```
  }];
  let hasFolder = 1;
}

def MHLO_InverseOp: MHLO_UnaryElementwiseOp<"inverse",
    [Pure, HLO_CompatibleOperandsAndResultType], MHLO_FieldTensor> {
  let summary = "Inverse operation";
  let description = [{
    Performs element-wise multiplicative inverse of `operand` tensor and
    produces a `result` tensor.
  }];
}

def MHLO_NotOp: MHLO_UnaryElementwiseOp<"not",
    [Pure, HLO_CompatibleOperandsAndResultType], MHLO_PredOrIntTensor> {
  let summary = "Not operation";
  let description = [{
    Performs element-wise NOT of tensor `operand` of type integer and produces
    a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#not

    Example:
    ```mlir
    %result = mhlo.not %operand : tensor<5x3x1xi1>
    ```
  }];
  let hasFolder = 1;
}

def MHLO_PopulationCountOp: MHLO_UnaryElementwiseOp<"popcnt",
    [Pure, HLO_CompatibleOperandsAndResultType], MHLO_IntTensor> {
  let summary = "PopulationCount operation";
  let description = [{
    Performs element-wise count of the number of bits set in the `operand`
    tensor and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#popcnt

    Example:
    ```mlir
    %result = mhlo.popcnt %operand : tensor<4xi8>
    ```
  }];
}

def MHLO_SignOp: MHLO_UnaryElementwiseOp<"sign",
    [Pure, HLO_CompatibleOperandsAndResultType],
    RankedTensorOf<[MHLO_SInt]>> {
  let summary = "Sign operation";
  let description = [{
    Returns the sign of the `operand` element-wise and produces a `result`
    tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#sign

    Example:
    ```mlir
    %result = mhlo.sign %operand : tensor<7xi32>
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// MHLO binary elementwise op definitions.
//===----------------------------------------------------------------------===//
// See https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations

class MHLO_BinaryElementwiseOp<string mnemonic, list<Trait> traits,
    Type OperandType = MHLO_Tensor, Type ResultType = OperandType> :
    MHLO_Op<mnemonic, traits # [InferShapedTypeOpInterface,
    SameOperandsAndResultShape, Elementwise]> {
  let arguments = (ins
    OperandType:$lhs,
    OperandType:$rhs
  );

  let extraClassDeclaration = [{
    LogicalResult reifyReturnTypeShapes(
        OpBuilder& builder, ValueRange operands,
        SmallVectorImpl<Value>& reifiedReturnShapes) {
      return ::mlir::hlo::deriveShapeFromOperand(&builder, getOperation(),
                                                 operands.front(),
                                                 &reifiedReturnShapes);
    }
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      return mlir::hlo::isCompatibleForHloTypeInference(l, r);
    }
  }];

  let results = (outs ResultType:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict
      `:` custom<SameOperandsAndResultType>(type($lhs), type($rhs), type($result))
  }];
}

def MHLO_AddOp : MHLO_BinaryElementwiseOp<"add",
      [Commutative, Pure, HLO_CompatibleOperandsAndResultType]> {
  let summary = "Add operation";
  let description = [{
    Performs element-wise addition of two tensors `lhs` and `rhs` and produces a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#add

    Example:
    ```mlir
    %result = mhlo.add %lhs, %rhs : tensor<2x2xi32>
    ```
  }];
  let hasFolder = 1;
}

def MHLO_DivOp : MHLO_BinaryElementwiseOp<"divide",
      [Pure, HLO_CompatibleOperandsAndResultType], MHLO_IntOrFieldTensor> {
  let summary = "Div operation";
  let description = [{
    Performs element-wise division of dividend `lhs` and divisor `rhs` tensors
    and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#divide

    Example:
    ```mlir
    %result = mhlo.divide %lhs, %rhs : tensor<4xi32>
    ```
  }];
  let hasFolder = 1;
}

def MHLO_DotGeneralOp : MHLO_ShapedInterfaceOp<"dot_general",
      [Pure, ConditionallySpeculatable]> {
  let summary = "DotGeneral operation";
  let description = [{
    Computes dot products between slices of `lhs` and slices of `rhs` and
    produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#dot_general

    Example:
    ```mlir
    %result = "mhlo.dot_general"(%lhs, %rhs) {
      dot_dimension_numbers = #mhlo.dot<
        lhs_batching_dimensions = [0],
        rhs_batching_dimensions = [0],
        lhs_contracting_dimensions = [2],
        rhs_contracting_dimensions = [1]
      >
    } : (tensor<2x3x4xi32>, tensor<2x4x5xi32>) -> tensor<2x3x5xi32>
    ```
  }];
  let arguments = (ins
    MHLO_Tensor:$lhs,
    MHLO_Tensor:$rhs,
    MHLO_DotDimensionNumbers:$dot_dimension_numbers
  );

  let results = (outs MHLO_Tensor:$result);
  let hasVerifier = 1;
  let hasCustomHLOConverter = 1;

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      return mlir::hlo::isCompatibleForHloTypeInference(l, r);
    }
    /// Interface method for ConditionallySpeculatable.
    mlir::Speculation::Speculatability getSpeculatability();
  }];
}

def MHLO_MaxOp : MHLO_BinaryElementwiseOp<"maximum",
      [Commutative, Pure, HLO_CompatibleOperandsAndResultType]> {
  let summary = "Max operation";
  let description = [{
    Performs element-wise max operation on tensors `lhs` and `rhs` and produces
    a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#maximum

    Example:
    ```mlir
    %result = mhlo.maximum %lhs, %rhs : tensor<4xi32>
    ```
  }];
  let hasFolder = 1;
}

def MHLO_MinOp : MHLO_BinaryElementwiseOp<"minimum",
      [Commutative, Pure, HLO_CompatibleOperandsAndResultType]> {
  let summary = "Min operation";
  let description = [{
    Performs element-wise min operation on tensors `lhs` and `rhs` and produces a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#minimum

    Example:
    ```mlir
    %result = mhlo.minimum %lhs, %rhs : tensor<4xi32>
    ```
  }];
  let hasFolder = 1;
}

def MHLO_MulOp : MHLO_BinaryElementwiseOp<"multiply",
      [Commutative, Pure, HLO_CompatibleOperandsAndResultType]> {
  let summary = "Mul operation";
  let description = [{
    Performs element-wise product of two tensors `lhs` and `rhs` and produces a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#multiply

    Example:
    ```mlir
    %result = mhlo.multiply %lhs, %rhs : tensor<2xi32>
    ```
  }];
  let hasFolder = 1;
}

def MHLO_PowOp : MHLO_BinaryElementwiseOp<"power",
      [Pure, HLO_CompatibleOperandsAndResultType], MHLO_IntOrFieldTensor> {
  let summary = "Pow operation";
  let description = [{
    Performs element-wise exponentiation of `lhs` tensor by `rhs` tensor and
    produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#power

    Example:
    ```mlir
    %result = mhlo.power %lhs, %rhs : tensor<6xi32>
    ```
  }];
}

def MHLO_RemOp : MHLO_BinaryElementwiseOp<"remainder",
      [Pure, HLO_CompatibleOperandsAndResultType], MHLO_IntTensor> {
  let summary = "Rem operation";
  let description = [{
    Performs element-wise remainder of dividend `lhs` and divisor `rhs` tensors
    and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#remainder

    Example:
    ```mlir
    %result = mhlo.remainder %lhs, %rhs : tensor<4xi64>
    ```
  }];
  let hasFolder = 1;
}

def MHLO_ShiftLeftOp : MHLO_BinaryElementwiseOp<"shift_left",
      [Pure, HLO_CompatibleOperandsAndResultType], MHLO_IntTensor> {
  let summary = "ShiftLeft operation";
  let description = [{
    Performs element-wise left-shift operation on the `lhs` tensor by `rhs`
    number of bits and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#shift_left

    Example:
    ```mlir
    %result = mhlo.shift_left %lhs, %rhs : tensor<6xi8>
    ```
  }];
}

def MHLO_ShiftRightArithmeticOp : MHLO_BinaryElementwiseOp<"shift_right_arithmetic",
      [Pure, HLO_CompatibleOperandsAndResultType], MHLO_IntTensor> {
  let summary = "ShiftRightArithmetic operation";
  let description = [{
    Performs element-wise arithmetic right-shift operation on the `lhs` tensor
    by `rhs` number of bits and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#shift_right_arithmetic

    Example:
    ```mlir
    %result = mhlo.shift_right_arithmetic %lhs, %rhs : tensor<6xi8>
    ```
  }];
}

def MHLO_ShiftRightLogicalOp : MHLO_BinaryElementwiseOp<"shift_right_logical",
      [Pure, HLO_CompatibleOperandsAndResultType], MHLO_IntTensor> {
  let summary = "ShiftRightLogical operation";
  let description = [{
    Performs element-wise logical right-shift operation on the `lhs` tensor by
    `rhs` number of bits and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#shift_right_logical

    Example:
    ```mlir
    %result = mhlo.shift_right_logical %lhs, %rhs : tensor<6xi8>
    ```
  }];
}

def MHLO_SubtractOp : MHLO_BinaryElementwiseOp<"subtract",
      [Pure, HLO_CompatibleOperandsAndResultType], MHLO_IntOrFieldTensor> {
  let summary = "Subtract operation";
  let description = [{
    Performs element-wise subtraction of two tensors `lhs` and `rhs` and
    produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#subtract

    Example:
    ```mlir
    %result = mhlo.subtract %lhs, %rhs : tensor<2xi32>
    ```
  }];
  let hasFolder = 1;
  let hasCustomHLOConverter = 1;
}

//===----------------------------------------------------------------------===//
// MHLO binary logical elementwise op definitions.
//===----------------------------------------------------------------------===//

// See https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations
class MHLO_BinaryBiwiseOrLogicalElementwiseOp<string mnemonic> :
        MHLO_BinaryElementwiseOp<mnemonic,
          [Commutative, Pure, HLO_CompatibleOperandsAndResultType]> {
  let arguments = (ins
    MHLO_PredOrIntTensor:$lhs,
    MHLO_PredOrIntTensor:$rhs
  );

  let hasFolder = 1;
}

def MHLO_AndOp: MHLO_BinaryBiwiseOrLogicalElementwiseOp<"and"> {
  let summary = "And operation";
  let description = [{
    Performs element-wise AND of two tensors `lhs` and `rhs` and produces a
    `result` tensor

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#and

    Example:
    ```mlir
    %result = mhlo.and %lhs, %rhs : tensor<2x2xi32>
    ```
  }];
}

def MHLO_OrOp: MHLO_BinaryBiwiseOrLogicalElementwiseOp<"or"> {
  let summary = "Or operation";
  let description = [{
    Performs element-wise OR of two tensors `lhs` and `rhs` and produces a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#or

    Example:
    ```mlir
    %result = mhlo.or %lhs, %rhs : tensor<2xi1>
    ```
  }];
}

def MHLO_XorOp : MHLO_BinaryBiwiseOrLogicalElementwiseOp<"xor"> {
  let summary = "Xor operation";
  let description = [{
    Performs element-wise XOR of two tensors `lhs` and `rhs` and produces a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#xor

    Example:
    ```mlir
    %result = mhlo.xor %lhs, %rhs : tensor<2xi32>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// MHLO control flow op definitions.
//===----------------------------------------------------------------------===//

// Zkx Client API has two separate calls for indexed and predicated conditional,
// although both eventually map to kConditional HLO. CaseOp maps to indexed
// conditional use of kConditional HLO.
def MHLO_CaseOp: MHLO_Op<"case", [
      RecursiveMemoryEffects,
      SingleBlockImplicitTerminator<"ReturnOp">,
      DeclareOpInterfaceMethods<InferTypeOpInterface>
    ]> {
  let summary = "Case operation";
  let description = [{
    Produces the output from executing exactly one `function` from `branches`
    depending on the value of `index`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#case

    Example:
    ```mlir
    %result0, %result1 = "mhlo.case"(%index) ({
      mhlo.return %result_branch0, %result_branch0 : tensor<2xi64>, tensor<2xi64>
    }, {
      mhlo.return %result_branch1, %result_branch1 : tensor<2xi64>, tensor<2xi64>
    }) : (tensor<i32>) -> (tensor<2xi64>, tensor<2xi64>)
    ```
  }];

  let arguments = (ins
    I32Tensor:$index
  );

  let regions = (region VariadicRegion<SizedRegion<1>>:$branches);

  let results = (outs Variadic<MHLO_TensorOrToken>);

  let hasCustomHLOConverter = 1;

  let hasCanonicalizer = 1;
}

// Zkx Client API has two separate calls for indexed and predicated conditional,
// although both eventually map to kConditional HLO. IfOp maps to predicated
// conditional use of kConditional HLO.
def MHLO_IfOp: MHLO_Op<"if", [
    RecursiveMemoryEffects,
    SingleBlockImplicitTerminator<"ReturnOp">,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "If operation";
  let description = [{
    Produces the output from executing exactly one branch from `true_branch` or
    `false_branch` depending on the value of `pred`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#if

    Example:
    %result = "mhlo.if"(%pred) ({
      "mhlo.return"(%result_true_branch) : (tensor<i32>) -> ()
    }, {
      "mhlo.return"(%result_false_branch) : (tensor<i32>) -> ()
    }) : (tensor<i1>) -> tensor<i32>
  }];

  let arguments = (ins
    MHLO_PredTensor:$pred
  );

  let regions = (region SizedRegion<1>:$true_branch,
                        SizedRegion<1>:$false_branch);

  let results = (outs Variadic<MHLO_TensorOrToken>);

  // TODO(b/129422361): ConditionalOp has special conversion logic to HLO.
  let hasCustomHLOConverter = 1;

  let hasCanonicalizer = 1;
}

def MHLO_WhileOp: MHLO_Op<"while", [
      RecursiveMemoryEffects,
      SingleBlockImplicitTerminator<"ReturnOp">,
      DeclareOpInterfaceMethods<InferTypeOpInterface>,
      OpAsmOpInterface
    ]> {
  let summary = "While operation";
  let description = [{
    Produces the output from executing `body` function 0 or more times while the
    `cond` function outputs `true`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#while

    Example:
    ```mlir
    %results0, %results1 = "mhlo.while"(%operand0, %operand1) ({
      ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
        %0 = "mhlo.compare"(%arg0, %arg1) {
          comparison_direction = #stablehlo<comparison_direction LT>
        } : (tensor<i32>, tensor<i32>) -> tensor<i1>
        "mhlo.return"(%0) : (tensor<i1>) -> ()
    }, {
      ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
        %0 = "mhlo.add"(%arg0, %constant0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
        "mhlo.return"(%0, %arg1) : (tensor<i32>, tensor<i32>) -> ()
    }) : (tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
    ```
  }];
  let arguments = (ins Variadic<MHLO_TensorOrToken>:$operand);

  let regions = (region SizedRegion<1>:$cond, SizedRegion<1>:$body);

  let results = (outs Variadic<MHLO_TensorOrToken>);

  let extraClassDeclaration = [{
    // Method of OpAsmOpInterface used during custom printing to name the block
    // arguments in the nested regions. We name both the condition and the body
    // regions entry arguments the same way, with a `iterArg` prefix. Since the
    // two regions are side-by-side they will have the same name, which allows
    // us to print them once and share it for the two regions, and still be able
    // to parse them back.
    void getAsmBlockArgumentNames(Region &region, OpAsmSetValueNameFn setNameFn) {
      for (BlockArgument arg : region.getArguments())
        setNameFn(arg, "iterArg");
    }
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      return mlir::hlo::isCompatibleForHloTypeInference(l, r);
    }
  }];
  // TODO(b/129422361): WhileOp has special conversion logic to HLO.
  let hasCustomHLOConverter = 1;
  let hasCanonicalizer = 1;
  let hasCustomAssemblyFormat = 1;
  let hasFolder = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// MHLO tuple op definitions.
//===----------------------------------------------------------------------===//

def MHLO_GetTupleElementOp: MHLO_Op<"get_tuple_element", [Pure,
     DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "GetTupleElement operation";
  let description = [{
    Extracts element at `index` position of the `operand` tuple and produces a
    `result`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#get_tuple_element

    Example:
    ```mlir
    %result = mhlo.get_tuple_element %operand[0] : (tuple<tensor<2xi32>, tuple<tensor<i32>>>) -> tensor<2xi32>
    ```
  }];
  let arguments = (ins
    MHLO_Tuple:$operand,
    ConfinedAttr<I32Attr, [IntNonNegative]>:$index
  );

  let results = (outs MHLO_TensorOrTokenOrTuple);

  let hasFolder = 1;

  let assemblyFormat = [{
    $operand `[` $index `]` attr-dict `:` functional-type(operands, results)
  }];
}

def MHLO_TupleOp : MHLO_Op<"tuple", [Pure,
     DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Tuple operation";
  let description = [{
    Produces a `result` tuple from values `val`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#tuple

    Example:
    ```mlir
    %result = mhlo.tuple %val0, %val1 : tuple<tensor<2xi32>, tuple<tensor<i32>>>
    ```
   }];
  let arguments = (ins Variadic<MHLO_TensorOrTokenOrTuple>:$val);
  let results = (outs MHLO_Tuple:$result);

  let hasCanonicalizer = 1;

  let assemblyFormat = [{
    $val attr-dict `:` custom<TupleOpType>(type($val), type($result))
  }];
}

//===----------------------------------------------------------------------===//
// MHLO Slice definitions.
//===----------------------------------------------------------------------===//

def MHLO_DynamicSliceOp: MHLO_Op<"dynamic_slice",
      [Pure, AllElementTypesMatch<["operand", "result"]>,
       InferTensorType]> {
  let summary = "DynamicSlice operation";
  let description = [{
    Extracts a slice from the `operand` using dynamically-computed starting
    indices and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#dynamic_slice

    Example:
    ```mlir
    %result = mhlo.dynamic_slice %operand, %start_indices0, %start_indices1, sizes = [2, 2]
      : (tensor<4x4xi32>, tensor<i64>, tensor<i64>) -> tensor<2x2xi32>
    ```
  }];
  let arguments = (ins
    MHLO_Tensor:$operand,
    Variadic<MHLO_ScalarIntTensor>:$start_indices,
    I64ElementsAttr:$slice_sizes
  );

  let results = (outs MHLO_Tensor:$result);
  let hasCanonicalizer = 1;
}

def MHLO_DynamicUpdateSliceOp: MHLO_Op<"dynamic_update_slice",
      [Pure, AllElementTypesMatch<["operand", "update", "result"]>,
       InferTensorType]> {
  let summary = "DynamicUpdateSlice operation";
  let description = [{
    Produces a `result` tensor which is equal to the `operand` tensor except
    that the slice starting at `start_indices` is updated with the values in
    `update`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#dynamic_update_slice

    Example:
    ```mlir
    %result = mhlo.dynamic_update_slice %operand, %update, %start_indices0, %start_indices1
      : (tensor<4x4xi32>, tensor<2x2xi32>, tensor<i64>, tensor<i64>) -> tensor<4x4xi32>
    ```
  }];
  let arguments = (ins
    MHLO_Tensor:$operand,
    MHLO_Tensor:$update,
    Variadic<MHLO_ScalarIntTensor>:$start_indices
  );
  let results = (outs MHLO_Tensor:$result);
  let hasFolder = 1;

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MHLO_SliceOp: MHLO_Op<
      "slice",
      [Pure, SameOperandsAndResultElementType,
       AllTypesMatch<["start_indices", "limit_indices", "strides"]>,
       DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Slice operation";
  let description = [{
    Extracts a slice from the `operand` using statically-computed starting
    indices and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#slice

    Example:
    ```mlir
    %result = "mhlo.slice" (%operand) {
      start_indices = dense<[1, 2]> : tensor<2xi64>,
      limit_indices = dense<[3, 4]> : tensor<2xi64>,
      strides = dense<1> : tensor<2xi64>
    } : (tensor<3x4xi64>) -> tensor<2x2xi64>
    ```
  }];
  let arguments = (ins
    MHLO_Tensor:$operand,
    I64ElementsAttr:$start_indices,
    I64ElementsAttr:$limit_indices,
    I64ElementsAttr:$strides
  );

  let results = (outs MHLO_Tensor);

  // TODO(chokobole): Uncomment this. Dependency: ConcatenateOp
  // let hasCanonicalizer = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// MHLO Other op definitions.
//===----------------------------------------------------------------------===//

// This is an op for purposes internal to ZKX/GPU.
def MHLO_BitcastOp : MHLO_Op<"bitcast", [Pure]> {
  let summary = "Bitcast operation";
  let description = [{
    This operation is private to the ZKX compiler, so it is does not yet have
    a specification.

    Informally, this operation changes the shape of the input in the way that
    the physical arrangement of elements are unchanged.

    This operation needs layout information to make sense of "physical
    arrangement of elements", and layout support in MHLO is currently a work
    in progress.

    Example:
    ```mlir
    %0 = mhlo.bitcast %arg0 : (tensor<3x4xu32>) -> tensor<3x4x1xu32>
    ```
  }];

  let arguments = (ins MHLO_Tensor:$operand);
  let results = (outs MHLO_Tensor);
  let hasCustomHLOConverter = 1;
  let hasFolder = 1;

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MHLO_BitcastConvertOp : MHLO_ShapedInterfaceOp<"bitcast_convert",
    [Pure]> {
  let summary = "BitcastConvert operation";
  let description = [{
    Performs a bitcast operation on `operand` tensor and produces a `result`
    tensor where the bits of the entire `operand` tensor are reinterpreted using
    the type of the `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#bitcast_convert

    Example:
    ```mlir
    %result = mhlo.bitcast_convert %operand : (tensor<2xu32>) -> tensor<2x4xi8>
    ```
  }];

  let arguments = (ins MHLO_Tensor:$operand);
  let results = (outs MHLO_Tensor);
  let hasVerifier = 1;
  let hasCustomHLOConverter = 1;

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MHLO_BroadcastOp : MHLO_ShapedInterfaceOp<"broadcast",
    [Pure, SameOperandsAndResultElementType, InferTensorType]> {
  let summary = "Broadcast operation";
  let description = [{
    This operation is on its way out of StableHLO, so it is not included in
    the specification: https://github.com/openxla/stablehlo/issues/3.

    Informally, this operation does the same thing as ZKX's Broadcast:
    https://www.tensorflow.org/xla/operation_semantics#broadcast

    Example:
    ```mlir
    %result = mhlo.broadcast %operand, sizes = [1, 2] : (tensor<3xi32>) -> tensor<1x2x3xi32>
    ```
  }];
  let arguments = (ins
    MHLO_Tensor:$operand,
    I64ElementsAttr:$broadcast_sizes
  );

  let results = (outs MHLO_Tensor);

  let hasFolder = 1;
}

def MHLO_BroadcastInDimOp : MHLO_Op<"broadcast_in_dim",
      [Pure, HLO_CompatibleOperandsAndResultElementType]> {
  let summary = "BroadcastInDim operation";
  let description = [{
    Expands the dimensions and/or rank of an input tensor by duplicating the
    data in the `operand` tensor and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#broadcast_in_dim

    Example:
    ```mlir
    %result = mhlo.broadcast_in_dim %operand, dims = [2, 1] : (tensor<1x3xi32>) -> tensor<2x3x2xi32>
    ```
  }];
  let arguments = (ins
    MHLO_Tensor:$operand,
    MHLO_BroadcastDimAttr:$broadcast_dimensions
  );

  let results = (outs MHLO_StaticShapeOrBoundedDimTensor);

  let hasFolder = 1;
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
  // Only handles a static subset of the legacy format.
  let hasCustomHLOConverter = 1;
}

def MHLO_ClampOp : MHLO_ShapedInterfaceOp<"clamp", [Pure,
  SameOperandsAndResultElementType, HLO_BroadcastingElementwise,
  InferTensorType]> {
  let summary = "Clamp operation";
  let description = [{
    Clamps every element of the `operand` tensor between a minimum and maximum
    value and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#clamp

    Example:
    ```mlir
    %result = mhlo.clamp %min, %operand, %max : tensor<3xi32>
    ```
  }];

  let arguments = (ins
    MHLO_Tensor:$min,
    MHLO_Tensor:$operand,
    MHLO_Tensor:$max
  );
  let results = (outs MHLO_Tensor:$result);

  let hasFolder = 1;

  let assemblyFormat = [{
    $min `,` $operand `,` $max attr-dict
      `:` custom<SameOperandsAndResultType>(type($min), type($operand), type($max), type($result))
  }];
}

def MHLO_CompareOp: MHLO_Op<"compare", [Pure, SameOperandsElementType,
    SameOperandsAndResultShape, Elementwise, InferTensorTypeWithReify]> {
  let summary = "Compare operation";
  let description = [{
    Performs element-wise comparison of `lhs` and `rhs` tensors according to
    `comparison_direction`, and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#compare

    Example:
    ```mlir
    %result = mhlo.compare LT, %lhs, %rhs : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
    ```
  }];
  let arguments = (ins
    MHLO_Tensor:$lhs,
    MHLO_Tensor:$rhs,
    MHLO_ComparisonDirectionAttr:$comparison_direction
  );
  let results = (outs MHLO_PredTensor);

  let hasFolder = 1;

  let hasCustomHLOConverter = 1;

  let assemblyFormat = [{
    $comparison_direction `,` $lhs `,` $rhs
      attr-dict `:` functional-type(operands, results)
  }];
}

def MHLO_ConcatenateOp : MHLO_ShapedInterfaceOp<"concatenate",
    [Pure, SameOperandsAndResultElementType,
     DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Concatenate operation";
  let description = [{
    Concatenates a variadic number of tensors in `inputs` along `dimension`
    dimension in the same order as the given arguments and produces a `result`
    tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#concatenate

    Example:
    ```mlir
    %result = mhlo.concatenate %input0, %input1, dim = 0 : (tensor<3x2xi64>, tensor<1x2xi64>) -> tensor<4x2xi64>
    ```
  }];

  let arguments = (ins
    Variadic<MHLO_Tensor>:$val,
    ConfinedAttr<I64Attr, [IntNonNegative]>:$dimension
  );

  let results = (outs MHLO_Tensor);

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def MHLO_DynamicBroadcastInDimOp : MHLO_ShapedInterfaceOp<
    "dynamic_broadcast_in_dim", [Pure]> {
  let summary = "DynamicBroadcastInDim operation";
  let description = [{
    This operation is functionally identical to
    [broadcast_in_dim](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#broadcast_in_dim)
    op, but the result shape is specified dynamically via `output_dimensions`.

    It also accepts optional attributes to express static knowledge about the
    expanding behavior of dimensions. If not specified, all dimensions are
    assumed to be possibly expanding. The sets of dimensions that are known to
    be expanding and the set of dimensions that are known to be non-expanding
    must be disjoint and they must be a subset of the operand's dimensions.

    See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#dynamic_broadcast_in_dim

    Example:
    ```mlir
    %operand = mhlo.constant dense<[[1, 2, 3]]> : tensor<1x3xi64>
    %output_dimensions = mhlo.constant dense<[2, 3, 2]> : tensor<3xi64>
    %result = "mhlo.dynamic_broadcast_in_dim"(%operand, %output_dimensions) {
      broadcast_dimensions = array<i64: 2, 1>,
      known_expanding_dimensions = array<i64: 0>,
      known_nonexpanding_dimensions = array<i64: 1>
    } : (tensor<1x3xi64>, tensor<3xi64>) -> tensor<2x3x2xi64>
    ```
  }];

  let arguments = (ins
    MHLO_Tensor:$operand,
    MHLO_DimensionTensor:$output_dimensions,
    MHLO_BroadcastDimAttr:$broadcast_dimensions,
    OptionalAttr<MHLO_BroadcastDimAttr>:$known_expanding_dimensions,
    OptionalAttr<MHLO_BroadcastDimAttr>:$known_nonexpanding_dimensions
  );

  let results = (outs MHLO_Tensor);

  let builders = [
    OpBuilder<(ins
        "Type":$result_type, "Value":$operand, "Value":$output_dimensions,
        "DenseIntElementsAttr":$broadcast_dimensions), [{
      build($_builder, $_state, result_type, operand, output_dimensions,
          broadcast_dimensions, /*known_expanding_dimensions=*/{},
          /*known_nonexpanding_dimensions=*/{});
    }]>
  ];

  let hasCanonicalizer = 1;
  let hasVerifier = 1;
  // Cannot be exported to legacy formats.
  let hasCustomHLOConverter = 1;
}

// TODO(chokobole): Do we need this? Dependency: interior_padding
def MHLO_DynamicPadOp: MHLO_ShapedInterfaceOp<"dynamic_pad",
      [Pure, AllElementTypesMatch<["operand", "padding_value", "result"]>,
      AllTypesMatch<["edge_padding_low", "edge_padding_high"]>]> {
  let summary = "DynamicPad operation";
  let description = [{
    This operation is a work in progress, so it is not yet included in
    the specification: https://github.com/openxla/stablehlo/issues/8.

    Informally, this operation does the same thing as PadOp except
    that `edge_padding_low` and `edge_padding_high` are
    specified dynamically:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#pad

    Example:
    ```mlir
    %result = mhlo.dynamic_pad %operand, %padding_value,
                %edge_padding_low, %edge_padding_high
           : (tensor<?x?xu32>, tensor<u32>, tensor<2xindex>, tensor<2xindex>) -> tensor<?x?xu32>
    ```
  }];
  let arguments = (ins
    MHLO_Tensor:$operand,
    MHLO_Tensor:$padding_value,
    MHLO_DimensionTensor:$edge_padding_low,
    MHLO_DimensionTensor:$edge_padding_high
  );
  let results = (outs MHLO_Tensor:$result);
  let description = [{
    Dynamically Pads the `operand`, with amount of padding added at
    low-end/high-end is passed through input tensors.
  }];
  let hasCanonicalizer = 1;
  let hasCustomHLOConverter = 1;
  let hasVerifier = 1;

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MHLO_DynamicReshapeOp: MHLO_ShapedInterfaceOp<"dynamic_reshape", [Pure]> {
  let summary = "DynamicReshape operation";
  let description = [{
    This operation is functionally identical to
    [reshape](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#reshape)
    op, but the result shape is specified dynamically via `output_shape`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#dynamic_reshape

    Example:
    ```mlir
    %output_shape = mhlo.constant dense<[3, 2]> : tensor<2xi64>
    %result = mhlo.dynamic_reshape %operand, %output_shape : (tensor<2x3xi64>, tensor<2xi64>) -> tensor<3x2xi64>
    ```
  }];

  let arguments = (ins MHLO_AnyTensor:$operand, MHLO_DimensionTensor:$output_shape);
  let results = (outs MHLO_AnyTensor:$result);

  let hasCanonicalizer = 1;
  // Cannot be exported to legacy formats.
  let hasCustomHLOConverter = 1;
  let hasVerifier = 1;

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MHLO_GetDimensionSizeOp: MHLO_Op<"get_dimension_size",
      [Pure, InferTensorType]> {
  let summary = "GetDimensionSize operation";
  let description = [{
    Produces the size of the given `dimension` of the `operand`.

    See
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#get_dimension_size

    Example:
    ```mlir
    %result = mhlo.get_dimension_size %operand, dim = 1 : (tensor<2x3xu32>) -> tensor<i32>
    ```
  }];
  let arguments = (ins
    MHLO_Tensor:$operand,
    ConfinedAttr<I64Attr, [IntNonNegative]>:$dimension
  );
  // TODO(hinsu): Allow 64-bit result types once ZKX HLO dialect based on the
  // ZKX semantics is available. This limitation is because of the current ZKX
  // implementation.
  let results = (outs I32Tensor);

  let hasFolder = 1;
}

def MHLO_MapOp: MHLO_ShapedInterfaceOp<"map",
      [RecursiveMemoryEffects, SameOperandsAndResultShape,
       SingleBlockImplicitTerminator<"ReturnOp">, InferTensorTypeWithReify]> {
  let summary = "Map operation";
  let description = [{
    Applies a map function `computation` to `inputs` along the `dimensions` and
    produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#map

    Example:
    ```mlir
    %result = "mhlo.map"(%input0, %input1) ({
      ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
        %0 = mhlo.multiply %arg0, %arg1 : tensor<i32>
        mhlo.return %0 : tensor<i32>
    }) {
      dimensions = dense<[0, 1]> : tensor<2xi64>
    } : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
    ```
  }];
  let arguments = (ins
    Variadic<MHLO_Tensor>:$inputs,
    I64ElementsAttr:$dimensions
  );
  let regions = (region SizedRegion<1>:$computation);
  let results = (outs MHLO_Tensor);
  let hasFolder = 1;
  let hasCustomHLOConverter = 1;
}

// TODO(chokobole): Do we need this? Dependency: interior_padding
def MHLO_PadOp: MHLO_ShapedInterfaceOp<"pad",
      [Pure, SameOperandsAndResultElementType,
      DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Pad operation";
  let description = [{
    Expands `operand` by padding around the tensor as well as between the
    elements of the tensor with the given `padding_value`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#pad

    Example:
    ```mlir
    %0 = mhlo.pad %arg0, %arg1, low = [0, 1], high = [2, 1]
      : (tensor<2x3xi32>, tensor<i32>) -> tensor<5x9xi32>
    ```
  }];
  let arguments = (ins
    MHLO_Tensor:$operand,
    MHLO_Tensor:$padding_value,
    I64ElementsAttr:$edge_padding_low,
    I64ElementsAttr:$edge_padding_high
  );

  let results = (outs MHLO_Tensor);

  // TODO(b/129422361): PadOp has a custom constructor for HLO.
  let hasCustomHLOConverter = 1;

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def MHLO_RealDynamicSliceOp: MHLO_ShapedInterfaceOp<
      "real_dynamic_slice",
      [Pure, AllElementTypesMatch<["operand", "result"]>,
       AllTypesMatch<["start_indices", "limit_indices", "strides"]>]> {
  let summary = "RealDynamicSlice operation";
  let description = [{
    This operation is a work in progress, so it is not yet included in
    the specification: https://github.com/openxla/stablehlo/issues/8.

    Informally, this operation does the same thing as SliceOp except
    that `start_indices`, `limit_indices` and `strides` are specified dynamically:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#slice

    Example:
    ```mlir
    %result = mhlo.real_dynamic_slice %operand,
                %start_indices, %limit_indices, %strides
           : (tensor<256x?xu32>, tensor<2xindex>, tensor<2xindex>, tensor<2xindex>) -> tensor<256x?xu32>
    ```
  }];
  let arguments = (ins
    MHLO_Tensor:$operand,
    MHLO_DimensionTensor:$start_indices,
    MHLO_DimensionTensor:$limit_indices,
    MHLO_DimensionTensor:$strides
  );
  let results = (outs MHLO_Tensor:$result);
  let hasCanonicalizer = 1;
  let hasCustomHLOConverter = 1;
  let hasVerifier = 1;

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MHLO_ReduceOp: MHLO_ShapedInterfaceOp<"reduce", [
      RecursiveMemoryEffects,
      SameVariadicOperandSize,
      SingleBlockImplicitTerminator<"ReturnOp">,
      InferTensorType,
    ]> {
  let summary = "Reduce operation";
  let description = [{
    Applies a reduction function `body` to `inputs` and `init_values` along the
    `dimensions` and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#reduce

    Example:
    ```mlir
    %result = "mhlo.reduce"(%input, %init_value) ({
      ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
        %0 = "mhlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
        "mhlo.return"(%0) : (tensor<i32>) -> ()
    }) {
      dimensions = dense<1> : tensor<1xi64>
    } : (tensor<1x6xi32>, tensor<i32>) -> tensor<1xi32>
    ```
  }];
  let arguments = (ins
    Variadic<MHLO_Tensor>:$inputs,
    Variadic<MHLO_Tensor>:$init_values,
    I64ElementsAttr:$dimensions
  );

  let results = (outs Variadic<MHLO_Tensor>);

  let hasCanonicalizer = 1;
  let hasCustomAssemblyFormat = 1;
  let hasFolder = 1;
  let hasVerifier = 1;

  // TODO(hinsu): Verify that the attached body arguments and results are
  // compatible with reduce op's operands.
  let regions = (region SizedRegion<1>:$body);

  // Builder
  let builders = [
    OpBuilder<(ins "ValueRange":$inputs, "ValueRange":$init_values,
      "DenseIntElementsAttr":$dimensions, "TypeRange":$element_types)>,
  ];

  // TODO(b/129422361): ReduceOp has special conversion logic to HLO.
  let hasCustomHLOConverter = 1;
}

def MHLO_ReshapeOp: MHLO_Op<"reshape",
      [Pure, HLO_CompatibleOperandsAndResultElementType]> {
  let summary = "Reshape operation";
  let description = [{
    Performs reshape of `operand` tensor to a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#reshape

    Example:
    ```mlir
    %result = mhlo.reshape %operand : (tensor<2xu32>) -> tensor<1x2xu32>
    ```
  }];

  let arguments = (ins MHLO_AnyTensor:$operand);

  let results = (outs MHLO_StaticShapeOrBoundedDimTensor);
  let hasFolder = 1;
  let hasCanonicalizer = 1;
  let hasVerifier = 1;

  let hasCustomHLOConverter = 1;

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MHLO_ReturnOp : MHLO_Op<"return", [Pure, Terminator]> {
  let summary = "Return operation";
  let summary = [{
    This operation is a work in progress, so it is not yet included in
    the specification: https://github.com/openxla/stablehlo/issues/425.

    Informally, this operation serves as a terminator for regions defined by
    the StableHLO ops. Non-StableHLO ops, e.g. `func.func`, have their own
    terminators, e.g. `func.return`.

    Example:
    ```mlir
    %result = "mhlo.reduce"(%input, %init_value) ({
      ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
        %0 = "mhlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
        "mhlo.return"(%0) : (tensor<i32>) -> ()
    }) {
      dimensions = dense<1> : tensor<1xi64>
    } : (tensor<1x6xi32>, tensor<i32>) -> tensor<1xi32>
    ```
  }];

  let arguments = (ins
    Variadic<MHLO_TensorOrTokenOrTuple >:$results
  );

  // Disable conversion operator for return op as the op is not an actual ZKX
  // instruction and is only used as a terminator for regions.
  let hasCustomHLOConverter = 1;

  let assemblyFormat = "$results attr-dict (`:` type($results)^)?";
}

def MHLO_BitReverseOp: MHLO_Op<"bit_reverse",
      [Pure, HLO_CompatibleOperandsAndResultType]> {
  let summary = "BitReverse operation";
  let description = [{
    Applies bit-reversal permutation to elements along the specified
    `dimensions` and produces a `result` tensor.

    Each dimension size must be a power of 2. For a dimension of size n,
    the element at index i is moved to index bit_reverse(i, log₂(n)).

    Example:
    ```mlir
    %result = mhlo.bit_reverse %operand, dims = [0] : tensor<8xi32>
    ```
  }];
  let arguments = (ins
    MHLO_Tensor:$operand,
    I64ElementsAttr:$dimensions
  );

  let hasVerifier = 1;

  let results = (outs MHLO_Tensor);

  let hasFolder = 1;
}

def MHLO_ReverseOp: MHLO_Op<"reverse",
      [Pure, HLO_CompatibleOperandsAndResultType]> {
  let summary = "Reverse operation";
  let description = [{
    Reverses the order of elements in the `operand` along the specified
    `dimensions` and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#reverse

    Example:
    ```mlir
    %result = mhlo.reverse %operand, dims = [1] : tensor<3x2xi32>
    ```
  }];
  let arguments = (ins
    MHLO_Tensor:$operand,
    I64ElementsAttr:$dimensions
  );

  let hasVerifier = 1;

  let results = (outs MHLO_Tensor);

  let hasFolder = 1;
}

def MHLO_GatherOp: MHLO_Op<"gather", [InferTensorTypeWithReify, Pure]> {
  let summary = "Gather operation";
  let description = [{
    Gathers slices from `operand` tensor from offsets specified in
    `start_indices` and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#gather

    Example:
    ```mlir
    %result = "mhlo.gather"(%operand, %start_indices) {
      dimension_numbers = #mhlo.gather<
        offset_dims = [3, 4],
        collapsed_slice_dims = [1],
        operand_batching_dims = [0],
        start_indices_batching_dims = [1],
        start_index_map = [2, 1],
        index_vector_dim = 3>,
      slice_sizes = dense<[0, 2, 2]> : tensor<3xi64>,
      indices_are_sorted = false
    } : (tensor<2x3x4x2xi64>, tensor<2x2x3x2xi64>) -> tensor<2x2x3x2x2xi64>
    ```
  }];

  let arguments = (ins
    MHLO_Tensor:$operand,
    MHLO_IntTensor:$start_indices,
    MHLO_GatherDimensionNumbers:$dimension_numbers,
    I64ElementsAttr:$slice_sizes,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$indices_are_sorted
  );

  let results = (outs MHLO_Tensor);

  let hasCanonicalizer = 1;
}

def MHLO_ScatterOp: MHLO_Op<"scatter",
      [SameVariadicOperandSize, RecursiveMemoryEffects,
      DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Scatter operation";
  let description = [{
    Produces `results` tensors which are equal to `inputs` tensors except that
    several slices specified by `scatter_indices` are updated with the values
    `updates` using `update_computation`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#scatter

   Example:
   ```mlir
   %result = "mhlo.scatter"(%input, %scatter_indices, %update) ({
     ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
       %0 = mhlo.add %arg0, %arg1 : tensor<i32>
       mhlo.return %0 : tensor<i32>
   }) {
     scatter_dimension_numbers = #mhlo.scatter<
       update_window_dims = [3, 4],
       inserted_window_dims = [1],
       input_batching_dims = [0],
       scatter_indices_batching_dims = [1],
       scatter_dims_to_operand_dims = [2, 1],
       index_vector_dim = 3>,
     indices_are_sorted = false,
     unique_indices = false
   } : (tensor<2x3x4x2xi64>, tensor<2x2x3x2xi64>, tensor<2x2x3x2x2xi64>) -> tensor<2x3x4x2xi64>
   ```
  }];
  let arguments = (ins
    Variadic<MHLO_Tensor>:$inputs,
    RankedTensorOf<[AnyInteger, Index]>:$scatter_indices,
    Variadic<MHLO_Tensor>:$updates,
    MHLO_ScatterDimensionNumbers:$scatter_dimension_numbers,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$indices_are_sorted,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$unique_indices
  );

  let regions = (region SizedRegion<1>:$update_computation);

  let results = (outs Variadic<MHLO_Tensor>);

  let hasCustomHLOConverter = 1;

  let hasFolder = 1;
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def MHLO_SelectOp: MHLO_Op<"select", [Pure, HLO_BroadcastingElementwise,
    InferTensorTypeWithReify]> {
  let summary = "Select operation";
  let description = [{
    Produces a `result` tensor where each element is selected from `on_true` or
    `on_false` tensor based on the value of the corresponding element of `pred`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#select

    Example:
    ```mlir
    %result = mhlo.select %pred, %on_true, %on_false : tensor<2x2xi1>, tensor<2x2xi32>
    ```
  }];
  let arguments = (ins
    MHLO_PredTensor:$pred,
    MHLO_Tensor:$on_true,
    MHLO_Tensor:$on_false
  );

  let results = (outs MHLO_Tensor:$result);

  let hasFolder = 1;
  let hasCanonicalizer = 1;

  let assemblyFormat = [{
    operands attr-dict `:`
      custom<SelectOpType>(type($pred), type($on_true), type($on_false), type($result))
  }];
}

def MHLO_SetDimensionSizeOp: MHLO_Op<"set_dimension_size", [Pure,
      InferTensorType]> {
  let summary = "SetDimensionSize operation";
  let description = [{
    This operation is a work in progress, so it is not yet included in
    the specification: https://github.com/openxla/stablehlo/issues/8.

    Informally, this operation does the same thing as ZKX's SetDimensionSize:
    https://www.tensorflow.org/xla/operation_semantics#setdimensionsize

    Example:
    ```mlir
    %0 = mhlo.set_dimension_size %arg0, %arg1, dim = 1 : (tensor<4x2xu32>, tensor<i32>) -> tensor<4x2xu32>
    ```
  }];
  let arguments = (ins
    MHLO_Tensor:$operand,
    I32Tensor:$size,
    ConfinedAttr<I64Attr, [IntNonNegative]>:$dimension
  );
  let results = (outs MHLO_Tensor);

  let hasFolder = 1;
  let hasCustomHLOConverter = 1;
}

def MHLO_SortOp : MHLO_Op<"sort",
      [RecursiveMemoryEffects, SameOperandsAndResultShape, InferTensorType]> {
  let summary = "Sort operation";
  let description = [{
    Sorts a variadic number of tensors in `inputs` together, according to a
    custom `comparator`, along the given `dimension` and produces a variadic
    number of tensors as `results`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#sort

    Example:
    ```mlir
    %result0, %result1 = "mhlo.sort"(%input0, %input1) ({
      ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>):
        %predicate = "mhlo.compare"(%arg0, %arg1) {
          comparison_direction = #stablehlo<comparison_direction GT>
          } : (tensor<i32>, tensor<i32>) -> tensor<i1>
        "mhlo.return"(%predicate) : (tensor<i1>) -> ()
    }) {
      dimension = 0 : i64,
      is_stable = true
    } : (tensor<2x3xi32>, tensor<2x3xi32>) -> (tensor<2x3xi32>, tensor<2x3xi32>)
    ```
  }];
  let arguments = (ins
    Variadic<MHLO_Tensor>:$inputs,
    DefaultValuedOptionalAttr<I64Attr, "-1">:$dimension,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$is_stable
  );

  let results = (outs Variadic<MHLO_Tensor>);

  let regions = (region SizedRegion<1>:$comparator);

  let builders = [
    OpBuilder<(ins "ValueRange":$inputs, CArg<"int64_t", "-1">:$dimension,
      CArg<"bool", "false">:$is_stable)>];

  // TODO(b/129422361): SortOp has special conversion logic to HLO.
  let hasCustomHLOConverter = 1;

  let hasCanonicalizer = 1;

  let hasVerifier = 1;
}

def MHLO_TransposeOp: MHLO_ShapedInterfaceOp<"transpose",
      [Pure, HLO_CompatibleOperandsAndResultElementType,
      DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Transpose operation";
  let description = [{
    Permutes the dimensions of `operand` tensor using `permutation` and produces
    a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#transpose

    Example:
    ```mlir
    %0 = mhlo.transpose %arg0, dims = [2, 1, 0] : (tensor<1x2x3xi32>) -> tensor<3x2x1xi32>
    ```
  }];
  let arguments = (ins
    MHLO_Tensor:$operand,
    I64ElementsAttr:$permutation
  );
  let results = (outs MHLO_Tensor);

  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def MHLO_ReduceWindowOp: MHLO_Op<"reduce_window", [
      RecursiveMemoryEffects,
      SameVariadicOperandSize,
      SingleBlockImplicitTerminator<"ReturnOp">,
      InferTensorType,
    ]> {
  let summary = "ReduceWindow operation";
  let description = [{
    Applies a reduction function `body` to windows of `inputs` and `init_values`
    and produces `results`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#reduce_window

    Example:
    ```mlir
    %result = "mhlo.reduce_window"(%input, %init_value) ({
      ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
        %0 = mhlo.add %arg0, %arg1 : tensor<i64>
        "mhlo.return"(%0) : (tensor<i64>) -> ()
    }) {
      window_dimensions = dense<[2, 1]> : tensor<2xi64>,
      window_strides = dense<[4, 1]> : tensor<2xi64>,
      base_dilations = dense<[2, 1]> : tensor<2xi64>,
      window_dilations = dense<[3, 1]> : tensor<2xi64>,
      padding = dense<[[2, 1], [0, 0]]> : tensor<2x2xi64>
    } : (tensor<3x2xi64>, tensor<i64>) -> tensor<2x2xi64>
    ```
  }];
  let arguments = (ins
    Variadic<MHLO_Tensor>:$inputs,
    Variadic<MHLO_Tensor>:$init_values,
    I64ElementsAttr:$window_dimensions,
    OptionalAttr<I64ElementsAttr>:$window_strides,
    OptionalAttr<I64ElementsAttr>:$base_dilations,
    OptionalAttr<I64ElementsAttr>:$window_dilations,
    OptionalAttr<I64ElementsAttr>:$padding
  );

  let results = (outs Variadic<MHLO_Tensor>);

  let regions = (region SizedRegion<1>:$body);

  let hasVerifier = 1;

  let hasCustomHLOConverter = 1;
}


def MHLO_PairingCheckOp: MHLO_Op<"pairing_check", [Pure]> {
  let summary = "PairingCheck operation";
  let description = [{
    Multi-pairing check: verifies that the product of pairings equals the
    identity in GT:
      e(P₁, Q₁) * e(P₂, Q₂) * ... * e(Pₙ, Qₙ) = 1

    Both inputs must be 1-D tensors of affine points on compatible
    pairing-friendly curves (e.g., BN254 G1 and G2). Both tensors must
    have matching first dimension.

    Returns a scalar PRED tensor.
  }];
  let arguments = (ins
    MHLO_ECTensor:$g1_points,
    MHLO_ECTensor:$g2_points
  );

  let results = (outs MHLO_ScalarTensor:$result);

  let hasCustomHLOConverter = 1;

  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];
}

def MHLO_MsmOp : MHLO_Op<"msm", [Pure]> {
  let summary = "Multi-scalar multiplication operation";
  let description = [{
    Computes the multi-scalar multiplication (MSM) of scalars and elliptic curve
    points: result = Σᵢ scalarsᵢ * basesᵢ.
  }];
  let arguments = (ins
    MHLO_Tensor:$scalars,
    MHLO_Tensor:$bases,
    DefaultValuedOptionalAttr<I32Attr, "0">:$window_bits,
    DefaultValuedOptionalAttr<I32Attr, "0">:$precompute_factor,
    DefaultValuedOptionalAttr<I32Attr, "0">:$bitsize,
    DefaultValuedOptionalAttr<I32Attr, "0">:$batch_size,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$are_points_shared
  );
  let results = (outs MHLO_Tensor:$result);
  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];
  let hasCustomHLOConverter = 1;
}

#endif // ZKX_MLIR_HLO_MHLO_IR_HLO_OPS_TD
