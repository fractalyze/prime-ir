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

#ifndef ZKIR_TENSOREXT_IR_TENSOREXT_CANONICALIZATION_TD
#define ZKIR_TENSOREXT_IR_TENSOREXT_CANONICALIZATION_TD

include "mlir/IR/PatternBase.td"
include "zkir/Dialect/Field/IR/FieldOps.td"
include "zkir/Dialect/TensorExt/IR/TensorExtOps.td"

// bit_reverse(bit_reverse(x)) -> x
def BitReverseIsInvolution : Pat<(TensorExt_BitReverseOp (TensorExt_BitReverseOp $x, $dst0), $dst1), (replaceWithValue $x)>;

// bit_reverse(mul(bit_reverse(x), y)) -> mul(x, bit_reverse(y))
def BitReverseMulBitReverse :
    Pat<(TensorExt_BitReverseOp (Field_MulOp (TensorExt_BitReverseOp $x, $dst0), $y), $dst1),
        (Field_MulOp $x, (TensorExt_BitReverseOp $y, $y))>;


#endif // ZKIR_TENSOREXT_IR_TENSOREXT_CANONICALIZATION_TD
