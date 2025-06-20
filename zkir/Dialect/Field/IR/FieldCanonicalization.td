#ifndef ZKIR_FIELD_IR_FIELD_CANONICALIZATION_TD
#define ZKIR_FIELD_IR_FIELD_CANONICALIZATION_TD

include "mlir/IR/PatternBase.td"
include "zkir/Dialect/Field/IR/FieldOps.td"
include "zkir/Dialect/TensorExt/IR/TensorExtOps.td"

// bit_reverse(mul(bit_reverse(x), y)) -> mul(x, bit_reverse(y))
def BitReverseMulBitReverse : Pat<(TensorExt_BitReverseOp (Field_MulOp (TensorExt_BitReverseOp $x), $y)), (Field_MulOp $x, (TensorExt_BitReverseOp $y))>;

#endif // ZKIR_FIELD_IR_FIELD_CANONICALIZATION_TD
