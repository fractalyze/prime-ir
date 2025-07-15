#ifndef ZKIR_TENSOREXT_IR_TENSOREXT_CANONICALIZATION_TD
#define ZKIR_TENSOREXT_IR_TENSOREXT_CANONICALIZATION_TD

include "mlir/IR/PatternBase.td"
include "zkir/Dialect/TensorExt/IR/TensorExtOps.td"

// bit_reverse(bit_reverse(x)) -> x
def IsInvolution : Pat<(TensorExt_BitReverseOp (TensorExt_BitReverseOp $x, $y), $z), (replaceWithValue $x)>;

#endif // ZKIR_TENSOREXT_IR_TENSOREXT_CANONICALIZATION_TD
