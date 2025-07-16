#ifndef ZKIR_DIALECT_POLY_IR_POLYCANONICALIZATION_TD_
#define ZKIR_DIALECT_POLY_IR_POLYCANONICALIZATION_TD_

include "zkir/Dialect/Poly/IR/PolyOps.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"

def Equal : Constraint<CPred<"$0 == $1">>;
def BoolNotEqual : Constraint<CPred<"$0.getValue() != $1.getValue()">, "not equal">;
def BoolEqual : Constraint<CPred<"$0.getValue() == $1.getValue()">, "equal">;

def INTTAfterNTT : Pat<
  (Poly_NTTOp (Poly_NTTOp $tensor, $twiddles1, $dest1, $r1, $bitReverse1, $inverse1), $twiddles2, $dest2, $r2, $bitReverse2, $inverse2),
  (replaceWithValue $tensor),
  [(Equal $twiddles1, $twiddles2), (Equal $r1, $r2), (BoolNotEqual $inverse1, $inverse2), (BoolEqual $bitReverse1, $bitReverse2)]
>;

def FromToTensor : Pat<
  (Poly_FromTensorOp (Poly_ToTensorOp $poly)),
  (replaceWithValue $poly)
>;

def ToFromTensor : Pat<
  (Poly_ToTensorOp (Poly_FromTensorOp $tensor)),
  (replaceWithValue $tensor)
>;

#endif  // ZKIR_DIALECT_POLY_IR_POLYCANONICALIZATION_TD_
