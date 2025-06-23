#ifndef ZKIR_DIALECT_POLY_IR_POLYCANONICALIZATION_TD_
#define ZKIR_DIALECT_POLY_IR_POLYCANONICALIZATION_TD_

include "zkir/Dialect/Poly/IR/PolyOps.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"

def Equal : Constraint<CPred<"$0 == $1">>;

def INTTAfterNTT : Pat<
  (Poly_INTTOp (Poly_NTTOp $tensor, $r1, $bitReverse1), $r2, $bitReverse2),
  (replaceWithValue $tensor),
  [(Equal $r1, $r2), (Equal $bitReverse1, $bitReverse2)]
>;

def NTTAfterINTT : Pat<
  (Poly_NTTOp (Poly_INTTOp $tensor, $r1, $bitReverse1), $r2, $bitReverse2),
  (replaceWithValue $tensor),
  [(Equal $r1, $r2), (Equal $bitReverse1, $bitReverse2)]
>;

def FromToTensor : Pat<
  (Poly_FromTensorOp (Poly_ToTensorOp $poly)),
  (replaceWithValue $poly)
>;

def ToFromTensor : Pat<
  (Poly_ToTensorOp (Poly_FromTensorOp $tensor)),
  (replaceWithValue $tensor)
>;

#endif  // LIB_DIALECT_POLYNOMIAL_IR_POLYNOMIALCANONICALIZATION_TD_
