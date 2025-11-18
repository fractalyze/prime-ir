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

#ifndef ZKIR_DIALECT_POLY_IR_POLYCANONICALIZATION_TD_
#define ZKIR_DIALECT_POLY_IR_POLYCANONICALIZATION_TD_

include "zkir/Dialect/Poly/IR/PolyOps.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"

def Equal : Constraint<CPred<"$0 == $1">>;
def BoolNotEqual : Constraint<CPred<"$0.getValue() != $1.getValue()">, "not equal">;
def BoolEqual : Constraint<CPred<"$0.getValue() == $1.getValue()">, "equal">;

def INTTAfterNTT : Pat<
  (Poly_NTTOp (Poly_NTTOp $tensor, $dest1, $twiddles1, $r1, $tileCap1, $gridSize1, $bitReverse1, $inverse1), $dest2, $twiddles2, $r2, $tileCap2, $gridSize2, $bitReverse2, $inverse2),
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
