/* Copyright 2025 The PrimeIR Authors.

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

#ifndef PRIME_IR_DIALECT_RING_CONVERSIONS_RINGTOMODARITH_TD_
#define PRIME_IR_DIALECT_RING_CONVERSIONS_RINGTOMODARITH_TD_

include "mlir/Pass/PassBase.td"

def RingToModArith : Pass<"ring-to-mod-arith", "ModuleOp"> {
  let summary = "Lower the `ring` dialect (RNS) to per-limb `mod_arith`";
  let description = [{
    Lowers `!ring.rq<[q0 ... qL], N>` to `tensor<LxNxi64>` (one raw residue per
    (limb, coefficient)) and expands the RNS ops into per-limb modular arithmetic
    on `mod_arith.int<q_i>`, so every reduction goes through the ModArith
    Barrett/Montgomery reducers (NOT `arith.remui`, which has no native GPU
    integer divide). `ring.base_convert` becomes the CRT fast-basis-extension:
    `y_i = x_i * yHatInv_i mod q_i` (a `mod_arith.mul` in q_i), then
    `out_j = sum_i y_i * (Q/q_i mod p_j) mod p_j` (`mod_arith.mul`/`add` in p_j).
    Barrett reduces each PRODUCT, so a raw `y_i` reinterpreted into `mod_arith<p_j>`
    needs no separate reduction. CRT constants precomputed (APInt) from the moduli.
  }];
  let dependentDialects = [
    "tensor::TensorDialect",
    "prime_ir::field::FieldDialect",
    "prime_ir::mod_arith::ModArithDialect",
    "prime_ir::poly::PolyDialect",
    "prime_ir::ring::RingDialect",
  ];
}

#endif // PRIME_IR_DIALECT_RING_CONVERSIONS_RINGTOMODARITH_TD_
