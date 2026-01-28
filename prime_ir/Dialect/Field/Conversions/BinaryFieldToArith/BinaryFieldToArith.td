/* Copyright 2026 The PrimeIR Authors.

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

#ifndef PRIME_IR_DIALECT_FIELD_CONVERSIONS_BINARYFIELDTOARITH_BINARYFIELDTOARITH_TD_
#define PRIME_IR_DIALECT_FIELD_CONVERSIONS_BINARYFIELDTOARITH_BINARYFIELDTOARITH_TD_

include "mlir/IR/PatternBase.td"
include "mlir/Pass/PassBase.td"

def BinaryFieldToArith : Pass<"binary-field-to-arith", "ModuleOp"> {
  let summary = "Lower binary field operations to `arith` dialect.";
  let description = [{
    This pass lowers binary field (GF(2^n)) operations directly to their
    `arith` equivalents. Unlike prime fields which use modular arithmetic,
    binary fields use XOR-based operations:

    - field.add -> arith.xori (XOR)
    - field.sub -> arith.xori (same as add in characteristic 2)
    - field.negate -> identity (no-op in characteristic 2)
    - field.double -> constant 0 (2x = 0 in characteristic 2)
    - field.mul -> Karatsuba tower multiplication
    - field.square -> optimized squaring
    - field.inverse -> a^(2^bits - 2) via repeated squaring
  }];
  let dependentDialects = [
    "arith::ArithDialect",
    "scf::SCFDialect",
  ];
}

#endif  // PRIME_IR_DIALECT_FIELD_CONVERSIONS_BINARYFIELDTOARITH_BINARYFIELDTOARITH_TD_
