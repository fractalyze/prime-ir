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
    "func::FuncDialect",
    // The outlined helpers carry llvm.linkage / llvm.no_inline attributes.
    "LLVM::LLVMDialect",
    "scf::SCFDialect",
  ];
  let options = [
    Option<"outlineTowerOps", "outline-tower-ops", "bool", /*default=*/"false",
           "Emit tower-basis mul/square as calls to shared `__prime_ir_bf_*` "
           "helpers instead of expanding the 3^k Karatsuba recursion inline. "
           "Cuts IR size by orders of magnitude for wide towers (a bf<7> "
           "divide is ~68k arith ops inline), at the cost of one call per "
           "tower level. Off by default: the helpers are `noinline`, so like "
           "the Montgomery outlining in #367 this trades runtime for compile "
           "time and the right default is target-dependent.">,
    Option<"outlineMinTowerLevel", "outline-min-tower-level", "unsigned",
           /*default=*/"4",
           "Lowest tower level emitted as an outlined helper when "
           "`outline-tower-ops` is set. Levels below this still expand "
           "inline, which bounds call depth: a lower threshold means smaller "
           "IR but more calls per multiply.">,
  ];
}

#endif  // PRIME_IR_DIALECT_FIELD_CONVERSIONS_BINARYFIELDTOARITH_BINARYFIELDTOARITH_TD_
