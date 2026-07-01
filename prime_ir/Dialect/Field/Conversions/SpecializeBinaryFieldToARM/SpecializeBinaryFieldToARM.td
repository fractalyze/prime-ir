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

#ifndef PRIME_IR_DIALECT_FIELD_CONVERSIONS_SPECIALIZEBINARYFIELDTOARM_TD
#define PRIME_IR_DIALECT_FIELD_CONVERSIONS_SPECIALIZEBINARYFIELDTOARM_TD

include "mlir/Pass/PassBase.td"

def SpecializeBinaryFieldToARM : Pass<"specialize-binary-field-to-arm", "ModuleOp"> {
  let summary = "Specialize binary field operations to ARM NEON/PMULL instructions.";
  let description = [{
    This pass specializes binary field operations to use ARM hardware-accelerated
    instructions when available:

    - **PMULL (vmull_p64)**: For 64-bit and 128-bit binary field multiplication
      using carryless multiply with polynomial reduction.
      Available on ARMv8 CPUs with Crypto extensions.

    - **PMULL (vmull_p8)**: For packed 8-bit binary field multiplication.
      Available on ARMv8 CPUs with NEON.

    This pass should run BEFORE `binary-field-to-arith` to specialize
    operations that can benefit from hardware acceleration. Operations not
    specialized by this pass will fall through to the portable Karatsuba
    implementation in `binary-field-to-arith`.

    Performance notes:
    - PMULL provides significant speedup for field multiplication
    - Falls back to portable implementation on unsupported hardware
  }];

  let dependentDialects = [
    "arith::ArithDialect",
    "LLVM::LLVMDialect",
    "vector::VectorDialect",
  ];

  let options = [
    Option<"usePMULL", "use-pmull", "bool", /*default=*/"true",
           "Use PMULL instructions for binary field operations">,
  ];
}

#endif // PRIME_IR_DIALECT_FIELD_CONVERSIONS_SPECIALIZEBINARYFIELDTOARM_TD
