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

#ifndef PRIME_IR_DIALECT_FIELD_CONVERSIONS_SPECIALIZEBINARYFIELDTOX86_TD
#define PRIME_IR_DIALECT_FIELD_CONVERSIONS_SPECIALIZEBINARYFIELDTOX86_TD

include "mlir/Pass/PassBase.td"

def SpecializeBinaryFieldToX86 : Pass<"specialize-binary-field-to-x86", "ModuleOp"> {
  let summary = "Specialize binary field operations to x86 GFNI/PCLMULQDQ instructions.";
  let description = [{
    This pass specializes binary field operations to use x86 hardware-accelerated
    instructions when available:

    - **GFNI (vgf2p8mulb)**: For packed 8-bit binary field multiplication.
      Requires isomorphism transformation between canonical tower field and
      AES field. Available on Ice Lake+ CPUs.

    - **PCLMULQDQ (vpclmulqdq)**: For 64-bit and 128-bit binary field
      multiplication using carryless multiply with polynomial reduction.
      Available on most modern x86 CPUs (Westmere+).

    This pass should run BEFORE `binary-field-to-arith` to specialize
    operations that can benefit from hardware acceleration. Operations not
    specialized by this pass will fall through to the portable Karatsuba
    implementation in `binary-field-to-arith`.

    Performance notes:
    - GFNI provides ~10x speedup for packed 8-bit field multiplication
    - PCLMULQDQ provides ~5x speedup for 128-bit field multiplication
    - Falls back to portable implementation on unsupported hardware
  }];

  let dependentDialects = [
    "arith::ArithDialect",
    "LLVM::LLVMDialect",
    "vector::VectorDialect",
  ];

  let options = [
    Option<"useGFNI", "use-gfni", "bool", /*default=*/"true",
           "Use GFNI instructions for 8-bit packed binary field operations">,
    Option<"usePCLMULQDQ", "use-pclmulqdq", "bool", /*default=*/"true",
           "Use PCLMULQDQ for 64/128-bit binary field multiplication">,
  ];
}

#endif // PRIME_IR_DIALECT_FIELD_CONVERSIONS_SPECIALIZEBINARYFIELDTOX86_TD
