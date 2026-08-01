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

#ifndef PRIME_IR_DIALECT_FIELD_CONVERSIONS_SPECIALIZEBINARYFIELDTONVPTX_TD
#define PRIME_IR_DIALECT_FIELD_CONVERSIONS_SPECIALIZEBINARYFIELDTONVPTX_TD

include "mlir/Pass/PassBase.td"

def SpecializeBinaryFieldToNVPTX : Pass<"specialize-binary-field-to-nvptx", "ModuleOp"> {
  let summary = "Specialize binary field operations to NVPTX clmad instructions.";
  let description = [{
    Specializes GHASH-basis binary field multiplication to the hardware
    carryless multiply-add instruction `clmad` (PTX ISA 9.3) for the CUDA GPU
    backend.

    The GPU analog of `SpecializeBinaryFieldToX86` (PCLMULQDQ) and
    `SpecializeBinaryFieldToARM` (PMULL): it rewrites `field.mul` on
    `bf<7, ghash>` into a 128x128 carryless product built from eight
    `clmad.{lo,hi}.u64`, then the shared GHASH polynomial reduction
    (x¹²⁸ + x⁷ + x² + x + 1). Tower `bf<4>`/`bf<5>` multiplies also
    specialize: a carryless product does not compute a tower product
    directly, so they convert through the isomorphic flat polynomial basis
    (constants in `TowerFlatBasis.h`), multiply with one `clmad.lo.u64`
    plus a low-weight reduction, and convert back. Native flat multiplies
    (`bf<3, aes>` and `bf<4|5, flat>`) take the same clmad product with no
    conversion at all — the fast path for emitters that hoist the basis
    change out of a kernel. Operations not specialized fall through to the
    portable software carryless multiply in `binary-field-to-arith`.

    The flat-basis multiplies (`bf<7, ghash>`, `bf<3, aes>`) also handle
    shaped (tensor/vector) operands: since `clmad` is scalar inline asm, a
    shaped `field.mul` is unrolled lane by lane into scalar clmad ladders
    reassembled with `tensor.from_elements`/`vector.from_elements`. This is
    what makes the batched gf8 additive-NTT butterfly multiply (flock
    zerocheck) emit `clmad` rather than the software shift-XOR fallback.
    Tower multiplies stay scalar-only.

    This pass must run BEFORE `binary-field-to-arith`. On the GPU codegen path
    it is inserted directly ahead of `createBinaryFieldToArith` — the GPU
    pipeline does not use the `field-to-llvm` pipeline where the x86/ARM
    specializers sit.

    The emitted `clmad` requires a CUDA-13.3+ ptxas to assemble; on older
    toolchains leave this pass disabled and the portable path is used.
  }];

  let dependentDialects = [
    "arith::ArithDialect",
    "LLVM::LLVMDialect",
    "tensor::TensorDialect",
    "vector::VectorDialect",
  ];

  let options = [
    Option<"useClmad", "use-clmad", "bool", /*default=*/"true",
           "Use clmad instructions for GHASH-basis binary field multiplication">,
  ];
}

#endif // PRIME_IR_DIALECT_FIELD_CONVERSIONS_SPECIALIZEBINARYFIELDTONVPTX_TD
