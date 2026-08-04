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

#ifndef PRIME_IR_DIALECT_ARITHEXT_CONVERSIONS_SPECIALIZEARITHTOAVX_SPECIALIZEARITHTOAVX_TD_
#define PRIME_IR_DIALECT_ARITHEXT_CONVERSIONS_SPECIALIZEARITHTOAVX_SPECIALIZEARITHTOAVX_TD_

include "mlir/Dialect/LLVMIR/LLVMOps.td"
include "mlir/IR/PatternBase.td"
include "mlir/Pass/PassBase.td"

def SpecializeArithToAVX : Pass<"specialize-arith-to-avx", "ModuleOp"> {
  let summary = "Specialize some `arith` ops to assembly";
  let description = [{
    This pass specializes some `arith` ops on `vector<16xi32>` to dual-lane
    (even/odd) sequences. The `flavor` option selects the target ISA:

    - `avx512` (default): 512-bit zmm inline asm using opmask-merged
      `vmovsldup`/`vmovshdup` interleaves. Requires AVX-512F at runtime.
    - `avx2`: the same dual-lane dataflow expressed as target-independent IR
      that legalizes to 2x ymm on AVX2-only targets: extended products are
      emitted as the masked 64-bit multiply idiom X86 selects to
      `vpmuludq`/`vpmuldq`, and interleaves as `vector.shuffle`.

    The caller is responsible for matching the flavor to the JIT/AOT target
    features; the pass itself does not inspect the host CPU.
  }];
  let options = [
    Option<"flavor", "flavor", "mlir::prime_ir::arith_ext::AVXFlavor",
           /*default=*/"mlir::prime_ir::arith_ext::AVXFlavor::kAVX512",
           "Instruction-set flavor to specialize for",
           [{::llvm::cl::values(
             clEnumValN(mlir::prime_ir::arith_ext::AVXFlavor::kAVX512,
                        "avx512",
                        "512-bit zmm inline asm with opmask interleaves"),
             clEnumValN(mlir::prime_ir::arith_ext::AVXFlavor::kAVX2, "avx2",
                        "2x 256-bit ymm via pmuludq idiom and shuffles")
           )}]>
  ];
  let dependentDialects = [
    "arith::ArithDialect",
    "LLVM::LLVMDialect",
    "vector::VectorDialect",
  ];
}

#endif // PRIME_IR_DIALECT_ARITHEXT_CONVERSIONS_SPECIALIZEARITHTOAVX_SPECIALIZEARITHTOAVX_TD_
