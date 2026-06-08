# Copyright 2025 The PrimeIR Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# buildifier: disable=module-docstring
LLVM_COMMIT = "815edc3ff646392bfee2b381d37dd35e4b04f9c5"

LLVM_SHA256 = "bbc6fa4993162bdc7dd39b53927906a455940b87c782d0a37e00127f8bf8c696"

# TODO(chokobole): We must review the applied patches below and remove any that
# are not strictly necessary for this project.
# NOTE(chokobole): The order of the patches is important. If you update this,
# please update the order in the tools/setup_llvm_clone.sh script.
LLVM_PATCHES = [
    # Add visited set to MemRefDependenceGraph::hasDependencePath. Without it
    # the DFS path search enumerates every path through the MDG, which is
    # exponential when many memref ops touch the same buffer (e.g. fully
    # unrolled SIMD-like bodies). affine-loop-fusion hangs at 99% CPU on
    # such inputs. Pending upstream submission.
    "@prime_ir//third_party/llvm-project:affine_loop_fusion_visited_set.patch",
    # Cache MemRefDependenceGraph edge lookups by-reference in
    # hasDependencePath / hasEdge. With the visited-set fix above bounding
    # the DFS, the next dominant cost on dense MDG inputs is DenseMap::lookup
    # returning SmallVector<Edge> by value on every iteration. Pending
    # upstream submission.
    "@prime_ir//third_party/llvm-project:affine_dependence_path_lookup_cache.patch",
    # NOTE(chokobole): Patches for supporting PrimeIR Dialects.
    "@prime_ir//third_party/llvm-project:linalg_type_support.patch",
    "@prime_ir//third_party/llvm-project:tensor_type_support.patch",
    "@prime_ir//third_party/llvm-project:vector_type_support.patch",
    "@prime_ir//third_party/llvm-project:lazy_linking.patch",
    "@prime_ir//third_party/llvm-project:elementwise_op_fusion_constant_support.patch",
    "@prime_ir//third_party/llvm-project:constant_like_interface.patch",
    # Adds OpAsmParser::resetToken — used by parseOptionalFieldConstant for
    # the speculative-parse-then-rewind pattern that disambiguates
    # field-typed dense literals from f32/i32 ones (the first token is the
    # same `dense` keyword in both cases, so MLIR's first-token-dispatch
    # convention can't be applied).
    "@prime_ir//third_party/llvm-project:asm_parser_rewind.patch",
]
