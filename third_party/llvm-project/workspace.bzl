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

"""Loads llvm-project and the two compression libraries its Bazel overlay needs.

Both dependency paths go through `repo()`: `WORKSPACE.bazel` calls it directly,
and `//bazel:llvm_deps.bzl` wraps it in the `llvm_deps` module extension. The
archives are declared here rather than at either call site so that the pin, the
patch list and their order have one home.
"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

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

def repo():
    """Declares llvm-raw at the pinned revision, plus llvm_zstd and llvm_zlib."""
    http_archive(
        name = "llvm-raw",
        build_file_content = "# empty",
        patch_args = ["-p1"],
        patches = LLVM_PATCHES,
        sha256 = LLVM_SHA256,
        strip_prefix = "llvm-project-" + LLVM_COMMIT,
        urls = ["https://github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT)],
    )

    # Uncomment this, plus its load at the top of the file, and comment out the
    # llvm-raw http_archive above, when following the llvm patch workflow from
    # CONTRIBUTING.md to point Bazel at a local clone. It is the Starlark
    # `new_local_repository` rather than the WORKSPACE native one, so the swap
    # works on the bzlmod lane too.
    # load("@bazel_tools//tools/build_defs/repo:local.bzl", "new_local_repository")
    # new_local_repository(
    #     name = "llvm-raw",
    #     build_file_content = "# empty",
    #     path = "../llvm-project",
    # )

    # This is needed since https://reviews.llvm.org/D143344.
    # Not sure if it's a bug or a feature, but it doesn't hurt to keep an additional
    # dependency here.
    http_archive(
        name = "llvm_zstd",
        build_file = "@llvm-raw//utils/bazel/third_party_build:zstd.BUILD",
        sha256 = "7c42d56fac126929a6a85dbc73ff1db2411d04f104fae9bdea51305663a83fd0",
        strip_prefix = "zstd-1.5.2",
        urls = [
            "https://github.com/facebook/zstd/releases/download/v1.5.2/zstd-1.5.2.tar.gz",
        ],
    )

    # This is needed since https://reviews.llvm.org/D143320
    # Not sure if it's a bug or a feature, but it doesn't hurt to keep an additional
    # dependency here.
    http_archive(
        name = "llvm_zlib",
        build_file = "@llvm-raw//utils/bazel/third_party_build:zlib-ng.BUILD",
        sha256 = "e36bb346c00472a1f9ff2a0a4643e590a254be6379da7cddd9daeb9a7f296731",
        strip_prefix = "zlib-ng-2.0.7",
        urls = [
            "https://github.com/zlib-ng/zlib-ng/archive/refs/tags/2.0.7.zip",
        ],
    )
