# Copyright 2025 The ZKIR Authors.
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
LLVM_COMMIT = "5ed852f7f72855710eeff53179e6a6f2271a3c2a"

LLVM_SHA256 = "95792e50d5f84847721545b645a6ca2c2b3b7610d02e3de07d65a6148e68508c"

# TODO(chokobole): We must review the applied patches below and remove any that
# are not strictly necessary for this project.
# NOTE(chokobole): The order of the patches is important. If you update this,
# please update the order in the tools/setup_llvm_clone.sh script.
LLVM_PATCHES = [
    # TODO(chokobole): Remove once the issues are resolved upstream.
    "@zkir//third_party/llvm-project:cuda_runtime.patch",
    "@zkir//third_party/llvm-project:kernel_outlining.patch",
    "@zkir//third_party/llvm-project:nvptx_lowering.patch",
    # TODO(chokobole): Remove owning_memref_free.patch once we upgrade the version of LLVM.
    # See https://github.com/llvm/llvm-project/pull/153133
    "@zkir//third_party/llvm-project:owning_memref_free.patch",
    # TODO(chokobole): Remove owning_memref_memset.patch once we upgrade the version of LLVM.
    # See https://github.com/llvm/llvm-project/pull/158200
    "@zkir//third_party/llvm-project:owning_memref_memset.patch",
    # NOTE(chokobole): Patches for supporting ZKIR Dialects.
    "@zkir//third_party/llvm-project:linalg_type_support.patch",
    "@zkir//third_party/llvm-project:tensor_type_support.patch",
    "@zkir//third_party/llvm-project:vector_type_support.patch",
    "@zkir//third_party/llvm-project:memref_folding.patch",
    "@zkir//third_party/llvm-project:lazy_linking.patch",
]
