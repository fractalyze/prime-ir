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
LLVM_COMMIT = "668f1c01624b2a7b15bc1639c49f6d0b39cd2e32"

LLVM_SHA256 = "ba4e2a11a50c52b1ba459da349fb723596d3fc0e79edac10a2811dca441a3b9b"

# TODO(chokobole): We must review the applied patches below and remove any that
# are not strictly necessary for this project.
# NOTE(chokobole): The order of the patches is important. If you update this,
# please update the order in the tools/setup_llvm_clone.sh script.
LLVM_PATCHES = [
    # TODO(chokobole): Remove once the issues are resolved upstream.
    "@prime_ir//third_party/llvm-project:cuda_runtime.patch",
    "@prime_ir//third_party/llvm-project:nvptx_lowering.patch",
    # Compatibility patch for rules_cc version mismatch
    "@prime_ir//third_party/llvm-project:rules_cc_compat.patch",
    # NOTE(chokobole): Patches for supporting PrimeIR Dialects.
    "@prime_ir//third_party/llvm-project:linalg_type_support.patch",
    "@prime_ir//third_party/llvm-project:tensor_type_support.patch",
    "@prime_ir//third_party/llvm-project:vector_type_support.patch",
    # NOTE: memref_folding.patch is no longer needed - the fix has been applied upstream
    # "@prime_ir//third_party/llvm-project:lazy_linking.patch",
]
