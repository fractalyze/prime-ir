# Copyright 2026 The PrimeIR Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Produces the `@llvm-project` the first-party BUILD files depend on.

LLVM ships no BUILD files of its own; `llvm_configure` overlays the ones under
`utils/bazel` onto the source archive `//bazel:llvm_deps.bzl` fetched.

The `load()` below is why this is a second file and a second extension: it reads
a `.bzl` out of `@llvm-raw`, so it cannot be evaluated until that repository
exists. MODULE.bazel declares the two in that order.
"""

load("@llvm-raw//utils/bazel:configure.bzl", "llvm_configure")

def _llvm_project_impl(_module_ctx):
    llvm_configure(name = "llvm-project")

llvm_project = module_extension(implementation = _llvm_project_impl)
