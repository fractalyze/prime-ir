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

"""Fetches the patched llvm-project source archive, before it is configured.

Split from `//bazel:llvm_configure.bzl` because that file has to `load()` out of
`@llvm-raw`, which cannot exist until this extension has run. Two extensions in
two files is what orders them: MODULE.bazel `use_repo`s `llvm-raw` from here, so
the other file's load resolves.

llvm-project is patched, and bzlmod applies `patches` only for the root module
when they hang off a `bazel_dep` override. Here they hang off an `http_archive`
inside a module extension, which runs the same way whoever the root module is —
so a consumer of prime_ir gets the patched LLVM, not a silently unpatched one.
"""

load("//third_party/llvm-project:workspace.bzl", llvm_project = "repo")

def _llvm_deps_impl(_module_ctx):
    llvm_project()

llvm_deps = module_extension(implementation = _llvm_deps_impl)
