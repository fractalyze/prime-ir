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

"""Tooling a contributor runs by hand; no build target depends on it.

`WORKSPACE.bazel` calls `prime_ir_dev_deps()` and MODULE.bazel wraps it in a
`dev_dependency` extension, so a consumer of prime_ir never fetches any of it.
"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def prime_ir_dev_deps():
    """Declares the repositories only a contributor's own commands need."""

    # Hedron's Compile Commands Extractor for Bazel, which backs
    # `bazel run @hedron_compile_commands//:refresh_all`.
    # https://github.com/hedronvision/bazel-compile-commands-extractor
    http_archive(
        name = "hedron_compile_commands",
        patch_args = ["-p1"],
        patches = [Label("@zk_dtypes//third_party/bazel-compile-commands-extractor:allow_header_as_a_source_file.patch")],
        strip_prefix = "bazel-compile-commands-extractor-ed994039a951b736091776d677f324b3903ef939",
        url = "https://github.com/hedronvision/bazel-compile-commands-extractor/archive/ed994039a951b736091776d677f324b3903ef939.tar.gz",
    )

def _dev_deps_impl(_module_ctx):
    prime_ir_dev_deps()

dev_deps = module_extension(implementation = _dev_deps_impl)
