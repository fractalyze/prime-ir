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

"""The bzlmod half of `//bazel:prime_ir_deps.bzl`.

`prime_ir_deps()` is what `WORKSPACE.bazel` calls; this extension calls the same
function, so the two lanes fetch one set of pins from one place. Everything it
declares either ships no Bazel module (nanobind, pybind11 and robin_map are
built from BUILD files this repository carries, which a registry module would
replace) or is generated on the host (`local_config_omp`). A dependency the
registry does carry belongs in MODULE.bazel as a `bazel_dep` instead, so that
consumers resolve one shared version of it with us.
"""

load("//bazel:prime_ir_deps.bzl", "prime_ir_deps")

def _non_module_deps_impl(_module_ctx):
    prime_ir_deps()

non_module_deps = module_extension(implementation = _non_module_deps_impl)
