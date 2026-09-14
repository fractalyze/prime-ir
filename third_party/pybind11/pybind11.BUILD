# Copyright 2025 The PrimeIR Authors.
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

load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "pybind11",
    hdrs = glob(
        include = ["include/pybind11/**/*.h"],
        exclude = [
            # Deprecated file that just emits a warning
            "include/pybind11/common.h",
        ],
    ),
    # The Python headers arrive through a toolchain-resolved alias, and
    # `layering_check` cannot follow `current_py_cc_headers` to a module map, so
    # it reports `Python.h` as undeclared even though the dependency below is
    # exactly the one that provides it.
    features = ["-layering_check"],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = ["@rules_python//python/cc:current_py_cc_headers"],
)
