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

load("@rules_cc//cc:cc_library.bzl", "cc_library")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "nanobind",
    srcs = glob(
        [
            "src/*.cpp",
        ],
        exclude = ["src/nb_combined.cpp"],
    ),
    defines = select({
        "@rules_python//python/config_settings:is_py_freethreaded": [
            "NB_FREE_THREADED=1",
            "NB_BUILD=1",
            "NB_SHARED=1",
        ],
        "//conditions:default": [
            "NB_BUILD=1",
            "NB_SHARED=1",
        ],
    }),
    # The Python headers arrive through a toolchain-resolved alias, and
    # `layering_check` cannot follow `current_py_cc_headers` to a module map, so
    # it reports `Python.h` as undeclared even though the dependency below is
    # exactly the one that provides it.
    features = ["-layering_check"],
    includes = ["include"],
    textual_hdrs = glob(
        [
            "include/**/*.h",
            "src/*.h",
        ],
    ),
    visibility = ["//visibility:public"],
    deps = [
        "@robin_map",
        "@rules_python//python/cc:current_py_cc_headers",
    ],
)
