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

package(default_visibility = ["//visibility:public"])

HEADERS = [
    "omp.h",
    "ompx.h",
    "omp-tools.h",
    "ompt.h",
]

# NOTE: This is only for macos.
#       On other platforms openmp should work without local config.
genrule(
    name = "link_headers",
    outs = HEADERS,
    cmd = select({
        "@prime_ir//:macos_x86_64": """
      mkdir -p $(@D)/
      for file in $(OUTS); do
        file=$${file##*/}
        ln -sf /usr/local/include/$$file $(@D)/$$file
      done
    """,
        "@prime_ir//:macos_aarch64": """
      mkdir -p $(@D)/
      for file in $(OUTS); do
        file=$${file##*/}
        ln -sf /opt/homebrew/opt/libomp/include/$$file $(@D)/$$file
      done
    """,
        "//conditions:default": "",
    }),
)

cc_library(
    name = "omp",
    hdrs = select({
        "@prime_ir//:macos_x86_64": HEADERS,
        "@prime_ir//:macos_aarch64": HEADERS,
        "//conditions:default": [],
    }),
    copts = select({
        "@prime_ir//:macos_x86_64": ["-Xclang -fopenmp"],
        "@prime_ir//:macos_aarch64": ["-Xclang -fopenmp"],
        "//conditions:default": [],
    }),
    include_prefix = "third_party/omp/include",
    includes = ["."],
    linkopts = select({
        "@prime_ir//:macos_x86_64": [
            "-L/usr/local/opt/libomp/lib",
            "-lomp",
        ],
        "@prime_ir//:macos_aarch64": [
            "-L/opt/homebrew/opt/libomp/lib",
            "-lomp",
        ],
        "//conditions:default": ["-lomp5"],
    }),
)
