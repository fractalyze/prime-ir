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

"""Provides the repo macro to import nlohmann/json."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

NLOHMANN_JSON_VERSION = "3.11.3"
NLOHMANN_JSON_SHA256 = "a22461d13119ac5c78f205d3df1db13403e58ce1bb1794edc9313677313f4a9d"

def repo():
    """Imports nlohmann/json."""
    http_archive(
        name = "nlohmann_json",
        sha256 = NLOHMANN_JSON_SHA256,
        urls = ["https://github.com/nlohmann/json/releases/download/v{version}/include.zip".format(version = NLOHMANN_JSON_VERSION)],
        build_file_content = """
load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "json",
    hdrs = glob(["include/**/*.hpp"]),
    includes = ["include"],
    visibility = ["//visibility:public"],
)
""",
    )
