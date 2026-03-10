# Copyright 2026 The ZKX Authors.
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

"""Provides the repo macro to import StableHLO."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def repo():
    STABLEHLO_COMMIT = "db02fb59375e463faefe5edab4fa36c98cb8036c"
    STABLEHLO_SHA256 = "d8699da61c616200f76e219a2645743a58478763f3e6ed1116119cdf9d496cc7"
    http_archive(
        name = "stablehlo",
        sha256 = STABLEHLO_SHA256,
        strip_prefix = "stablehlo-{commit}".format(commit = STABLEHLO_COMMIT),
        urls = ["https://github.com/fractalyze/stablehlo/archive/{commit}/stablehlo-{commit}.tar.gz".format(commit = STABLEHLO_COMMIT)],
    )
    # Uncomment this for development!
    # native.local_repository(
    #     name = "stablehlo",
    #     path = "../stablehlo",
    # )
