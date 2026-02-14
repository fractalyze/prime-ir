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
    STABLEHLO_COMMIT = "7b9d1021a0b3309d120ea2cb7cab50c6a95d068c"
    STABLEHLO_SHA256 = "63b9e7f3012ff7b724f2e399715dfdc23e894b483c4aa0ccf3eb3f61466ec0dc"
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
