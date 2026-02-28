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
    STABLEHLO_COMMIT = "a3523150d0de61325abb8543f6d6cb80690d9529"
    STABLEHLO_SHA256 = "9778f98a5336c56e1179b9cd4548b0be2907b319c229395f967187af21f38c15"
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
