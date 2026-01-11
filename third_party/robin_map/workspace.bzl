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

"""Loads the robin_map library."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def repo():
    http_archive(
        name = "robin_map",
        strip_prefix = "robin-map-1.3.0",
        sha256 = "a8424ad3b0affd4c57ed26f0f3d8a29604f0e1f2ef2089f497f614b1c94c7236",
        urls = ["https://github.com/Tessil/robin-map/archive/refs/tags/v1.3.0.tar.gz"],
        build_file = "//third_party/robin_map:robin_map.BUILD",
    )
