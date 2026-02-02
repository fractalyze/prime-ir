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

"""Provides the repo macro to import zkbench-cpp."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

ZKBENCH_CPP_COMMIT = "7263538c429e7be93a6d9a95c4011cfe59843637"
ZKBENCH_CPP_SHA256 = ""  # TODO: Add SHA256 after stable release

def repo():
    """Imports zkbench-cpp."""
    http_archive(
        name = "zkbench_cpp",
        sha256 = ZKBENCH_CPP_SHA256,
        strip_prefix = "zkbench-cpp-{commit}".format(commit = ZKBENCH_CPP_COMMIT),
        urls = ["https://github.com/fractalyze/zkbench-cpp/archive/{commit}.tar.gz".format(commit = ZKBENCH_CPP_COMMIT)],
    )
