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

"""Provides the repository macro to import zkbench-cpp."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

ZKBENCH_CPP_COMMIT = "2e282b6d1228b2d3cf40e6288097ee3cde07fdc6"
ZKBENCH_CPP_SHA256 = "85cd66f6d5d5e93de57521cba38c4279beaa376a9834ced689efff3d1c62287e"

def repo():
    """Imports zkbench-cpp."""

    http_archive(
        name = "zkbench_cpp",
        sha256 = ZKBENCH_CPP_SHA256,
        strip_prefix = "zkbench-cpp-{commit}".format(commit = ZKBENCH_CPP_COMMIT),
        urls = ["https://github.com/fractalyze/zkbench-cpp/archive/{commit}.tar.gz".format(commit = ZKBENCH_CPP_COMMIT)],
        repo_mapping = {
            "@google_benchmark": "@com_google_benchmark",
            "@googletest": "@com_google_googletest",
        },
    )
