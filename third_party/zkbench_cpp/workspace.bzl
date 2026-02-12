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

ZKBENCH_CPP_COMMIT = "46ee11729b6fb2cc48699632e3a4c6fa90ba2a54"
ZKBENCH_CPP_SHA256 = "c4647240af3f84e2b15f3f6859b10872d87be497b6a24181605b053b8fbb8a70"

# repo_mapping resolves naming differences between bzlmod (used by zkbench-cpp)
# and WORKSPACE (used by prime-ir):
#   @google_benchmark  -> @com_google_benchmark
#   @googletest        -> @com_google_googletest
_REPO_MAPPING = {
    "@google_benchmark": "@com_google_benchmark",
    "@googletest": "@com_google_googletest",
}

def repo():
    """Imports zkbench-cpp and sets up dependency aliases."""

    http_archive(
        name = "zkbench_cpp",
        sha256 = ZKBENCH_CPP_SHA256,
        strip_prefix = "zkbench-cpp-{commit}".format(commit = ZKBENCH_CPP_COMMIT),
        urls = ["https://github.com/fractalyze/zkbench-cpp/archive/{commit}.tar.gz".format(commit = ZKBENCH_CPP_COMMIT)],
        repo_mapping = _REPO_MAPPING,
    )
    # Uncomment this for development!
    # native.local_repository(
    #     name = "zkbench_cpp",
    #     path = "../zkbench-cpp",
    #     repo_mapping = _REPO_MAPPING,
    # )
