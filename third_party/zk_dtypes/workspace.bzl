# Copyright 2025 The PrimeIR Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# buildifier: disable=module-docstring
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def repo():
    ZK_DTYPES_COMMIT = "d880d1c465baa4efb5706b5ba38b70cc6f2501e1"
    ZK_DTYPES_SHA256 = "c044948de97063bc5504f6b4c537a4a924f203147ef1c64e9d781c8ad0db2ad9"
    http_archive(
        name = "zk_dtypes",
        sha256 = ZK_DTYPES_SHA256,
        strip_prefix = "zk_dtypes-{commit}".format(commit = ZK_DTYPES_COMMIT),
        urls = ["https://github.com/fractalyze/zk_dtypes/archive/{commit}/zk_dtypes-{commit}.tar.gz".format(commit = ZK_DTYPES_COMMIT)],
    )
    # Uncomment this for development!
    # native.local_repository(
    #     name = "zk_dtypes",
    #     path = "../zk_dtypes",
    # )
