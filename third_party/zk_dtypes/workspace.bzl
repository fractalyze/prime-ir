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
    ZK_DTYPES_COMMIT = "760107d1e95f0af3eb1242d4f19f2d731e9e011b"
    ZK_DTYPES_SHA256 = "c80156921eca72225667ceddf985f93640da36a487b213acd8b4a404c3617fd4"
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
