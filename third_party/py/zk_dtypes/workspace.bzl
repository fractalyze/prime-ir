# Copyright 2025 The ZKX Authors.
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

"""Provides the repo macro to import zk_dtypes.

zk_dtypes provides ZK-specific data-types like babybear.
"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def repo():
    ZK_DTYPES_COMMIT = "447ae5bfa94aa79b13d46b53e0d194ab0d2da7f6"
    ZK_DTYPES_SHA256 = "798f2ba932154671ee46bf214b361d999b6402a3dc0239a7eeb793e5c98f88fb"
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
