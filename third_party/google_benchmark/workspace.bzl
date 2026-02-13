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

"""Provides the repository macro to import Google Benchmark."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

VERSION = "1.9.4"

def repo():
    """Imports Google Benchmark."""

    http_archive(
        name = "com_google_benchmark",
        sha256 = "b334658edd35efcf06a99d9be21e4e93e092bd5f95074c1673d5c8705d95c104",
        strip_prefix = "benchmark-{version}".format(version = VERSION),
        urls = ["https://github.com/google/benchmark/archive/refs/tags/v{version}.tar.gz".format(version = VERSION)],
        repo_mapping = {"@googletest": "@com_google_googletest"},
    )
