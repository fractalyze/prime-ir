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

"""Loads the pybind11 library."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def repo():
    http_archive(
        name = "pybind11",
        strip_prefix = "pybind11-2.10.3",
        sha256 = "201966a61dc826f1b1879a24a3317a1ec9214a918c8eb035be2f30c3e9cfbdcb",
        urls = ["https://github.com/pybind/pybind11/archive/v2.10.3.zip"],
        build_file = "//third_party/pybind11:pybind11.BUILD",
    )
