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

"""
This module configures dependencies for the PrimeIR project.
"""

load("//third_party/benchmark:workspace.bzl", benchmark = "repo")
load("//third_party/nanobind:workspace.bzl", nanobind = "repo")
load("//third_party/omp:omp_configure.bzl", "omp_configure")
load("//third_party/pybind11:workspace.bzl", pybind11 = "repo")
load("//third_party/robin_map:workspace.bzl", robin_map = "repo")

# buildifier: disable=function-docstring
def prime_ir_deps():
    omp_configure(name = "local_config_omp")

    benchmark()
    nanobind()
    robin_map()
    pybind11()
