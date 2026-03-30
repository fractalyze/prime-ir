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

"""RISC-V Witness C++ build macros with project-wide compiler settings."""

load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")

_PRIME_IR_COPTS = [
    "-Werror=all",
]

def prime_ir_cc_library(copts = [], **kwargs):
    cc_library(
        copts = _PRIME_IR_COPTS + copts,
        **kwargs
    )

def prime_ir_cc_binary(copts = [], **kwargs):
    cc_binary(
        copts = _PRIME_IR_COPTS + copts,
        **kwargs
    )

def prime_ir_cc_test(copts = [], **kwargs):
    cc_test(
        copts = _PRIME_IR_COPTS + copts,
        **kwargs
    )
