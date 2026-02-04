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

package(default_visibility = ["//visibility:public"])

licenses(["unencumbered"])  # Public Domain or MIT

cc_library(
    name = "jsoncpp",
    srcs = glob([
        "src/lib_json/*.cpp",
        "src/lib_json/*.h",
        "src/lib_json/*.inl",
    ]),
    hdrs = glob(["include/json/*.h"]),
    includes = ["include"],
    copts = ["-Iexternal/jsoncpp/src/lib_json"],
)
