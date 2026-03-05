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

"""ZKX rules."""

# See https://semver.org/
VERSION_MAJOR = 1
VERSION_MINOR = 0
VERSION_PATCH = 0
VERSION_PRERELEASE = ""
VERSION = ".".join([str(VERSION_MAJOR), str(VERSION_MINOR), str(VERSION_PATCH)])

# Platform specific conditions
def if_android(a, b = []):
    return select({
        "@platforms//os:android": a,
        "//conditions:default": b,
    })

def if_aarch64(a, b = []):
    return select({
        "@platforms//cpu:aarch64": a,
        "//conditions:default": b,
    })

def if_arm(a, b = []):
    return select({
        "@platforms//cpu:arm": a,
        "//conditions:default": b,
    })

def if_linux(a, b = []):
    return select({
        "@platforms//os:linux": a,
        "//conditions:default": b,
    })

def if_macos(a, b = []):
    return select({
        "@platforms//os:macos": a,
        "//conditions:default": b,
    })

def if_posix(a, b = []):
    return select({
        "@platforms//os:windows": b,
        "//conditions:default": a,
    })

def if_windows(a, b = []):
    return select({
        "@platforms//os:windows": a,
        "//conditions:default": b,
    })

def if_x86_32(a, b = []):
    return select({
        "@platforms//cpu:x86_32": a,
        "//conditions:default": b,
    })

def if_x86_64(a, b = []):
    return select({
        "@platforms//cpu:x86_64": a,
        "//conditions:default": b,
    })

# Feature specific conditions
def if_has_exception(a, b = []):
    return select({
        "@zkx//:zkx_has_exception": a,
        "//conditions:default": b,
    })
