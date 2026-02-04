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

"""ZKX dependencies."""

load("@zkx//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")
load("@zkx//third_party/dlpack:workspace.bzl", dlpack = "repo")
load("@zkx//third_party/eigen3:workspace.bzl", eigen3 = "repo")
load("@zkx//third_party/farmhash:workspace.bzl", farmhash = "repo")
load("@zkx//third_party/gloo:workspace.bzl", gloo = "repo")
load("@zkx//third_party/gpus/cuda/hermetic:cuda_json_init_repository.bzl", "cuda_json_init_repository")
load("@zkx//third_party/highwayhash:workspace.bzl", highwayhash = "repo")
load("@zkx//third_party/implib_so:workspace.bzl", implib_so = "repo")
load("@zkx//third_party/jsoncpp:workspace.bzl", jsoncpp = "repo")
load("@zkx//third_party/nanobind:workspace.bzl", nanobind = "repo")
load("@zkx//third_party/prime_ir:workspace.bzl", prime_ir = "repo")
load("@zkx//third_party/robin_map:workspace.bzl", robin_map = "repo")
load("@zkx//third_party/stablehlo:workspace.bzl", stablehlo = "repo")
load("@zkx//third_party/uv:workspace.bzl", uv = "repo")
load("@zkx//third_party/version:workspace.bzl", version = "repo")

def zkx_deps():
    """ZKX dependencies."""

    eigen3()
    farmhash()
    gloo()
    highwayhash()
    implib_so()
    nanobind()
    dlpack()
    robin_map()
    stablehlo()
    uv()
    version()
    prime_ir()

    cuda_json_init_repository()

    # `apple_support` is needed to build on recent macOS versions (e.g., Tahoe 26.2)
    # to fix "missing LC_UUID load command" errors.
    # See https://github.com/tensorflow/tensorflow/pull/104948
    tf_http_archive(
        name = "build_bazel_apple_support",
        sha256 = "1ae6fcf983cff3edab717636f91ad0efff2e5ba75607fdddddfd6ad0dbdfaf10",
        urls = tf_mirror_urls("https://github.com/bazelbuild/apple_support/releases/download/1.24.5/apple_support.1.24.5.tar.gz"),
    )

    # TODO(chokobole): Delete after removing `build --noenable_bzlmod` from .bazelrc
    tf_http_archive(
        name = "bazel_skylib",
        sha256 = "bc283cdfcd526a52c3201279cda4bc298652efa898b10b4db0837dc51652756f",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.7.1/bazel-skylib-1.7.1.tar.gz",
            "https://github.com/bazelbuild/bazel-skylib/releases/download/1.7.1/bazel-skylib-1.7.1.tar.gz",
        ],
    )

    tf_http_archive(
        name = "com_google_absl",
        sha256 = "9b2b72d4e8367c0b843fa2bcfa2b08debbe3cee34f7aaa27de55a6cbb3e843db",
        strip_prefix = "abseil-cpp-20250814.0",
        urls = tf_mirror_urls("https://github.com/abseil/abseil-cpp/archive/refs/tags/20250814.0.tar.gz"),
        patch_file = [
            "@zkx//third_party/absl:btree.patch",
            "@zkx//third_party/absl:build_dll.patch",
            "@zkx//third_party/absl:check_op.patch",
            "@zkx//third_party/absl:check_op_2.patch",
            "@zkx//third_party/absl:endian.patch",
            "@zkx//third_party/absl:if_constexpr.patch",
            "@zkx//third_party/absl:rules_cc.patch",
        ],
        repo_mapping = {
            "@googletest": "@com_google_googletest",
        },
    )

    # Needed by com_google_googletest
    tf_http_archive(
        name = "com_googlesource_code_re2",
        sha256 = "ef516fb84824a597c4d5d0d6d330daedb18363b5a99eda87d027e6bdd9cba299",
        strip_prefix = "re2-03da4fc0857c285e3a26782f6bc8931c4c950df4",
        urls = tf_mirror_urls("https://github.com/google/re2/archive/03da4fc0857c285e3a26782f6bc8931c4c950df4.tar.gz"),
    )

    tf_http_archive(
        name = "com_google_googletest",
        # Use the commit on 2025/6/09:
        # https://github.com/google/googletest/commit/28e9d1f26771c6517c3b4be10254887673c94018
        sha256 = "f253ca1a07262f8efde8328e4b2c68979e40ddfcfc001f70d1d5f612c7de2974",
        strip_prefix = "googletest-28e9d1f26771c6517c3b4be10254887673c94018",
        # Patch googletest to:
        #   - avoid dependencies on @fuchsia_sdk,
        #   - refer to re2 as @com_googlesource_code_re2,
        #   - refer to abseil as @com_google_absl.
        #
        # To update the patch, run:
        # $ cd ~
        # $ mkdir -p github
        # $ cd github
        # $ git clone https://github.com/google/googletest.git
        # $ cd googletest
        # $ git checkout 28e9d1f26771c6517c3b4be10254887673c94018
        # ... make local changes to googletest ...
        # $ git diff > <client-root>/third_party/googletest/googletest.patch
        #
        # The patch path is relative to the workspace root.
        patch_file = ["//third_party/googletest:googletest.patch"],
        urls = tf_mirror_urls("https://github.com/google/googletest/archive/28e9d1f26771c6517c3b4be10254887673c94018.zip"),
    )

    tf_http_archive(
        name = "com_github_grpc_grpc",
        sha256 = "dd6a2fa311ba8441bbefd2764c55b99136ff10f7ea42954be96006a2723d33fc",
        strip_prefix = "grpc-1.74.0",
        patch_file = ["//third_party/grpc:grpc.patch"],
        urls = tf_mirror_urls("https://github.com/grpc/grpc/archive/refs/tags/v1.74.0.tar.gz"),
    )

    # Needed by com_google_protobuf
    tf_http_archive(
        name = "zlib",
        build_file = "//third_party:zlib.BUILD",
        sha256 = "9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23",
        strip_prefix = "zlib-1.3.1",
        system_build_file = "@zkx//third_party/systemlibs:zlib.BUILD",
        urls = tf_mirror_urls("https://zlib.net/zlib-1.3.1.tar.gz"),
    )

    tf_http_archive(
        name = "com_github_tencent_rapidjson",
        build_file = "//third_party:rapidjson/rapidjson.BUILD",
        sha256 = "c97766f7b8a3129e49fcd36d041a790d063a62f7e06b600759541d08dc0156d5",
        strip_prefix = "rapidjson-862c39be371278a45a88d4d1d75164be57bb7e2d",
        urls = tf_mirror_urls("https://github.com/Tencent/rapidjson/archive/862c39be371278a45a88d4d1d75164be57bb7e2d.zip"),
    )

    tf_http_archive(
        name = "pybind11",
        urls = tf_mirror_urls("https://github.com/pybind/pybind11/archive/v2.13.6.tar.gz"),
        sha256 = "e08cb87f4773da97fa7b5f035de8763abc656d87d5773e62f6da0587d1f0ec20",
        strip_prefix = "pybind11-2.13.6",
        build_file = "@zkx//third_party:pybind11.BUILD",
    )

    jsoncpp()
