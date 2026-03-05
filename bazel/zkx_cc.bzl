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

"""ZKX cc rules."""

load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")
load(
    "//bazel:zkx.bzl",
    "if_has_exception",
)

def zkx_safe_code():
    return [
        "-Wall",
        "-Werror",
        # NOTE(chokobole): See the whole list of warnings in https://github.com/openxla/xla/tree/8bac4a2/warnings.bazelrc.
        # TODO(chokobole): Remove this warning once resolved. This is required to compile async_value.h.
        # ./xla/tsl/concurrency/async_value.h:722:19: warning: offset of on non-standard-layout type 'ConcreteAsyncValue<C>' [-Winvalid-offsetof]
        # 722 |     static_assert(offsetof(ConcreteAsyncValue<T>, data_store_.data_) ==
        "-Wno-invalid-offsetof",
        # /usr/lib/gcc/x86_64-linux-gnu/13/../../../../include/c++/13/functional:552:2: error: 'result_of<(lambda at external/zkx/zkx/service/cpu/runtime_fork_join.cc:97:9) &()>' is deprecated: use 'std::invoke_result' instead [-Werror,-Wdeprecated-declarations]
        # 552 |         using _Res_type_impl
        #     |         ^
        # /usr/lib/gcc/x86_64-linux-gnu/13/../../../../include/c++/13/functional:556:2: note: in instantiation of template type alias '_Res_type_impl' requested here
        # 556 |         using _Res_type = _Res_type_impl<_Functor, _CallArgs, _Bound_args...>;
        #     |         ^
        # /usr/lib/gcc/x86_64-linux-gnu/13/../../../../include/c++/13/functional:586:28: note: in instantiation of template type alias '_Res_type' requested here
        # 586 |                typename _Result = _Res_type<tuple<_Args...>>>
        #     |                                   ^
        # external/eigen_archive/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceThreadPool.h:160:23: note: in instantiation of template class 'std::_Bind<(lambda at external/zkx/zkx/service/cpu/runtime_fork_join.cc:97:9) ()>' requested here
        # 160 |       pool_->Schedule(std::bind(std::forward<Function>(f), args...));
        #     |                       ^
        # external/zkx/zkx/service/cpu/runtime_fork_join.cc:96:42: note: in instantiation of function template specialization 'Eigen::ThreadPoolDevice::enqueueNoNotification<(lambda at external/zkx/zkx/service/cpu/runtime_fork_join.cc:97:9)>' requested here
        # 96 |     run_options->intra_op_thread_pool()->enqueueNoNotification(
        #    |                                          ^
        # /usr/lib/gcc/x86_64-linux-gnu/13/../../../../include/c++/13/type_traits:2590:9: note: 'result_of<(lambda at external/zkx/zkx/service/cpu/runtime_fork_join.cc:97:9) &()>' has been explicitly marked deprecated here
        # 2590 |     { } _GLIBCXX17_DEPRECATED_SUGGEST("std::invoke_result");
        #      |         ^
        # /usr/lib/gcc/x86_64-linux-gnu/13/../../../../include/x86_64-linux-gnu/c++/13/bits/c++config.h:122:45: note: expanded from macro '_GLIBCXX17_DEPRECATED_SUGGEST'
        # 122 | # define _GLIBCXX17_DEPRECATED_SUGGEST(ALT) _GLIBCXX_DEPRECATED_SUGGEST(ALT)
        #     |                                             ^
        # /usr/lib/gcc/x86_64-linux-gnu/13/../../../../include/x86_64-linux-gnu/c++/13/bits/c++config.h:98:19: note: expanded from macro '_GLIBCXX_DEPRECATED_SUGGEST'
        # 98 |   __attribute__ ((__deprecated__ ("use '" ALT "' instead")))
        #    |                   ^
        # 1 error generated.
        "-Wno-deprecated-declarations",
        "-Wno-nullability-completeness",
    ]

def zkx_warnings(safe_code):
    warnings = []
    if safe_code:
        warnings.extend(zkx_safe_code())
    return warnings

def zkx_exceptions(force_exceptions):
    return if_has_exception(["-fexceptions"], (["-fexceptions"] if force_exceptions else ["-fno-exceptions"]))

def zkx_copts(safe_code = True):
    return zkx_warnings(safe_code)

def zkx_cxxopts(safe_code = True, force_exceptions = False):
    return zkx_copts(safe_code) + zkx_exceptions(force_exceptions)

def zkx_defines():
    return []

def zkx_local_defines():
    return []

def zkx_linkopts():
    return []

def zkx_cc_library(
        name,
        copts = [],
        defines = [],
        local_defines = [],
        linkopts = [],
        alwayslink = True,
        safe_code = True,
        force_exceptions = False,
        **kwargs):
    cc_library(
        name = name,
        copts = copts + zkx_cxxopts(safe_code = safe_code, force_exceptions = force_exceptions),
        defines = defines + zkx_defines(),
        local_defines = local_defines + zkx_local_defines(),
        linkopts = linkopts + zkx_linkopts(),
        alwayslink = alwayslink,
        **kwargs
    )

def zkx_cc_binary(
        name,
        copts = [],
        defines = [],
        local_defines = [],
        linkopts = [],
        safe_code = True,
        force_exceptions = False,
        **kwargs):
    cc_binary(
        name = name,
        copts = copts + zkx_cxxopts(safe_code = safe_code, force_exceptions = force_exceptions),
        defines = defines + zkx_defines(),
        local_defines = local_defines + zkx_local_defines(),
        linkopts = linkopts + zkx_linkopts(),
        **kwargs
    )

def zkx_cc_test(
        name,
        copts = [],
        defines = [],
        local_defines = [],
        linkopts = [],
        linkstatic = True,
        deps = [],
        safe_code = True,
        force_exceptions = False,
        **kwargs):
    cc_test(
        name = name,
        copts = copts + zkx_cxxopts(safe_code = safe_code, force_exceptions = force_exceptions),
        defines = defines + zkx_defines(),
        local_defines = local_defines + zkx_local_defines(),
        linkopts = linkopts + zkx_linkopts(),
        linkstatic = linkstatic,
        deps = deps + ["@com_google_googletest//:gtest_main"],
        **kwargs
    )

def zkx_cc_unittest(
        name,
        size = "small",
        **kwargs):
    zkx_cc_test(
        name = name,
        size = size,
        **kwargs
    )
