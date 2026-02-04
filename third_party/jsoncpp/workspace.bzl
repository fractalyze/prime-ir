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

"""Provides the repository macro to import jsoncpp."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports jsoncpp."""

    tf_http_archive(
        name = "jsoncpp",
        sha256 = "f93b6dd7ce796b13d02c108bc9f79812245a82e577581c4c9aabe57075c90ea2",
        strip_prefix = "jsoncpp-1.9.6",
        build_file = "//third_party/jsoncpp:jsoncpp.BUILD",
        urls = tf_mirror_urls("https://github.com/open-source-parsers/jsoncpp/archive/refs/tags/1.9.6.tar.gz"),
    )
