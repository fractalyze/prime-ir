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

"""Provides the repo macro to import prime_ir.

prime_ir provides MLIR dialects for cryptographic computations.
"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    PRIME_IR_COMMIT = "1af4332a4bab540cd1fd122ebeb53754318e1ec2"
    PRIME_IR_SHA256 = "6e4bf0ee446bafe8f7f5dd8ed98ce273c095e9de4927177e2a0204e5b832cfcb"
    tf_http_archive(
        name = "prime_ir",
        sha256 = PRIME_IR_SHA256,
        strip_prefix = "prime-ir-{commit}".format(commit = PRIME_IR_COMMIT),
        urls = tf_mirror_urls("https://github.com/fractalyze/prime-ir/archive/{commit}.tar.gz".format(commit = PRIME_IR_COMMIT)),
    )
    # Uncomment this for development!
    # native.local_repository(
    #     name = "prime_ir",
    #     path = "../prime-ir",
    # )
