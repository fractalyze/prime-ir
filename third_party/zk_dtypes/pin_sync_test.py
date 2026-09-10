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

"""Checks that both dependency lanes pin the same zk_dtypes archive.

zk_dtypes is not in any registry, so the bzlmod lane reaches it through an
`archive_override` in MODULE.bazel and the WORKSPACE lane through an
`http_archive` in `workspace.bzl`. MODULE.bazel cannot `load()`, so the pin
cannot be single-sourced and the two copies can drift — leaving the lanes
building different revisions of the dependency the rest of the chain hangs on.
"""

import base64
import binascii
import re

from absl.testing import absltest

_COMMIT_RE = re.compile(r'ZK_DTYPES_COMMIT = "([0-9a-f]{40})"')
_SHA256_RE = re.compile(r'ZK_DTYPES_SHA256 = "([0-9a-f]{64})"')
_OVERRIDE_RE = re.compile(
    r"archive_override\(\s*"
    r'module_name = "zk_dtypes",\s*'
    r'integrity = "sha256-([A-Za-z0-9+/=]+)",\s*'
    r'strip_prefix = "zk_dtypes-([0-9a-f]{40})",'
)


def _read(path):
  with open(path, encoding="utf-8") as f:
    return f.read()


class PinSyncTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    workspace_bzl = _read("third_party/zk_dtypes/workspace.bzl")
    self.workspace_commit = _COMMIT_RE.search(workspace_bzl).group(1)
    self.workspace_sha256 = _SHA256_RE.search(workspace_bzl).group(1)

    override = _OVERRIDE_RE.search(_read("MODULE.bazel"))
    self.assertIsNotNone(
        override, "MODULE.bazel has no zk_dtypes archive_override"
    )
    self.module_integrity, self.module_commit = override.groups()

  def test_commits_match(self):
    self.assertEqual(self.workspace_commit, self.module_commit)

  def test_hashes_match(self):
    integrity = base64.b64encode(
        binascii.unhexlify(self.workspace_sha256)
    ).decode()
    self.assertEqual(integrity, self.module_integrity)


if __name__ == "__main__":
  absltest.main()
