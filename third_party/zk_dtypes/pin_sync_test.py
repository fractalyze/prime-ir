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

Both files put the pin behind `ZK_DTYPES_`-prefixed variables so that one
substitution finds it in either, which is what lets
`.github/workflows/pin-bump.yml` hand both paths to the same bump action. This
test is the other half of that arrangement: it fails if a hand edit, or a bump
that reached only one file, leaves them disagreeing.

The digest is the same hash written two ways — `http_archive` takes hex,
`archive_override` takes base64 — so comparing it means converting first.
"""

import base64
import binascii
import re

from absl.testing import absltest

_COMMIT_RE = re.compile(r'ZK_DTYPES_COMMIT = "([0-9a-f]{40})"')
_SHA256_RE = re.compile(r'ZK_DTYPES_SHA256 = "([0-9a-f]{64})"')
_INTEGRITY_RE = re.compile(r'ZK_DTYPES_INTEGRITY = "sha256-([A-Za-z0-9+/=]+)"')


def _read(path):
  with open(path, encoding="utf-8") as f:
    return f.read()


def _search(pattern, contents, path):
  match = pattern.search(contents)
  if not match:
    raise AssertionError(f"{path} has no {pattern.pattern}")
  return match.group(1)


class PinSyncTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    workspace_bzl = _read("third_party/zk_dtypes/workspace.bzl")
    module_bazel = _read("MODULE.bazel")

    self.workspace_commit = _search(_COMMIT_RE, workspace_bzl, "workspace.bzl")
    self.workspace_sha256 = _search(_SHA256_RE, workspace_bzl, "workspace.bzl")
    self.module_commit = _search(_COMMIT_RE, module_bazel, "MODULE.bazel")
    self.module_integrity = _search(_INTEGRITY_RE, module_bazel, "MODULE.bazel")

  def test_commits_match(self):
    self.assertEqual(self.workspace_commit, self.module_commit)

  def test_hashes_match(self):
    as_integrity = base64.b64encode(
        binascii.unhexlify(self.workspace_sha256)
    ).decode()
    self.assertEqual(as_integrity, self.module_integrity)


if __name__ == "__main__":
  absltest.main()
