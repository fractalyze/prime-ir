#!/bin/bash
# Copyright 2026 The PrimeIR Authors.
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

set -o errexit
set -o nounset
set -o pipefail

if [[ $# -ne 0 && $# -ne 3 ]] ; then
  echo "Usage: $0 [<bazel-diff.jar> <start_commit> <end_commit>]"
  echo "   Builds and tests all bazel targets."
  echo "   If start_commit and end_commit are specified, uses bazel-diff to"
  echo "   determine the set of targets to build based on code changes."
  echo
  echo "Example: Build all affected targets between current branch and main"
  echo "  $ curl -Lo /tmp/bazel-diff.jar https://github.com/Tinder/bazel-diff/releases/latest/download/bazel-diff_deploy.jar"
  echo "  $ ci_build_bazel.sh /tmp/bazel-diff.jar main \$(git rev-parse --abbrev-ref HEAD)"
  exit 1
fi

# CI options live in .bazelrc.ci as :ci-namespaced configs.
bazel-ci() {
  bazel --bazelrc=.bazelrc.ci "$@" --config ci
}

bazel-test-all() {
  bazel-ci test //...
}

bazel-test-diff() {
  set -e
  WORKSPACE_PATH=$(git rev-parse --show-toplevel)
  BAZEL_DIFF_JAR="$1"
  PREVIOUS_REV="$2" # Starting Revision SHA
  FINAL_REV="$3" # Final Revision SHA
  BAZEL_PATH="$(which bazel)"

  # Per-invocation scratch dir: fixed /tmp names collide when several runner
  # slots on one host execute this script concurrently.
  SCRATCH_DIR="$(mktemp -d -t bazel-diff.XXXXXX)"
  trap 'rm -rf "$SCRATCH_DIR"' EXIT
  STARTING_HASHES_JSON="$SCRATCH_DIR/starting_hashes.json"
  FINAL_HASHES_JSON="$SCRATCH_DIR/final_hashes.json"
  IMPACTED_TARGETS_PATH="$SCRATCH_DIR/impacted_targets.txt"
  FILTERED_TARGETS_PATH="$SCRATCH_DIR/filtered_targets.txt"

  bazel-diff() {
    java -jar "$BAZEL_DIFF_JAR" "$@"
  }

  # Build-configuration files live outside the target graph: editing them
  # changes what every target means but no target hash. Seed them into every
  # hash so such a change falls back to a full run.
  SEED_FILEPATHS="$SCRATCH_DIR/seed_filepaths.txt"
  seed-filepaths() {
    : > "$SEED_FILEPATHS"
    for f in .bazelrc .bazelrc.ci .bazelversion WORKSPACE.bazel MODULE.bazel; do
      [[ -f "$WORKSPACE_PATH/$f" ]] && echo "$WORKSPACE_PATH/$f" >> "$SEED_FILEPATHS"
    done
  }

  git -C "$WORKSPACE_PATH" checkout "$PREVIOUS_REV" --quiet

  echo "Generating Hashes for Revision '$PREVIOUS_REV'"
  seed-filepaths
  bazel-diff generate-hashes -w "$WORKSPACE_PATH" -b "$BAZEL_PATH" -s "$SEED_FILEPATHS" $STARTING_HASHES_JSON

  UNCOMMITTED_CHANGES="$(git status -s)"
  if [[ -n "$UNCOMMITTED_CHANGES" ]]; then
    echo "[WARNING] Uncommitted changes found, likely build byproducts:"
    echo "$UNCOMMITTED_CHANGES"
  fi

  git reset --hard "$PREVIOUS_REV"
  git -C "$WORKSPACE_PATH" checkout "$FINAL_REV" --quiet

  echo "Generating Hashes for Revision '$FINAL_REV'"
  seed-filepaths
  bazel-diff generate-hashes -w "$WORKSPACE_PATH" -b "$BAZEL_PATH" -s "$SEED_FILEPATHS" $FINAL_HASHES_JSON

  echo "Determining Impacted Targets"
  bazel-diff get-impacted-targets -sh $STARTING_HASHES_JSON -fh $FINAL_HASHES_JSON -o $IMPACTED_TARGETS_PATH -w "$WORKSPACE_PATH"
  echo ""

  impacted_targets=()
  IFS=$'\n' read -d '' -r -a impacted_targets < $IMPACTED_TARGETS_PATH || true
  formatted_impacted_targets=$(IFS=$'\n'; echo "${impacted_targets[*]}")
  if [[ -z "$formatted_impacted_targets" ]]; then
    echo "No impacted targets from change."
    exit 0
  fi

  NUM_IMPACTED=$(echo "$formatted_impacted_targets" | wc -l)
  echo "[$NUM_IMPACTED] Impacted Targets between $PREVIOUS_REV and $FINAL_REV:"
  echo "$formatted_impacted_targets"
  echo ""

  # Remove external and duplicate targets.
  sort "$IMPACTED_TARGETS_PATH" | uniq | grep -v '//external' > "$FILTERED_TARGETS_PATH" || true

  # Mirror wildcard (//...) semantics: keep only rule targets and drop
  # `manual`-tagged ones. bazel-diff also lists output-file labels and manual
  # rules; naming those in --target_pattern_file would force-build targets a
  # plain `bazel test //...` run never touches (e.g. the Windows-only
  # _prime_ir_common_def stub, which fails outright on Linux).
  if [[ -s "$FILTERED_TARGETS_PATH" ]]; then
    QUERY_FILE="$SCRATCH_DIR/wildcard_semantics_query.txt"
    {
      printf 'let t = set('
      tr '\n' ' ' < "$FILTERED_TARGETS_PATH"
      printf ') in kind(rule, $t) except attr("tags", "(^\\[|, )manual(, |\\]$)", $t)'
    } > "$QUERY_FILE"
    bazel query --query_file="$QUERY_FILE" > "$FILTERED_TARGETS_PATH.rules"
    mv "$FILTERED_TARGETS_PATH.rules" "$FILTERED_TARGETS_PATH"
  fi

  # Build and Test impacted targets. A single `test` invocation also builds
  # the impacted non-test targets — a separate `build` would compile a second
  # configuration, since test:ci carries build settings (--//:has_avx512).
  # Exit code 4 means everything built but the impacted set contains no tests.
  if [[ -s "$FILTERED_TARGETS_PATH" ]]; then
    echo "Building and Testing Impacted (Non-External) Targets..."
    bazel-ci test --target_pattern_file="$FILTERED_TARGETS_PATH" || {
      ec=$?
      if [[ $ec -eq 4 ]]; then
        echo "No test targets among the impacted set; build succeeded."
      else
        exit $ec
      fi
    }
  else
    echo "No non-external impacted targets to build and test."
  fi
}

# Run bazel build and test
if [[ $# -eq 0 ]] ; then
  bazel-test-all
else
  bazel-test-diff "$@"
fi
