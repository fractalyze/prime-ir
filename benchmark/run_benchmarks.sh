#!/usr/bin/env bash
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

# Builds and runs all PrimeIR benchmarks, then merges the JSON results
# into a single file for zkbench consumption.
#
# Usage: run_benchmarks.sh <output.json>

set -euo pipefail

OUTPUT="${1:?Usage: run_benchmarks.sh <output.json>}"
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

echo "=== Building all benchmarks ==="
bazel --bazelrc=.bazelrc.ci build --config ci //benchmark/...

targets=(
  "benchmark/ntt/ntt_benchmark_test"
  "benchmark/mod_arith/mul_benchmark_test"
  "benchmark/msm/msm_benchmark_test"
  "benchmark/poseidon2/poseidon2_benchmark"
  "benchmark/vectorize/vectorize_benchmark"
  "benchmark/binary_field/binary_field_benchmark"
)

for target in "${targets[@]}"; do
  name=$(basename "$target")
  echo "=== Running $name ==="
  ./bazel-bin/"$target" --zkbench_out="$TMPDIR/${name}.json" || true
done

echo "=== Merging benchmark results ==="
python3 -c "
import json, glob, sys
files = sorted(glob.glob('$TMPDIR/*.json'))
if not files:
    sys.exit('No benchmark results found')
with open(files[0]) as f:
    merged = json.load(f)
for p in files[1:]:
    with open(p) as f:
        merged['benchmarks'].update(json.load(f)['benchmarks'])
with open('$OUTPUT', 'w') as f:
    json.dump(merged, f, indent=2)
print(f'Merged {len(files)} results -> $OUTPUT')
"
