#!/usr/bin/env bash
# Copyright 2025 The PrimeIR Authors.
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

# Microbenchmark harness for SpecializeArithToAVX (not wired into CI; run
# manually). Measures the packed Montgomery chain kernel in three configs:
#
#   generic-avx2      LLVM generic legalization, JIT capped to AVX2
#   flavor-avx2       -specialize-arith-to-avx=flavor=avx2, JIT capped to AVX2
#   flavor-avx512     -specialize-arith-to-avx (default), native host ISA
#                     (reference; requires an AVX-512 host)
#
# The AVX2 cap is applied with mlir-runner --mattr=-avx512f (JitRunner adds
# -mattr on top of the detected host features, and clearing avx512f also
# clears everything that implies it). The script verifies the cap from the
# dumped JIT object: no zmm registers or opmask merges may appear, and the
# flavor-avx2 config must actually contain ymm vpmuludq.
#
# Each config runs RUNS times (default 5); the per-round minimum is reported.
# Printed result vectors must be identical across configs (bit-exactness).
#
# Usage: benchmarks/ArithExt/run_bench.sh
#   RUNS=<n>           number of repetitions per config
#   BENCH_CPU=<core>   core to pin to (default: last core)
#   BENCH_OUT_DIR=<d>  keep lowered IR, JIT objects, and raw runs there
#   SKIP_AVX512=1      skip the native AVX-512 reference config

set -euo pipefail
cd "$(dirname "$0")/../.."

RUNS="${RUNS:-5}"
# Default to the highest CPU this process may run on; nproc is a count and
# breaks in sparse or non-zero-based cpusets.
CPU="${BENCH_CPU:-$(awk '/Cpus_allowed_list/ {
  count = split($2, ranges, ",")
  bounds = split(ranges[count], range, "-")
  print range[bounds]
}' /proc/self/status)}"
SRC=benchmarks/ArithExt/packed_mont_chain.mlir
if [[ -n "${BENCH_OUT_DIR:-}" ]]; then
  OUT=$BENCH_OUT_DIR
  mkdir -p "$OUT"
else
  OUT=$(mktemp -d)
  trap 'rm -rf "$OUT"' EXIT
fi

echo "== load average: $(cat /proc/loadavg) (benchmarking wants an idle box)"
echo "== building tools"
nice -n 10 bazel build //tools:prime-ir-opt \
  @llvm-project//mlir:mlir-runner \
  @llvm-project//mlir:libmlir_c_runner_utils.so

OPT=bazel-bin/tools/prime-ir-opt
RUNNER=bazel-bin/external/llvm-project/mlir/mlir-runner
LIBS=$(readlink -f bazel-bin/external/llvm-project/mlir/libmlir_c_runner_utils.so)

# lower <pass-flags...>
lower() {
  # -convert-vector-to-llvm handles vector.print, which the generic
  # -convert-to-llvm interface does not lower.
  "$OPT" "$SRC" "$@" -convert-scf-to-cf -convert-vector-to-llvm \
    -convert-to-llvm
}

# run_config <name> <mattr-flags> — expects $OUT/<name>.mlir to exist.
# Prints "chain l1 l2" minima; dumps the JIT object on the first run.
run_config() {
  local name=$1 mattr=$2
  local best_chain= best_l1= best_l2=
  for i in $(seq "$RUNS"); do
    local dump=()
    if [[ $i == 1 ]]; then
      dump=(--dump-object-file --object-filename="$OUT/$name.o")
    fi
    # shellcheck disable=SC2086
    taskset -c "$CPU" nice -n -0 "$RUNNER" "$OUT/$name.mlir" \
      -e main -entry-point-result=void -O3 $mattr \
      -shared-libs="$LIBS" "${dump[@]}" > "$OUT/$name.run$i.txt"
    mapfile -t ns < <(grep -E '^[0-9]+\.[0-9]+(e[-+]?[0-9]+)?$' \
      "$OUT/$name.run$i.txt")
    if [[ ${#ns[@]} -ne 3 ]]; then
      echo "unexpected output in $OUT/$name.run$i.txt" >&2
      exit 1
    fi
    grep -E '^-?[0-9]+$' "$OUT/$name.run$i.txt" > "$OUT/$name.vectors.txt"
    best_chain=$(printf '%s\n' "${best_chain:-inf}" "${ns[0]}" | sort -g | head -1)
    best_l1=$(printf '%s\n' "${best_l1:-inf}" "${ns[1]}" | sort -g | head -1)
    best_l2=$(printf '%s\n' "${best_l2:-inf}" "${ns[2]}" | sort -g | head -1)
  done
  echo "$best_chain $best_l1 $best_l2"
}

echo "== lowering"
lower > "$OUT/generic-avx2.mlir"
lower -specialize-arith-to-avx=flavor=avx2 > "$OUT/flavor-avx2.mlir"
lower -specialize-arith-to-avx > "$OUT/flavor-avx512.mlir"

echo "== running generic-avx2 (baseline)"
GEN=$(run_config generic-avx2 --mattr=-avx512f)
echo "== running flavor-avx2"
AVX2=$(run_config flavor-avx2 --mattr=-avx512f)
if [[ "${SKIP_AVX512:-}" != 1 ]]; then
  echo "== running flavor-avx512 (native reference)"
  AVX512=$(run_config flavor-avx512 "")
fi

echo "== verifying the AVX2 cap from the dumped JIT objects"
# grep -q would SIGPIPE objdump on first match, which pipefail turns into a
# failure — disassemble to a file instead.
for name in generic-avx2 flavor-avx2; do
  objdump -d "$OUT/$name.o" > "$OUT/$name.disasm"
  if grep -qE 'zmm|\{%k[0-7]\}' "$OUT/$name.disasm"; then
    echo "FAIL: $name object uses zmm/opmask; the AVX2 cap did not hold" >&2
    exit 1
  fi
done
if ! grep -qE 'vpmuludq.*ymm' "$OUT/flavor-avx2.disasm"; then
  echo "FAIL: flavor-avx2 object contains no ymm vpmuludq" >&2
  exit 1
fi
echo "   ok: no zmm/opmask in capped objects; ymm vpmuludq present"

echo "== verifying bit-exact results across configs"
if ! diff -q "$OUT/generic-avx2.vectors.txt" "$OUT/flavor-avx2.vectors.txt"; then
  echo "FAIL: flavor-avx2 results differ from generic" >&2
  exit 1
fi
if [[ "${SKIP_AVX512:-}" != 1 ]] && \
   ! diff -q "$OUT/generic-avx2.vectors.txt" "$OUT/flavor-avx512.vectors.txt"; then
  echo "FAIL: flavor-avx512 results differ from generic" >&2
  exit 1
fi
echo "   ok"

echo
echo "ns/round (min of $RUNS runs)      chain      sweep-16KiB  sweep-1MiB"
printf 'generic-avx2 (baseline)    %10s %12s %12s\n' $GEN
printf 'flavor-avx2                %10s %12s %12s\n' $AVX2
if [[ "${SKIP_AVX512:-}" != 1 ]]; then
  printf 'flavor-avx512 (native)     %10s %12s %12s\n' $AVX512
fi
