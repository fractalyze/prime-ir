#!/usr/bin/env python3
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
"""Generate AOT runtime MLIR op files from emit_aot_types output.

Reads the type list produced by emit_aot_types (one line per type) and
generates EC and extension field operation MLIR files.

Pipeline:
    emit_aot_types (C++) → aot_types.txt → gen_aot_runtime.py → .mlir

Input format:
    curve <lower_snake> <rank>
    ext_field <lower_snake>

Mont variants are emitted by WITH_MONT (e.g., bn254_g1_affine_mont).
The generator groups std/mont pairs and emits both in one file.

Usage (Bazel):
    genrule(cmd = "$(execpath :emit_aot_types) | $(execpath :gen) --outdir $(@D)")

Usage (manual):
    ./emit_aot_types | python gen_aot_runtime.py --types /dev/stdin --outdir out/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

LICENSE = """\
// Copyright 2026 The PrimeIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================
"""

# ---- MLIR type alias conventions ----
# Must match the shared defs .mlir file (e.g., bn254_defs.mlir).

EC_TYPES = {
    1: {  # G1
        False: {"affine": "affine", "jacobian": "jacobian", "xyzz": "xyzz"},
        True: {"affine": "affinem", "jacobian": "jacobianm", "xyzz": "xyzzm"},
    },
    2: {  # G2
        False: {
            "affine": "g2affine",
            "jacobian": "g2jacobian",
            "xyzz": "g2xyzz",
        },
        True: {
            "affine": "g2affinem",
            "jacobian": "g2jacobianm",
            "xyzz": "g2xyzzm",
        },
    },
}

# Extension field MLIR aliases use the lower_snake name directly.
# e.g., bn254_bfx2 → !bn254_bfx2 (std), !bn254_bfx2m (mont)


def gen_ec_ops(base: str, rank: int) -> str:
  """Generate EC ops for one curve (both std and mont)."""
  lines = [f"// AOT-compiled {base.upper()} EC operations.\n"]
  for is_mont in [False, True]:
    ms = "_mont" if is_mont else ""
    form = "Montgomery" if is_mont else "Standard"
    t = EC_TYPES[rank][is_mont]
    sf = "SFm" if is_mont else "SF"
    lines.append(f"""\
// ===== {form} form =====

// --- XYZZ ({form}) ---

func.func @ec_add_{base}_xyzz{ms}(%a: !{t['xyzz']}, %b: !{t['xyzz']}) -> !{t['xyzz']}
    attributes {{ llvm.emit_c_interface }} {{
  %r = elliptic_curve.add %a, %b : !{t['xyzz']}, !{t['xyzz']} -> !{t['xyzz']}
  return %r : !{t['xyzz']}
}}

func.func @ec_double_{base}_xyzz{ms}(%a: !{t['xyzz']}) -> !{t['xyzz']}
    attributes {{ llvm.emit_c_interface }} {{
  %r = elliptic_curve.double %a : !{t['xyzz']} -> !{t['xyzz']}
  return %r : !{t['xyzz']}
}}

func.func @ec_negate_{base}_xyzz{ms}(%a: !{t['xyzz']}) -> !{t['xyzz']}
    attributes {{ llvm.emit_c_interface }} {{
  %r = elliptic_curve.negate %a : !{t['xyzz']}
  return %r : !{t['xyzz']}
}}

func.func @ec_mixed_add_{base}_xyzz{ms}(%a: !{t['xyzz']}, %b: !{t['affine']}) -> !{t['xyzz']}
    attributes {{ llvm.emit_c_interface }} {{
  %r = elliptic_curve.add %a, %b : !{t['xyzz']}, !{t['affine']} -> !{t['xyzz']}
  return %r : !{t['xyzz']}
}}

// --- Jacobian ({form}) ---

func.func @ec_add_{base}_jacobian{ms}(%a: !{t['jacobian']}, %b: !{t['jacobian']}) -> !{t['jacobian']}
    attributes {{ llvm.emit_c_interface }} {{
  %r = elliptic_curve.add %a, %b : !{t['jacobian']}, !{t['jacobian']} -> !{t['jacobian']}
  return %r : !{t['jacobian']}
}}

func.func @ec_double_{base}_jacobian{ms}(%a: !{t['jacobian']}) -> !{t['jacobian']}
    attributes {{ llvm.emit_c_interface }} {{
  %r = elliptic_curve.double %a : !{t['jacobian']} -> !{t['jacobian']}
  return %r : !{t['jacobian']}
}}

func.func @ec_mixed_add_{base}_jacobian{ms}(%a: !{t['jacobian']}, %b: !{t['affine']}) -> !{t['jacobian']}
    attributes {{ llvm.emit_c_interface }} {{
  %r = elliptic_curve.add %a, %b : !{t['jacobian']}, !{t['affine']} -> !{t['jacobian']}
  return %r : !{t['jacobian']}
}}

// --- Cross-type: affine → jacobian ({form}) ---

func.func @ec_add_{base}_affine_to_jacobian{ms}(%a: !{t['affine']}, %b: !{t['affine']}) -> !{t['jacobian']}
    attributes {{ llvm.emit_c_interface }} {{
  %r = elliptic_curve.add %a, %b : !{t['affine']}, !{t['affine']} -> !{t['jacobian']}
  return %r : !{t['jacobian']}
}}

func.func @ec_double_{base}_affine_to_jacobian{ms}(%a: !{t['affine']}) -> !{t['jacobian']}
    attributes {{ llvm.emit_c_interface }} {{
  %r = elliptic_curve.double %a : !{t['affine']} -> !{t['jacobian']}
  return %r : !{t['jacobian']}
}}

// --- Scalar multiply ({form}) ---

func.func @ec_scalar_mul_{base}_jacobian{ms}(%k: !{sf}, %p: !{t['affine']}) -> !{t['jacobian']}
    attributes {{ llvm.emit_c_interface }} {{
  %r = elliptic_curve.scalar_mul %k, %p : !{sf}, !{t['affine']} -> !{t['jacobian']}
  return %r : !{t['jacobian']}
}}

func.func @ec_scalar_mul_jac_{base}_jacobian{ms}(%k: !{sf}, %p: !{t['jacobian']}) -> !{t['jacobian']}
    attributes {{ llvm.emit_c_interface }} {{
  %r = elliptic_curve.scalar_mul %k, %p : !{sf}, !{t['jacobian']} -> !{t['jacobian']}
  return %r : !{t['jacobian']}
}}

// --- Conversions ({form}) ---

func.func @ec_jacobian_to_affine_{base}{ms}(%a: !{t['jacobian']}) -> !{t['affine']}
    attributes {{ llvm.emit_c_interface }} {{
  %r = elliptic_curve.convert_point_type %a : !{t['jacobian']} -> !{t['affine']}
  return %r : !{t['affine']}
}}

func.func @ec_xyzz_to_affine_{base}{ms}(%a: !{t['xyzz']}) -> !{t['affine']}
    attributes {{ llvm.emit_c_interface }} {{
  %r = elliptic_curve.convert_point_type %a : !{t['xyzz']} -> !{t['affine']}
  return %r : !{t['affine']}
}}

func.func @ec_affine_to_jacobian_{base}{ms}(%a: !{t['affine']}) -> !{t['jacobian']}
    attributes {{ llvm.emit_c_interface }} {{
  %r = elliptic_curve.convert_point_type %a : !{t['affine']} -> !{t['jacobian']}
  return %r : !{t['jacobian']}
}}

func.func @ec_affine_to_xyzz_{base}{ms}(%a: !{t['affine']}) -> !{t['xyzz']}
    attributes {{ llvm.emit_c_interface }} {{
  %r = elliptic_curve.convert_point_type %a : !{t['affine']} -> !{t['xyzz']}
  return %r : !{t['xyzz']}
}}
""")
  return "\n".join(lines)


def gen_ef_ops(alias: str) -> str:
  """Generate expensive extension field ops (mul, square, inverse)."""
  lines = [
      f"// AOT-compiled {alias} extension field operations.",
      "// Only expensive operations (mul, square, inverse) are AOT-compiled.\n",
  ]
  for is_mont in [False, True]:
    ms = "_mont" if is_mont else ""
    m = "m" if is_mont else ""
    ef = alias + m  # e.g., bn254_bfx2 / bn254_bfx2m
    form = "Montgomery" if is_mont else "Standard"
    lines.append(f"""\
// ===== {form} form =====

func.func @ef_mul_{alias}{ms}(%a: !{ef}, %b: !{ef}) -> !{ef}
    attributes {{ llvm.emit_c_interface }} {{
  %r = field.mul %a, %b : !{ef}
  return %r : !{ef}
}}

func.func @ef_square_{alias}{ms}(%a: !{ef}) -> !{ef}
    attributes {{ llvm.emit_c_interface }} {{
  %r = field.square %a : !{ef}
  return %r : !{ef}
}}

func.func @ef_inverse_{alias}{ms}(%a: !{ef}) -> !{ef}
    attributes {{ llvm.emit_c_interface }} {{
  %r = field.inverse %a : !{ef}
  return %r : !{ef}
}}
""")
  return "\n".join(lines)


def main():
  parser = argparse.ArgumentParser(description="Generate AOT runtime MLIR")
  parser.add_argument(
      "--types", required=True, help="emit_aot_types output file"
  )
  parser.add_argument("--ec-outdir", required=True, help="EC ops output dir")
  parser.add_argument(
      "--field-outdir", required=True, help="Field ops output dir"
  )
  args = parser.parse_args()

  ec_outdir = Path(args.ec_outdir)
  field_outdir = Path(args.field_outdir)
  ec_outdir.mkdir(parents=True, exist_ok=True)
  field_outdir.mkdir(parents=True, exist_ok=True)

  curves: dict[str, int] = {}
  ext_fields: list[str] = []

  with open(args.types) as f:
    for line in f:
      parts = line.strip().split()
      if not parts:
        continue
      kind, alias = parts[0], parts[1]
      if alias.endswith("_mont"):
        continue
      if kind == "curve":
        base = alias.removesuffix("_affine")
        curves[base] = int(parts[2])
      elif kind == "ext_field":
        ext_fields.append(alias)

  for base, rank in curves.items():
    path = ec_outdir / f"ec_ops_{base}.mlir"
    path.write_text(LICENSE + "\n" + gen_ec_ops(base, rank))
    print(f"Generated {path}", file=sys.stderr)

  for alias in ext_fields:
    content = gen_ef_ops(alias)
    if not content:
      continue
    path = field_outdir / f"field_ops_{alias}.mlir"
    path.write_text(LICENSE + "\n" + content)
    print(f"Generated {path}", file=sys.stderr)


if __name__ == "__main__":
  sys.exit(main() or 0)
