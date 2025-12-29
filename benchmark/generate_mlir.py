#!/usr/bin/env python3
# Copyright 2025 The ZKIR Authors.
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

"""Generate MLIR files from templates with parameterized values.

This script fills in template placeholders with concrete values based on
the log2 size parameter.
"""

import argparse
import sys

# Pre-calculated roots of unity for BN254 Fr field
ROOTS_OF_UNITY = {
    10: {
        "size": 1024,
        "root": "3161067157621608152362653341354432744960400845131437947728257924963983317266",
    },
    11: {
        "size": 2048,
        "root": "1120550406532664055539694724667294622065367841900378087843176726913374367458",
    },
    12: {
        "size": 4096,
        "root": "4158865282786404163413953114870269622875596290766033564087307867933865333818",
    },
    13: {
        "size": 8192,
        "root": "197302210312744933010843010704445784068657690384188106020011018676818793232",
    },
    14: {
        "size": 16384,
        "root": "20619701001583904760601357484951574588621083236087856586626117568842480512645",
    },
    15: {
        "size": 32768,
        "root": "20402931748843538985151001264530049874871572933694634836567070693966133783803",
    },
    16: {
        "size": 65536,
        "root": "421743594562400382753388642386256516545992082196004333756405989743524594615",
    },
    17: {
        "size": 131072,
        "root": "12650941915662020058015862023665998998969191525479888727406889100124684769509",
    },
    18: {
        "size": 262144,
        "root": "11699596668367776675346610687704220591435078791727316319397053191800576917728",
    },
    19: {
        "size": 524288,
        "root": "15549849457946371566896172786938980432421851627449396898353380550861104573629",
    },
    20: {
        "size": 1048576,
        "root": "17220337697351015657950521176323262483320249231368149235373741788599650842711",
    },
    21: {
        "size": 2097152,
        "root": "13536764371732269273912573961853310557438878140379554347802702086337840854307",
    },
    22: {
        "size": 4194304,
        "root": "12143866164239048021030917283424216263377309185099704096317235600302831912062",
    },
    23: {
        "size": 8388608,
        "root": "934650972362265999028062457054462628285482693704334323590406443310927365533",
    },
    24: {
        "size": 16777216,
        "root": "5709868443893258075976348696661355716898495876243883251619397131511003808859",
    },
    25: {
        "size": 33554432,
        "root": "19200870435978225707111062059747084165650991997241425080699860725083300967194",
    },
    26: {
        "size": 67108864,
        "root": "7419588552507395652481651088034484897579724952953562618697845598160172257810",
    },
    27: {
        "size": 134217728,
        "root": "2082940218526944230311718225077035922214683169814847712455127909555749686340",
    },
    28: {
        "size": 268435456,
        "root": "19103219067921713944291392827692070036145651957329286315305642004821462161904",
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate MLIR from template with parameterized values"
    )
    parser.add_argument(
        "--template", required=True, help="Path to template file"
    )
    parser.add_argument(
        "--output", required=True, help="Path to output file"
    )
    parser.add_argument(
        "--log-size",
        type=int,
        help="Log2 of the size (e.g., 16 for 2^16=65536)",
    )
    parser.add_argument(
        "--placeholder-size",
        default="NUM_COEFFS",
        help="Placeholder for size in template",
    )
    parser.add_argument(
        "--placeholder-root",
        default="ROOT_OF_UNITY",
        help="Placeholder for root of unity in template",
    )
    parser.add_argument(
        "--extra-replacements",
        action="append",
        default=[],
        help="Extra replacements in format PLACEHOLDER=VALUE (can be specified multiple times)",
    )

    args = parser.parse_args()

    # Read template
    with open(args.template, "r") as f:
        content = f.read()

    # Apply log-size based replacements if provided
    if args.log_size is not None:
        if args.log_size not in ROOTS_OF_UNITY:
            print(
                f"Error: log_size {args.log_size} not supported. "
                f"Supported values: {sorted(ROOTS_OF_UNITY.keys())}",
                file=sys.stderr,
            )
            return 1

        # Get values
        data = ROOTS_OF_UNITY[args.log_size]
        size = data["size"]
        root = data["root"]

        # Replace placeholders
        content = content.replace(args.placeholder_size, str(size))
        content = content.replace(args.placeholder_root, root)

    # Apply extra replacements
    for replacement in args.extra_replacements:
        if "=" not in replacement:
            print(
                f"Error: Invalid replacement format '{replacement}'. "
                f"Expected PLACEHOLDER=VALUE",
                file=sys.stderr,
            )
            return 1
        placeholder, value = replacement.split("=", 1)
        content = content.replace(placeholder, value)

    # Write output
    with open(args.output, "w") as f:
        f.write(content)

    return 0


if __name__ == "__main__":
    sys.exit(main())
