#!/usr/bin/env python3
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
"""Generate PJRT_Buffer_Type enum from zkx_data.proto and zk_dtypes."""

import re
import sys


def parse_primitive_type_enum(proto_content):
    """Extract PrimitiveType enum values from proto content."""
    match = re.search(r"enum PrimitiveType \{([^}]+)\}", proto_content, re.DOTALL)
    if not match:
        raise ValueError("PrimitiveType enum not found")

    enum_body = match.group(1)
    values = []

    for line in enum_body.split("\n"):
        line = line.strip()
        # Match: NAME = VALUE;
        m = re.match(r"(\w+)\s*=\s*(\d+)\s*;", line)
        if m:
            name, value = m.groups()
            values.append((name, int(value)))

    return values


def parse_zk_dtypes_list(list_content):
    """Parse type list from emit_zk_dtypes output."""
    types = set()
    for line in list_content.strip().split("\n"):
        type_name = line.strip()
        if type_name:
            types.add(type_name)
    return types


def generate_c_header(values, zk_dtypes):
    """Generate C header with PJRT_Buffer_Type enum."""
    # Filter to only PJRT-supported types
    pjrt_types = {
        "PRIMITIVE_TYPE_INVALID": "INVALID",
        "PRED": "PRED",
        "S2": "S2",
        "S4": "S4",
        "S8": "S8",
        "S16": "S16",
        "S32": "S32",
        "S64": "S64",
        "U2": "U2",
        "U4": "U4",
        "U8": "U8",
        "U16": "U16",
        "U32": "U32",
        "U64": "U64",
        "U128": "U128",
        "U256": "U256",
        "TOKEN": "TOKEN",
    }

    # Add zk_dtypes types (they use the same name in both proto and pjrt)
    for dtype in zk_dtypes:
        pjrt_types[dtype] = dtype

    lines = [
        "/* Copyright 2026 The ZKX Authors.",
        "",
        'Licensed under the Apache License, Version 2.0 (the "License");',
        "you may not use this file except in compliance with the License.",
        "You may obtain a copy of the License at",
        "",
        "    http://www.apache.org/licenses/LICENSE-2.0",
        "",
        "Unless required by applicable law or agreed to in writing, software",
        'distributed under the License is distributed on an "AS IS" BASIS,',
        "WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.",
        "See the License for the specific language governing permissions and",
        "limitations under the License.",
        "==============================================================================*/",
        "",
        "/* Auto-generated from zkx_data.proto and zk_dtypes - DO NOT EDIT */",
        "",
        "#ifndef ZKX_PJRT_C_PJRT_C_API_DATA_TYPES_H_",
        "#define ZKX_PJRT_C_PJRT_C_API_DATA_TYPES_H_",
        "",
        "typedef enum {",
    ]

    for name, value in values:
        if name in pjrt_types:
            pjrt_name = f"PJRT_Buffer_Type_{pjrt_types[name]}"
            lines.append(f"  {pjrt_name} = {value},")

    lines.extend(
        [
            "} PJRT_Buffer_Type;",
            "",
            "#endif  // ZKX_PJRT_C_PJRT_C_API_DATA_TYPES_H_",
            "",
        ]
    )

    return "\n".join(lines)


def main():
    if len(sys.argv) != 4:
        print(
            f"Usage: {sys.argv[0]} <proto_path> <zk_dtypes_list_path> <output_path>",
            file=sys.stderr,
        )
        sys.exit(1)

    proto_path = sys.argv[1]
    zk_dtypes_list_path = sys.argv[2]
    output_path = sys.argv[3]

    with open(proto_path) as f:
        proto_content = f.read()

    with open(zk_dtypes_list_path) as f:
        zk_dtypes_content = f.read()

    values = parse_primitive_type_enum(proto_content)
    zk_dtypes = parse_zk_dtypes_list(zk_dtypes_content)
    header = generate_c_header(values, zk_dtypes)

    with open(output_path, "w") as f:
        f.write(header)


if __name__ == "__main__":
    main()
