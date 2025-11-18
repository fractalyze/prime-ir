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

"""A rule for running zkir-opt."""

def executable_attr(label):
    """A helper for declaring executable dependencies."""
    return attr.label(
        default = Label(label),
        executable = True,
        # commenting this out breaks cross-compilation, but this should not be a problem
        # for developer builds
        # cfg = "exec",
        cfg = "target",
    )

_ZKIR_OPT = "@zkir//tools:zkir-opt"

def _zkir_opt_impl(ctx):
    generated_file = ctx.outputs.generated_filename
    args = ctx.actions.args()
    args.add_all(ctx.attr.pass_flags)
    args.add_all(["-o", generated_file.path])
    args.add(ctx.file.src)

    ctx.actions.run(
        inputs = ctx.attr.src.files,
        outputs = [generated_file],
        arguments = [args],
        executable = ctx.executable._zkir_opt_binary,
    )
    return [
        DefaultInfo(files = depset([generated_file, ctx.file.src])),
    ]

zkir_opt = rule(
    doc = """
      This rule takes MLIR input and runs zkir-opt on it to produce
      a single output file after applying the given MLIR passes.
      """,
    implementation = _zkir_opt_impl,
    attrs = {
        "src": attr.label(
            doc = "A single MLIR source file to opt.",
            allow_single_file = [".mlir"],
        ),
        "pass_flags": attr.string_list(
            doc = """
            The pass flags passed to zkir-opt, e.g., --canonicalize
            """,
        ),
        "generated_filename": attr.output(
            doc = """
            The name used for the output file, including the extension (e.g.,
            <filename>.mlir).
            """,
            mandatory = True,
        ),
        "_zkir_opt_binary": executable_attr(_ZKIR_OPT),
    },
)
