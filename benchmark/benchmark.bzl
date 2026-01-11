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

"""A rule for running PrimeIR benchmark"""

load("@prime_ir//tools:tools.bzl", "prime_ir_opt")
load("@rules_cc//cc:defs.bzl", "cc_import", "cc_test")

def executable_attr(label):
    """A helper for declaring executable dependencies."""
    return attr.label(
        default = Label(label),
        allow_single_file = True,
        executable = True,
        # commenting this out breaks cross-compilation, but this should not be a problem
        # for developer builds
        # cfg = "exec",
        cfg = "target",
    )

_LLC = "@llvm-project//llvm:llc"
_MLIR_TRANSLATE = "@llvm-project//mlir:mlir-translate"

def _binary_impl(ctx):
    generated_file = ctx.outputs.generated_filename
    args = ctx.actions.args()
    args.add_all(ctx.attr.pass_flags)
    args.add_all(["-o", generated_file.path])
    args.add(ctx.file.src)

    ctx.actions.run(
        inputs = ctx.attr.src.files,
        outputs = [generated_file],
        arguments = [args],
        executable = ctx.executable._binary,
        toolchain = None,
    )
    return [
        DefaultInfo(files = depset([generated_file, ctx.file.src])),
    ]

llc = rule(
    doc = """
      This rule runs llc
      """,
    implementation = _binary_impl,
    attrs = {
        "src": attr.label(
            doc = "A single LLVM IR source file to translate.",
            allow_single_file = [".ll"],
        ),
        "pass_flags": attr.string_list(
            doc = """
            The pass flags passed to llc, e.g., --filetype=obj
            """,
        ),
        "generated_filename": attr.output(
            doc = """
            The name used for the output file, including the extension (e.g.,
            <filename>.rs for rust files).
            """,
            mandatory = True,
        ),
        "_binary": executable_attr(_LLC),
    },
)

mlir_translate = rule(
    doc = """
      This rule takes MLIR input and runs mlir-translate on it to produce
      a single generated source file in some target language.
      """,
    implementation = _binary_impl,
    attrs = {
        "src": attr.label(
            doc = "A single MLIR source file to translate.",
            allow_single_file = [".mlir"],
        ),
        "pass_flags": attr.string_list(
            doc = """
            The pass flag passed to mlir-translate, e.g., --mlir-to-llvmir
            """,
        ),
        "generated_filename": attr.output(
            doc = """
            The name used for the output file, including the extension (e.g.,
            <filename>.rs for rust files).
            """,
            mandatory = True,
        ),
        "_binary": executable_attr(_MLIR_TRANSLATE),
    },
)

def prime_ir_mlir_cc_import(name, mlir_src, prime_ir_opt_flags = [], llc_flags = [], tags = [], **kwargs):
    """Compiles an MLIR file to an object file and exposes it via cc_import.

    This rule runs the following pipeline:
      MLIR -> (prime-ir-opt) -> LLVM IR -> Object (.o) -> cc_import

    Args:
      name: The name of the cc_import target.
      mlir_src: The source .mlir file to compile.
      prime_ir_opt_flags: Optional list of flags to pass to prime-ir-opt.
      llc_flags: Optional list of flags to pass to llc.
      tags: Optional Bazel tags to apply to each build step.
      **kwargs: Additional arguments to pass to each build step.
    """
    prime_ir_opt_name = "%s_prime_ir_opt" % name
    generated_prime_ir_opt_name = "%s_prime_ir_opt.mlir" % name
    llvmir_target = "%s_mlir_translate" % name
    generated_llvmir_name = "%s_llvmir.ll" % name
    obj_name = "%s_object" % name
    generated_obj_name = "%s.o" % name

    if prime_ir_opt_flags:
        prime_ir_opt(
            name = prime_ir_opt_name,
            src = mlir_src,
            pass_flags = prime_ir_opt_flags,
            tags = tags,
            generated_filename = generated_prime_ir_opt_name,
            **kwargs
        )
    else:
        generated_prime_ir_opt_name = mlir_src

    mlir_translate(
        name = llvmir_target,
        src = generated_prime_ir_opt_name,
        pass_flags = ["--mlir-to-llvmir"],
        tags = tags,
        generated_filename = generated_llvmir_name,
        **kwargs
    )

    llc(
        name = obj_name,
        src = generated_llvmir_name,
        pass_flags = ["-relocation-model=pic", "-filetype=obj"] + llc_flags,
        tags = tags,
        generated_filename = generated_obj_name,
        **kwargs
    )
    cc_import(
        name = name,
        objects = [generated_obj_name],
        data = [generated_obj_name],
        tags = tags,
        **kwargs
    )

def prime_ir_benchmark(name, srcs, deps, data = [], copts = [], linkopts = [], tags = [], **kwargs):
    """A rule for running a benchmark test."""

    cc_test(
        name = name,
        srcs = srcs,
        deps = deps + [
            "@com_google_benchmark//:benchmark_main",
            "@com_google_googletest//:gtest",
            "@llvm-project//mlir:mlir_runner_utils",
            "//prime_ir/Dialect/ModArith/IR:ModArith",
            "//prime_ir/Dialect/Field/IR:Field",
            "//prime_ir/Dialect/EllipticCurve/IR:EllipticCurve",
        ],
        tags = tags,
        copts = copts,
        linkopts = linkopts,
        data = data,
        **kwargs
    )
