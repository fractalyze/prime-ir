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

"""A rule for running ZKIR benchmark"""

load("@rules_cc//cc:defs.bzl", "cc_import", "cc_test")
load("@zkir//tools:tools.bzl", "zkir_opt")

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

def zkir_mlir_cc_import(name, mlir_src, zkir_opt_flags = [], llc_flags = [], tags = [], **kwargs):
    """Compiles an MLIR file to an object file and exposes it via cc_import.

    This rule runs the following pipeline:
      MLIR -> (zkir-opt) -> LLVM IR -> Object (.o) -> cc_import

    Args:
      name: The name of the cc_import target.
      mlir_src: The source .mlir file to compile.
      zkir_opt_flags: Optional list of flags to pass to zkir-opt.
      llc_flags: Optional list of flags to pass to llc.
      tags: Optional Bazel tags to apply to each build step.
      **kwargs: Additional arguments to pass to each build step.
    """
    zkir_opt_name = "%s_zkir_opt" % name
    generated_zkir_opt_name = "%s_zkir_opt.mlir" % name
    llvmir_target = "%s_mlir_translate" % name
    generated_llvmir_name = "%s_llvmir.ll" % name
    obj_name = "%s_object" % name
    generated_obj_name = "%s.o" % name

    if zkir_opt_flags:
        zkir_opt(
            name = zkir_opt_name,
            src = mlir_src,
            pass_flags = zkir_opt_flags,
            tags = tags,
            generated_filename = generated_zkir_opt_name,
            **kwargs
        )
    else:
        generated_zkir_opt_name = mlir_src

    mlir_translate(
        name = llvmir_target,
        src = generated_zkir_opt_name,
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

def zkir_mlir_template(name, template_src, replacements_quick, replacements_comprehensive):
    """Creates quick and comprehensive MLIR variants from a template using sed.

    DEPRECATED: Use zkir_mlir_template_parameterized instead for better flexibility.

    This macro generates two genrules that produce MLIR files from a template
    by replacing placeholders with concrete values via sed substitutions.

    Args:
      name: Base name for the generated rules and output files.
      template_src: The .mlir.template file containing placeholders.
      replacements_quick: Dictionary of placeholder->value for quick mode.
      replacements_comprehensive: Dictionary of placeholder->value for comprehensive mode.

    Example:
      zkir_mlir_template(
          name = "msm_benchmark_mlir",
          template_src = "msm_benchmark.mlir.template",
          replacements_quick = {
              "TENSOR_SIZE": "65536",
              "MSM_DEGREE": "16",
          },
          replacements_comprehensive = {
              "TENSOR_SIZE": "1048576",
              "MSM_DEGREE": "20",
          },
      )
    """

    # Build sed command for quick mode
    quick_sed_exprs = []
    for placeholder, value in replacements_quick.items():
        quick_sed_exprs.append("s/{}/{}/g".format(placeholder, value))
    quick_sed_cmd = "sed '{}' $< > $@".format("; ".join(quick_sed_exprs))

    # Build sed command for comprehensive mode
    comprehensive_sed_exprs = []
    for placeholder, value in replacements_comprehensive.items():
        comprehensive_sed_exprs.append("s/{}/{}/g".format(placeholder, value))
    comprehensive_sed_cmd = "sed '{}' $< > $@".format("; ".join(comprehensive_sed_exprs))

    native.genrule(
        name = name + "_quick",
        srcs = [template_src],
        outs = [name + "_quick.mlir"],
        cmd = quick_sed_cmd,
    )

    native.genrule(
        name = name + "_comprehensive",
        srcs = [template_src],
        outs = [name + "_comprehensive.mlir"],
        cmd = comprehensive_sed_cmd,
    )

def zkir_mlir_template_parameterized(name, template_src, log_sizes, size_flag, default_log_size = 16, extra_replacements_fn = None):
    """Creates parameterized MLIR files from a template.

    This macro generates multiple genrules, one for each supported log_size,
    and uses config_setting + select() to choose the appropriate one based
    on a build flag.

    Args:
      name: Base name for the generated rules and output files.
      template_src: The .mlir.template file with NUM_COEFFS and ROOT_OF_UNITY placeholders.
      log_sizes: List of log2 sizes to support (e.g., [10, 16, 20, 24]).
      size_flag: Label of the string_flag that specifies the log_size (e.g., ":ntt_log_size").
      default_log_size: Default log_size to use (default: 16).
      extra_replacements_fn: Optional function that takes log_size and returns a dict of
                             PLACEHOLDER->VALUE for additional replacements.

    Example:
      zkir_mlir_template_parameterized(
          name = "ntt_benchmark_mlir",
          template_src = "ntt_benchmark.mlir.template",
          log_sizes = [10, 16, 20, 24],
          size_flag = ":ntt_log_size",
          default_log_size = 16,
      )

      Then use: bazel build --//benchmark/ntt:ntt_log_size=20 //benchmark/ntt:ntt_benchmark_mlir

      With extra replacements:
      def msm_replacements(log_size):
          return {"MSM_DEGREE": str(log_size)}

      zkir_mlir_template_parameterized(
          name = "msm_benchmark_mlir",
          template_src = "msm_benchmark.mlir.template",
          log_sizes = [16, 20],
          size_flag = ":msm_log_size",
          extra_replacements_fn = msm_replacements,
      )
    """

    # Create a genrule for each log_size
    genrule_targets = {}
    for log_size in log_sizes:
        target_name = "{}_size_{}".format(name, log_size)
        output_name = "{}_size_{}.mlir".format(name, log_size)

        # Build command with extra replacements if provided
        cmd_parts = ["$(location //benchmark:generate_mlir)", "--template $<", "--output $@", "--log-size {}".format(log_size)]

        if extra_replacements_fn:
            replacements = extra_replacements_fn(log_size)
            for placeholder, value in replacements.items():
                cmd_parts.append("--extra-replacements {}={}".format(placeholder, value))

        cmd = " ".join(cmd_parts)

        native.genrule(
            name = target_name,
            srcs = [template_src],
            outs = [output_name],
            cmd = cmd,
            tools = ["//benchmark:generate_mlir"],
        )

        genrule_targets[str(log_size)] = ":" + target_name

    # Create config_settings for each log_size
    config_settings = {}
    for log_size in log_sizes:
        setting_name = "_{}_size_{}".format(name, log_size)
        native.config_setting(
            name = setting_name,
            flag_values = {size_flag: str(log_size)},
        )
        config_settings[log_size] = ":" + setting_name

    # Create select() mapping
    select_map = {}
    for log_size in log_sizes:
        select_map[config_settings[log_size]] = genrule_targets[str(log_size)]

    # Set default
    select_map["//conditions:default"] = genrule_targets[str(default_log_size)]

    # Create alias to select the appropriate target
    native.alias(
        name = name,
        actual = select(select_map),
    )

def msm_extra_replacements(log_size):
    """Returns extra replacements for MSM template.

    For MSM, TENSOR_SIZE = 2^log_size and MSM_DEGREE = log_size
    """

    # Pre-calculated powers of 2
    sizes = {
        12: 4096,
        14: 16384,
        16: 65536,
        18: 262144,
        20: 1048576,
        22: 4194304,
    }

    return {
        "TENSOR_SIZE": str(sizes[log_size]),
        "MSM_DEGREE": str(log_size),
    }

def vecops_extra_replacements(log_size):
    """Returns extra replacements for vecops template.

    For vecops, MATRIX_ROWS = 2^log_size
    """

    # Pre-calculated powers of 2
    sizes = {
        12: 4096,
        14: 16384,
        16: 65536,
        18: 262144,
        20: 1048576,
        22: 4194304,
    }

    return {
        "MATRIX_ROWS": str(sizes[log_size]),
    }

def zkir_benchmark(name, srcs, deps, data = [], copts = [], linkopts = [], tags = [], **kwargs):
    """A rule for running a benchmark test."""

    cc_test(
        name = name,
        srcs = srcs,
        deps = deps + [
            "@com_google_benchmark//:benchmark_main",
            "@com_google_googletest//:gtest",
            "@llvm-project//mlir:mlir_runner_utils",
        ],
        tags = tags,
        copts = copts,
        linkopts = linkopts,
        data = data,
        **kwargs
    )
