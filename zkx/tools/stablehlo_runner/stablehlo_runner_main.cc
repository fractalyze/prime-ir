/* Copyright 2026 The ZKX Authors.

Licensed under the Apache License, Version 2.0 (the "License")
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/time/time.h"

#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/command_line_flags.h"
#include "zkx/debug_options_flags.h"
#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/literal.h"
#include "zkx/literal_util.h"
#include "zkx/primitive_util.h"
#include "zkx/service/hlo_runner.h"
#include "zkx/service/platform_util.h"
#include "zkx/shape_util.h"
#include "zkx/tools/stablehlo_runner/stablehlo_utils.h"

namespace {

const char* const kUsage = R"(
Run StableHLO MLIR by lowering it to HLO and executing with HloRunner.

Usage:
  bazel run //zkx/tools/stablehlo_runner:stablehlo_runner_main -- /path/to/module.mlir

Flags:
  --run=true|false             Whether to execute after lowering (default: true)
  --print_output=true|false    Whether to print output literal (default: true)
  --input="v0,v1,...,vN"       Comma-separated input values (parsed per element type)
  --use_random_inputs          Populate inputs with random data (default: false)
  --dump_hlo_to=/path/to/file  Write lowered HLO text to file
)";

struct Options {
  bool run = true;
  bool print_output = true;
  bool use_random_inputs = false;
  int32_t iterations = 1;
  std::string input;
  std::string dump_hlo_to;
  std::string platform_name = "cpu";
};

}  // namespace

namespace zkx {
namespace {

absl::StatusOr<Literal> ParseInputLiteral(const Shape& shape,
                                          std::string_view input_str) {
  std::vector<std::string_view> tokens = absl::StrSplit(input_str, ',');
  int64_t num_elements = ShapeUtil::ElementsIn(shape);
  if (static_cast<int64_t>(tokens.size()) != num_elements) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Expected ", num_elements, " values but got ", tokens.size()));
  }

  Literal literal(shape);
  PrimitiveType elem_type = shape.element_type();

  return primitive_util::PrimitiveTypeSwitch<absl::StatusOr<Literal>>(
      [&](auto primitive_type_constant) -> absl::StatusOr<Literal> {
        if constexpr (primitive_type_constant == PRIMITIVE_TYPE_INVALID ||
                      !primitive_util::IsArrayType(primitive_type_constant)) {
          return absl::InvalidArgumentError(absl::StrCat(
              "Unsupported element type: ", PrimitiveType_Name(elem_type)));
        } else {
          using NativeT = primitive_util::NativeTypeOf<primitive_type_constant>;
          auto data = literal.data<NativeT>();
          for (int64_t i = 0; i < num_elements; ++i) {
            TF_ASSIGN_OR_RETURN(
                data[i],
                primitive_util::NativeTypeFromDecString<NativeT>(tokens[i]));
          }
          return std::move(literal);
        }
      },
      elem_type);
}

absl::StatusOr<std::vector<Literal>> CreateInputLiterals(
    const HloModule& module, const Options& options) {
  std::vector<Literal> literals;
  const HloComputation* entry = module.entry_computation();
  literals.reserve(entry->num_parameters());

  if (!options.input.empty()) {
    if (entry->num_parameters() != 1) {
      return absl::InvalidArgumentError(
          "--input is only supported for modules with a single parameter");
    }
    const Shape& shape = entry->parameter_instruction(0)->shape();
    TF_ASSIGN_OR_RETURN(Literal literal,
                        ParseInputLiteral(shape, options.input));
    literals.push_back(std::move(literal));
    return literals;
  }

  for (int64_t i = 0; i < entry->num_parameters(); ++i) {
    const Shape& shape = entry->parameter_instruction(i)->shape();
    TF_ASSIGN_OR_RETURN(Literal literal,
                        MakeFakeLiteral(shape, options.use_random_inputs));
    literals.push_back(std::move(literal));
  }
  return literals;
}

absl::Status RunStablehlo(const Options& options, const char* module_path) {
  std::string module_text;
  TF_RETURN_IF_ERROR(
      tsl::ReadFileToString(tsl::Env::Default(), module_path, &module_text));

  mlir::MLIRContext context;
  TF_ASSIGN_OR_RETURN(auto stablehlo_module,
                      ParseStablehloModule(module_text, &context));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> hlo_module,
                      ConvertStablehloToHloModule(*stablehlo_module));

  if (!options.dump_hlo_to.empty()) {
    TF_RETURN_IF_ERROR(tsl::WriteStringToFile(
        tsl::Env::Default(), options.dump_hlo_to, hlo_module->ToString()));
  }

  if (!options.run) {
    return absl::OkStatus();
  }

  TF_ASSIGN_OR_RETURN(auto platform,
                      PlatformUtil::GetPlatform(options.platform_name));
  HloRunner runner(platform);
  TF_ASSIGN_OR_RETURN(std::vector<Literal> literals,
                      CreateInputLiterals(*hlo_module, options));
  std::vector<const Literal*> literal_ptrs;
  literal_ptrs.reserve(literals.size());
  for (const Literal& literal : literals) {
    literal_ptrs.push_back(&literal);
  }

  TF_ASSIGN_OR_RETURN(
      auto executable,
      runner.CreateExecutable(std::move(hlo_module), /*run_hlo_passes=*/true));

  Literal output;
  if (options.iterations > 1) {
    // Warmup
    TF_ASSIGN_OR_RETURN(
        output, runner.ExecuteWithExecutable(executable.get(), literal_ptrs,
                                             /*profile=*/nullptr));

    absl::Time start = absl::Now();
    for (int32_t i = 0; i < options.iterations; ++i) {
      TF_ASSIGN_OR_RETURN(
          output, runner.ExecuteWithExecutable(executable.get(), literal_ptrs,
                                               /*profile=*/nullptr));
    }
    absl::Duration elapsed = absl::Now() - start;
    std::cout << options.iterations << " iterations in "
              << absl::FormatDuration(elapsed) << " ("
              << absl::FormatDuration(elapsed / options.iterations) << "/iter)"
              << std::endl;
  } else {
    TF_ASSIGN_OR_RETURN(
        output, runner.ExecuteWithExecutable(executable.get(), literal_ptrs,
                                             /*profile=*/nullptr));
  }

  if (options.print_output) {
    std::cout << output.ToString() << std::endl;
  }
  return absl::OkStatus();
}

}  // namespace
}  // namespace zkx

int main(int argc, char** argv) {
  Options options;
  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("run", &options.run,
                "Whether to execute after lowering to HLO."),
      tsl::Flag("print_output", &options.print_output,
                "Whether to print the output literal."),
      tsl::Flag("input", &options.input,
                "Comma-separated input values (parsed per element type)."),
      tsl::Flag("use_random_inputs", &options.use_random_inputs,
                "Populate inputs with random data (otherwise zeros)."),
      tsl::Flag("dump_hlo_to", &options.dump_hlo_to,
                "Write lowered HLO text to the given file path."),
      tsl::Flag("platform", &options.platform_name,
                "Execution platform: cpu, gpu/cuda, rocm, or interpreter."),
      tsl::Flag("iterations", &options.iterations,
                "Number of execution iterations for benchmarking."),
  };

  zkx::AppendDebugOptionsFlags(&flag_list);

  // The usage string includes the message at the top of the file, the
  // DebugOptions flags and the flags defined above.
  const std::string kUsageString =
      absl::StrCat(kUsage, "\n\n", tsl::Flags::Usage(argv[0], flag_list));
  bool parse_ok = tsl::Flags::Parse(&argc, argv, flag_list);
  if (!parse_ok || argc < 2) {
    LOG(ERROR) << kUsageString;
    return 1;
  }

  absl::Status s = zkx::RunStablehlo(options, argv[1]);
  if (!s.ok()) {
    LOG(ERROR) << s;
    return 1;
  }
  return 0;
}
