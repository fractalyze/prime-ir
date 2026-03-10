/* Copyright 2026 The ZKX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// MSM benchmark using StableHLO/HLO → HloRunner pipeline.
//
// Benchmarks BN254 G1/G2 MSM with size 2²⁰ (1,048,576 scalar multiplications).
// Supports both StableHLO MLIR (.stablehlo.mlir) and raw HLO (.hlo) input.
// The group (G1/G2) is detected from the module name.

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/strip.h"
#include "benchmark/benchmark.h"
#include "mlir/IR/MLIRContext.h"
#include "zkbench/benchmark_context.h"
#include "zkbench/benchmark_main.h"
#include "zkbench/hash.h"

#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/literal.h"
#include "zkx/service/hlo_runner.h"
#include "zkx/service/platform_util.h"
#include "zkx/tools/stablehlo_runner/stablehlo_utils.h"

namespace {

struct BenchmarkState {
  std::unique_ptr<zkx::HloRunner> runner;
  std::unique_ptr<zkx::OpaqueExecutable> executable;
  std::vector<zkx::Literal> inputs;
  std::vector<const zkx::Literal*> input_ptrs;
};

BenchmarkState g_state;

// Detect G1 vs G2 from the module path.
std::string DetectGroup(std::string_view path) {
  if (path.find("g2") != std::string_view::npos ||
      path.find("G2") != std::string_view::npos) {
    return "G2";
  }
  return "G1";
}

absl::StatusOr<std::string> SetupBenchmark(const char* module_path,
                                           std::string_view platform_name) {
  std::string module_text;
  TF_RETURN_IF_ERROR(
      tsl::ReadFileToString(tsl::Env::Default(), module_path, &module_text));

  std::unique_ptr<zkx::HloModule> hlo_module;
  mlir::MLIRContext context;
  TF_ASSIGN_OR_RETURN(auto stablehlo_module,
                      zkx::ParseStablehloModule(module_text, &context));
  TF_ASSIGN_OR_RETURN(hlo_module,
                      zkx::ConvertStablehloToHloModule(*stablehlo_module));

  TF_ASSIGN_OR_RETURN(auto platform,
                      zkx::PlatformUtil::GetPlatform(platform_name));
  g_state.runner = std::make_unique<zkx::HloRunner>(platform);

  // Create valid random input literals (Random() for scalars, Generator() for
  // base points — scalar values are the primary MSM performance factor).
  const zkx::HloComputation* entry = hlo_module->entry_computation();
  g_state.inputs.reserve(entry->num_parameters());
  for (int64_t i = 0; i < entry->num_parameters(); ++i) {
    const zkx::Shape& shape = entry->parameter_instruction(i)->shape();
    zkx::Literal literal = zkx::Literal::CreateFromShape(shape);
    zkx::FillLiteralWithRandom(literal);
    g_state.inputs.push_back(std::move(literal));
  }
  g_state.input_ptrs.reserve(g_state.inputs.size());
  for (const auto& literal : g_state.inputs) {
    g_state.input_ptrs.push_back(&literal);
  }

  // Compile the executable.
  TF_ASSIGN_OR_RETURN(g_state.executable,
                      g_state.runner->CreateExecutable(
                          std::move(hlo_module), /*run_hlo_passes=*/true));

  // Warm up and capture output for test vectors.
  auto warmup_result = g_state.runner->ExecuteWithExecutable(
      g_state.executable.get(), g_state.input_ptrs, /*profile=*/nullptr);
  bool verified = warmup_result.ok();

  // Compute test vector hashes over all input/output bytes.
  std::string input_hash;
  if (!g_state.inputs.empty()) {
    size_t total_input_bytes = 0;
    for (const auto& literal : g_state.inputs) {
      total_input_bytes += literal.size_bytes();
    }
    std::vector<uint8_t> input_bytes;
    input_bytes.reserve(total_input_bytes);
    for (const auto& literal : g_state.inputs) {
      auto* data = static_cast<const uint8_t*>(literal.untyped_data());
      input_bytes.insert(input_bytes.end(), data, data + literal.size_bytes());
    }
    input_hash = zkbench::ComputeHash(input_bytes.data(), input_bytes.size());
  }
  std::string output_hash;
  if (verified) {
    zkx::Literal output = std::move(warmup_result).value();
    output_hash =
        zkbench::ComputeHash(static_cast<const uint8_t*>(output.untyped_data()),
                             output.size_bytes());
  }

  std::string group = DetectGroup(module_path);
  std::string bench_name = "BM_Msm" + group;
  zkbench::BenchmarkContext::SetTestVectors(bench_name, input_hash, output_hash,
                                            verified);
  zkbench::BenchmarkContext::SetMetadata(
      bench_name, {{"curve", "BN254"},
                   {"group", group},
                   {"log_size", 20},
                   {"size", 1048576},
                   {"platform", std::string(platform_name)}});

  return bench_name;
}

void BM_Msm(benchmark::State& state) {
  for (auto _ : state) {
    auto result = g_state.runner->ExecuteWithExecutable(
        g_state.executable.get(), g_state.input_ptrs, /*profile=*/nullptr);
    if (!result.ok()) {
      state.SkipWithError(result.status().ToString());
      return;
    }
  }
  state.counters["items_per_second"] = benchmark::Counter(
      static_cast<double>(state.iterations()), benchmark::Counter::kIsRate);
}

void ParseCustomFlags(int* argc, char** argv, std::string* platform,
                      std::string* module_path) {
  *platform = "gpu";
  int write_idx = 1;
  for (int i = 1; i < *argc; ++i) {
    std::string_view arg(argv[i]);
    if (absl::ConsumePrefix(&arg, "--platform=")) {
      *platform = std::string(arg);
    } else if (!arg.empty() && arg[0] != '-' && module_path->empty()) {
      *module_path = std::string(arg);
    } else {
      argv[write_idx++] = argv[i];
    }
  }
  *argc = write_idx;
}

}  // namespace

int main(int argc, char** argv) {
  std::string platform;
  std::string module_path;
  ParseCustomFlags(&argc, argv, &platform, &module_path);

  if (module_path.empty()) {
    LOG(ERROR) << "Usage: msm_benchmark <module_file> [--platform=cpu|gpu] "
                  "[--benchmark_repetitions=N] [--zkbench_out=<file>]";
    return 1;
  }

  absl::StatusOr bench_name_or = SetupBenchmark(module_path.c_str(), platform);
  if (!bench_name_or.ok()) {
    LOG(ERROR) << "Setup failed: " << bench_name_or.status();
    return 1;
  }

  benchmark::RegisterBenchmark(*bench_name_or, BM_Msm);
  int ret = zkbench::BenchmarkMain(argc, argv, "zkx-msm", "0.1.0");

  // Tear down global state before CUDA context is destroyed at exit.
  g_state = {};
  return ret;
}
