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

// ICICLE sumcheck benchmark using StableHLO -> HLO -> HloRunner pipeline.
//
// Benchmarks the ICICLE sumcheck prover (E · (A · B − C) combine function,
// Poseidon2 Fiat-Shamir transcript) on BabyBear field at multiple log_sizes.
// The StableHLO modules are pre-compiled from a JAX implementation.

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/strip.h"
#include "benchmark/benchmark.h"
#include "zkbench/benchmark_context.h"
#include "zkbench/benchmark_main.h"
#include "zkbench/hash.h"

#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/literal.h"
#include "zkx/literal_util.h"
#include "zkx/primitive_util.h"
#include "zkx/service/executable.h"
#include "zkx/service/hlo_runner.h"
#include "zkx/service/platform_util.h"
#include "zkx/service/shaped_buffer.h"
#include "zkx/shape_util.h"
#include "zkx/tools/stablehlo_runner/stablehlo_utils.h"
#include "zkx/tools/stablehlo_runner/sumcheck_testlib.h"

namespace {

using zkx::sumcheck_testlib::ComputeDeterministicClaimedSum;
using zkx::sumcheck_testlib::FillDeterministicChallenges;
using zkx::sumcheck_testlib::FillDeterministicPolys;
using zkx::sumcheck_testlib::FillDeterministicScalar;
using zkx::sumcheck_testlib::FromMontgomery;

// Static config — add a new line here to benchmark an additional log_size.
struct SumcheckConfig {
  int log_size;
  const char* mlir_path;
};

constexpr SumcheckConfig kConfigs[] = {
    {20, "zkx/tools/stablehlo_runner/icicle_sumcheck_fs_20.stablehlo.mlir"},
    {22, "zkx/tools/stablehlo_runner/icicle_sumcheck_fs_22.stablehlo.mlir"},
    {24, "zkx/tools/stablehlo_runner/icicle_sumcheck_fs_24.stablehlo.mlir"},
    {26, "zkx/tools/stablehlo_runner/icicle_sumcheck_fs_26.stablehlo.mlir"},
};

struct BenchmarkState {
  std::unique_ptr<zkx::HloRunner> runner;
  std::unique_ptr<zkx::OpaqueExecutable> executable;
  std::vector<zkx::Literal> inputs;
  std::vector<const zkx::Literal*> input_ptrs;
  // Device-resident buffers for the timing loop (avoids H2D per iteration).
  std::vector<zkx::ScopedShapedBuffer> device_buffers;
};

absl::Status SetupBenchmark(const char* mlir_path,
                            std::string_view platform_name, bool deterministic,
                            size_t config_idx, BenchmarkState& state) {
  std::string module_text;
  TF_RETURN_IF_ERROR(
      tsl::ReadFileToString(tsl::Env::Default(), mlir_path, &module_text));

  mlir::MLIRContext context;
  TF_ASSIGN_OR_RETURN(auto stablehlo_module,
                      zkx::ParseStablehloModule(module_text, &context));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<zkx::HloModule> hlo_module,
                      zkx::ConvertStablehloToHloModule(*stablehlo_module));

  TF_ASSIGN_OR_RETURN(auto platform,
                      zkx::PlatformUtil::GetPlatform(platform_name));
  state.runner = std::make_unique<zkx::HloRunner>(platform);

  // Create input literals (zero-filled, then optionally overwritten).
  const zkx::HloComputation* entry = hlo_module->entry_computation();
  state.inputs.reserve(entry->num_parameters());
  for (int64_t i = 0; i < entry->num_parameters(); ++i) {
    const zkx::Shape& shape = entry->parameter_instruction(i)->shape();
    TF_ASSIGN_OR_RETURN(zkx::Literal literal,
                        zkx::MakeFakeLiteral(shape, /*use_random=*/false));
    state.inputs.push_back(std::move(literal));
  }
  if (deterministic && state.inputs.size() >= 2) {
    FillDeterministicPolys(state.inputs[0]);
    // Detect FS variant: second param is claimed_sum scalar (rank 0) vs
    // challenge vector (rank 1).
    if (state.inputs[1].shape().rank() == 0) {
      // FS variant: compute and fill claimed_sum = Σ E · (A · B − C).
      int64_t poly_size = state.inputs[0].shape().dimensions(1);
      uint32_t claimed_sum = ComputeDeterministicClaimedSum(poly_size);
      LOG(INFO) << "Computed claimed_sum = " << claimed_sum;
      FillDeterministicScalar(state.inputs[1], claimed_sum);
    } else {
      // Non-FS variant: fill challenge vector.
      FillDeterministicChallenges(state.inputs[1]);
    }
  }
  if (deterministic) {
    for (size_t i = 0; i < state.inputs.size(); ++i) {
      const auto& lit = state.inputs[i];
      bool is_mont =
          zkx::primitive_util::IsMontgomeryForm(lit.shape().element_type());
      LOG(INFO) << "input[" << i
                << "] shape: " << zkx::ShapeUtil::HumanString(lit.shape());
      const auto* d = static_cast<const uint32_t*>(lit.untyped_data());
      size_t n = std::min<size_t>(lit.size_bytes() / sizeof(uint32_t), 5);
      std::string vals;
      for (size_t j = 0; j < n; ++j) {
        if (j > 0) vals += ", ";
        uint32_t v = is_mont ? FromMontgomery(d[j]) : d[j];
        vals += std::to_string(v);
      }
      LOG(INFO) << "  first values (std): [" << vals << "]";
    }
  }
  state.input_ptrs.reserve(state.inputs.size());
  for (const auto& literal : state.inputs) {
    state.input_ptrs.push_back(&literal);
  }

  // Compile the executable.
  TF_ASSIGN_OR_RETURN(state.executable,
                      state.runner->CreateExecutable(std::move(hlo_module),
                                                     /*run_hlo_passes=*/true));

  // Pre-stage inputs on device so the timing loop avoids H2D transfers.
  TF_ASSIGN_OR_RETURN(state.device_buffers,
                      state.runner->TransferLiteralsToDevice(state.input_ptrs));

  // Warm up and capture output for test vectors.
  auto warmup_result = state.runner->ExecuteWithDeviceBuffers(
      state.executable.get(), state.device_buffers);
  bool verified = warmup_result.ok();

  // Compute test vector hashes over all input/output bytes.
  std::string input_hash;
  if (!state.inputs.empty()) {
    size_t total_input_bytes = 0;
    for (const auto& literal : state.inputs) {
      total_input_bytes += literal.size_bytes();
    }
    std::vector<uint8_t> input_bytes;
    input_bytes.reserve(total_input_bytes);
    for (const auto& literal : state.inputs) {
      auto* data = static_cast<const uint8_t*>(literal.untyped_data());
      input_bytes.insert(input_bytes.end(), data, data + literal.size_bytes());
    }
    input_hash = zkbench::ComputeHash(input_bytes.data(), input_bytes.size());
  }
  std::string output_hash;
  if (verified) {
    TF_ASSIGN_OR_RETURN(zkx::Literal output,
                        state.runner->TransferLiteralFromDevice(
                            std::move(warmup_result).value().Result()));

    // The output may be a tuple (multiple return values). Concatenate all
    // leaf element bytes for hashing.
    std::vector<uint8_t> output_bytes;
    if (output.shape().IsTuple()) {
      std::vector<zkx::Literal> elements = output.DecomposeTuple();

      // Print round polynomials if using deterministic inputs.
      if (deterministic) {
        bool out_is_mont = false;
        for (size_t e = 0; e < elements.size(); ++e) {
          LOG(INFO) << "output[" << e << "] shape: "
                    << zkx::ShapeUtil::HumanString(elements[e].shape());
          if (zkx::primitive_util::IsMontgomeryForm(
                  elements[e].shape().element_type())) {
            out_is_mont = true;
          }
        }
        // elements[0] = tensor<NxMx!pf_babybear> (round polys)
        if (!elements.empty() && elements[0].shape().rank() == 2) {
          const auto& rp = elements[0];
          const auto* rp_data = static_cast<const uint32_t*>(rp.untyped_data());
          int64_t num_rounds = rp.shape().dimensions(0);
          int64_t num_coeffs = rp.shape().dimensions(1);
          int64_t print_rounds = std::min(num_rounds, int64_t{3});
          for (int64_t r = 0; r < print_rounds; ++r) {
            std::string vals;
            for (int64_t j = 0; j < num_coeffs; ++j) {
              if (j > 0) vals += ", ";
              uint32_t v = rp_data[r * num_coeffs + j];
              if (out_is_mont) v = FromMontgomery(v);
              vals += std::to_string(v);
            }
            LOG(INFO) << "round_poly[" << r << "] = [" << vals << "]";
          }
        }
      }

      for (const auto& elem : elements) {
        auto* data = static_cast<const uint8_t*>(elem.untyped_data());
        output_bytes.insert(output_bytes.end(), data, data + elem.size_bytes());
      }
    } else {
      auto* data = static_cast<const uint8_t*>(output.untyped_data());
      output_bytes.insert(output_bytes.end(), data, data + output.size_bytes());
    }
    output_hash =
        zkbench::ComputeHash(output_bytes.data(), output_bytes.size());
  }

  std::string bench_name =
      absl::StrCat("sumcheck_icicle_2p", kConfigs[config_idx].log_size);
  zkbench::BenchmarkContext::SetTestVectors(bench_name, input_hash, output_hash,
                                            verified);
  zkbench::BenchmarkContext::SetMetadata(
      bench_name, {{"field", "BabyBear"},
                   {"statement", "E · (A · B − C)"},
                   {"log_size", kConfigs[config_idx].log_size},
                   {"platform", std::string(platform_name)}});

  // Free device buffers after warmup. They will be re-staged per benchmark
  // run so that only one config's data resides on GPU at a time.
  state.device_buffers.clear();

  return absl::OkStatus();
}

// Extracts --platform=<name> and --deterministic from argv, removing them so
// Google Benchmark doesn't see unknown flags.
void ParseCustomFlags(int* argc, char** argv, std::string* platform,
                      bool* deterministic) {
  *platform = "cpu";
  *deterministic = false;
  int write_idx = 1;
  for (int i = 1; i < *argc; ++i) {
    std::string_view arg(argv[i]);
    if (absl::ConsumePrefix(&arg, "--platform=")) {
      *platform = std::string(arg);
    } else if (arg == "--deterministic") {
      *deterministic = true;
    } else {
      argv[write_idx++] = argv[i];
    }
  }
  *argc = write_idx;
}

}  // namespace

int main(int argc, char** argv) {
  std::string platform;
  bool deterministic = false;
  ParseCustomFlags(&argc, argv, &platform, &deterministic);

  std::vector<BenchmarkState> states(std::size(kConfigs));

  // Compile each config and register its benchmark dynamically.
  // Device buffers are staged per benchmark run (not upfront) so that only one
  // config's data resides on GPU at a time.
  for (size_t i = 0; i < std::size(kConfigs); ++i) {
    auto status = SetupBenchmark(kConfigs[i].mlir_path, platform, deterministic,
                                 i, states[i]);
    if (!status.ok()) {
      LOG(ERROR) << "Setup failed for log_size=" << kConfigs[i].log_size << ": "
                 << status;
      return 1;
    }

    std::string name = absl::StrCat("sumcheck_icicle_2p", kConfigs[i].log_size);
    benchmark::RegisterBenchmark(name, [&state =
                                            states[i]](benchmark::State& st) {
      // Stage inputs to device before the timing loop.
      auto result = state.runner->TransferLiteralsToDevice(state.input_ptrs);
      if (!result.ok()) {
        st.SkipWithError(result.status().ToString());
        return;
      }
      state.device_buffers = std::move(result).value();

      for (auto _ : st) {
        auto result = state.runner->ExecuteWithDeviceBuffers(
            state.executable.get(), state.device_buffers);
        if (!result.ok()) {
          st.SkipWithError(result.status().ToString());
          return;
        }
      }
      st.counters["items_per_second"] = benchmark::Counter(
          static_cast<double>(st.iterations()), benchmark::Counter::kIsRate);

      // Free device buffers so the next config can use GPU memory.
      state.device_buffers.clear();
    })->Unit(benchmark::kMillisecond);
  }

  int ret = zkbench::BenchmarkMain(argc, argv, "zkx-icicle-sumcheck", "0.1.0");

  // Tear down in reverse dependency order: device buffers must be freed while
  // the runner (and its allocator) is still alive.
  for (auto& state : states) {
    state.device_buffers.clear();
    state.executable.reset();
    state.input_ptrs.clear();
    state.inputs.clear();
    state.runner.reset();
  }
  return ret;
}
