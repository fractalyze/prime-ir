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
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/time/time.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"

#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveDialect.h"
#include "prime_ir/Dialect/Field/IR/FieldDialect.h"
#include "stablehlo/dialect/Register.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/command_line_flags.h"
#include "zkx/debug_options_flags.h"
#include "zkx/hlo/evaluator/hlo_evaluator.h"
#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/literal.h"
#include "zkx/literal_util.h"
#include "zkx/mlir/utils/error_util.h"
#include "zkx/pjrt/mlir_to_hlo.h"
#include "zkx/primitive_util.h"
#include "zkx/service/executable.h"
#include "zkx/service/gpu/gpu_executable_run_options.h"
#include "zkx/service/hlo_runner.h"
#include "zkx/service/platform_util.h"
#include "zkx/service/service_executable_run_options.h"
#include "zkx/shape_util.h"
#include "zkx/tools/stablehlo_runner/stablehlo_utils.h"

namespace zkx {
namespace {

namespace se = stream_executor;

// Pooling allocator that caches freed device buffers and returns them on
// subsequent allocations of the same size. Eliminates per-iteration
// cudaMalloc/cudaFree overhead in benchmark loops.
class PoolingAllocator : public se::DeviceMemoryAllocator {
 public:
  explicit PoolingAllocator(se::DeviceMemoryAllocator* real)
      : DeviceMemoryAllocator(real->platform()), real_(real) {}

  ~PoolingAllocator() override {
    for (auto& [key, bufs] : pool_) {
      for (auto& buf : bufs) {
        real_->Deallocate(std::get<0>(key), buf).IgnoreError();
      }
    }
  }

  absl::StatusOr<se::OwningDeviceMemory> Allocate(
      int device_ordinal, uint64_t size, bool retry_on_failure,
      int64_t memory_space) override {
    if (size == 0) {
      return se::OwningDeviceMemory();
    }
    auto key = std::make_tuple(device_ordinal, size, memory_space);
    auto it = pool_.find(key);
    if (it != pool_.end() && !it->second.empty()) {
      auto buf = it->second.back();
      it->second.pop_back();
      return se::OwningDeviceMemory(buf, device_ordinal, this);
    }
    TF_ASSIGN_OR_RETURN(
        auto mem,
        real_->Allocate(device_ordinal, size, retry_on_failure, memory_space));
    auto base = mem.Release();
    return se::OwningDeviceMemory(base, device_ordinal, this);
  }

  absl::Status Deallocate(int device_ordinal,
                          se::DeviceMemoryBase mem) override {
    if (mem.is_null()) return absl::OkStatus();
    pool_[{device_ordinal, mem.size(), /*memory_space=*/0}].push_back(mem);
    return absl::OkStatus();
  }

  bool AllowsAsynchronousDeallocation() const override {
    return real_->AllowsAsynchronousDeallocation();
  }

  absl::StatusOr<se::Stream*> GetStream(int device_ordinal) override {
    return real_->GetStream(device_ordinal);
  }

 private:
  se::DeviceMemoryAllocator* real_;  // not owned
  std::map<std::tuple<int, uint64_t, int64_t>,
           std::vector<se::DeviceMemoryBase>>
      pool_;
};

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
  bool eval = false;
  int32_t iterations = 1;
  std::string input;
  std::string dump_hlo_to;
  std::string platform_name = "cpu";
};

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ParseStablehloModule(
    std::string_view module_text, mlir::MLIRContext* context) {
  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  registry.insert<mlir::prime_ir::field::FieldDialect>();
  registry.insert<mlir::prime_ir::elliptic_curve::EllipticCurveDialect>();
  context->appendDialectRegistry(registry);
  context->loadAllAvailableDialects();

  mlir::BaseScopedDiagnosticHandler diagnostic_handler(context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(
          llvm::StringRef(module_text.data(), module_text.size()),
          mlir::ParserConfig{context});
  if (!module) {
    mlir::emitError(mlir::UnknownLoc::get(context))
        << "Failed to parse StableHLO module";
    return diagnostic_handler.ConsumeStatus();
  }
  return std::move(module);
}

absl::StatusOr<std::unique_ptr<HloModule>> ConvertStablehloToHloModule(
    mlir::ModuleOp module) {
  ZkxComputation computation;
  TF_RETURN_IF_ERROR(MlirToZkxComputation(
      module, computation, /*use_tuple_args=*/false, /*return_tuple=*/false,
      /*use_shardy=*/false));
  TF_ASSIGN_OR_RETURN(HloModuleConfig config,
                      HloModule::CreateModuleConfigFromProto(
                          computation.proto(), GetDebugOptionsFromFlags()));
  return HloModule::CreateFromProto(computation.proto(), config);
}

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

  if (options.eval) {
    // Evaluation mode: use HloEvaluator to interpret the HLO
    // without any GPU compilation. This verifies HLO correctness
    // independently of the GPU emission pipeline.
    TF_ASSIGN_OR_RETURN(std::vector<Literal> literals,
                        CreateInputLiterals(*hlo_module, options));
    std::vector<const Literal*> literal_ptrs;
    literal_ptrs.reserve(literals.size());
    for (const Literal& literal : literals) {
      literal_ptrs.push_back(&literal);
    }
    HloEvaluator evaluator;
    TF_ASSIGN_OR_RETURN(Literal output,
                        evaluator.Evaluate(*hlo_module, literal_ptrs));
    if (options.print_output) {
      std::cout << output.ToString() << std::endl;
    }
    return absl::OkStatus();
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

  if (options.iterations > 1) {
    // Transfer inputs to device once.
    TF_ASSIGN_OR_RETURN(auto argument_buffers,
                        runner.TransferLiteralsToDevice(literal_ptrs));

    // Borrow a single stream for the entire benchmark.
    TF_ASSIGN_OR_RETURN(auto stream,
                        runner.backend().BorrowStream(
                            runner.backend().default_stream_executor()));

    // Warmup
    for (int i = 0; i < 3; ++i) {
      TF_ASSIGN_OR_RETURN(
          ExecutionOutput output,
          runner.ExecuteAsyncWithDeviceBuffers(executable.get(),
                                               argument_buffers, stream.get()));
      TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());
    }

    // Sync benchmark: measure latency (sync after each iteration).
    absl::Time sync_start = absl::Now();
    for (int32_t i = 0; i < options.iterations; ++i) {
      TF_ASSIGN_OR_RETURN(
          ExecutionOutput output,
          runner.ExecuteAsyncWithDeviceBuffers(executable.get(),
                                               argument_buffers, stream.get()));
      TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());
    }
    absl::Time sync_end = absl::Now();

    // Async benchmark: measure throughput (single sync at end).
    std::vector<ExecutionOutput> pending_outputs;
    pending_outputs.reserve(options.iterations);
    absl::Time async_start = absl::Now();
    for (int32_t i = 0; i < options.iterations; ++i) {
      TF_ASSIGN_OR_RETURN(
          ExecutionOutput output,
          runner.ExecuteAsyncWithDeviceBuffers(executable.get(),
                                               argument_buffers, stream.get()));
      pending_outputs.push_back(std::move(output));
    }
    TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());
    absl::Time async_end = absl::Now();
    pending_outputs.clear();

    // Lean benchmark: bypass ExecutionInput construction, call
    // ExecuteAsyncOnStreamWrapper with ShapedBuffer* directly.
    TF_ASSIGN_OR_RETURN(Executable * raw_exec,
                        runner.ExecutableFromWrapped(executable.get()));
    std::vector<const ShapedBuffer*> shaped_ptrs;
    shaped_ptrs.reserve(argument_buffers.size());
    for (const auto& buf : argument_buffers) {
      shaped_ptrs.push_back(&buf);
    }
    ExecutableRunOptions ero;
    ero.set_stream(stream.get());
    ero.set_allocator(runner.backend().memory_allocator());
    ero.set_device_ordinal(
        runner.backend().default_stream_executor()->device_ordinal());
    ero.set_local_device_count(runner.backend().device_count());
    ServiceExecutableRunOptions sro(ero);
    gpu::GpuExecutableRunOptions gpu_opts;
    sro.mutable_run_options()->set_gpu_executable_run_options(&gpu_opts);
    // Warmup lean path.
    for (int i = 0; i < 3; ++i) {
      TF_ASSIGN_OR_RETURN(
          ScopedShapedBuffer output,
          raw_exec->ExecuteAsyncOnStreamWrapper(&sro, shaped_ptrs));
      TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());
    }
    absl::Time lean_start = absl::Now();
    for (int32_t i = 0; i < options.iterations; ++i) {
      TF_ASSIGN_OR_RETURN(
          ScopedShapedBuffer output,
          raw_exec->ExecuteAsyncOnStreamWrapper(&sro, shaped_ptrs));
      TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());
    }
    absl::Time lean_end = absl::Now();

    // GPU-only timing: measure actual kernel execution time via CUDA events.
    // Uses PoolingAllocator to eliminate per-iteration cudaMalloc/cudaFree,
    // matching cuPQC's pre-allocation methodology.
    bool has_gpu_timer = (options.platform_name == "gpu");
    absl::Duration gpu_dur;
    if (has_gpu_timer) {
      PoolingAllocator pool_alloc(runner.backend().memory_allocator());
      ExecutableRunOptions pool_ero;
      pool_ero.set_stream(stream.get());
      pool_ero.set_allocator(&pool_alloc);
      pool_ero.set_device_ordinal(
          runner.backend().default_stream_executor()->device_ordinal());
      pool_ero.set_local_device_count(runner.backend().device_count());
      ServiceExecutableRunOptions pool_sro(pool_ero);
      gpu::GpuExecutableRunOptions pool_gpu_opts;
      pool_sro.mutable_run_options()->set_gpu_executable_run_options(
          &pool_gpu_opts);
      // Warmup with pooling allocator to populate the buffer pool.
      for (int i = 0; i < 3; ++i) {
        TF_ASSIGN_OR_RETURN(
            ScopedShapedBuffer output,
            raw_exec->ExecuteAsyncOnStreamWrapper(&pool_sro, shaped_ptrs));
        TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());
      }
      auto* executor = runner.backend().default_stream_executor();
      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<stream_executor::EventBasedTimer> gpu_timer,
          executor->CreateEventBasedTimer(stream.get(),
                                          /*use_delay_kernel=*/false));
      for (int32_t i = 0; i < options.iterations; ++i) {
        TF_ASSIGN_OR_RETURN(
            ScopedShapedBuffer output,
            raw_exec->ExecuteAsyncOnStreamWrapper(&pool_sro, shaped_ptrs));
      }
      TF_ASSIGN_OR_RETURN(gpu_dur, gpu_timer->GetElapsedDuration());
    }

    absl::Duration sync_dur = sync_end - sync_start;
    absl::Duration async_dur = async_end - async_start;
    absl::Duration lean_dur = lean_end - lean_start;
    absl::Duration sync_avg = sync_dur / options.iterations;
    absl::Duration async_avg = async_dur / options.iterations;
    absl::Duration lean_avg = lean_dur / options.iterations;

    std::cout << "Benchmark: " << options.iterations << " iterations\n";
    std::cout << "\n  Sync (latency):\n";
    std::cout << "    Total: " << absl::FormatDuration(sync_dur) << "\n";
    std::cout << "    Avg:   " << absl::FormatDuration(sync_avg) << "/iter\n";
    std::cout << "\n  Async (throughput):\n";
    std::cout << "    Total: " << absl::FormatDuration(async_dur) << "\n";
    std::cout << "    Avg:   " << absl::FormatDuration(async_avg) << "/iter\n";
    std::cout << "\n  Lean (ShapedBuffer*, sync):\n";
    std::cout << "    Total: " << absl::FormatDuration(lean_dur) << "\n";
    std::cout << "    Avg:   " << absl::FormatDuration(lean_avg) << "/iter\n";
    if (has_gpu_timer) {
      absl::Duration gpu_avg = gpu_dur / options.iterations;
      std::cout << "\n  GPU-only (CUDA events):\n";
      std::cout << "    Total: " << absl::FormatDuration(gpu_dur) << "\n";
      std::cout << "    Avg:   " << absl::FormatDuration(gpu_avg) << "/iter\n";
    }
    return absl::OkStatus();
  }

  TF_ASSIGN_OR_RETURN(Literal output, runner.ExecuteWithExecutable(
                                          executable.get(), literal_ptrs,
                                          /*profile=*/nullptr));

  if (options.print_output) {
    std::cout << output.ToString() << std::endl;
  }
  return absl::OkStatus();
}

}  // namespace
}  // namespace zkx

int main(int argc, char** argv) {
  zkx::Options options;
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
      tsl::Flag("eval", &options.eval,
                "Evaluate HLO using software interpreter (HloEvaluator)."),
  };

  zkx::AppendDebugOptionsFlags(&flag_list);

  // The usage string includes the message at the top of the file, the
  // DebugOptions flags and the flags defined above.
  const std::string kUsageString =
      absl::StrCat(zkx::kUsage, "\n\n", tsl::Flags::Usage(argv[0], flag_list));
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
