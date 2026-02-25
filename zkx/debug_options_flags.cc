/* Copyright 2017 The OpenXLA Authors.
Copyright 2025 The ZKX Authors.

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

#include "zkx/debug_options_flags.h"

#include "absl/base/call_once.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_map.h"
#include "absl/debugging/leak_check.h"
#include "absl/log/check.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_join.h"

#include "zkx/debug_options_parsers.h"
#include "zkx/parse_flags_from_env.h"
#include "zkx/stream_executor/cuda/nvjitlink_support.h"
#include "zkx/stream_executor/cuda/ptx_compiler_support.h"

namespace zkx {

DebugOptions DefaultDebugOptionsIgnoringFlags() {
  DebugOptions opts;
  opts.set_zkx_llvm_enable_invariant_load_metadata(true);
  opts.set_zkx_llvm_disable_expensive_passes(false);
  opts.set_zkx_backend_optimization_level(3);
  opts.set_zkx_gpu_cuda_data_dir("./cuda_sdk_lib");
  opts.set_zkx_gpu_generate_debug_info(false);
  opts.set_zkx_gpu_generate_line_info(false);

  opts.set_zkx_dump_hlo_as_html(false);
  opts.set_zkx_dump_fusion_visualization(false);
  opts.set_zkx_dump_include_timestamp(false);
  opts.set_zkx_dump_max_hlo_modules(-1);
  opts.set_zkx_dump_module_metadata(false);
  opts.set_zkx_dump_hlo_as_long_text(false);
  opts.set_zkx_dump_large_constants(false);
  opts.set_zkx_dump_enable_mlir_pretty_form(true);
  opts.set_zkx_annotate_with_emitter_loc(false);
  opts.set_zkx_debug_buffer_assignment_show_max(15);
  opts.set_zkx_cpu_parallel_codegen_split_count(32);
  opts.set_zkx_cpu_copy_insertion_use_region_analysis(false);
  opts.set_zkx_cpu_enable_concurrency_optimized_scheduler(true);
  opts.set_zkx_cpu_prefer_vector_width(256);
  opts.set_zkx_cpu_max_isa("");

  opts.set_zkx_gpu_collectives_use_persistent_cliques(false);

  opts.set_zkx_force_host_platform_device_count(1);
  opts.set_zkx_gpu_enable_approx_costly_collectives(false);

  opts.set_zkx_gpu_unsafe_fallback_to_driver_on_ptxas_not_found(false);
  opts.set_zkx_multiheap_size_constraint_per_heap(-1);
  opts.set_zkx_enable_dumping(true);

  opts.set_zkx_gpu_nccl_termination_timeout_seconds(-1);
  opts.set_zkx_gpu_enable_shared_constants(true);
  opts.set_zkx_gpu_enable_nccl_user_buffers(false);
  opts.set_zkx_gpu_enable_nccl_comm_splitting(true);
  opts.set_zkx_gpu_nccl_init_max_rank_per_root_ratio(0);

  opts.set_zkx_gpu_temp_buffer_use_separate_color(false);
  opts.set_zkx_gpu_require_exclusive_lock(false);

  opts.set_zkx_dump_latency_hiding_schedule(false);
  opts.set_zkx_gpu_enable_latency_hiding_scheduler(false);
  opts.set_zkx_gpu_pgle_profile_file_or_directory_path("");
  opts.set_zkx_gpu_enable_highest_priority_async_stream(true);
  opts.set_zkx_gpu_memory_limit_slop_factor(95);
  opts.set_zkx_gpu_enable_pipelined_p2p(false);

  opts.set_zkx_gpu_experimental_pipeline_parallelism_opt_level(
      DebugOptions::PIPELINE_PARALLELISM_OPT_LEVEL_DISABLE);

  opts.set_zkx_llvm_force_inline_before_split(true);

  opts.set_zkx_gpu_filter_kernels_spilling_registers_on_autotuning(true);
  opts.set_zkx_gpu_fail_ptx_compilation_on_register_spilling(false);
  opts.set_zkx_gpu_target_config_filename("");

  opts.set_zkx_gpu_enable_llvm_module_compilation_parallelism(false);
  opts.set_zkx_gpu_enable_libnvptxcompiler(se::IsLibNvPtxCompilerSupported());
  opts.set_zkx_gpu_libnvjitlink_mode(DebugOptions::LIB_NV_JIT_LINK_MODE_AUTO);

  opts.set_zkx_gpu_nccl_collective_max_nchannels(0);
  opts.set_zkx_gpu_nccl_p2p_max_nchannels(0);

  opts.set_zkx_gpu_experimental_stream_annotation(false);

  opts.set_zkx_gpu_nccl_terminate_on_error(false);

  opts.set_zkx_syntax_sugar_async_ops(false);

  opts.set_zkx_gpu_pgle_accuracy_checker(
      DebugOptions::PGLE_STRICTNESS_LEVEL_WARN);

  opts.set_zkx_gpu_executable_warn_stuck_timeout_seconds(10);
  opts.set_zkx_gpu_executable_terminate_timeout_seconds(30);
  opts.set_zkx_gpu_experimental_parallel_collective_overlap_limit(1);
  opts.set_zkx_pjrt_allow_auto_layout_in_hlo(false);
  return opts;
}

namespace {

absl::once_flag flags_init;
DebugOptions* flag_values;
std::vector<tsl::Flag>* flag_objects;

// Maps pass -> initial fuel values (parsed when AllocateFlags was run).
auto* const initial_fuel =
    absl::IgnoreLeak(new absl::flat_hash_map<std::string, int64_t>());

// Maps pass -> whether fuel was ever consumed for that pass.
auto* const fuel_ever_consumed =
    absl::IgnoreLeak(new absl::node_hash_map<std::string, std::atomic<bool>>());

// Maps pass -> remaining fuel.
//
// All threads start off using this global fuel pool, but ResetThreadLocalFuel()
// switches them to a thread-local fuel pool.
auto* const global_fuel = absl::IgnoreLeak(
    new absl::node_hash_map<std::string, std::atomic<int64_t>>());

// If we're using thread-local fuel, this stores it.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
thread_local std::unique_ptr<
    absl::node_hash_map<std::string, std::atomic<int64_t>>>
    thread_fuel;

// Logs a warning if a pass's fuel was never consumed, on the theory that this
// may be a typo in the flag value.  Called atexit.
void WarnIfFuelWasNeverConsumed() {
  CHECK(fuel_ever_consumed != nullptr);
  for (const auto& kv : *fuel_ever_consumed) {
    std::string_view pass = kv.first;
    bool was_consumed = kv.second;
    if (!was_consumed) {
      LOG(ERROR) << absl::StreamFormat(
          "Compiler fuel for \"%s\" was never consumed. This may be a typo in "
          "the --zkx_fuel flag you passed.",
          pass);
    }
  }
}

}  // namespace

void MakeDebugOptionsFlags(std::vector<tsl::Flag>* flag_list,
                           DebugOptions* debug_options) {
  // Returns a lambda that calls "member_setter" on "debug_options" with the
  // argument passed in to the lambda.
  auto bool_setter_for =
      [debug_options](void (DebugOptions::*member_setter)(bool)) {
        return [debug_options, member_setter](bool value) {
          (debug_options->*member_setter)(value);
          return true;
        };
      };

  // Returns a lambda that calls "member_setter" on "debug_options" with the
  // argument passed in to the lambda.
  auto int32_setter_for =
      [debug_options](void (DebugOptions::*member_setter)(int32_t)) {
        return [debug_options, member_setter](int32_t value) {
          (debug_options->*member_setter)(value);
          return true;
        };
      };

  auto int64_setter_for =
      [debug_options](void (DebugOptions::*member_setter)(int64_t)) {
        return [debug_options, member_setter](int64_t value) {
          (debug_options->*member_setter)(value);
          return true;
        };
      };

  auto string_setter_for = [debug_options](void (DebugOptions::*member_setter)(
                               const std::string& value)) {
    return [debug_options, member_setter](const std::string& value) {
      (debug_options->*member_setter)(value);
      return true;
    };
  };

  auto uppercase_string_setter_for =
      [debug_options](
          void (DebugOptions::*member_setter)(const std::string& value)) {
        return [debug_options, member_setter](const std::string& value) {
          (debug_options->*member_setter)(absl::AsciiStrToUpper(value));
          return true;
        };
      };

  auto make_repeated_string_setter =
      [debug_options](void (DebugOptions::*adder)(const std::string&)) {
        return [debug_options, adder](std::string_view comma_separated_values) {
          for (const std::string_view passname :
               absl::StrSplit(comma_separated_values, ',')) {
            (debug_options->*adder)(std::string(passname));
          }
          return true;
        };
      };

  // Custom "sub-parser" lambda for zkx_disable_hlo_passes.
  auto setter_for_zkx_disable_hlo_passes =
      make_repeated_string_setter(&DebugOptions::add_zkx_disable_hlo_passes);

  // Custom "sub-parser" lambda for zkx_enable_hlo_passes_only.
  auto setter_for_zkx_enable_hlo_passes_only = make_repeated_string_setter(
      &DebugOptions::add_zkx_enable_hlo_passes_only);

  // Custom "sub-parser" lambda for zkx_gpu_ptx_file.
  auto setter_for_zkx_gpu_ptx_file = [debug_options](const std::string& value) {
    debug_options->add_zkx_gpu_ptx_file(value);
    return true;
  };

  // Custom "sub-parser" lambda for zkx_gpu_llvm_ir_file.
  auto setter_for_zkx_gpu_llvm_ir_file =
      [debug_options](const std::string& value) {
        debug_options->add_zkx_gpu_llvm_ir_file(value);
        return true;
      };

  // Custom "sub-parser" lambda for zkx_backend_extra_options.
  auto setter_for_zkx_backend_extra_options =
      [debug_options](std::string_view comma_separated_values) {
        auto* extra_options_map =
            debug_options->mutable_zkx_backend_extra_options();
        parse_zkx_backend_extra_options(extra_options_map,
                                        comma_separated_values);
        return true;
      };

  // Custom "sub-parser" lambda for
  // zkx_gpu_experimental_pipeline_parallelism_opt_level.
  auto setter_for_zkx_gpu_experimental_pipeline_parallelism_opt_level =
      [debug_options](const std::string& value) {
        DebugOptions::PipelineParallelismOptLevel level;
        if (!DebugOptions::PipelineParallelismOptLevel_Parse(value, &level)) {
          return false;
        }
        debug_options->set_zkx_gpu_experimental_pipeline_parallelism_opt_level(
            level);
        return true;
      };

  // Custom "sub-parser" for zkx_fuel.  Note that ConsumeFuel does not do any
  // locking on the fuel global variables.  This means that it's
  // illegal/undefined behavior to modify this flag value while the compiler is
  // running.
  auto setter_for_zkx_fuel = [](std::string zkx_fuel_value) {
    initial_fuel->clear();
    global_fuel->clear();
    fuel_ever_consumed->clear();

    for (const auto& kv : absl::StrSplit(zkx_fuel_value, ',')) {
      std::vector<std::string> pass_and_fuel = absl::StrSplit(kv, '=');
      if (pass_and_fuel.size() != 2) {
        LOG(ERROR) << absl::StreamFormat(
            "Illegal value for --zkx_fuel. Saw %s, but expected token %s to "
            "have format X=INTEGER.",
            zkx_fuel_value, kv);
        return false;
      }
      const auto& pass = pass_and_fuel[0];
      const auto& fuel_str = pass_and_fuel[1];
      int64_t fuel;
      if (!absl::SimpleAtoi(fuel_str, &fuel)) {
        LOG(ERROR) << absl::StreamFormat(
            "Illegal value for --zkx_fuel. Saw %s, but expected token %s to be "
            "an integer.",
            zkx_fuel_value, fuel_str);
        return false;
      }
      initial_fuel->emplace(pass, fuel);
      global_fuel->emplace(pass, fuel);
      fuel_ever_consumed->emplace(pass, false);
    }

    // If --zkx_fuel was specified, register an atexit handler which logs a
    // warning if a pass was specified but never consumed any fuel, on the
    // theory that this is may be a typo.
    if (!initial_fuel->empty()) {
      static absl::once_flag register_atexit_once;
      absl::call_once(
          register_atexit_once,
          +[] { std::atexit(WarnIfFuelWasNeverConsumed); });
    }
    return true;
  };

  auto collective_op_types_to_string =
      [](google::protobuf::RepeatedField<int> collective_ops) -> std::string {
    struct Formatter {
      void operator()(std::string* out, int type) const {
        absl::StrAppend(out, DebugOptions::CollectiveOpType_Name(type));
      }
    };
    return absl::StrJoin(collective_ops, ", ", Formatter());
  };

  // Custom parser for zkx_gpu_disable_async_collectives.
  auto setter_for_zkx_gpu_disable_async_collectives =
      [debug_options](std::string_view input) {
        std::vector<std::string_view> values = absl::StrSplit(input, ',');
        std::vector<DebugOptions::CollectiveOpType> parsed_ops;
        parsed_ops.reserve(values.size());
        bool has_all_collectives = false;

        for (const std::string_view value : values) {
          DebugOptions::CollectiveOpType op_type;
          if (!DebugOptions::CollectiveOpType_Parse(
                  absl::AsciiStrToUpper(value), &op_type)) {
            // Return an error if flag value was not recognized.
            return false;
          }
          if (op_type == DebugOptions::ALLCOLLECTIVES) {
            has_all_collectives = true;
          }
          parsed_ops.push_back(op_type);
        }

        debug_options->clear_zkx_gpu_disable_async_collectives();
        if (has_all_collectives) {
          for (int i = static_cast<int>(DebugOptions::ALLREDUCE);
               i < static_cast<int>(DebugOptions::ALLCOLLECTIVES); ++i) {
            debug_options->add_zkx_gpu_disable_async_collectives(
                static_cast<DebugOptions::CollectiveOpType>(i));
          }
        } else {
          for (DebugOptions::CollectiveOpType op_type : parsed_ops) {
            debug_options->add_zkx_gpu_disable_async_collectives(op_type);
          }
        }
        return true;
      };

  // Custom "sub-parser" lambda for zkx_gpu_pgle_accuracy_checker.
  auto setter_for_zkx_gpu_pgle_accuracy_checker =
      [debug_options](const std::string& value) {
        DebugOptions::PGLEStrictnessLevel strictness_level;
        if (!DebugOptions::PGLEStrictnessLevel_Parse(value,
                                                     &strictness_level)) {
          return false;
        }
        debug_options->set_zkx_gpu_pgle_accuracy_checker(strictness_level);
        return true;
      };

  // Don't use an initializer list for initializing the vector; this would
  // create a temporary copy, and exceeds the stack space when compiling with
  // certain configurations.
  flag_list->push_back(tsl::Flag(
      "zkx_llvm_enable_invariant_load_metadata",
      bool_setter_for(
          &DebugOptions::set_zkx_llvm_enable_invariant_load_metadata),
      debug_options->zkx_llvm_enable_invariant_load_metadata(),
      "In LLVM-based backends, enable the emission of !invariant.load metadata "
      "in the generated IR."));
  flag_list->push_back(tsl::Flag(
      "zkx_llvm_disable_expensive_passes",
      bool_setter_for(&DebugOptions::set_zkx_llvm_disable_expensive_passes),
      debug_options->zkx_llvm_disable_expensive_passes(),
      "In LLVM-based backends, disable a custom set of expensive optimization "
      "passes."));
  flag_list->push_back(tsl::Flag(
      "zkx_backend_optimization_level",
      int32_setter_for(&DebugOptions::set_zkx_backend_optimization_level),
      debug_options->zkx_backend_optimization_level(),
      "Numerical optimization level for the ZKX compiler backend."));
  flag_list->push_back(tsl::Flag(
      "zkx_disable_hlo_passes", setter_for_zkx_disable_hlo_passes, "",
      "Comma-separated list of hlo passes to be disabled. These names must "
      "exactly match the passes' names; no whitespace around commas."));
  flag_list->push_back(tsl::Flag(
      "zkx_enable_hlo_passes_only", setter_for_zkx_enable_hlo_passes_only, "",
      "Comma-separated list of hlo passes to be enabled. These names must "
      "exactly match the passes' names; no whitespace around commas. The "
      "unspecified passes are all disabled."));
  flag_list->push_back(tsl::Flag(
      "zkx_disable_all_hlo_passes",
      bool_setter_for(&DebugOptions::set_zkx_disable_all_hlo_passes), false,
      "Disables all HLO passes. Notes that some passes are necessary for "
      "correctness and the invariants that must be satisfied by 'fully "
      "optimized' HLO are different for different devices and may change "
      "over time. The only 'guarantee', such as it is, is that if you compile "
      "ZKX and dump the optimized HLO for some graph, you should be able to "
      "run it again on the same device with the same build of ZKX."));
  flag_list->push_back(tsl::Flag(
      "zkx_unsupported_crash_on_hlo_pass_fix_max_iterations",
      bool_setter_for(
          &DebugOptions::
              set_zkx_unsupported_crash_on_hlo_pass_fix_max_iterations),
      debug_options->zkx_unsupported_crash_on_hlo_pass_fix_max_iterations(),
      "Crash if HloPassFix can not converge after a fixed number of "
      "iterations."));
  flag_list->push_back(
      tsl::Flag("zkx_embed_ir_in_executable",
                bool_setter_for(&DebugOptions::set_zkx_embed_ir_in_executable),
                debug_options->zkx_embed_ir_in_executable(),
                "Embed the compiler IR as a string in the executable."));
  flag_list->push_back(tsl::Flag(
      "zkx_gpu_cuda_data_dir", debug_options->mutable_zkx_gpu_cuda_data_dir(),
      "If non-empty, specifies a local directory containing ptxas and nvvm "
      "libdevice files; otherwise we use those from runfile directories."));
  flag_list->push_back(tsl::Flag(
      "zkx_gpu_ptx_file", setter_for_zkx_gpu_ptx_file, "",
      "If non-empty, specifies a file containing ptx to use. The filename "
      "prefix must have the same pattern as PTX dumped by ZKX. This allows to "
      "match one specific module. General workflow. Get the generated module "
      "ptx from ZKX, modify it, then pass it back via this option."));
  flag_list->push_back(tsl::Flag(
      "zkx_gpu_llvm_ir_file", setter_for_zkx_gpu_llvm_ir_file, "",
      "If non-empty, specifies a file containing textual LLVM IR to use. The "
      "filename prefix must have the same pattern as LLVM dumped by ZKX "
      "(i.e. module_0001.ir-no-opt.ll -> module_0001.MY_NEW_FILE.ll). This "
      "allows to match one specific module. General workflow. Get the not "
      "optimized LLVM IR from ZKX, modify it, then pass it back via this "
      "option."));
  flag_list->push_back(tsl::Flag(
      "zkx_hlo_profile", bool_setter_for(&DebugOptions::set_zkx_hlo_profile),
      debug_options->zkx_hlo_profile(),
      "Instrument the computation to collect per-HLO cycle counts"));
  flag_list->push_back(tsl::Flag(
      "zkx_backend_extra_options", setter_for_zkx_backend_extra_options, "",
      "Extra options to pass to a backend; comma-separated list of 'key=val' "
      "strings (=val may be omitted); no whitespace around commas."));
  flag_list->push_back(tsl::Flag(
      "zkx_cpu_parallel_codegen_split_count",
      int32_setter_for(&DebugOptions::set_zkx_cpu_parallel_codegen_split_count),
      debug_options->zkx_cpu_parallel_codegen_split_count(),
      "Split LLVM module into at most this many parts before codegen to enable "
      "parallel compilation for the CPU backend."));
  flag_list->push_back(tsl::Flag(
      "zkx_cpu_copy_insertion_use_region_analysis",
      bool_setter_for(
          &DebugOptions::set_zkx_cpu_copy_insertion_use_region_analysis),
      debug_options->zkx_cpu_copy_insertion_use_region_analysis(),
      "Use region based analysis in copy insertion pass."));
  flag_list->push_back(tsl::Flag(
      "zkx_cpu_enable_concurrency_optimized_scheduler",
      bool_setter_for(
          &DebugOptions::set_zkx_cpu_enable_concurrency_optimized_scheduler),
      debug_options->zkx_cpu_enable_concurrency_optimized_scheduler(),
      "Use HLO module scheduler that is optimized for extracting concurrency "
      "from an HLO module by trading off extra memory pressure."));
  flag_list->push_back(tsl::Flag(
      "zkx_cpu_prefer_vector_width",
      int32_setter_for(&DebugOptions::set_zkx_cpu_prefer_vector_width),
      debug_options->zkx_cpu_prefer_vector_width(),
      "Preferred vector width for the ZKX:CPU LLVM backend."));
  flag_list->push_back(tsl::Flag(
      "zkx_cpu_max_isa",
      uppercase_string_setter_for(&DebugOptions::set_zkx_cpu_max_isa),
      debug_options->zkx_cpu_max_isa(),
      "Maximum ISA that ZKX:CPU LLVM backend will codegen, i.e., it will not "
      "use newer instructions. Available values: SSE4_2, AVX, AVX2, AVX512, "
      "AVX512_VNNI, AVX512_BF16, AMX, and AMX_FP16. (`AMX` will enable both "
      "`AMX_BF16` and `AMX_INT8` instructions.)"));
  flag_list->push_back(tsl::Flag(
      "zkx_force_host_platform_device_count",
      int32_setter_for(&DebugOptions::set_zkx_force_host_platform_device_count),
      debug_options->zkx_force_host_platform_device_count(),
      "Force the host platform to pretend that there are these many host "
      "\"devices\". All of these host devices are backed by the same "
      "threadpool. Setting this to anything other than 1 can increase overhead "
      "from context switching but we let the user override this behavior to "
      "help run tests on the host that run models in parallel across multiple "
      "devices."));
  flag_list->push_back(tsl::Flag(
      "zkx_gpu_disable_gpuasm_optimizations",
      bool_setter_for(&DebugOptions::set_zkx_gpu_disable_gpuasm_optimizations),
      debug_options->zkx_gpu_disable_gpuasm_optimizations(),
      "In ZKX:GPU run ptxas in -O0 (default is -O3)."));
  flag_list->push_back(
      tsl::Flag("zkx_gpu_generate_debug_info",
                bool_setter_for(&DebugOptions::set_zkx_gpu_generate_debug_info),
                debug_options->zkx_gpu_generate_debug_info(),
                "Generate debug info for codegened CUDA kernels."));
  flag_list->push_back(
      tsl::Flag("zkx_gpu_generate_line_info",
                bool_setter_for(&DebugOptions::set_zkx_gpu_generate_line_info),
                debug_options->zkx_gpu_generate_line_info(),
                "Generate line info for codegened CUDA kernels."));
  flag_list->push_back(tsl::Flag(
      "zkx_fuel", setter_for_zkx_fuel, /*default_value_for_display=*/"",
      "Sets compiler fuel, useful for bisecting bugs in passes. Format "
      "--zkx_fuel=PASS1=NUM1,PASS2=NUM2,..."));
  flag_list->push_back(tsl::Flag(
      "zkx_dump_to", string_setter_for(&DebugOptions::set_zkx_dump_to),
      debug_options->zkx_dump_to(),
      "Directory into which debugging data is written. If not specified but "
      "another dumping flag is passed, data will be written to stdout. To "
      "explicitly write to stdout, set this to \"-\". The values \"sponge\" "
      "and \"test_undeclared_outputs_dir\" have a special meaning: They cause "
      "us to dump into the directory specified by the environment variable "
      "TEST_UNDECLARED_OUTPUTS_DIR."));
  flag_list->push_back(tsl::Flag(
      "zkx_flags_reset", bool_setter_for(&DebugOptions::set_zkx_flags_reset),
      debug_options->zkx_flags_reset(),
      "Whether to reset ZKX_FLAGS next time to parse."));
  flag_list->push_back(tsl::Flag(
      "zkx_annotate_with_emitter_loc",
      bool_setter_for(&DebugOptions::set_zkx_annotate_with_emitter_loc),
      debug_options->zkx_annotate_with_emitter_loc(),
      "Forces emitters that use MLIR to annotate all the created MLIR "
      "instructions with the emitter's C++ source file and line number. The "
      "annotations should appear in the MLIR dumps. The emitters should use "
      "EmitterLocOpBuilder for that."));
  flag_list->push_back(tsl::Flag(
      "zkx_dump_hlo_as_text",
      bool_setter_for(&DebugOptions::set_zkx_dump_hlo_as_text),
      debug_options->zkx_dump_hlo_as_text(),
      "Dumps HLO modules as text before and after optimizations. debug_options "
      "are written to the --zkx_dump_to dir, or, if no dir is specified, to "
      "stdout."));
  flag_list->push_back(tsl::Flag(
      "zkx_dump_hlo_as_long_text",
      bool_setter_for(&DebugOptions::set_zkx_dump_hlo_as_long_text),
      debug_options->zkx_dump_hlo_as_long_text(),
      "Dumps HLO modules as long text before and after optimizations. "
      "debug_options are written to the --zkx_dump_to dir, or, if no dir is "
      "specified, to stdout. Ignored unless zkx_dump_hlo_as_text is true."));
  flag_list->push_back(
      tsl::Flag("zkx_dump_large_constants",
                bool_setter_for(&DebugOptions::set_zkx_dump_large_constants),
                debug_options->zkx_dump_large_constants(),
                "Dumps HLO modules including large constants before and after "
                "optimizations. debug_options are written to the --zkx_dump_to "
                "dir, or, if no dir is specified, to stdout. Ignored unless "
                "zkx_dump_hlo_as_text is true."));
  flag_list->push_back(
      tsl::Flag("zkx_dump_hlo_as_proto",
                bool_setter_for(&DebugOptions::set_zkx_dump_hlo_as_proto),
                debug_options->zkx_dump_hlo_as_proto(),
                "Dumps HLO modules as HloProtos to the directory specified by "
                "--zkx_dump_to."));
  flag_list->push_back(
      tsl::Flag("zkx_gpu_experimental_dump_fdo_profiles",
                bool_setter_for(
                    &DebugOptions::set_zkx_gpu_experimental_dump_fdo_profiles),
                debug_options->zkx_gpu_experimental_dump_fdo_profiles(),
                "Dumps FDO profiles as text to the directory specified "
                "by --zkx_dump_to."));
  flag_list->push_back(
      tsl::Flag("zkx_dump_hlo_as_dot",
                bool_setter_for(&DebugOptions::set_zkx_dump_hlo_as_dot),
                debug_options->zkx_dump_hlo_as_dot(),
                "Dumps HLO modules rendered as dot files to the "
                "directory specified by --zkx_dump_to."));
  flag_list->push_back(
      tsl::Flag("zkx_dump_hlo_as_html",
                bool_setter_for(&DebugOptions::set_zkx_dump_hlo_as_html),
                debug_options->zkx_dump_hlo_as_html(),
                "Dumps HLO modules rendered as HTML files to the "
                "directory specified by --zkx_dump_to."));
  flag_list->push_back(tsl::Flag(
      "zkx_dump_hlo_as_url",
      bool_setter_for(&DebugOptions::set_zkx_dump_hlo_as_url),
      debug_options->zkx_dump_hlo_as_url(),
      "Tries to dump HLO modules rendered as URLs to stdout (and also to the "
      "directory specified by --zkx_dump_to). This is not implemented by "
      "default; you need to add a plugin which calls "
      "RegisterGraphToURLRenderer()."));
  flag_list->push_back(tsl::Flag(
      "zkx_dump_fusion_visualization",
      bool_setter_for(&DebugOptions::set_zkx_dump_fusion_visualization),
      debug_options->zkx_dump_fusion_visualization(),
      "Tries to generate HLO fusion visualization as an HTML page to the "
      "directory specified by --zkx_dump_to. This is not implemented by "
      "default; you need to add a plugin which calls "
      "RegisterGraphToURLRenderer(). Generates a file per computation. "
      "Currently only implemented for the GPU backend."));
  flag_list->push_back(tsl::Flag(
      "zkx_dump_hlo_snapshots",
      bool_setter_for(&DebugOptions::set_zkx_dump_hlo_snapshots),
      debug_options->zkx_dump_hlo_snapshots(),
      "Every time an HLO module is run, dumps an HloSnapshot to the directory "
      "specified by --zkx_dump_to."));
  flag_list->push_back(tsl::Flag(
      "zkx_dump_hlo_module_re",
      string_setter_for(&DebugOptions::set_zkx_dump_hlo_module_re),
      debug_options->zkx_dump_hlo_module_re(),
      "Limits dumping only to modules which match this regular expression. "
      "Default is to dump all modules."));
  flag_list->push_back(tsl::Flag(
      "zkx_dump_hlo_pass_re",
      string_setter_for(&DebugOptions::set_zkx_dump_hlo_pass_re),
      debug_options->zkx_dump_hlo_pass_re(),
      "If specified, dumps HLO before and after optimization passes which "
      "match this regular expression, in addition to dumping at the very "
      "beginning and end of compilation."));
  flag_list->push_back(
      tsl::Flag("zkx_dump_include_timestamp",
                bool_setter_for(&DebugOptions::set_zkx_dump_include_timestamp),
                debug_options->zkx_dump_include_timestamp(),
                "If specified, includes a timestamp in the dumped filenames."));
  flag_list->push_back(
      tsl::Flag("zkx_dump_max_hlo_modules",
                int32_setter_for(&DebugOptions::set_zkx_dump_max_hlo_modules),
                debug_options->zkx_dump_max_hlo_modules(),
                "Max number of hlo module dumps in a directory. Set to < 0 for "
                "unbounded."));
  flag_list->push_back(tsl::Flag(
      "zkx_dump_module_metadata",
      bool_setter_for(&DebugOptions::set_zkx_dump_module_metadata),
      debug_options->zkx_dump_module_metadata(),
      "Dumps HloModuleMetadata as text protos to the directory specified "
      "by --zkx_dump_to."));
  flag_list->push_back(
      tsl::Flag("zkx_dump_compress_protos",
                bool_setter_for(&DebugOptions::set_zkx_dump_compress_protos),
                debug_options->zkx_dump_compress_protos(),
                "Gzip-compress protos dumped by --zkx_dump_hlo_as_proto."));
  flag_list->push_back(tsl::Flag(
      "zkx_gpu_unsafe_fallback_to_driver_on_ptxas_not_found",
      bool_setter_for(
          &DebugOptions::
              set_zkx_gpu_unsafe_fallback_to_driver_on_ptxas_not_found),
      debug_options->zkx_gpu_unsafe_fallback_to_driver_on_ptxas_not_found(),
      "If true, ZKX GPU falls back to the driver if ptxas is not found. Note "
      "that falling back to the driver can have drawbacks like using more "
      "memory and/or other bugs during compilation, so we recommend setting "
      "this flag to false."));
  flag_list->push_back(tsl::Flag(
      "zkx_multiheap_size_constraint_per_heap",
      int32_setter_for(
          &DebugOptions::set_zkx_multiheap_size_constraint_per_heap),
      debug_options->zkx_multiheap_size_constraint_per_heap(),
      "Generates multiple heaps (i.e., temp buffers) with a size "
      "constraint on each heap to avoid Out-of-Memory due to memory "
      "fragmentation. The constraint is soft, so it works with tensors "
      "larger than the given constraint size. -1 corresponds to no "
      "constraints."));
  flag_list->push_back(tsl::Flag(
      "zkx_gpu_force_compilation_parallelism",
      int32_setter_for(
          &DebugOptions::set_zkx_gpu_force_compilation_parallelism),
      debug_options->zkx_gpu_force_compilation_parallelism(),
      "Overrides normal multi-threaded compilation setting to use this many "
      "threads. Setting to 0 (the default value) means no enforcement."));
  flag_list->push_back(tsl::Flag(
      "zkx_gpu_enable_llvm_module_compilation_parallelism",
      bool_setter_for(
          &DebugOptions::
              set_zkx_gpu_enable_llvm_module_compilation_parallelism),
      debug_options->zkx_gpu_enable_llvm_module_compilation_parallelism(),
      "Decides whether we can do LLVM module compilation in a parallelised "
      "way. If set to false, then it will be single threaded, otherwise the "
      "number of threads depends on the "
      "--zkx_gpu_force_compilation_parallelism flag and the thread pool "
      "supplied to GpuCompiler."));

  flag_list->push_back(
      tsl::Flag("zkx_gpu_deterministic_ops",
                bool_setter_for(&DebugOptions::set_zkx_gpu_deterministic_ops),
                debug_options->zkx_gpu_deterministic_ops(),
                "Guarantees run-to-run determinism on GPU."));
  flag_list->push_back(tsl::Flag(
      "zkx_gpu_exclude_nondeterministic_ops",
      bool_setter_for(&DebugOptions::set_zkx_gpu_exclude_nondeterministic_ops),
      debug_options->zkx_gpu_exclude_nondeterministic_ops(),
      "Excludes non-deterministic ops from compiled executables."));
  flag_list->push_back(tsl::Flag(
      "zkx_gpu_disable_async_collectives",
      setter_for_zkx_gpu_disable_async_collectives,
      collective_op_types_to_string(
          debug_options->zkx_gpu_disable_async_collectives()),
      "This disables a certain set of async collectives and turn them into "
      "synchronous ones. By default, this is empty which indicates enabling "
      "async execution for all collectives. A sample usage is: "
      "--zkx_gpu_disable_async_collectives=ALLREDUCE,REDUCESCATTER"));
  flag_list->push_back(tsl::Flag(
      "zkx_gpu_enable_approx_costly_collectives",
      bool_setter_for(
          &DebugOptions::set_zkx_gpu_enable_approx_costly_collectives),
      debug_options->zkx_gpu_enable_approx_costly_collectives(),
      "Enables more accurate latency approximation of collectives. Used in "
      "`ApproximateLatencyEstimator` scheduler."));

  flag_list->push_back(tsl::Flag(
      "zkx_gpu_filter_kernels_spilling_registers_on_autotuning",
      bool_setter_for(
          &DebugOptions::
              set_zkx_gpu_filter_kernels_spilling_registers_on_autotuning),
      debug_options->zkx_gpu_filter_kernels_spilling_registers_on_autotuning(),
      "Filter out kernels that spill registers during autotuning"));
  flag_list->push_back(tsl::Flag(
      "zkx_gpu_fail_ptx_compilation_on_register_spilling",
      bool_setter_for(
          &DebugOptions::set_zkx_gpu_fail_ptx_compilation_on_register_spilling),
      debug_options->zkx_gpu_fail_ptx_compilation_on_register_spilling(),
      "Fails the PTX compilation if a kernel spills registers."));
  flag_list->push_back(tsl::Flag(
      "zkx_debug_buffer_assignment_show_max",
      int64_setter_for(&DebugOptions::set_zkx_debug_buffer_assignment_show_max),
      debug_options->zkx_debug_buffer_assignment_show_max(),
      "Number of buffers to display when debugging the buffer assignment"));
  flag_list->push_back(tsl::Flag(
      "zkx_gpu_target_config_filename",
      string_setter_for(&DebugOptions::set_zkx_gpu_target_config_filename),
      debug_options->zkx_gpu_target_config_filename(),
      "Filename for GPU TargetConfig. Triggers devicless compilation: attached "
      "device is ignored, and the proto is queried instead"));

  flag_list->push_back(tsl::Flag(
      "zkx_gpu_enable_libnvptxcompiler",
      [debug_options](bool enabled) {
        if (enabled && !se::IsLibNvPtxCompilerSupported()) {
          // This feature can't be enabled when ZKX was built without
          // libnvptxcompiler support.
          return false;
        }
        debug_options->set_zkx_gpu_enable_libnvptxcompiler(enabled);
        return true;
      },
      debug_options->zkx_gpu_enable_libnvptxcompiler(),
      "Use libnvptxcompiler for PTX-to-GPU-assembly compilation instead of "
      "calling ptxas."));
  flag_list->push_back(tsl::Flag(
      "zkx_gpu_enable_libnvjitlink",
      [debug_options](bool enabled) {
        debug_options->set_zkx_gpu_libnvjitlink_mode(
            enabled ? DebugOptions::LIB_NV_JIT_LINK_MODE_ENABLED
                    : DebugOptions::LIB_NV_JIT_LINK_MODE_DISABLED);
        return true;
      },
      se::IsLibNvJitLinkSupported(),
      "Use libnvjitlink for PTX-to-GPU-assembly compilation instead of "
      "calling ptxas."));
  flag_list->push_back(tsl::Flag(
      "zkx_gpu_nccl_collective_max_nchannels",
      int64_setter_for(
          &DebugOptions::set_zkx_gpu_nccl_collective_max_nchannels),
      debug_options->zkx_gpu_nccl_collective_max_nchannels(),
      "Specify the maximum number of channels(SMs) NCCL will use "
      "for collective operations. Default is 0 which is to let NCCL decide."));
  flag_list->push_back(tsl::Flag(
      "zkx_gpu_nccl_p2p_max_nchannels",
      int64_setter_for(&DebugOptions::set_zkx_gpu_nccl_p2p_max_nchannels),
      debug_options->zkx_gpu_nccl_p2p_max_nchannels(),
      "Specify the maximum number of channels(SMs) NCCL will use "
      "for p2p operations. Default is 0 which is to let NCCL decide."));
  flag_list->push_back(tsl::Flag(
      "zkx_gpu_pgle_accuracy_checker", setter_for_zkx_gpu_pgle_accuracy_checker,
      DebugOptions::PGLEStrictnessLevel_Name(
          debug_options->zkx_gpu_pgle_accuracy_checker()),
      "If an FDO profile is specified and latency hiding scheduler encounters "
      "missing instructions in the profile, then the compilation will halt "
      "(ERROR), or a warning will be emitted (WARN), or the checker is "
      "disabled (OFF)"));

  flag_list->push_back(tsl::Flag(
      "zkx_gpu_executable_warn_stuck_timeout",
      int32_setter_for(
          &DebugOptions::set_zkx_gpu_executable_warn_stuck_timeout_seconds),
      debug_options->zkx_gpu_executable_warn_stuck_timeout_seconds(),
      "Set timeout for Rendezvous stuck warning"));
  flag_list->push_back(tsl::Flag(
      "zkx_gpu_executable_terminate_timeout",
      int32_setter_for(
          &DebugOptions::set_zkx_gpu_executable_terminate_timeout_seconds),
      debug_options->zkx_gpu_executable_terminate_timeout_seconds(),
      "Set timeout for Rendezvous termination"));
  flag_list->push_back(
      tsl::Flag("zkx_gpu_experimental_stream_annotation",
                bool_setter_for(
                    &DebugOptions::set_zkx_gpu_experimental_stream_annotation),
                debug_options->zkx_gpu_experimental_stream_annotation(),
                "Enable the experimental explicit stream annotation support. "
                "If false, the annotations are ignored."));
  flag_list->push_back(tsl::Flag(
      "zkx_gpu_experimental_parallel_collective_overlap_limit",
      int32_setter_for(
          &DebugOptions::
              set_zkx_gpu_experimental_parallel_collective_overlap_limit),
      debug_options->zkx_gpu_experimental_parallel_collective_overlap_limit(),
      "This controls how many in-flight collectives latency hiding scheduler "
      "can schedule."));
  flag_list->push_back(tsl::Flag(
      "zkx_pjrt_allow_auto_layout_in_hlo",
      bool_setter_for(&DebugOptions::set_zkx_pjrt_allow_auto_layout_in_hlo),
      debug_options->zkx_pjrt_allow_auto_layout_in_hlo(),
      "Experimental: Make unset entry computation layout mean auto layout "
      "instead of default layout in HLO when run through PjRT. In other cases "
      "(StableHLO or non-PjRT) the auto layout is already used."));

  flag_list->push_back(tsl::Flag(
      "zkx_llvm_force_inline_before_split",
      bool_setter_for(&DebugOptions::set_zkx_llvm_force_inline_before_split),
      debug_options->zkx_llvm_force_inline_before_split(),
      "Decide whether to force inline before llvm module split to get a more "
      "balanced splits for parallel compilation"));

  flag_list->push_back(
      tsl::Flag("zkx_gpu_dump_llvmir",
                bool_setter_for(&DebugOptions::set_zkx_gpu_dump_llvmir),
                debug_options->zkx_gpu_dump_llvmir(), "Dump LLVM IR."));
  flag_list->push_back(tsl::Flag(
      "zkx_gpu_dump_hlo_unoptimized_snapshots",
      bool_setter_for(
          &DebugOptions::set_zkx_gpu_dump_hlo_unoptimized_snapshots),
      debug_options->zkx_gpu_dump_hlo_unoptimized_snapshots(),
      "Every time an HLO module is run, dumps an HloUnoptimizedSnapshot to the "
      "directory specified by --zkx_dump_to."));
  flag_list->push_back(tsl::Flag(
      "zkx_gpu_collectives_use_persistent_cliques",
      bool_setter_for(
          &DebugOptions::set_zkx_gpu_collectives_use_persistent_cliques),
      debug_options->zkx_gpu_collectives_use_persistent_cliques(),
      "Use persistent per-process ZKX:GPU collectives cliques"));
  flag_list->push_back(
      tsl::Flag("zkx_dump_disable_metadata",
                bool_setter_for(&DebugOptions::set_zkx_dump_disable_metadata),
                debug_options->zkx_dump_disable_metadata(),
                "Disable dumping HLO metadata in HLO dumps."));
  flag_list->push_back(tsl::Flag(
      "zkx_dump_hlo_pipeline_re",
      string_setter_for(&DebugOptions::set_zkx_dump_hlo_pipeline_re),
      debug_options->zkx_dump_hlo_pipeline_re(),
      "If specified, dumps HLO before and after optimization passes in the "
      "pass pipelines that match this regular expression."));
  flag_list->push_back(tsl::Flag(
      "zkx_dump_enable_mlir_pretty_form",
      bool_setter_for(&DebugOptions::set_zkx_dump_enable_mlir_pretty_form),
      debug_options->zkx_dump_enable_mlir_pretty_form(),
      "Enable dumping MLIR using pretty print form. If set to false, the "
      "dumped MLIR will be in the llvm-parsable format and can be processed by "
      "mlir-opt tools. Pretty print form is not legal MLIR."));
  flag_list->push_back(tsl::Flag(
      "zkx_gpu_nccl_termination_timeout_seconds",
      int64_setter_for(
          &DebugOptions::set_zkx_gpu_nccl_termination_timeout_seconds),
      debug_options->zkx_gpu_nccl_termination_timeout_seconds(),
      "Timeout in seconds before terminating jobs stuck in NCCL Rendezvous."));
  flag_list->push_back(tsl::Flag(
      "zkx_gpu_enable_shared_constants",
      bool_setter_for(&DebugOptions::set_zkx_gpu_enable_shared_constants),
      debug_options->zkx_gpu_enable_shared_constants(),
      "Enable constant sharing between GPU executables"));
  flag_list->push_back(tsl::Flag(
      "zkx_gpu_enable_nccl_user_buffers",
      bool_setter_for(&DebugOptions::set_zkx_gpu_enable_nccl_user_buffers),
      debug_options->zkx_gpu_enable_nccl_user_buffers(),
      "Enables NCCL User Buffer Registration. collective_memory_size in the "
      "allocator config must also be set to a non-zero value that is large "
      "enough to meet peak collective memory usage."));
  flag_list->push_back(tsl::Flag(
      "zkx_gpu_temp_buffer_use_separate_color",
      bool_setter_for(
          &DebugOptions::set_zkx_gpu_temp_buffer_use_separate_color),
      debug_options->zkx_gpu_temp_buffer_use_separate_color(),
      "Enables temp User Buffer Registration. Enabling this flag will cause a "
      "separate cuda async memory allocator to allocate a temp buffer to the "
      "fixed address on every iteration"));

  flag_list->push_back(tsl::Flag(
      "zkx_gpu_require_exclusive_lock",
      bool_setter_for(&DebugOptions::set_zkx_gpu_require_exclusive_lock),
      debug_options->zkx_gpu_require_exclusive_lock(),
      "if true, running gpu executable will require exclusive lock on gpu, so "
      "there is no multi thread conflicts on gpu. this can enable some "
      "optimizations that reduce the cost of resource management, e.g, "
      "command buffer update to ensure correctness when running in multi "
      "thread mode."));

  flag_list->push_back(tsl::Flag(
      "zkx_gpu_enable_nccl_comm_splitting",
      bool_setter_for(&DebugOptions::set_zkx_gpu_enable_nccl_comm_splitting),
      debug_options->zkx_gpu_enable_nccl_comm_splitting(),
      "Enables NCCL communicator splitting which allows sharing NCCL resources "
      "between different NCCL cliques."));
  flag_list->push_back(tsl::Flag(
      "zkx_gpu_nccl_init_max_rank_per_root_ratio",
      int64_setter_for(
          &DebugOptions::set_zkx_gpu_nccl_init_max_rank_per_root_ratio),
      debug_options->zkx_gpu_nccl_init_max_rank_per_root_ratio(),
      "Maximum number of ranks associated with a root rank to initialize a "
      "NCCL communicator via ncclCommInitRankScalable. "
      "A value of zero will lead to a single root."));
  flag_list->push_back(tsl::Flag(
      "zkx_dump_latency_hiding_schedule",
      bool_setter_for(&DebugOptions::set_zkx_dump_latency_hiding_schedule),
      debug_options->zkx_dump_latency_hiding_schedule(),
      "Dump the schedule from the latency-hiding scheduler."));
  flag_list->push_back(
      tsl::Flag("zkx_gpu_enable_latency_hiding_scheduler",
                bool_setter_for(
                    &DebugOptions::set_zkx_gpu_enable_latency_hiding_scheduler),
                debug_options->zkx_gpu_enable_latency_hiding_scheduler(),
                "Enable latency-hiding scheduler for ZKX:GPU"));
  flag_list->push_back(tsl::Flag(
      "zkx_gpu_pgle_profile_file_or_directory_path",
      string_setter_for(
          &DebugOptions::set_zkx_gpu_pgle_profile_file_or_directory_path),
      debug_options->zkx_gpu_pgle_profile_file_or_directory_path(),
      "Directory or file for PGLE profiles in ZKX:GPU"));
  flag_list->push_back(tsl::Flag(
      "zkx_gpu_memory_limit_slop_factor",
      int32_setter_for(&DebugOptions::set_zkx_gpu_memory_limit_slop_factor),
      debug_options->zkx_gpu_memory_limit_slop_factor(),
      "Slop factor for memory limits in ZKX:GPU. This flag serves as a "
      "multiplier applied to the total available memory, creating a threshold "
      "that guides the Latency Hiding Scheduler (LHS) in balancing memory "
      "reduction and latency hiding optimizations. This factor effectively "
      "establishes a memory limit for compiler passes, determining when the "
      "scheduler should prioritize: "
      "  1. Memory reduction: When memory usage approaches or exceeds the "
      "calculated "
      "     threshold. "
      "  2. Latency hiding: When memory usage is below the threshold, allowing "
      "for "
      "     more aggressive optimizations that may temporarily increase memory "
      "usage "
      "     but improve overall performance. "
      "By adjusting this factor, users can fine-tune the trade-off between "
      "memory efficiency and performance optimizations. The default value is "
      "95."));
  flag_list->push_back(tsl::Flag(
      "zkx_gpu_enable_highest_priority_async_stream",
      bool_setter_for(
          &DebugOptions::set_zkx_gpu_enable_highest_priority_async_stream),
      debug_options->zkx_gpu_enable_highest_priority_async_stream(),
      "Enable async stream to have the highest priority."));
  flag_list->push_back(tsl::Flag(
      "zkx_gpu_enable_pipelined_p2p",
      bool_setter_for(&DebugOptions::set_zkx_gpu_enable_pipelined_p2p),
      debug_options->zkx_gpu_enable_pipelined_p2p(),
      "Enable pipelinling of P2P instructions."));
  flag_list->push_back(tsl::Flag(
      "zkx_gpu_experimental_pipeline_parallelism_opt_level",
      setter_for_zkx_gpu_experimental_pipeline_parallelism_opt_level,
      DebugOptions::PipelineParallelismOptLevel_Name(
          debug_options->zkx_gpu_experimental_pipeline_parallelism_opt_level()),
      "Experimental optimizations for SPMD-based pipeline parallelism on "
      "GPU."));
  flag_list->push_back(tsl::Flag(
      "zkx_gpu_dump_autotune_results_to",
      string_setter_for(&DebugOptions::set_zkx_gpu_dump_autotune_results_to),
      debug_options->zkx_gpu_dump_autotune_results_to(),
      "File to write autotune results to. It will be a binary file unless the "
      "name ends with .txt or .textproto. Warning: The results are written at "
      "every compilation, possibly multiple times per process. This only works "
      "on CUDA. In tests, the TEST_UNDECLARED_OUTPUTS_DIR prefix can be used "
      "to write to their output directory."));
  flag_list->push_back(tsl::Flag(
      "zkx_gpu_dump_autotuned_gemm_fusions",
      bool_setter_for(&DebugOptions::set_zkx_gpu_dump_autotuned_gemm_fusions),
      debug_options->zkx_gpu_dump_autotuned_gemm_fusions(),
      "Dumps autotuned GEMM fusions to the directory specified by "
      "zkx_dump_to or stdout. Each fusion is dumped only once, as an optimized "
      "HLO."));
  flag_list->push_back(tsl::Flag(
      "zkx_gpu_dump_autotune_logs_to",
      string_setter_for(&DebugOptions::set_zkx_gpu_dump_autotune_logs_to),
      debug_options->zkx_gpu_dump_autotune_logs_to(),
      "File to write autotune logs to. It will be a binary file unless the "
      "name ends with .txt or .textproto."));
  flag_list->push_back(tsl::Flag(
      "zkx_gpu_nccl_terminate_on_error",
      bool_setter_for(&DebugOptions::set_zkx_gpu_nccl_terminate_on_error),
      debug_options->zkx_gpu_nccl_terminate_on_error(),
      "If set, then NCCL errors will terminate the process."));
  flag_list->push_back(
      tsl::Flag("zkx_syntax_sugar_async_ops",
                bool_setter_for(&DebugOptions::set_zkx_syntax_sugar_async_ops),
                debug_options->zkx_syntax_sugar_async_ops(),
                "Enable syntax sugar for async ops in HLO dumps."));
  flag_list->push_back(
      tsl::Flag("zkx_gpu_kernel_cache_file",
                string_setter_for(&DebugOptions::set_zkx_gpu_kernel_cache_file),
                debug_options->zkx_gpu_kernel_cache_file(),
                "Path to a file to cache compiled kernels. Cached kernels get "
                "reused in further compilations; not yet cached kernels are "
                "compiled as usual and get appended to the cache file whenever "
                "possible."));
}  // NOLINT(readability/fn_size)

namespace {

// Allocates flag_values and flag_objects; this function must not be called more
// than once - its call done via call_once.
void AllocateFlags(DebugOptions* defaults) {
  if (defaults == nullptr) {
    defaults =
        absl::IgnoreLeak(new DebugOptions(DefaultDebugOptionsIgnoringFlags()));
  }
  flag_values = defaults;
  flag_objects = absl::IgnoreLeak(new std::vector<tsl::Flag>());
  MakeDebugOptionsFlags(flag_objects, flag_values);
  ParseFlagsFromEnvAndDieIfUnknown("ZKX_FLAGS", *flag_objects);
}

}  // namespace

void AppendDebugOptionsFlags(std::vector<tsl::Flag>* flag_list,
                             DebugOptions* debug_options) {
  absl::call_once(flags_init, &AllocateFlags, debug_options);
  flag_list->insert(flag_list->end(), flag_objects->begin(),
                    flag_objects->end());
}

DebugOptions GetDebugOptionsFromFlags() {
  absl::call_once(flags_init, &AllocateFlags, nullptr);
  if (flag_values->zkx_flags_reset()) {
    ParseFlagsFromEnvAndDieIfUnknown("ZKX_FLAGS", *flag_objects,
                                     /*reset_envvar=*/true);
  }
  return *flag_values;
}

void ResetThreadLocalFuel() {
  absl::call_once(flags_init, &AllocateFlags, nullptr);

  thread_fuel = std::make_unique<
      absl::node_hash_map<std::string, std::atomic<int64_t>>>();
  CHECK(initial_fuel != nullptr);
  for (const auto& kv : *initial_fuel) {
    thread_fuel->emplace(kv.first, kv.second);
  }
}

bool ConsumeFuel(std::string_view pass, bool* just_ran_out) {
  absl::call_once(flags_init, &AllocateFlags, nullptr);
  if (just_ran_out != nullptr) {
    *just_ran_out = false;
  }
  auto* fuel_pool = thread_fuel ? thread_fuel.get() : global_fuel;
  if (fuel_pool->empty()) {
    return true;
  }
  auto it = fuel_pool->find(pass);
  if (it == fuel_pool->end()) {
    return true;
  }
  std::atomic<int64_t>& remaining_fuel = it->second;
  std::atomic<bool>& fuel_has_been_consumed = fuel_ever_consumed->at(pass);
  fuel_has_been_consumed = true;

  int64_t remaining = remaining_fuel.fetch_sub(1);
  if (just_ran_out != nullptr) {
    *just_ran_out = remaining == 0;
  }
  return remaining > 0;
}

}  // namespace zkx
