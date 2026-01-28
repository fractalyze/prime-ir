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

#include "zkx/service/cpu/cpu_compiler.h"

#include <functional>
#include <vector>

#include "absl/debugging/leak_check.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/Utils/SplitModule.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

#include "xla/tsl/platform/cpu_info.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/thread_pool.h"
#include "xla/tsl/profiler/lib/traceme.h"
#include "zkx/backends/cpu/codegen/cpu_features.h"
#include "zkx/backends/cpu/codegen/jit_compiler.h"
#include "zkx/backends/cpu/codegen/object_loader.h"
#include "zkx/backends/cpu/runtime/thunk_proto_serdes.h"
#include "zkx/hlo/pass/hlo_pass_pipeline.h"
#include "zkx/hlo/transforms/simplifiers/hlo_dce.h"
#include "zkx/hlo/transforms/simplifiers/hlo_memory_scheduler.h"
#include "zkx/service/buffer_value.h"
#include "zkx/service/copy_insertion.h"
#include "zkx/service/cpu/cpu_instruction_fusion.h"
#include "zkx/service/cpu/cpu_options.h"
#include "zkx/service/cpu/executable.pb.h"
#include "zkx/service/cpu/runtime_symbol_generator.h"
#include "zkx/service/cpu/thunk_emitter.h"
#include "zkx/service/dump.h"
#include "zkx/service/llvm_ir/llvm_command_line_options.h"
#include "zkx/service/scatter_expander.h"
#include "zkx/shape_util.h"
#include "zkx/stream_executor/host/host_platform_id.h"
#include "zkx/util.h"

namespace zkx::cpu {

namespace {

// A module identifier (prefix) for emitted LLVM modules.
constexpr std::string_view kZkxModuleIdentifier = "__compute_module";

// Align buffers to XLA:CPU minimal alignment.
int64_t memory_alignment(LogicalBuffer::Color) {
  return cpu_function_runtime::MinAlign();
}

std::pair<LlvmCompiler::ModuleHook, LlvmCompiler::ModuleHook> GetIRModuleHooks(
    const HloModule& hlo_module,
    const LlvmCompiler::ModuleHook& user_pre_optimization_hook,
    const LlvmCompiler::ModuleHook& user_post_optimization_hook) {
  // Create the IR hooks. If applicable, each IR hook does the following:
  //
  //  * Calls the user supplied module hook.
  //  * Writes out the IR to a file in the output directory designated by
  //    --zkx_dump_to
  const HloModule* hlo_module_ptr = &hlo_module;
  auto hook = [user_pre_optimization_hook, user_post_optimization_hook,
               hlo_module_ptr](bool optimized,
                               const llvm::Module& llvm_module) {
    const auto& user_hook =
        !optimized ? user_pre_optimization_hook : user_post_optimization_hook;
    if (user_hook) {
      user_hook(llvm_module);
    }

    // Include LLVM module identifier suffix in case `llvm_module` is just a
    // part of the original LLVM module constructed by the ZKX.
    std::string_view id = llvm_module.getModuleIdentifier();
    size_t pos = std::min(id.size(), 1 + kZkxModuleIdentifier.size());
    DumpIrIfEnabled(*hlo_module_ptr, llvm_module, optimized,
                    /*filename_suffix=*/id.substr(pos));
  };
  return {[hook](const llvm::Module& llvm_module) {
            return hook(/*optimized=*/false, llvm_module);
          },
          [hook](const llvm::Module& llvm_module) {
            return hook(/*optimized=*/true, llvm_module);
          }};
}

absl::Status VerifyLlvmModule(const llvm::Module& llvm_module) {
  // TODO(chokobole): Uncomment this. Dependency: XLA_SCOPED_LOGGING_TIMER
  // XLA_SCOPED_LOGGING_TIMER("CpuCompiler - Running LLVM verifier");

  std::string err;
  llvm::raw_string_ostream err_stream(err);

  // verifyModule() returns true if the module is broken.
  TF_RET_CHECK(!llvm::verifyModule(llvm_module, &err_stream))
      << "Invalid LLVM IR before optimizations:\n"
      << err_stream.str()
      << "\nThis probably indicates a bug in the HLO -> LLVM IR lowering. "
         "Rerun with --zkx_dump_to to get the IR. ";
  return absl::OkStatus();
}

// Returns a global (per-process) thread pool for ZKX CPU compilation tasks.
tsl::thread::ThreadPool* GetCompilationThreadPool() {
  // LLVM compilation has a lot of memory-bound pointer chasing and not
  // so much CPU-bound work. Based on profiling a few examples, 32 threads seems
  // to be enough to achieve maximum parallel compilation speedup.
  static constexpr int kMaxCompilationThreads = 32;
  static auto* thread_pool = absl::IgnoreLeak(new tsl::thread::ThreadPool(
      tsl::Env::Default(), "zkx-cpu-llvm-codegen",
      std::min(kMaxCompilationThreads, tsl::port::MaxParallelism())));
  return thread_pool;
}

// Returns task runner that uses the global compilation thread pool.
JitCompiler::TaskRunner GetCompilationTaskRunner() {
  return [](JitCompiler::Task task) {
    GetCompilationThreadPool()->Schedule(std::move(task));
  };
}

// If LLVM module has large constants constructed from literals, we don't want
// to split it, because it will cause us to copy large constants across module
// parts. We should not be storing large constants in LLVM IR in a first place,
// but while we do that, we have to be extra-careful, or it leads to extremely
// long compilation times, OOMs and timeouts.
//
// TODO(b/361800465): Figure out how to avoid putting large constants into
// LLVM IR in the first place.
bool HasLargeConstants(llvm::Module& module) {
  static constexpr int kMaxConstantSize = 10000;
  for (llvm::GlobalVariable& g : module.globals()) {
    if (!g.hasInitializer()) {
      continue;
    }

    llvm::Constant* initializer = g.getInitializer();
    if (auto* arr = llvm::dyn_cast<llvm::ArrayType>(initializer->getType())) {
      if (arr->getNumElements() > kMaxConstantSize) return true;
    }
  }
  return false;
}

inline void VlogMaxIsa(std::string_view max_cpu_isa) {
  if (VLOG_IS_ON(1) && !max_cpu_isa.empty()) {
    if (tsl::port::IsX86CPU()) {
      VLOG(1) << "`zkx_cpu_max_isa` is set. Will not use features newer than: "
              << max_cpu_isa;
    } else {
      VLOG(1) << "`zkx_cpu_max_isa` is set to `" << max_cpu_isa
              << "`. This flag is not supported on non-x86 CPUs yet.";
    }
  }
}

// We keep HloProto in the CpuExecutable, but we don't need to keep literals
// payload in it as we use it only for debugging and memory analysis.
void StripPayloadFromLiteralProto(HloProto& proto) {
  auto* module = proto.mutable_hlo_module();
  for (auto& computation : *module->mutable_computations()) {
    for (auto& instruction : *computation.mutable_instructions()) {
      // We only keep literal shape to correctly estimate memory usage of the
      // HLO module, but we don't need the actual literal data.
      if (instruction.has_literal()) {
        LiteralProto literal;
        *literal.mutable_shape() = instruction.literal().shape();
        *instruction.mutable_literal() = std::move(literal);
      }
    }
  }
}

// Post-compilation callback functor for use by SimpleOrcJIT.
//
// Dumps machine code if dumping is enabled for the module.
std::function<void(const llvm::Module&, const llvm::object::ObjectFile&)>
CreateOrcJITPostCompilationHook(const HloModule* hlo_module,
                                std::vector<std::string>* obj_files) {
  return [=](const llvm::Module& llvm_module,
             const llvm::object::ObjectFile& obj_file) {
    if (obj_files) obj_files->push_back(obj_file.getData().str());

    if (DumpingEnabledForHloModule(*hlo_module)) {
      std::string_view id = llvm_module.getModuleIdentifier();
      size_t pos = std::min(id.size(), 1 + kZkxModuleIdentifier.size());
      DumpToFileInDir(
          *hlo_module, /*file_prefix=*/"",
          /*file_suffix=*/absl::StrCat("obj-file.", id.substr(pos), ".o"),
          std::string_view(obj_file.getData().data(),
                           obj_file.getData().size()));
    }
  };
}

absl::StatusOr<CpuExecutable::ConstantAllocation> LiteralToConstantAllocation(
    BufferAllocation::Index index, const Literal& literal) {
  // TODO(ezhulenev): This code is almost identical to code in ZKX:GPU, we
  // should standardize it. See `zkx/service/gpu/ir_emission_utils.cc`.
  PrimitiveType element_type = literal.shape().element_type();
  if (!primitive_util::IsArrayType(element_type)) {
    return absl::InternalError(
        "Only array literals can be converted to constant allocations");
  }

  int64_t size_bytes = literal.size_bytes();
  const void* untyped_data = literal.untyped_data();

  // Pack sub-byte types into a ZKX storage format.
  if (primitive_util::IsSubByteNonPredType(element_type)) {
    int bit_width = primitive_util::BitWidth(element_type);
    int packed_size_bytes = CeilOfRatio<int64_t>(size_bytes, 8 / bit_width);

    // Use Literal as a storage for packed data as it allocates underlying
    // buffer with correct alignment. Keep it allocated on heap to avoid
    // capturing stack address that will be invalidated by a move below.
    auto packed = std::make_unique<Literal>(
        ShapeUtil::MakeShape(U8, {packed_size_bytes}));

    PackIntN(
        bit_width,
        absl::MakeSpan(reinterpret_cast<const char*>(untyped_data), size_bytes),
        absl::MakeSpan(reinterpret_cast<char*>(packed->untyped_data()),
                       packed->size_bytes()));

    return CpuExecutable::ConstantAllocation{index, std::move(packed)};
  }

  // Create a constant allocation from the literal's untyped data.
  return CpuExecutable::ConstantAllocation{
      index, absl::Span<const uint8_t>(
                 reinterpret_cast<const uint8_t*>(untyped_data), size_bytes)};
}

// Creates a vector of constant allocations from the given buffer assignment.
absl::StatusOr<std::vector<CpuExecutable::ConstantAllocation>>
CreateConstantAllocations(const BufferAssignment& assignment) {
  std::vector<CpuExecutable::ConstantAllocation> constants;

  for (const BufferAllocation& allocation : assignment.Allocations()) {
    if (!allocation.is_constant()) {
      continue;
    }

    // Find the constant instruction defining the value for allocation.
    HloInstruction* const_instr = nullptr;
    for (const auto& [value, _] : allocation.assigned_buffers()) {
      // Multiple aliasing instructions can share the allocation, we need to
      // find the original constant instruction that defines the value.
      if (value->instruction()->opcode() == HloOpcode::kConstant) {
        if (const_instr != nullptr) {
          return absl::InternalError(
              absl::StrCat("Multiple constant instructions define buffer ",
                           allocation.ToString()));
        }
        const_instr = value->instruction();
      }
    }
    if (const_instr == nullptr) {
      return absl::InternalError(
          absl::StrCat("Could not find constant instruction defining buffer ",
                       allocation.ToString()));
    }

    VLOG(3) << "Create constant allocation for index " << allocation.index()
            << " from constant literal " << const_instr->name()
            << "; shape=" << const_instr->literal().shape();
    TF_ASSIGN_OR_RETURN(constants.emplace_back(),
                        LiteralToConstantAllocation(allocation.index(),
                                                    const_instr->literal()));
  }

  return constants;
}

// Removes unused globals and function declarations from the LLVM module.
//
// After splitting LLVM module into multiple parts, we end up with unused
// symbols in each part: external globals and function declarations. We don't
// support linking across modules added to SimpleOrcJIT, and we don't need it,
// because we never construct LLVM IR that might require cross-module linking,
// so we can just remove unused symbols from each part.
void RemoveUnusedSymbols(llvm::Module& module) {
  llvm::SmallVector<llvm::GlobalVariable*> unused_globals;
  llvm::SmallVector<llvm::Function*> unused_functions;

  for (llvm::GlobalVariable& gv : module.globals()) {
    if (gv.use_empty()) unused_globals.push_back(&gv);
  }
  for (llvm::Function& f : module.functions()) {
    if (f.isDeclaration() && f.use_empty()) unused_functions.push_back(&f);
  }

  for (llvm::GlobalVariable* gv : unused_globals) {
    module.eraseGlobalVariable(gv);
  }
  for (llvm::Function* f : unused_functions) {
    f->eraseFromParent();
  }
}

// Clones a ThreadSafeModule from the given LLVM module in a new LLVM context.
//
// To enable parallel compilation, each LLVM module has to be owned by a
// separate LLVM context. We take each part of the original module after a
// split, and clone it into a new LLVM context.
llvm::orc::ThreadSafeModule CloneAsThreadSafeModule(
    int64_t part, std::unique_ptr<llvm::Module> module) {
  tsl::profiler::TraceMe trace([&] {
    return tsl::profiler::TraceMeEncode("CpuCompiler::CloneAsThreadSafeModule",
                                        {{"part", part}});
  });

  // There is no way to clone a module from one context to another, so we need
  // to serialize the module to bitcode and parse it back into the new context.
  llvm::SmallString<0> bc;
  llvm::raw_svector_ostream bcos(bc);
  llvm::WriteBitcodeToFile(*module, bcos);

  // Parse module back into its own LLVM context.
  auto clone_context = std::make_unique<llvm::LLVMContext>();
  auto clone_module = llvm::parseBitcodeFile(
      llvm::MemoryBufferRef(
          llvm::StringRef(bc.data(), bc.size()),
          absl::StrFormat("%s_part_%02d", kZkxModuleIdentifier, part)),
      *clone_context);

  return llvm::orc::ThreadSafeModule(std::move(*clone_module),
                                     std::move(clone_context));
}

}  // namespace

se::Platform::Id CpuAotCompilationOptions::PlatformId() const {
  return se::host::kHostPlatformId;
}

absl::StatusOr<std::vector<std::unique_ptr<Executable>>> CpuCompiler::Compile(
    std::unique_ptr<HloModuleGroup> module_group,
    std::vector<std::vector<se::StreamExecutor*>> executors,
    const CompileOptions& options) {
  for (const std::vector<se::StreamExecutor*>& se_vector : executors) {
    if (se_vector.size() != 1) {
      return absl::UnimplementedError(
          "Model partitioning not implemented for the CPU compiler");
    }
  }
  return LlvmCompiler::Compile(std::move(module_group), executors, options);
}

absl::Status CpuCompiler::RunHloPassesThroughLayoutAssn(
    HloModule* module, bool is_aot_compile,
    TargetMachineFeatures* target_machine_features) {
  HloPassPipeline pipeline("HLO passes through layout assignment");
  pipeline.AddPass<ScatterExpander>(ScatterExpander::kEliminateAllScatters);

  return pipeline.Run(module).status();
}

absl::Status CpuCompiler::RunHloPassesAfterLayoutAssn(
    HloModule* module, bool is_aot_compile,
    TargetMachineFeatures* target_machine_features,
    const CompileOptions& compile_options) {
  HloPassPipeline pipeline("HLO passes after layout assignment");

  // Add a fusion pass now that layout assignment is done.
  pipeline.AddPass<CpuInstructionFusion>();

  // Copy insertion should be performed immediately before IR emission to
  // avoid inserting unnecessary copies (later pass adds an instruction which
  // materializes the value) or missing a necessary copy (later pass removes
  // an instruction which materializes a value). DCE must be run immediately
  // before (and sometime after) copy insertion, to avoid dead code from
  // interfering with the rewrites.
  pipeline.AddPass<HloDCE>();
  // TODO(chokobole): Uncomment this. Dependency: OptimizeInputOutputBufferAlias
  // pipeline.AddPass<OptimizeInputOutputBufferAlias>(true);

  // If enabled we'll use more precise region based analysis for copy removal.
  if (module->config()
          .debug_options()
          .zkx_cpu_copy_insertion_use_region_analysis()) {
    pipeline.AddPass<CopyInsertion>(
        /*can_share_buffer=*/nullptr,
        /*use_region_based_live_range_analysis=*/-1);
  } else {
    pipeline.AddPass<CopyInsertion>();
  }

  pipeline.AddPass<HloDCE>();
  return pipeline.Run(module).status();
}

absl::Status CpuCompiler::RunHloPasses(HloModule* module, bool is_aot_compile,
                                       llvm::TargetMachine* target_machine,
                                       const CompileOptions& compile_options) {
  TargetMachineFeatures target_machine_features(target_machine);
  TF_RETURN_IF_ERROR(RunHloPassesThroughLayoutAssn(module, is_aot_compile,
                                                   &target_machine_features));

  return RunHloPassesAfterLayoutAssn(module, is_aot_compile,
                                     &target_machine_features, compile_options);
}

absl::StatusOr<std::unique_ptr<HloModule>> CpuCompiler::RunHloPasses(
    std::unique_ptr<HloModule> module, se::StreamExecutor* /*stream_exec*/,
    const CompileOptions& options) {
  const HloModuleConfig& config = module->config();

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<llvm::TargetMachine> jit_target_machine,
      JitCompiler::InferTargetMachine(
          llvm::TargetOptions(), IrCompiler::GetCodeGenOptLevel(config),
          CpuFeatureFromString(config.debug_options().zkx_cpu_max_isa())));

  TF_RETURN_IF_ERROR(RunHloPasses(module.get(), /*is_aot_compile=*/false,
                                  jit_target_machine.get(),
                                  /*compile_options=*/options));
  return std::move(module);
}

absl::StatusOr<std::unique_ptr<CpuExecutable>>
CpuCompiler::CompileCpuExecutable(std::unique_ptr<HloModule> module) {
  tsl::profiler::TraceMe trace([&] {
    return tsl::profiler::TraceMeEncode("CpuCompiler::CompileCpuExecutable",
                                        {{"name", module->name()}});
  });

  ModuleHook pre_optimization_ir_hook;
  ModuleHook post_optimization_ir_hook;
  std::tie(pre_optimization_ir_hook, post_optimization_ir_hook) =
      GetIRModuleHooks(*module, user_pre_optimization_hook_,
                       user_post_optimization_hook_);

  // Compile must be thread-safe so create a new LLVM context for the module.
  mlir::DialectRegistry registry;
  auto mlir_context = std::make_unique<mlir::MLIRContext>(registry);
  auto llvm_context = std::make_unique<llvm::LLVMContext>();
  auto llvm_module =
      std::make_unique<llvm::Module>(kZkxModuleIdentifier, *llvm_context);

  const DebugOptions& debug_options = module->config().debug_options();

  // We collect compiled object files (machine code) so we can export
  // CpuExecutable to an AOT compilation result.
  std::vector<std::string> obj_files;

  // We split LLVM module and distribute it across separate DyLibs to enable
  // parallel compilation at run time.
  size_t parallel_codegen_split_count =
      debug_options.zkx_cpu_parallel_codegen_split_count();
  VlogMaxIsa(debug_options.zkx_cpu_max_isa());

  const HloModuleConfig& config = module->config();

  // Options for compiling LLVM IR to machine code.
  IrCompiler::Options ir_compiler_options{
      /*optimization_level=*/IrCompiler::GetCodeGenOptLevel(config),
      /*optimize_for_size=*/options::OptimizeForSizeRequested(config),
      /*disable_expensive_passes=*/
      debug_options.zkx_llvm_disable_expensive_passes(),
      /*slp_vectorizer_disabled=*/options::SlpVectorizerDisabled(config),
      /*disable_loop_unrolling=*/options::DisableLoopUnrolling(config),
  };

  // Compiler hooks to intercept compiled LLVM IR modules.
  IrCompiler::CompilationHooks ir_compiler_hooks{
      pre_optimization_ir_hook,
      post_optimization_ir_hook,
      CreateOrcJITPostCompilationHook(module.get(), &obj_files),
  };

  // Definition generator to link with ZKX:CPU host runtime symbols.
  ExecutionEngine::DefinitionGenerator definition_generator =
      [](const llvm::DataLayout& data_layout) {
        return std::make_unique<RuntimeSymbolGenerator>(data_layout);
      };

  // Options for orchestrating the JIT compilation process.
  JitCompiler::Options jit_compiler_options{
      std::move(ir_compiler_options),
      std::move(ir_compiler_hooks),
      /*num_dylibs=*/parallel_codegen_split_count,
      /*definition_generator=*/std::move(definition_generator),
      /*max_cpu_isa=*/CpuFeatureFromString(debug_options.zkx_cpu_max_isa()),
  };

  TF_ASSIGN_OR_RETURN(JitCompiler jit_compiler,
                      JitCompiler::Create(llvm::TargetOptions(),
                                          std::move(jit_compiler_options),
                                          GetCompilationTaskRunner()));

  // TODO(chokobole): Uncomment this. Dependency: CreateHloProfilingArtifacts
  // absl::flat_hash_map<const HloInstruction*, int64_t>
  //     instruction_to_profile_idx;
  // absl::flat_hash_map<const HloComputation*, int64_t>
  //     computation_to_profile_idx;
  // std::unique_ptr<HloProfileIndexMap> hlo_profile_index_map;
  // std::unique_ptr<HloProfilePrinterData> hlo_profile_printer_data;
  // if (module->config().hlo_profiling_enabled()) {
  //   TF_RETURN_IF_ERROR(CreateHloProfilingArtifacts(
  //       *module, &instruction_to_profile_idx, &computation_to_profile_idx,
  //       &hlo_profile_index_map, &hlo_profile_printer_data));
  // }

  // Cache these flags here since we'll want to access them after the module's
  // ownership is std::moved.
  // clang-format off
  // TODO(chokobole): Uncomment this. Dependency: DebugOptions::zkx_embed_ir_in_executable
  // clang-format on
  // const bool embed_ir_in_executable =
  //     debug_options.zkx_embed_ir_in_executable();

  TF_ASSIGN_OR_RETURN(HloSchedule schedule, CreateHloSchedule(*module));
  TF_RETURN_IF_ERROR(module->set_schedule(std::move(schedule)));

  TF_ASSIGN_OR_RETURN(std::unique_ptr<BufferAssignment> assignment,
                      CreateBufferAssignment(*module));
  DumpHloModuleIfEnabled(*module, *assignment,
                         absl::StrCat("cpu_", kAfterOptimizationsDumpName));

  // Dump computation proto state and buffer assignment for
  // GetCompiledMemoryStats results.
  auto with_hlo_proto = [&](std::unique_ptr<CpuExecutable> cpu_executable) {
    auto hlo_proto = std::make_unique<HloProto>();
    *hlo_proto->mutable_hlo_module() = cpu_executable->module().ToProto();
    *hlo_proto->mutable_buffer_assignment() =
        cpu_executable->buffer_assignment().ToProto();
    StripPayloadFromLiteralProto(*hlo_proto);
    cpu_executable->set_hlo_proto(std::move(hlo_proto));
    return cpu_executable;
  };

  // Thunk emitter is responsible for building a Thunk sequence that will
  // resolved kernels in the compiled LLVM module and execute them together
  // with Thunks implemented as library calls (e.g. oneDNN or Eigen).
  ThunkEmitter thunk_emitter(assignment.get(), mlir_context.get());
  TF_ASSIGN_OR_RETURN(ThunkSequence thunks,
                      thunk_emitter.EmitEntryComputation(*module));

  TF_RETURN_IF_ERROR(VerifyLlvmModule(*llvm_module));
  for (const auto& [name, module] : thunk_emitter.kernels()) {
    TF_RETURN_IF_ERROR(VerifyLlvmModule(*module.getModuleUnlocked()));
  }

  // We define the number of module parts based on the total number of
  // compiled functions (kernels and comparators) that are called from thunks,
  // and the maximum number of parts that we want to split the module into.
  size_t num_compiled_functions =
      thunk_emitter.kernels().size() + thunk_emitter.comparators().size();
  size_t num_parts =
      std::min(num_compiled_functions, parallel_codegen_split_count);

  if (HasLargeConstants(*llvm_module)) {
    VLOG(3) << "Skip parallel compilation due to large constants";
    num_parts = 1;
  }

  if (num_parts > 1) {
    VLOG(3) << "Split LLVM module into " << num_parts
            << " parts before codegen to enable parallel compilation"
            << " (max split count: " << parallel_codegen_split_count << ")";

    tsl::profiler::TraceMe trace([&] {
      return tsl::profiler::TraceMeEncode("SplitModule",
                                          {{"num_parts", num_parts}});
    });

    llvm::SplitModule(
        *llvm_module, num_parts,
        [&, n = 0](std::unique_ptr<llvm::Module> llvm_module_part) mutable {
          // Collect symbols that are compiled in this LLVM module part.
          RemoveUnusedSymbols(*llvm_module_part);
          // clang-format off
          // TODO(chokobole): Uncomment this. Dependency: CollectCompiledSymbolsPart
          // clang-format on
          // compiled_parts.push_back(
          //     CollectCompiledSymbolsPart(ir_emitter2, *llvm_module_part));

          // Clone LLVM module part into its own thread safe context.
          auto tsm = CloneAsThreadSafeModule(n, std::move(llvm_module_part));
          CHECK_OK(jit_compiler.AddModule(std::move(tsm), /*dylib_index=*/n++));
        },
        /*PreserveLocals=*/true, /*RoundRobin=*/true);

    // Free resources used by the original LLVM module.
    llvm_module.reset();
    llvm_context.reset();

  } else {
    VLOG(3) << "Compile LLVM module without splitting (max split count: "
            << parallel_codegen_split_count << ")";
    // TODO(chokobole): Uncomment this. Dependency: CollectCompiledSymbolsPart
    // compiled_parts.push_back(
    //     CollectCompiledSymbolsPart(ir_emitter2, *llvm_module));
    CHECK_OK(jit_compiler.AddModule(llvm::orc::ThreadSafeModule(
        std::move(llvm_module), std::move(llvm_context))));
  }

  // Collect compiled symbols from all LLVM module parts.
  std::vector<FunctionLibrary::Symbol> compiled_symbols;

  absl::flat_hash_map<FunctionLibrary::TypeId, SymbolProto::FunctionTypeId>
      symbol_type_id_to_function_type_id;
  for (auto& [name, module] : thunk_emitter.kernels()) {
    compiled_symbols.push_back(
        FunctionLibrary::Sym<FunctionLibrary::Kernel>(name));
    symbol_type_id_to_function_type_id.emplace(compiled_symbols.back().type_id,
                                               SymbolProto::KERNEL);
    // TODO(chokobole): Uncomment this. Dependency: kernel_dylib_index
    // CHECK_OK(jit_compiler.AddModule(std::move(module),
    // kernel_dylib_index));
    CHECK_OK(jit_compiler.AddModule(std::move(module)));
    // Simply roundrobin the kernel dylibs
    // kernel_dylib_index = (kernel_dylib_index + 1) % num_parts;
  }
  for (auto& [name, module] : thunk_emitter.comparators()) {
    compiled_symbols.push_back(
        FunctionLibrary::Sym<FunctionLibrary::Comparator>(name));
    symbol_type_id_to_function_type_id.emplace(compiled_symbols.back().type_id,
                                               SymbolProto::COMPARATOR);

    CHECK_OK(jit_compiler.AddModule(std::move(module)));
  }

  TF_ASSIGN_OR_RETURN(std::unique_ptr<FunctionLibrary> function_library,
                      std::move(jit_compiler).Compile(compiled_symbols));

  // Create constant allocations from the buffer assignment.
  TF_ASSIGN_OR_RETURN(std::vector<CpuExecutable::ConstantAllocation> constants,
                      CreateConstantAllocations(*assignment));

  TF_ASSIGN_OR_RETURN(
      auto cpu_executable,
      CpuExecutable::Create(std::move(function_library), std::move(assignment),
                            std::move(module), std::move(thunks),
                            std::move(constants)));

  // Save object files to be able to export them to AOT compilation
  // result.
  cpu_executable->set_obj_files(std::move(obj_files));

  // Save compiled symbols to be able to export them to AOT compilation
  // result.
  cpu_executable->set_compiled_symbols(std::move(compiled_symbols));

  // Save mapping between symbol type id and function type id to be able to
  // export them to AOT compilation result.
  cpu_executable->set_symbol_type_id_to_function_type_id(
      symbol_type_id_to_function_type_id);

  return with_hlo_proto(std::move(cpu_executable));
  // TODO(chokobole): Implement other branch.
  // if (!module->config().debug_options().xla_cpu_use_thunk_runtime()) {
}

absl::StatusOr<std::unique_ptr<Executable>> CpuCompiler::RunBackend(
    std::unique_ptr<HloModule> module,
    [[maybe_unused]] se::StreamExecutor* executor,
    [[maybe_unused]] const CompileOptions& options) {
  tsl::profiler::TraceMe trace([&] {
    return tsl::profiler::TraceMeEncode("CpuCompiler::RunBackend",
                                        {{"name", module->name()}});
  });

  VLOG(1) << "Compiling: " << module->name();
  // TODO(chokobole): Uncomment this. Dependency: RecordCpuCompilerStacktrace
  // RecordCpuCompilerStacktrace();
  // TODO(chokobole): Uncomment this. Dependency: XLA_SCOPED_LOGGING_TIMER
  // XLA_SCOPED_LOGGING_TIMER(
  //     absl::StrFormat("Compiling [%s] for CPU using JIT", module->name()));

  // TODO(chokobole): Uncomment this. Dependency: SlowCompilationAlarm
  // std::string slow_compilation_msg =
  //     absl::StrCat("Compiling module ", module->name());
  // auto slow_compile_alarm = SlowCompilationAlarm(slow_compilation_msg);
  auto llvm_options = llvm_ir::ExtractZkxBackendExtraOptions(
      module->config().debug_options().zkx_backend_extra_options());
  llvm_ir::LLVMCommandLineOptionsLock llvm_lock(llvm_options);

  std::unique_ptr<CpuExecutable> cpu_executable;
  TF_ASSIGN_OR_RETURN(cpu_executable, CompileCpuExecutable(std::move(module)));

  cpu_executable->set_debug_info(
      cpu_executable->buffer_assignment().StatsString(
          /*report_total_fragmentation=*/true));
  VLOG(1) << "Compilation finished";
  return std::unique_ptr<CpuExecutable>(std::move(cpu_executable));
}

se::Platform::Id CpuCompiler::PlatformId() const {
  return se::host::kHostPlatformId;
}

namespace {

// This is a result of exporting JIT compiled CpuExecutable to AOT compilation
// result that can be saved on disk and shipped over the wire.
class CpuExecutableAotCompilationResult : public AotCompilationResult {
 public:
  static absl::StatusOr<std::unique_ptr<CpuExecutableAotCompilationResult>>
  Create(const HloModule* hlo_module, const BufferAssignment* buffer_assignment,
         std::string_view function_name, std::vector<std::string> obj_files,
         std::vector<SymbolProto> symbols, const ThunkSequence* thunks,
         CompilationResultProto::ObjFileKind obj_file_kind) {
    std::optional<ThunkSequenceProto> thunk_proto;

    if (thunks != nullptr) {
      ThunkSequenceSerDesProtobuf thunk_sequence_serdes(
          &buffer_assignment->Allocations());
      TF_ASSIGN_OR_RETURN(thunk_proto, thunk_sequence_serdes.ToProto(*thunks));
    }

    return absl::WrapUnique(new CpuExecutableAotCompilationResult(
        hlo_module, buffer_assignment, function_name, std::move(obj_files),
        std::move(symbols), thunk_proto, obj_file_kind));
  }

  absl::StatusOr<std::string> SerializeAsString() const override {
    return proto_.SerializeAsString();
  }

  static absl::StatusOr<std::unique_ptr<CpuExecutableAotCompilationResult>>
  FromString(const std::string& serialized) {
    CompilationResultProto proto;
    if (!proto.ParseFromString(serialized)) {
      return absl::InternalError(
          "Failed to parse serialized CpuExecutableAotCompilationResult.");
    }

    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<HloModule> module,
        HloModule::CreateFromProtoWithConfig(proto.hlo_module()));

    return std::unique_ptr<CpuExecutableAotCompilationResult>(
        new CpuExecutableAotCompilationResult(proto, std::move(module)));
  }

  absl::StatusOr<std::unique_ptr<Executable>> LoadExecutable(
      Compiler* compiler, const se::StreamExecutor* stream_exec) const override;

  const HloModule* optimized_module() const override { return module_.get(); }

  std::unique_ptr<HloModule> consume_optimized_module() override {
    return std::move(module_);
  }

 private:
  CpuExecutableAotCompilationResult(
      const HloModule* hlo_module, const BufferAssignment* buffer_assignment,
      std::string_view function_name, std::vector<std::string> obj_files,
      std::vector<SymbolProto> symbols,
      const std::optional<ThunkSequenceProto>& thunks,
      CompilationResultProto::ObjFileKind obj_file_kind) {
    *proto_.mutable_hlo_module()->mutable_hlo_module() = hlo_module->ToProto();
    *proto_.mutable_hlo_module()->mutable_config() =
        hlo_module->config().ToProto();
    *proto_.mutable_buffer_assignment() = buffer_assignment->ToProto();
    proto_.set_entry_function_name(std::string(function_name));
    for (std::string& obj_file : obj_files) {
      proto_.add_obj_files(std::move(obj_file));
    }

    for (const auto& symbol : symbols) {
      auto* symbol_proto = proto_.add_compiled_symbols();
      *symbol_proto = symbol;
    }
    proto_.set_obj_files_kind(obj_file_kind);
    module_ = hlo_module->Clone();

    if (thunks.has_value()) {
      ThunkSequenceSerDesProtobuf thunk_sequence_serdes(
          &buffer_assignment->Allocations());
      *proto_.mutable_thunk_sequence() = *thunks;
    }
  }

  explicit CpuExecutableAotCompilationResult(CompilationResultProto proto,
                                             std::unique_ptr<HloModule> module)
      : proto_(std::move(proto)), module_(std::move(module)) {}

  CompilationResultProto proto_;
  std::unique_ptr<HloModule> module_;
};

}  // namespace

absl::StatusOr<std::unique_ptr<Executable>>
CpuExecutableAotCompilationResult::LoadExecutable(
    Compiler* compiler, const se::StreamExecutor* stream_exec) const {
  // Recreate HloModule from proto.
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> module,
      HloModule::CreateFromProtoWithConfig(proto_.hlo_module()));

  VLOG(2) << "Load ZKX:CPU executable for module: " << module->name();

  // Recreate BufferAssignment from proto.
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<BufferAssignment> buffer_assignment,
      BufferAssignment::FromProto(proto_.buffer_assignment(), module.get(),
                                  compiler->BufferSizeBytesFunction(),
                                  /*can_share_buffer=*/nullptr));

  const DebugOptions& debug_options = module->config().debug_options();
  VlogMaxIsa(debug_options.zkx_cpu_max_isa());
  const HloModuleConfig& config = module->config();

  // Infer target machine from the current host CPU.
  IrCompiler::TargetMachineBuilder target_machine_builder =
      JitCompiler::InferTargetMachineBuilder(
          llvm::TargetOptions(), IrCompiler::GetCodeGenOptLevel(config),
          CpuFeatureFromString(debug_options.zkx_cpu_max_isa()));
  TF_ASSIGN_OR_RETURN(auto target_machine, target_machine_builder());

  // Definition generator to link with ZKX:CPU host runtime symbols.
  ExecutionEngine::DefinitionGenerator definition_generator =
      [](const llvm::DataLayout& data_layout) {
        return std::make_unique<RuntimeSymbolGenerator>(data_layout);
      };

  ObjectLoader object_loader(/*num_dylibs=*/1,
                             target_machine->createDataLayout(),
                             definition_generator);

  for (size_t i = 0; i < object_loader.num_dylibs(); ++i) {
    object_loader.dylib(i).value()->addGenerator(
        std::make_unique<RuntimeSymbolGenerator>(
            target_machine->createDataLayout()));
  }

  // We might have a ZKX:CPU executable that has only runtime thunks and
  // doesn't have any corresponding object files, and it's absolutely fine.
  VLOG(2) << "Load ZKX:CPU executable from " << proto_.obj_files_size()
          << " object files; entry_function_name="
          << proto_.entry_function_name();

  size_t obj_file_index = 0;
  for (auto& obj_file : proto_.obj_files()) {
    llvm::StringRef data(obj_file.data(), obj_file.size());
    TF_RETURN_IF_ERROR(
        object_loader.AddObjFile(llvm::MemoryBuffer::getMemBuffer(
            data, absl::StrCat(proto_.entry_function_name(), "_",
                               obj_file_index++))));
  }

  std::unique_ptr<CpuExecutable> cpu_executable;

  CHECK_EQ(proto_.obj_files_kind(), CompilationResultProto::KERNELS);
  ThunkSequenceSerDesProtobuf thunk_sequence_serdes(
      &buffer_assignment->Allocations());
  TF_ASSIGN_OR_RETURN(std::unique_ptr<ThunkSequence> thunks,
                      thunk_sequence_serdes.FromProto(proto_.thunk_sequence()));

  VLOG(3) << "Loaded " << thunks->size() << " thunks.";

  std::vector<FunctionLibrary::Symbol> compiled_symbols;

  for (const auto& symbol_proto : proto_.compiled_symbols()) {
    switch (symbol_proto.function_type_id()) {
      case SymbolProto::KERNEL:
        compiled_symbols.push_back(
            FunctionLibrary::Sym<FunctionLibrary::Kernel>(symbol_proto.name()));
        break;
      case SymbolProto::COMPARATOR:
        compiled_symbols.push_back(
            FunctionLibrary::Sym<FunctionLibrary::Comparator>(
                symbol_proto.name()));
        break;
      default:
        return absl::InternalError(absl::StrFormat(
            "Unknown function type id %s",
            SymbolProto_FunctionTypeId_Name(symbol_proto.function_type_id())));
    }
  }

  VLOG(3) << "Collected " << compiled_symbols.size() << " compiled symbols";
  TF_ASSIGN_OR_RETURN(std::unique_ptr<FunctionLibrary> function_library,
                      std::move(object_loader).Load(compiled_symbols));

  // Create constant allocations from the buffer assignment.
  TF_ASSIGN_OR_RETURN(std::vector<CpuExecutable::ConstantAllocation> constants,
                      CreateConstantAllocations(*buffer_assignment));

  TF_ASSIGN_OR_RETURN(
      cpu_executable,
      CpuExecutable::Create(std::move(function_library),
                            std::move(buffer_assignment), std::move(module),
                            std::move(*thunks), std::move(constants)));

  // Dump computation proto state and buffer assignment for
  // GetCompiledMemoryStats results.
  auto hlo_proto = std::make_unique<HloProto>();
  *hlo_proto->mutable_hlo_module() = cpu_executable->module().ToProto();
  *hlo_proto->mutable_buffer_assignment() =
      cpu_executable->buffer_assignment().ToProto();
  cpu_executable->set_hlo_proto(std::move(hlo_proto));

  return cpu_executable;
}

absl::StatusOr<std::unique_ptr<AotCompilationResult>> CpuCompiler::Export(
    Executable* executable) const {
  auto* cpu_executable = tensorflow::down_cast<CpuExecutable*>(executable);
  if (!cpu_executable)
    return absl::InternalError(
        "Could not downcast Executable to CpuExecutable");

  // Export object files for all dylibs.
  std::vector<std::string> obj_files;
  for (const auto& obj_file : cpu_executable->obj_files()) {
    obj_files.push_back(std::string(obj_file));
  }

  CHECK(cpu_executable->has_thunks());
  auto kind = CompilationResultProto::KERNELS;
  const ThunkSequence* thunk_sequence =
      &cpu_executable->thunks().thunk_sequence();

  std::vector<SymbolProto> compiled_symbols =
      cpu_executable->get_compiled_symbols_proto();

  return CpuExecutableAotCompilationResult::Create(
      &cpu_executable->module(), &cpu_executable->buffer_assignment(),
      cpu_executable->module_name(), std::move(obj_files),
      std::move(compiled_symbols), thunk_sequence, kind);
}

absl::StatusOr<std::unique_ptr<AotCompilationResult>>
CpuCompiler::LoadAotCompilationResult(
    const std::string& serialized_aot_result) {
  return CpuExecutableAotCompilationResult::FromString(serialized_aot_result);
}

absl::StatusOr<HloSchedule> CpuCompiler::CreateHloSchedule(
    const HloModule& hlo_module) const {
  // Select a memory scheduler optimized for concurrency vs minimal memory.
  auto scheduler = hlo_module.config()
                           .debug_options()
                           .zkx_cpu_enable_concurrency_optimized_scheduler()
                       ? BFSMemoryScheduler
                       : DFSMemoryScheduler;

  // Select an order for emitting the HLO instructions for each
  // computation. Using this sequence enables tighter buffer liveness analysis
  // and reduced memory usage (as compared to using `DependencyHloOrdering`).
  return ScheduleModule(&hlo_module, BufferSizeBytesFunction(),
                        ComputationSchedulerToModuleScheduler(scheduler));
}

absl::StatusOr<std::unique_ptr<BufferAssignment>>
CpuCompiler::CreateBufferAssignment(const HloModule& module) const {
  // Run buffer allocation on the HLO graph.
  return BufferAssigner::Run(
      &module, std::make_unique<SequentialHloOrdering>(module.schedule()),
      BufferSizeBytesFunction(), memory_alignment,
      /*allocate_buffers_for_constants=*/true);
}

}  // namespace zkx::cpu
