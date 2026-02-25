/* Copyright 2024 The OpenXLA Authors.
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

#include "zkx/backends/cpu/codegen/jit_compiler.h"

#include <string>
#include <utility>

#include "absl/base/call_once.h"
#include "absl/log/log.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/Orc/SelfExecutorProcessControl.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/TargetParser/Host.h"

#include "xla/tsl/platform/statusor.h"
#include "zkx/backends/cpu/codegen/cpu_features.h"
#include "zkx/backends/cpu/codegen/object_loader.h"

namespace zkx::cpu {

namespace {

// Initialize LLVM the first time `JitCompiler` is created.
void InitializeLLVMTarget() {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
}

absl::once_flag initialize_llvm_flag;

}  // namespace

absl::StatusOr<std::unique_ptr<llvm::TargetMachine>>
JitCompiler::InferTargetMachine(
    const llvm::TargetOptions& target_options, llvm::CodeGenOptLevel opt_level,
    std::optional<tsl::port::CPUFeature> max_cpu_feature) {
  // Detect machine attributes for the target CPU.
  auto result = DetectMachineAttributes(max_cpu_feature);
  llvm::SmallVector<std::string> attrs(result.features.begin(),
                                       result.features.end());

  // If `max_cpu_feature` is newer than the host CPU, we should keep the host
  // CPU name, e.g., we don't want to set the target CPU to Skylake when we are
  // on a Broadwell host.
  std::string_view cpu = result.num_filtered_features
                             ? CpuTargetFromMaxFeature(*max_cpu_feature)
                             : std::string_view(llvm::sys::getHostCPUName());

  absl::call_once(initialize_llvm_flag, InitializeLLVMTarget);
  std::unique_ptr<llvm::TargetMachine> target_machine(
      llvm::EngineBuilder()
          .setTargetOptions(target_options)
          .setOptLevel(opt_level)
          .selectTarget(
              /*TargetTriple=*/llvm::Triple(), /*MArch=*/"",
              /*MCPU=*/cpu,
              /*MAttrs=*/attrs));

  if (target_machine == nullptr) {
    return absl::InternalError(
        absl::StrFormat("Failed to create target machine for CPU %s", cpu));
  }

  return std::move(target_machine);
}

IrCompiler::TargetMachineBuilder JitCompiler::InferTargetMachineBuilder(
    const llvm::TargetOptions& target_options, llvm::CodeGenOptLevel opt_level,
    std::optional<tsl::port::CPUFeature> max_cpu_feature) {
  return [target_options, opt_level, max_cpu_feature] {
    return InferTargetMachine(target_options, opt_level, max_cpu_feature);
  };
}

absl::StatusOr<JitCompiler> JitCompiler::Create(
    llvm::TargetOptions target_options, Options options,
    TaskRunner task_runner) {
  absl::call_once(initialize_llvm_flag, InitializeLLVMTarget);

  // Infer target machine from the current host CPU.
  IrCompiler::TargetMachineBuilder target_machine_builder =
      InferTargetMachineBuilder(std::move(target_options),
                                options.ir_compiler_options.opt_level,
                                options.max_cpu_feature);
  TF_ASSIGN_OR_RETURN(auto target_machine, target_machine_builder());

  // Dispatch compilation tasks using the provided task runner.
  auto task_dispatcher =
      std::make_unique<TaskDispatcher>(std::move(task_runner));
  TaskDispatcher* task_dispatcher_ptr = task_dispatcher.get();

  // Create a concrete epc for this process (cannot construct epc directly).
  auto epc = llvm::orc::SelfExecutorProcessControl::Create(
      /*SSP=*/nullptr, std::move(task_dispatcher));
  if (!epc) {
    // Convert LLVM Error -> absl::Status
    auto msg = llvm::toString(epc.takeError());
    return absl::InternalError(
        absl::StrCat("Failed to create SelfExecutorProcessControl: ", msg));
  }

  // LLVM execution session that holds JIT-compiled functions.
  auto execution_session =
      std::make_unique<llvm::orc::ExecutionSession>(std::move(*epc));

  execution_session->setErrorReporter([](llvm::Error err) {
    LOG(ERROR) << "LLVM compilation error: " << llvm::toString(std::move(err));
  });

  // Create an instance of IrCompiler for lowering LLVM modules to machine code.
  auto ir_compiler = std::make_unique<IrCompiler>(
      target_machine_builder, std::move(options.ir_compiler_options),
      std::move(options.ir_compiler_hooks));

  return JitCompiler(
      std::move(target_machine_builder), std::move(target_machine),
      task_dispatcher_ptr, std::move(execution_session), std::move(ir_compiler),
      options.num_dylibs, std::move(options.definition_generator));
}

namespace {

std::unique_ptr<llvm::orc::IRCompileLayer> CreateCompileLayer(
    llvm::orc::ExecutionSession& execution_session,
    llvm::orc::RTDyldObjectLinkingLayer& object_layer,
    std::unique_ptr<IrCompiler> ir_compiler) {
  return std::make_unique<llvm::orc::IRCompileLayer>(
      execution_session, object_layer, std::move(ir_compiler));
}

}  // namespace

JitCompiler::JitCompiler(
    IrCompiler::TargetMachineBuilder target_machine_builder,
    std::shared_ptr<llvm::TargetMachine> target_machine,
    TaskDispatcher* task_dispatcher,
    std::unique_ptr<llvm::orc::ExecutionSession> execution_session,
    std::unique_ptr<IrCompiler> ir_compiler, size_t num_dylibs,
    ExecutionEngine::DefinitionGenerator definition_generator)
    : target_machine_builder_(std::move(target_machine_builder)),
      target_machine_(std::move(target_machine)),
      task_dispatcher_(task_dispatcher),
      execution_engine_(std::make_unique<ExecutionEngine>(
          std::move(execution_session), target_machine_->createDataLayout(),
          definition_generator)),
      compile_layer_(CreateCompileLayer(*execution_engine_->execution_session(),
                                        *execution_engine_->object_layer(),
                                        std::move(ir_compiler))) {
  execution_engine_->AllocateDylibs(num_dylibs);
  execution_engine_->RegisterJITEventListeners();

  if (target_machine_->getTargetTriple().isOSBinFormatCOFF()) {
    execution_engine_->SetObjectLayerFlags();
  }
}

JitCompiler::~JitCompiler() = default;

absl::Status JitCompiler::AddModule(llvm::orc::ThreadSafeModule module,
                                    size_t dylib_index) {
  // Set up module for codegen for the target machine at hand.
  module.withModuleDo([&](llvm::Module& m) {
    m.setDataLayout(target_machine_->createDataLayout());
    m.setTargetTriple(target_machine_->getTargetTriple());
  });

  // Add module to the selected dynamic library.
  TF_ASSIGN_OR_RETURN(llvm::orc::JITDylib * dylib,
                      execution_engine_->dylib(dylib_index));
  if (auto err = compile_layer_->add(*dylib, std::move(module))) {
    return absl::InternalError(
        absl::StrFormat("Failed to add module to dylib %d: %s", dylib_index,
                        llvm::toString(std::move(err))));
  }

  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<FunctionLibrary>> JitCompiler::Compile(
    absl::Span<const Symbol> symbols) && {
  // TODO(chokobole): Uncomment this. Dependency: Tracer
  // TraceMe trace([&] {
  //   return TraceMeEncode("JitCompiler::Compile",
  //                        {{"num_symbols", symbols.size()}});
  // });
  ObjectLoader object_loader(std::move(execution_engine_));
  llvm::DataLayout data_layout = target_machine_->createDataLayout();
  TF_ASSIGN_OR_RETURN(auto symbol_map, object_loader.LookupSymbols(symbols));
  // Wait for all compilation tasks to finish.
  task_dispatcher_->shutdown();
  return std::move(object_loader)
      .CreateFunctionLibrary(std::move(symbols), symbol_map);
}

JitCompiler::TaskDispatcher::TaskDispatcher(TaskRunner task_runner)
    : task_runner_(std::move(task_runner)) {}

JitCompiler::TaskDispatcher::~TaskDispatcher() { shutdown(); }

void JitCompiler::TaskDispatcher::dispatch(
    std::unique_ptr<llvm::orc::Task> task) {
  // Dispatch task in the current thread if no task runner is provided.
  if (task_runner_ == nullptr) {
    task->run();
    return;
  }

  // Dispatch task using user-provided task runner.
  absl::MutexLock lock(&mu_);
  ++num_dispatched_tasks_;

  task_runner_([this, task = std::shared_ptr<llvm::orc::Task>(
                          std::move(task))]() mutable {
    // TODO(chokobole): Uncomment this. Dependency: Tracer
    // TraceMe trace("TaskDispatcher::dispatch");

    // We run and explicitly destroy the task before decrementing the counter
    // and notifying the condition variable, to ensure that the task is fully
    // executed and cleaned up before task dispatcher shut down.
    task->run();
    task.reset();

    absl::MutexLock lock(&mu_);
    --num_dispatched_tasks_;
  });
}

void JitCompiler::TaskDispatcher::shutdown() {
  auto all_tasks_finished = [this]() ABSL_SHARED_LOCKS_REQUIRED(mu_) {
    return num_dispatched_tasks_ == 0;
  };
  absl::MutexLock lock(&mu_, absl::Condition(&all_tasks_finished));
}

}  // namespace zkx::cpu
