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

#include "zkx/service/compiler.h"

#include "absl/base/const_init.h"
#include "absl/debugging/leak_check.h"
#include "absl/log/check.h"

#include "zkx/debug_options_flags.h"

namespace zkx {

// static
ABSL_CONST_INIT absl::Mutex Compiler::platform_compiler_mutex_(
    absl::kConstInit);

Compiler::TargetConfig::TargetConfig(se::StreamExecutor* s)
    : device_description(s->GetDeviceDescription()),
      platform_name(s->GetPlatform()->Name()),
      device_description_str(s->GetDeviceDescription().name()) {}

Compiler::TargetConfig::TargetConfig(const se::GpuTargetConfigProto& proto)
    : device_description({proto.gpu_device_info()}),
      platform_name(proto.platform_name()),
      device_description_str(proto.device_description_str()) {}

se::GpuTargetConfigProto Compiler::TargetConfig::ToProto() const {
  stream_executor::GpuTargetConfigProto proto;
  *proto.mutable_gpu_device_info() = device_description.ToGpuProto();
  proto.set_platform_name(platform_name);
  proto.set_device_description_str(device_description_str);
  return proto;
}

std::vector<std::unique_ptr<google::protobuf::Message>>
Compiler::ComputeBackendConfigs(const HloInstruction& hlo,
                                se::StreamExecutor* executor) const {
  CHECK(executor != nullptr);
  return {};
}

std::unique_ptr<google::protobuf::Message>
Compiler::ComputeDefaultBackendConfig(const HloInstruction& hlo,
                                      se::StreamExecutor* executor) const {
  CHECK(executor != nullptr);
  return nullptr;
}

// Define a default version where metadata is not used.
absl::StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
Compiler::CompileAheadOfTime(
    std::unique_ptr<HloModuleGroup> module_group,
    const AotCompilationOptions& options,
    std::unique_ptr<AotCompilationMetadata>* metadata) {
  if (metadata != nullptr) {
    return absl::UnimplementedError(
        "Populating AotCompilationMetadata is not implemented on this "
        "compiler.");
  }
  return CompileAheadOfTime(std::move(module_group), options);
}

// static
absl::flat_hash_map<se::Platform::Id, Compiler::CompilerFactory>*
Compiler::GetPlatformCompilerFactories() {
  static auto* r = absl::IgnoreLeak(
      new absl::flat_hash_map<se::Platform::Id, CompilerFactory>);
  return r;
}

// static
absl::flat_hash_map<se::Platform::Id, std::unique_ptr<Compiler>>*
Compiler::GetPlatformCompilers() {
  static auto* r = absl::IgnoreLeak(
      new absl::flat_hash_map<se::Platform::Id, std::unique_ptr<Compiler>>);
  return r;
}

// static
void Compiler::RegisterCompilerFactory(
    se::Platform::Id platform_id,
    std::function<std::unique_ptr<Compiler>()> compiler_factory) {
  absl::MutexLock lock(&platform_compiler_mutex_);
  auto* factories = GetPlatformCompilerFactories();
  CHECK(factories->find(platform_id) == factories->end())
      << "Compiler factory already registered for platform";
  (*factories)[platform_id] = std::move(compiler_factory);
}

// static
absl::StatusOr<Compiler*> Compiler::GetForPlatform(
    const se::Platform* platform) {
  absl::MutexLock lock(&platform_compiler_mutex_);

  auto* compilers = GetPlatformCompilers();
  // See if we already instantiated a compiler for this platform.
  {
    auto it = compilers->find(platform->id());
    if (it != compilers->end()) {
      return it->second.get();
    }

    // If not, we just fall through to try to create one with a registered
    // factory.
  }

  auto* factories = GetPlatformCompilerFactories();
  auto it = factories->find(platform->id());
  if (it == factories->end()) {
    return absl::NotFoundError(absl::StrFormat(
        "could not find registered compiler for platform %s -- was support for "
        "that platform linked in?",
        platform->Name()));
  }

  // And then we invoke the factory, placing the result into the mapping.
  compilers->insert(std::make_pair(platform->id(), it->second()));
  return compilers->at(platform->id()).get();
}

AotCompilationOptions::AotCompilationOptions()
    : debug_options_(GetDebugOptionsFromFlags()) {}

AotCompilationOptions::AotCompilationOptions(se::Platform::Id platform_id)
    : platform_id_(platform_id), debug_options_(GetDebugOptionsFromFlags()) {}

}  // namespace zkx
